"""
Metrics

Update metrics for displaying in TensorBoard during training or evaluation after
training

Usage during training (logging to a log file for TensorBoard):
    metrics = Metrics(log_dir, source_dataset,
        task_loss_fn, domain_loss_fn, domain_b_data is not None)

    # Evaluate model on a single training batch, update metrics, save to log file
    metrics.train(model, train_data_a, train_data_b, step, train_time)

    # Evaluate model on entire evaluation dataset, update metrics, save to log file
    validation_accuracy = metrics.test(model, eval_data_a, eval_data_b, step)

Usage after training (evaluating but not logging):
    metrics = Metrics(log_dir, source_dataset,
        None, None, domain_b_data is not None)

    # Evaluate on datasets
    metrics.train(model, train_data_a, train_data_b, evaluation=True)
    metrics.test(model, eval_data_a, eval_data_b, evaluation=True)

    # Get the results
    results = metrics.results()
"""
import time
import tensorflow as tf

from absl import flags

from plots import generate_plots
from datasets import inversions

FLAGS = flags.FLAGS

flags.DEFINE_boolean("best_source", False, "Calculate \"best\" model based on source (vs. target) validation data")


class Metrics:
    """
    Handles keeping track of metrics either over one batch or many batch, then
    after all (or just the one) batches are processed, saving this to a log file
    for viewing in TensorBoard.

    Note: enable_compile=True decorates some functions with tf.function, but when
    only evaluating once this can drastically increase the run time. Thus, set
    to enable_compile=False when only evaluating the metrics once.

    Accuracy values:
        accuracy_{domain,task}/{source,target}/{training,validation}
        {auc,precision,recall}_{task,target}/{source,target}/{training,validation}
        accuracy_{task,target}_class_{class1name,...}/{source,target}/{training,validation}
        rates_{task,target}_class_{class1name,...}/{TP,FP,TN,FN}/{source,target}/{training,validation}
    Loss values:
        loss/{total,task,domain}
    """
    def __init__(self, log_dir, source_dataset,
            task_loss, domain_loss, target_domain=True,
            target_classifier=False, enable_compile=True):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.source_dataset = source_dataset
        self.num_classes = source_dataset.num_classes
        self.num_domains = source_dataset.num_domains

        # If num_domains = None, then we're passing a "target" dataset as the source
        # for the upper bound, so we really only have one domain
        # (also in main.py, main_eval.py)
        if self.num_domains is None:
            self.num_domains = 1

        self.datasets = ["training", "validation"]
        self.task_loss = task_loss if task_loss is not None else lambda y_true, y_pred, training: 0
        self.domain_loss = domain_loss if domain_loss is not None else lambda y_true, y_pred: 0
        self.target_domain = target_domain  # whether we have just source or both
        self.has_target_classifier = target_classifier
        self.invertible = self.source_dataset.invert_name is not None

        if not target_domain:
            self.domains = ["source"]
        else:
            self.domains = ["source", "target"]

        if not target_classifier:
            self.classifiers = ["task"]
        else:
            self.classifiers = ["task", "target"]

        # Create all entire-batch metrics
        self.batch_metrics = {dataset: {} for dataset in self.datasets}
        self.map_batch_metrics = {dataset: {} for dataset in self.datasets}
        for domain in self.domains:
            for dataset in self.datasets:
                n = "accuracy_domain/%s/%s"%(domain, dataset)
                self.batch_metrics[dataset][n] = tf.keras.metrics.CategoricalAccuracy(name=n)

                for name in self.classifiers:
                    n = "accuracy_%s/%s/%s"%(name, domain, dataset)
                    self.batch_metrics[dataset][n] = tf.keras.metrics.CategoricalAccuracy(name=n)

                n = "mapping_rmse/to_%s/%s"%(domain, dataset)
                self.map_batch_metrics[dataset][n] = tf.keras.metrics.RootMeanSquaredError(name=n)

                n = "mapping_mse/to_%s/%s"%(domain, dataset)
                self.map_batch_metrics[dataset][n] = tf.keras.metrics.MeanSquaredError(name=n)

                n = "mapping_mae/to_%s/%s"%(domain, dataset)
                self.map_batch_metrics[dataset][n] = tf.keras.metrics.MeanAbsoluteError(name=n)

        for domain in self.domains:
            for dataset in self.datasets:
                for classifier in self.classifiers:
                    n = "auc_%s/%s/%s"%(classifier, domain, dataset)
                    self.batch_metrics[dataset][n] = tf.keras.metrics.AUC(name=n)

                    n = "precision_%s/%s/%s"%(classifier, domain, dataset)
                    self.batch_metrics[dataset][n] = tf.keras.metrics.Precision(name=n)

                    n = "recall_%s/%s/%s"%(classifier, domain, dataset)
                    self.batch_metrics[dataset][n] = tf.keras.metrics.Recall(name=n)

        # Create all per-class metrics
        self.per_class_metrics = {dataset: {} for dataset in self.datasets}
        for i in range(self.num_classes):
            class_name = self.source_dataset.int_to_label(i)

            for domain in self.domains:
                for dataset in self.datasets:
                    for classifier in self.classifiers:
                        n = "accuracy_%s_class_%s/%s/%s"%(classifier, class_name, domain, dataset)
                        self.per_class_metrics[dataset][n] = tf.keras.metrics.Accuracy(name=n)

                        n = "rates_%s_class_%s/TP/%s/%s"%(classifier, class_name, domain, dataset)
                        self.per_class_metrics[dataset][n] = tf.keras.metrics.TruePositives(name=n)

                        n = "rates_%s_class_%s/FP/%s/%s"%(classifier, class_name, domain, dataset)
                        self.per_class_metrics[dataset][n] = tf.keras.metrics.FalsePositives(name=n)

                        n = "rates_%s_class_%s/TN/%s/%s"%(classifier, class_name, domain, dataset)
                        self.per_class_metrics[dataset][n] = tf.keras.metrics.TrueNegatives(name=n)

                        n = "rates_%s_class_%s/FN/%s/%s"%(classifier, class_name, domain, dataset)
                        self.per_class_metrics[dataset][n] = tf.keras.metrics.FalseNegatives(name=n)

        # Losses
        self.loss_total = tf.keras.metrics.Mean(name="loss/total")
        self.loss_task = tf.keras.metrics.Mean(name="loss/task")
        self.loss_domain = tf.keras.metrics.Mean(name="loss/domain")

        # Compile frequent-running functions if the metrics will be updated
        # multiple times
        if enable_compile:
            self._run_single_batch_task_a = tf.function(self._run_single_batch_task_a)
            self._run_single_batch_target_a = tf.function(self._run_single_batch_target_a)
            self._run_single_batch_task_b = tf.function(self._run_single_batch_task_b)
            self._run_single_batch_target_b = tf.function(self._run_single_batch_target_b)

    def _reset_states(self, dataset):
        """ Reset states of all the Keras metrics """
        for _, metric in self.batch_metrics[dataset].items():
            metric.reset_states()

        for _, metric in self.per_class_metrics[dataset].items():
            metric.reset_states()

        for _, metric in self.map_batch_metrics[dataset].items():
            metric.reset_states()

        if dataset == "training":
            self.loss_total.reset_states()
            self.loss_task.reset_states()
            self.loss_domain.reset_states()

    def _process_losses(self, results):
        """ Update loss values """
        _, _, _, _, \
            total_loss, task_loss, domain_loss = results
        self.loss_total(total_loss)
        self.loss_task(task_loss)
        self.loss_domain(domain_loss)

    def _process_batch(self, results, classifier, domain, dataset):
        """ Update metrics for accuracy over entire batch for domain-dataset """
        task_y_true, task_y_pred, domain_y_true, domain_y_pred, \
            _, _, _ = results

        # Since we are now using sparse
        domain_y_true = tf.one_hot(tf.cast(domain_y_true, tf.int32), self.num_domains)

        domain_names = [
            "accuracy_domain/%s/%s",
        ]

        for n in domain_names:
            name = n%(domain, dataset)
            self.batch_metrics[dataset][name](domain_y_true, domain_y_pred)

        task_names = [
            "accuracy_%s/%s/%s",
            "auc_%s/%s/%s",
            "precision_%s/%s/%s",
            "recall_%s/%s/%s",
        ]

        # Since we are now using sparse
        task_y_true = tf.one_hot(tf.cast(task_y_true, tf.int32), self.num_classes)

        for n in task_names:
            name = n%(classifier, domain, dataset)
            self.batch_metrics[dataset][name](task_y_true, task_y_pred)

    def _process_map_batch(self, map_true, map_pred, domain, dataset):
        """ Update metrics for mapping error over entire batch for domain-dataset """
        task_names = [
            "mapping_rmse/to_%s/%s",
            "mapping_mse/to_%s/%s",
            "mapping_mae/to_%s/%s",
        ]

        # Reshape from (batch_size, time_steps, features) to (batch_size, values)
        # since RMSE is for normal single-prediction type of data, like with a
        # class label predicted but a real value. However, we "predict" something
        # for each time step.
        map_true = tf.reshape(map_true, [tf.shape(map_true)[0], -1])
        map_pred = tf.reshape(map_pred, [tf.shape(map_pred)[0], -1])

        for n in task_names:
            name = n%(domain, dataset)
            self.map_batch_metrics[dataset][name](map_true, map_pred)

    def _process_per_class(self, results, classifier, domain, dataset):
        """ Update metrics for accuracy over per-class portions of batch for domain-dataset """
        task_y_true, task_y_pred, _, _, _, _, _ = results
        batch_size = tf.shape(task_y_true)[0]

        # Since we are now using sparse
        task_y_true = tf.one_hot(tf.cast(task_y_true, tf.int32), self.num_classes)

        # If only predicting a single class (using softmax), then look for the
        # max value
        # e.g. [0.2 0.2 0.4 0.2] -> [0 0 1 0]
        per_class_predictions = tf.one_hot(
            tf.argmax(task_y_pred, axis=-1), self.num_classes)

        # List of per-class task metrics to update
        task_names = [
            "accuracy_%s_class_%s/%s/%s",
            "rates_%s_class_%s/TP/%s/%s",
            "rates_%s_class_%s/FP/%s/%s",
            "rates_%s_class_%s/TN/%s/%s",
            "rates_%s_class_%s/FN/%s/%s",
        ]

        for i in range(self.num_classes):
            class_name = self.source_dataset.int_to_label(i)

            # Get ith column (all groundtruth/predictions for ith class)
            y_true = tf.slice(task_y_true, [0, i], [batch_size, 1])  # if not sparse
            y_pred = tf.slice(per_class_predictions, [0, i], [batch_size, 1])

            # For single-class prediction, we want to first isolate which
            # examples in the batch were supposed to be class X. Then, of
            # those, calculate accuracy = correct / total.
            rows_of_class_y = tf.where(tf.equal(y_true, 1))  # i.e. have 1
            acc_y_true = tf.gather(y_true, rows_of_class_y)
            acc_y_pred = tf.gather(y_pred, rows_of_class_y)

            # Update metrics
            for n in task_names:
                name = n%(classifier, class_name, domain, dataset)
                self.per_class_metrics[dataset][name](acc_y_true, acc_y_pred)

    def _write_data(self, step, dataset, eval_time, train_time=None,
            additional_losses=None, log_mapping=False, log_task=False):
        """ Write either the training or validation data """
        assert dataset in self.datasets, "unknown dataset "+str(dataset)

        # Write all the values to the file
        with self.writer.as_default():
            # Mapping evaluation -- but only if the mapping was invertible,
            # otherwise we can't calculate the mapping error.
            if log_mapping and self.invertible:
                for key, metric in self.map_batch_metrics[dataset].items():
                    tf.summary.scalar(key, metric.result(), step=step)

            # Task evaluation
            if log_task:
                for key, metric in self.batch_metrics[dataset].items():
                    tf.summary.scalar(key, metric.result(), step=step)

                for key, metric in self.per_class_metrics[dataset].items():
                    tf.summary.scalar(key, metric.result(), step=step)

                # Only log losses on training data
                if dataset == "training":
                    tf.summary.scalar("loss/total", self.loss_total.result(), step=step)
                    tf.summary.scalar("loss/task", self.loss_task.result(), step=step)
                    tf.summary.scalar("loss/domain", self.loss_domain.result(), step=step)

            # Any other losses
            if additional_losses is not None:
                names, values = additional_losses

                for i, name in enumerate(names):
                    # If TensorFlow string (when using tf.function), get the
                    # value from it
                    if not isinstance(name, str):
                        name = name.numpy().decode("utf-8")

                    tf.summary.scalar("loss/%s"%(name), values[i], step=step)

            # Regardless of mapping/task, log times
            tf.summary.scalar("step_time/metrics/%s"%(dataset), eval_time, step=step)

            if train_time is not None:
                tf.summary.scalar("step_time/%s"%(dataset), train_time, step=step)

        # Make sure we sync to disk
        self.writer.flush()

    def _run_dataset(self, model, mapping_model, data_a, data_b, dataset, target):
        """ Run all the data A/B through the model -- data_a and data_b
        should both be of type tf.data.Dataset """
        if data_a is not None:
            func = self._run_single_batch_target_a if target else self._run_single_batch_task_a

            for x, task_y_true, domain_y_true in data_a:
                func(x, task_y_true, domain_y_true, model, mapping_model, dataset)

        if self.target_domain and data_b is not None:
            func = self._run_single_batch_target_b if target else self._run_single_batch_task_b

            for x, task_y_true, domain_y_true in data_b:
                func(x, task_y_true, domain_y_true, model, mapping_model, dataset)

    def _run_batch(self, model, mapping_model, data_a, data_b, dataset, target):
        """ Run a single batch of A/B data through the model -- data_a and data_b
        should both be a tuple of (x, task_y_true) """
        if data_a is not None:
            func = self._run_single_batch_target_a if target else self._run_single_batch_task_a
            func(*data_a, model, mapping_model, dataset)

        if self.target_domain and data_b is not None:
            func = self._run_single_batch_target_b if target else self._run_single_batch_task_b
            func(*data_b, model, mapping_model, dataset)

    def _run_single_batch(self, x, task_y_true, domain_y_true, model, mapping_model,
            dataset_name, domain_name, target):
        """
        Run a batch of data through the model. Call after_batch() afterwards:
            after_batch([labels_batch_a, task_y_pred, domains_batch_a, domain_y_pred,
                total_loss, task_loss, domain_loss], domain_name, dataset_name)
        """
        assert dataset_name in self.datasets, "unknown dataset "+str(dataset_name)
        assert domain_name in self.domains, "unknown domain "+str(domain_name)

        # If performing mapping (i.e. if we don't pass in mapping_model=None)
        # then if this data is source domain data, we first need to map to the
        # target domain since our classifier is for target-like data.
        if mapping_model is not None:
            if domain_name == "source":  # source -> target
                mapped = mapping_model.map_to_target(x)

                if self.invertible:
                    map_true = inversions.map_to_target[self.source_dataset.invert_name](x)

                # We'll run the task model on the mapped source to target data,
                # so keep the mapped target-like data
                x = mapped
            elif domain_name == "target":  # target -> source
                if self.invertible:
                    mapped = mapping_model.map_to_source(x)
                    map_true = inversions.map_to_source[self.source_dataset.invert_name](x)

            # Process this batch
            if self.invertible:
                map_to_name = "target" if domain_name == "source" else "source"
                self._process_map_batch(map_true, mapped, map_to_name, dataset_name)

        # If performing a task
        if model is not None:
            # Evaluate model on data
            task_y_pred, domain_y_pred, _ = model(x, target=target, training=False, domain=domain_name)

            # Calculate losses
            task_l = self.task_loss(task_y_true, task_y_pred, training=False)

            # For MDAN Smooth ideally we'd grab the loss from the right domain
            # classifier based on which domain the sample is from. However,
            # for now just grab the first classifier's output. TODO
            if isinstance(domain_y_pred, list):
                domain_y_pred = domain_y_pred[0] # TODO

            domain_l = self.domain_loss(domain_y_true, domain_y_pred)
            total_l = task_l + domain_l

            # We'll compute the accuracy based on a binary output, not the logits
            # which we used for the loss function
            domain_y_pred = tf.sigmoid(domain_y_pred)

            # Process this batch
            results = [
                task_y_true, task_y_pred, domain_y_true, domain_y_pred,
                total_l, task_l, domain_l,
            ]

            # Which classifier's task_y_pred are we looking at?
            classifier = "target" if target else "task"

            self._process_batch(results, classifier, domain_name, dataset_name)
            self._process_per_class(results, classifier, domain_name, dataset_name)

            # Only log losses on training data with the task classifier (not target)
            if dataset_name == "training" and not target:
                self._process_losses(results)

    # Compile separate _run_single_batch functions since if we pass in varying
    # values of target=True/False or dataset_name it ends up dying sometimes
    # with the error:
    #   "ValueError: tf.function-decorated function tried to create variables on
    #   non-first call."
    def _run_single_batch_task_a(self, *args, **kwargs):
        return self._run_single_batch(*args, domain_name="source", target=False, **kwargs)

    def _run_single_batch_task_b(self, *args, **kwargs):
        return self._run_single_batch(*args, domain_name="target", target=False, **kwargs)

    def _run_single_batch_target_a(self, *args, **kwargs):
        return self._run_single_batch(*args, domain_name="source", target=True, **kwargs)

    def _run_single_batch_target_b(self, *args, **kwargs):
        return self._run_single_batch(*args, domain_name="target", target=True, **kwargs)

    def train(self, model, mapping_model, non_mapped_data_a, data_a, data_b,
            step=None, train_time=None, additional_losses=None, evaluation=False):
        """
        Call this once after evaluating on the training data for domain A and
        domain B

        Note: leave off step and train_time if evaluation=True and make sure
        data_a and data_b are the entire training datasets rathe than a single
        batch as when evaluation=False.
        """
        dataset = "training"
        self._reset_states(dataset)
        t = time.time()

        if not self.target_domain:
            data_b = None

        # If we're going to calculate mapping error, then we need to pass the
        # non-mapped data so we can map it (now that the model is updated)
        # then calculate the error.
        #
        # Or, if no mapped data is passed (e.g. in evaluation), then use the
        # non-mapped data.
        if self.invertible or data_a is None:
            data_a = non_mapped_data_a
        # Otherwise, use the already-mapped data as is and don't map it again.
        # This ends up being a bit faster since we do one less mapping.
        else:
            mapping_model = None

        # evaluation=True is a tf.data.Dataset, otherwise a single batch
        if evaluation:
            self._run_dataset(model, mapping_model, data_a, data_b, dataset, False)

            if self.has_target_classifier:
                self._run_dataset(model, mapping_model, data_a, data_b, dataset, True)
        else:
            self._run_batch(model, mapping_model, data_a, data_b, dataset, False)

            if self.has_target_classifier:
                self._run_batch(model, mapping_model, data_a, data_b, dataset, True)

        t = time.time() - t

        if not evaluation:
            assert step is not None and train_time is not None, \
                "Must pass step and train_time to train() if evaluation=False"
            step = int(step)
            self._write_data(step, dataset, t, train_time,
                additional_losses=additional_losses,
                log_mapping=mapping_model is not None,
                log_task=model is not None)

    def test(self, model, mapping_model, eval_data_a, eval_data_b, step=None,
            evaluation=False):
        """
        Evaluate the model on domain A/B but batched to make sure we don't run
        out of memory

        Note: leave off step if evaluation=True

        Returns: source task validation accuracy
        """
        dataset = "validation"
        self._reset_states(dataset)
        t = time.time()

        if not self.target_domain:
            eval_data_b = None

        self._run_dataset(model, mapping_model, eval_data_a, eval_data_b, dataset, False)

        if self.has_target_classifier:
            self._run_dataset(model, mapping_model, eval_data_a, eval_data_b, dataset, True)

        # These are metrics only filled out when there's a task model
        if model is not None:
            # We use the validation accuracy to save the best model
            #
            # If best_source then use source validation accuracy (so we never look)
            # at labeled target data. However, as is commonly done, another approach
            # is tuning based on 1000 random labeled target samples.
            if FLAGS.best_source:
                acc = self.batch_metrics["validation"]["accuracy_task/source/validation"]

                if self.has_target_classifier:
                    target_acc = self.batch_metrics["validation"]["accuracy_target/source/validation"]
            else:
                acc = self.batch_metrics["validation"]["accuracy_task/target/validation"]

                if self.has_target_classifier:
                    target_acc = self.batch_metrics["validation"]["accuracy_target/target/validation"]

            validation_accuracy = float(acc.result())

            if self.has_target_classifier:
                target_validation_accuracy = float(target_acc.result())

        t = time.time() - t

        if not evaluation:
            assert step is not None, "Must pass step to test() if evaluation=False"
            step = int(step)
            self._write_data(step, dataset, t,
                log_mapping=mapping_model is not None,
                log_task=model is not None)

        # These validation accuracy values are for the task-based evaluation,
        # otherwise we don't have a "best" model on the task. TODO maybe we could
        # save the model with the lowest mapping error?
        if model is not None:
            if self.has_target_classifier:
                return validation_accuracy, target_validation_accuracy
            else:
                return validation_accuracy, 0
        else:
            return None, None

    def plots(self, model, mapping_model, eval_data_a, eval_data_b, adapt,
            global_step):
        """ Log plots """
        # Get first batch of data
        data_a = next(iter(eval_data_a))

        if not self.target_domain:
            data_b = None
        else:
            data_b = next(iter(eval_data_b))

        # We'll only plot the real plots once since they don't change
        step = int(global_step)
        first_time = step == 1

        # Generate plots
        t = time.time()
        plots = generate_plots(data_a, data_b, model, mapping_model, adapt,
            first_time, self.source_dataset.invert_name)
        t = time.time() - t

        # Write all the values to the file
        with self.writer.as_default():
            for name, plot in plots:
                tf.summary.image(name, plot, step=step)

            tf.summary.scalar("step_time/plots", t, step=step)

        # Make sure we sync to disk
        self.writer.flush()

    def results(self):
        """ Returns one dictionary of all the current metric results (floats) """
        results = {}

        for dataset in self.datasets:
            for key, metric in self.batch_metrics[dataset].items():
                results[key] = float(metric.result())

            for key, metric in self.per_class_metrics[dataset].items():
                results[key] = float(metric.result())

        results["loss/total"] = float(self.loss_total.result())
        results["loss/task"] = float(self.loss_task.result())
        results["loss/domain"] = float(self.loss_domain.result())

        return results
