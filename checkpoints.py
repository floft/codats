"""
Checkpoints

Keep both the latest and the best on validation data

Usage:
    # Create the checkpoint on the data you wish to save and the manager object
    checkpoint = tf.train.Checkpoint(model=model, opt=opt, ...)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)

    # Restore either the latest model with .restore_latest() to resume training
    # or the best model with .restore_best() for evaluation after trining
    checkpoint_manager.restore_latest()

    # During training, save at a particular step and if validation_accuracy is
    # higher than the best previous validation accuracy, then save a new "best"
    # model as well
    checkpoint_manager.save(step, validation_accuracy)
"""
import os
import tensorflow as tf

from absl import flags

from file_utils import get_best_valid, write_best_valid, \
    get_last_int

FLAGS = flags.FLAGS

flags.DEFINE_integer("latest_checkpoints", 1, "Max number of latest checkpoints to keep")
flags.DEFINE_integer("best_checkpoints", 1, "Max number of best checkpoints to keep")


class CheckpointManager:
    """
    Keep both the latest and the best on validation data

    Latest stored in model_dir and best stored in model_dir/best
    Saves the best validation accuracy in log_dir/best_valid_accuracy.txt
    """
    def __init__(self, checkpoint, model_dir, log_dir):
        self.checkpoint = checkpoint
        self.log_dir = log_dir

        # Keep track of the latest for restoring interrupted training
        self.latest_manager = tf.train.CheckpointManager(
            checkpoint, directory=model_dir, max_to_keep=FLAGS.latest_checkpoints)

        # Keeps track of our best model for use after training
        best_model_dir_source = os.path.join(model_dir, "best_source")

        self.best_manager_source = tf.train.CheckpointManager(
            checkpoint, directory=best_model_dir_source,
            max_to_keep=FLAGS.best_checkpoints)

        best_model_dir_target = os.path.join(model_dir, "best_target")
        self.best_manager_target = tf.train.CheckpointManager(
            checkpoint, directory=best_model_dir_target,
            max_to_keep=FLAGS.best_checkpoints)

        # Restore best from file or if no file yet, set it to zero
        self.best_validation_source = get_best_valid(self.log_dir,
            filename="best_valid_accuracy_source.txt")

        self.best_validation_target = get_best_valid(self.log_dir,
            filename="best_valid_accuracy_target.txt")

        # Do we have these checkpoints -- used to verify we were able to load
        # the previous or best checkpoint during evaluation
        self.found_last = len(self.latest_manager.checkpoints) != 0
        self.found_best_source = True
        self.found_best_target = True

        if self.best_validation_source is None:
            self.found_best_source = False
            self.best_validation_source = 0.0

        if self.best_validation_target is None:
            self.found_best_target = False
            self.best_validation_target = 0.0

    def restore_latest(self):
        """ Restore the checkpoint from the latest one """
        self.checkpoint.restore(self.latest_manager.latest_checkpoint).expect_partial()

    def restore_best_source(self):
        """ Restore the checkpoint from the best one on the source valid data """
        # Note: using expect_partial() so we don't get warnings about loading
        # only some of the weights
        self.checkpoint.restore(self.best_manager_source.latest_checkpoint).expect_partial()

    def restore_best_target(self):
        """ Restore the checkpoint from the best one on the target valid data """
        # Note: using expect_partial() so we don't get warnings about loading
        # only some of the weights
        self.checkpoint.restore(self.best_manager_target.latest_checkpoint).expect_partial()

    def latest_step(self):
        """ Return the step number from the latest checkpoint. Returns None if
        no checkpoints. """
        return self._get_step_from_manager(self.latest_manager)

    def best_step_source(self):
        """ Return the step number from the best source checkpoint. Returns None
        if no checkpoints. """
        return self._get_step_from_manager(self.best_manager_source)

    def best_step_target(self):
        """ Return the step number from the best target checkpoint. Returns None
        if no checkpoints. """
        return self._get_step_from_manager(self.best_manager_target)

    def _get_step_from_manager(self, manager):
        # If no checkpoints found
        if len(manager.checkpoints) == 0:
            return None

        # If one is found, the last checkpoint will be a string like
        #   "models/target-foldX-model-debugnum/ckpt-100'
        # and we want to step number at the end, e.g. 100 in this example
        last = manager.checkpoints[-1]  # sorted oldest to newest
        name = os.path.basename(last)
        step = get_last_int(name, only_one=True)

        return step

    def save(self, step, validation_accuracy_source=None,
            validation_accuracy_target=None):
        """ Save the latest model. If validation_accuracy_* specified and higher
        than the previous best, also save this model as the new best one. """
        # Always save the latest
        self.latest_manager.save(checkpoint_number=step)

        # Only save the "best" if it's better than the previous best
        if validation_accuracy_source is not None:
            if validation_accuracy_source > self.best_validation_source \
                    or not self.found_best_source:
                self.best_manager_source.save(checkpoint_number=step)
                self.best_validation_source = validation_accuracy_source
                write_best_valid(self.log_dir,
                    self.best_validation_source,
                    filename="best_valid_accuracy_source.txt")

        if validation_accuracy_target is not None:
            if validation_accuracy_target > self.best_validation_target \
                    or not self.found_best_target:
                self.best_manager_target.save(checkpoint_number=step)
                self.best_validation_target = validation_accuracy_target
                write_best_valid(self.log_dir,
                    self.best_validation_target,
                    filename="best_valid_accuracy_target.txt")
