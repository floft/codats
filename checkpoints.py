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

from file_utils import get_best_valid_accuracy, write_best_valid_accuracy, \
    get_best_target_valid_accuracy, write_best_target_valid_accuracy, \
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
    def __init__(self, checkpoint, model_dir, log_dir, target=False):
        self.checkpoint = checkpoint
        self.log_dir = log_dir
        self.target = target

        # Keep track of the latest for restoring interrupted training
        self.latest_manager = tf.train.CheckpointManager(
            checkpoint, directory=model_dir, max_to_keep=FLAGS.latest_checkpoints)

        # Keeps track of our best model for use after training
        best_model_dir = os.path.join(model_dir, "best")
        self.best_manager = tf.train.CheckpointManager(
            checkpoint, directory=best_model_dir, max_to_keep=FLAGS.best_checkpoints)

        # Keeps track of best model based on target classifier valid accuracy
        if self.target:
            best_target_model_dir = os.path.join(model_dir, "best_target")
            self.best_target_manager = tf.train.CheckpointManager(
                checkpoint, directory=best_target_model_dir,
                max_to_keep=FLAGS.best_checkpoints)

        # Restore best from file or if no file yet, set it to zero
        self.best_validation = get_best_valid_accuracy(self.log_dir)
        self.found = True

        if self.best_validation is None:
            self.found = False
            self.best_validation = 0.0

        # Best target
        if self.target:
            self.best_target_validation = get_best_target_valid_accuracy(self.log_dir)

            if self.best_target_validation is None:
                self.best_target_validation = 0.0

    def restore_latest(self):
        """ Restore the checkpoint from the latest one """
        self.checkpoint.restore(self.latest_manager.latest_checkpoint)

    def restore_best(self, target=False):
        """ Restore the checkpoint from the best one """
        if target and self.target:
            self.checkpoint.restore(self.best_target_manager.latest_checkpoint)
        else:
            self.checkpoint.restore(self.best_manager.latest_checkpoint)

    def latest_step(self):
        """ Return the step number from the latest checkpoint. Returns None if
        no checkpoints. """
        return self._get_step_from_manager(self.latest_manager)

    def best_step(self, target=False):
        """ Return the step number from the best checkpoint. Returns None if
        no checkpoints. """
        if target and self.target:
            return self._get_step_from_manager(self.best_target_manager)
        else:
            return self._get_step_from_manager(self.best_manager)

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

    def save(self, step, validation_accuracy=None, target_validation_accuracy=None):
        """ Save the latest model. If validation_accuracy specified and higher
        than the previous best, also save this model as the new best one. """
        # Always save the latest
        self.latest_manager.save(checkpoint_number=step)

        # Only save the "best" if it's better than the previous best
        if validation_accuracy is not None:
            if validation_accuracy > self.best_validation:
                self.best_manager.save(checkpoint_number=step)
                self.best_validation = validation_accuracy
                write_best_valid_accuracy(self.log_dir, self.best_validation)

        # Based on target classifier
        if target_validation_accuracy is not None and self.target:
            if target_validation_accuracy > self.best_target_validation:
                self.best_target_manager.save(checkpoint_number=step)
                self.best_target_validation = target_validation_accuracy
                write_best_target_valid_accuracy(self.log_dir,
                    self.best_target_validation)
