# pylint: disable=no-member
import tensorflow as tf
import numpy as np


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A custom learning rate scheduler that implements the 1-cycle policy.
    Its responsibility is solely to calculate the learning rate for a given training step.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        max_lr,
        steps_per_epoch,
        epochs,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=True,
    ):
        super().__init__()
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.total_steps = steps_per_epoch * epochs
        self.initial_lr = self.max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.three_phase = three_phase

        if three_phase:
            self.phase1_steps = tf.cast(int(self.total_steps * 0.3), tf.float32)
            self.phase2_steps = tf.cast(int(self.total_steps * 0.4), tf.float32)
            self.phase3_steps = tf.cast(
                self.total_steps - int(self.phase1_steps) - int(self.phase2_steps),
                tf.float32,
            )
        else:
            self.phase1_steps = tf.cast(int(self.total_steps * 0.3), tf.float32)
            self.phase2_steps = tf.cast(
                self.total_steps - int(self.phase1_steps), tf.float32
            )
            self.phase3_steps = tf.cast(0, tf.float32)

    def __call__(self, step):
        """Calculates the learning rate for the current step."""
        step = tf.cast(step, tf.float32)

        phase1_lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
            step / self.phase1_steps
        )

        phase2_step = step - self.phase1_steps
        if self.three_phase:
            phase2_lr = self.max_lr
        else:
            cosine_decay = 0.5 * (1 + np.cos(np.pi * phase2_step / self.phase2_steps))
            phase2_lr = self.initial_lr + (self.max_lr - self.initial_lr) * cosine_decay

        phase3_step = step - self.phase1_steps - self.phase2_steps
        if self.three_phase:
            cosine_decay = 0.5 * (1 + np.cos(np.pi * phase3_step / self.phase3_steps))
            phase3_lr = self.initial_lr * cosine_decay + self.min_lr * (
                1 - cosine_decay
            )
        else:
            phase3_lr = self.initial_lr

        lr = tf.where(
            step < self.phase1_steps,
            phase1_lr,
            tf.where(
                step < (self.phase1_steps + self.phase2_steps), phase2_lr, phase3_lr
            ),
        )

        return lr

    def get_config(self):
        """Enables the scheduler to be serialized."""
        return {
            "max_lr": self.max_lr.numpy(),
            "total_steps": self.total_steps,
            "initial_lr": self.initial_lr.numpy(),
            "min_lr": self.min_lr.numpy(),
            "three_phase": self.three_phase,
            "phase1_steps": self.phase1_steps.numpy(),
            "phase2_steps": self.phase2_steps.numpy(),
            "phase3_steps": self.phase3_steps.numpy(),
        }
