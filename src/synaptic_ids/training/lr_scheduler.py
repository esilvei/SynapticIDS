import math
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class OneCycleLR(LearningRateSchedule):
    """
    Implements the 1-Cycle Learning Rate policy.
    This schedule modifies the learning rate cyclically between a base
    and a max learning rate.

    The implementation is now fully compatible with TensorFlow's graph mode
    and can be correctly serialized.
    """

    def __init__(
        self,
        max_lr,
        steps_per_epoch,
        epochs,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        name=None,
    ):
        super().__init__()
        # Store all initial parameters to be used in get_config
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.name = name

        # Derived parameters
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor

        # Calculate total steps and phase boundaries
        total_steps = float(steps_per_epoch * epochs)
        self.phase1_steps = int(total_steps * 0.3)
        self.phase2_steps = int(total_steps - self.phase1_steps)

        # Use TensorFlow-native floats for graph compatibility
        self.total_steps_tf = tf.constant(total_steps, dtype=tf.float32)
        self.phase1_steps_tf = tf.constant(self.phase1_steps, dtype=tf.float32)

    def __call__(self, step):
        """Calculates the learning rate for a given step."""
        step = tf.cast(step, dtype=tf.float32)

        def phase1_fn():
            # Linear warmup from initial_lr to max_lr
            return self.initial_lr + (self.max_lr - self.initial_lr) * (
                step / self.phase1_steps_tf
            )

        def phase2_fn():
            # Cosine annealing from max_lr to min_lr
            phase2_step = step - self.phase1_steps_tf

            # Use tf.math for graph-compatible operations
            cosine_decay = 0.5 * (
                1 + tf.math.cos(math.pi * phase2_step / float(self.phase2_steps))
            )
            return (self.max_lr - self.min_lr) * cosine_decay + self.min_lr

        # Use tf.cond for conditional logic within the TensorFlow graph
        lr = tf.cond(step < self.phase1_steps_tf, phase1_fn, phase2_fn)

        return lr

    def get_config(self):
        """
        Returns the configuration of the scheduler as a JSON-serializable dict.
        This enables the scheduler to be saved and loaded with the model.
        """
        return {
            "max_lr": self.max_lr,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
            "three_phase": self.three_phase,
            "name": self.name,
        }
