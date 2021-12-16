
from transformers import (
    TrainerState,
    TrainerControl,
    TrainerCallback
)

# copie from https://github.com/pytorch/fairseq/blob/14c5bd027f04aae9dbb32f1bd7b34591b61af97f/fairseq/tasks/online_backtranslation.py#L41
class PiecewiseLinearFn:
    """Piecewise linear function. Can be configured with a string."""

    def __init__(self, pieces: Sequence[Tuple[int, float]]):
        assert pieces == sorted(
            pieces
        ), f"PiecewiseLinearFn configuration should be sorted, received: {pieces}"

        self.pieces = pieces

    def __call__(self, x: int) -> float:
        for i, (x_a, y_a) in enumerate(self.pieces[:-1]):
            x_b, y_b = self.pieces[i + 1]
            if x_a <= x <= x_b:
                return y_a + (x - x_a) * (y_b - y_a) / (x_b - x_a)

        return self.pieces[-1][1]

    @staticmethod
    def from_string(configuration: str) -> "PiecewiseLinearFn":
        """
        Parse the configuration of lambda coefficient (for scheduling).
        x = "3"                  # lambda will be a constant equal to x
        x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                                 # to 0 during the first 1000 iterations
        x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                                 # iterations, then will linearly increase to 1 until iteration 2000
        """
        if isinstance(configuration, float):
            return PiecewiseLinearFn([(0, configuration)])

        try:
            parts = configuration.split(",")
            if len(parts) == 1:
                v = float(configuration)
                return PiecewiseLinearFn([(0, v)])

            split = [s.split(":") for s in parts]
            pieces = [(int(t), float(v)) for t, v in split]
            return PiecewiseLinearFn(pieces)
        except Exception:
            raise ValueError(
                f"Invalid PiecewiseLinearFn configuration: {configuration!r}"
            )

    @staticmethod
    def one() -> "PiecewiseLinearFn":
        return PiecewiseLinearFn([(0, 1.0)])


class GatingCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that handles the default flow of the training loop for logs, evaluation
    and checkpoints.
    """

    def __init__(self, config_str):
        self.lambda_original = PiecewiseLinearFn.from_string(config_str)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if (
            args.logging_strategy == IntervalStrategy.STEPS
            and args.logging_steps > 0
            and state.global_step % args.logging_steps == 0
        ):
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step % args.eval_steps == 0:
            control.should_evaluate = True
            if args.load_best_model_at_end:
                control.should_save = True

        # Save
        if (
            not args.load_best_model_at_end
            and args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control
