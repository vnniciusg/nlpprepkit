class PipelineError(Exception):
    """Base class for all pipeline-related exceptions."""

    pass


class StepNotCallableError(PipelineError):
    """Raised when a step added to the pipeline is not callable."""

    def __init__(self, step):
        super().__init__(f"The step '{step}' must be a callable function.")


class ProfilingNotEnabledError(PipelineError):
    """Raised when profiling is accessed but not enabled."""

    def __init__(self):
        super().__init__("Profiling is not enabled. Set enable_profiling=True to enable profiling.")


class InvalidInputTypeError(PipelineError):
    """Raised when the input to the pipeline is not a string or a list of strings."""

    def __init__(self, input_type):
        super().__init__(f"Invalid input type: {input_type.__name__}. Input must be a string or a list of strings.")
