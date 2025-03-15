class TextPreprocessorError(Exception):
    """base exception class for TextPreprocessor errors."""

    pass


class ConfigurationError(TextPreprocessorError):
    """exception raised for errors in the configuration."""

    pass


class InputError(TextPreprocessorError):
    """exception raised for errors in the input data."""

    pass


class TokenizationError(TextPreprocessorError):
    """exception raised for errors during tokenization."""

    pass


class ParallelProcessingError(TextPreprocessorError):
    """exception raised for errors during parallel processing."""

    pass


class SerializationError(TextPreprocessorError):
    """exception raised for errors during serialization or deserialization."""

    pass
