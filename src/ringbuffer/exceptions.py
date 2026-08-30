class RingBufferError(Exception):
    """Base error for ring buffer problems."""


class InvalidConfigurationError(RingBufferError):
    """Raised when the ring buffer configuration is invalid."""


class DataFormatError(RingBufferError):
    """Raised when the input data shape or layout is invalid."""


class BufferOverflowError(RingBufferError):
    """Raised when an append would exceed the configured capacity in an invalid way."""
