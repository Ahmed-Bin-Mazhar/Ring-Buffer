from __future__ import annotations

import threading
import weakref
from typing import Optional, Sequence

import numpy as np

from .config import RingBufferConfig
from .dtype_utils import resolve_dtype
from .exceptions import BufferOverflowError, DataFormatError, InvalidConfigurationError


class RingBuffer:
    """Thread-safe circular buffer for a single signal with optional channel data."""

    _instances: weakref.WeakSet = weakref.WeakSet()
    _instances_lock = threading.Lock()

    def __init__(
        self,
        signal_id: int,
        num_channels: int,
        samples_per_channel: int,
        dtype=None,
        channel_ids: Optional[Sequence[int]] = None,
        input_layout: str = "channel_major",
        bytes_per_sample: Optional[int] = None,
    ):
        """Create a ring buffer for one signal.

        Parameters
        ----------
        signal_id:
            Unique ID used to identify the signal.
        num_channels:
            Number of logical channels belonging to the signal.
        samples_per_channel:
            Number of samples retained per channel.
        dtype:
            NumPy dtype for sample values. Defaults to int16 unless overridden.
        channel_ids:
            Optional integer IDs for each channel.
        input_layout:
            Either "channel_major" or "interleaved".
        bytes_per_sample:
            Legacy compatibility convenience parameter.
        """
        if not isinstance(signal_id, (int, np.integer)):
            raise TypeError("signal_id must be an integer")
        if not isinstance(num_channels, (int, np.integer)):
            raise TypeError("num_channels must be an integer")
        if int(num_channels) <= 0:
            raise ValueError("num_channels must be greater than 0")
        if not isinstance(samples_per_channel, (int, np.integer)):
            raise TypeError("samples_per_channel must be an integer")
        if int(samples_per_channel) <= 0:
            raise ValueError("samples_per_channel must be greater than 0")
        if input_layout not in ("channel_major", "interleaved"):
            raise InvalidConfigurationError("input_layout must be either 'channel_major' or 'interleaved'")

        if channel_ids is None:
            normalized_ids = tuple(range(int(num_channels)))
        else:
            normalized_ids = tuple(channel_ids)
            if len(normalized_ids) != int(num_channels):
                raise ValueError("channel_ids length must match num_channels")
            if len(set(normalized_ids)) != len(normalized_ids):
                raise ValueError("channel_ids must contain unique values")
            for channel_id in normalized_ids:
                if not isinstance(channel_id, (int, np.integer)):
                    raise TypeError("Every channel ID must be an integer")

        resolved_dtype = resolve_dtype(dtype=dtype, bytes_per_sample=bytes_per_sample)

        config = RingBufferConfig(
            signal_id=int(signal_id),
            num_channels=int(num_channels),
            samples_per_channel=int(samples_per_channel),
            dtype=np.dtype(resolved_dtype),
            channel_ids=tuple(int(ch) for ch in normalized_ids),
            input_layout=input_layout,
            bytes_per_sample=int(np.dtype(resolved_dtype).itemsize),
        )

        self.config = config
        self.signal_id = config.signal_id
        self.num_channels = config.num_channels
        self.samples_per_channel = config.samples_per_channel
        self.channel_ids = config.channel_ids
        self.input_layout = config.input_layout
        self.dtype = config.dtype
        self.bytes_per_sample = config.bytes_per_sample
        self.frame_bytes = self.num_channels * self.bytes_per_sample
        self.capacity = self.num_channels * self.samples_per_channel * self.bytes_per_sample

        self.buffer = np.zeros(self.capacity, dtype=np.uint8)
        self.write_index = 0
        self.read_index = 0
        self.size = 0
        self.step: Optional[int] = None
        self.timestamp: Optional[float] = None
        self.lock = threading.Lock()

        with RingBuffer._instances_lock:
            RingBuffer._instances.add(self)

    @classmethod
    def total_buffers_created(cls) -> int:
        """Return the number of currently active buffer instances."""
        with cls._instances_lock:
            return len(cls._instances)

    @classmethod
    def all_buffers_summary(cls):
        """Return a list summary for all active buffers."""
        with cls._instances_lock:
            buffers = list(cls._instances)
        return [
            {
                "buffer_index": index,
                "signal_id": buf.signal_id,
                "channel_ids": buf.channel_ids,
                "num_channels": buf.num_channels,
                "samples_per_channel": buf.samples_per_channel,
                "dtype": str(buf.dtype),
                "bytes_per_sample": buf.bytes_per_sample,
                "capacity_bytes": buf.capacity,
            }
            for index, buf in enumerate(buffers)
        ]

    @classmethod
    def buffer_report(cls) -> dict:
        """Return a summary of all active buffers."""
        summary = cls.all_buffers_summary()
        return {"total_buffers": len(summary), "total_active_buffers": len(summary), "buffers": summary}

    @staticmethod
    def _normalize_data(data: np.ndarray, *, num_channels: int, samples_per_channel: int, input_layout: str, dtype: np.dtype) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a NumPy ndarray")
        if data.size == 0:
            raise ValueError("Cannot append an empty array")

        arr = np.asarray(data, dtype=dtype)
        total_samples = num_channels * samples_per_channel

        if arr.ndim == 1:
            if num_channels == 1:
                if arr.size != samples_per_channel:
                    raise ValueError(
                        f"For single-channel signals, expected {samples_per_channel} samples, got {arr.size}."
                    )
                return arr.reshape(1, samples_per_channel)

            if arr.size != total_samples:
                raise ValueError(
                    f"Expected {total_samples} samples total for {num_channels} channels, got {arr.size}."
                )
            if input_layout == "channel_major":
                return arr.reshape(num_channels, samples_per_channel)
            return arr.reshape(samples_per_channel, num_channels).T

        if arr.ndim == 2:
            if input_layout == "channel_major":
                if arr.shape == (num_channels, samples_per_channel):
                    return arr
                if arr.shape == (samples_per_channel, num_channels):
                    return arr.T
                raise DataFormatError(
                    f"channel_major data must have shape ({num_channels}, {samples_per_channel}), got {arr.shape}"
                )

            if arr.shape == (samples_per_channel, num_channels):
                return arr.T
            if arr.shape == (num_channels, samples_per_channel):
                return arr
            raise DataFormatError(
                f"interleaved data must have shape ({samples_per_channel}, {num_channels}) or "
                f"({num_channels}, {samples_per_channel}), got {arr.shape}"
            )

        raise DataFormatError("data must be 1D or 2D")

    def append(self, data: np.ndarray, step: int, timestamp: float) -> None:
        """Append a full data frame to the circular buffer.

        Parameters
        ----------
        data:
            NumPy array containing the sample values. The array may be 1D for a
            single channel or 2D for multi-channel data.
        step:
            The logical sampling step or sequence number for this append.
        timestamp:
            Timestamp associated with the appended frame.
        """
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        if not isinstance(timestamp, (int, float, np.integer, np.floating)):
            raise TypeError("timestamp must be numeric")
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")

        normalized = self._normalize_data(
            data,
            num_channels=self.num_channels,
            samples_per_channel=self.samples_per_channel,
            input_layout=self.input_layout,
            dtype=self.dtype,
        )
        data_bytes = np.ascontiguousarray(normalized).view(np.uint8).reshape(-1)
        n_bytes = data_bytes.size

        if n_bytes > self.capacity:
            raise BufferOverflowError(
                f"Data size ({n_bytes} bytes) exceeds buffer capacity ({self.capacity} bytes)"
            )
        if n_bytes % self.bytes_per_sample != 0:
            raise ValueError("Data size is not aligned to sample boundaries")
        if n_bytes % self.frame_bytes != 0:
            raise ValueError("Data does not contain a complete number of multi-channel frames")

        with self.lock:
            first_part = min(n_bytes, self.capacity - self.write_index)
            self.buffer[self.write_index : self.write_index + first_part] = data_bytes[:first_part]
            remaining = n_bytes - first_part
            if remaining > 0:
                self.buffer[:remaining] = data_bytes[first_part:]

            self.write_index = (self.write_index + n_bytes) % self.capacity
            self.size = min(self.size + n_bytes, self.capacity)
            if self.size == self.capacity:
                self.read_index = self.write_index

            self.step = int(step)
            self.timestamp = float(timestamp)

    def read(self) -> bytes:
        """Return the raw byte stream currently stored in the buffer."""
        with self.lock:
            if self.size == 0:
                return b""
            first_part = min(self.size, self.capacity - self.read_index)
            output = bytearray(self.size)
            output[:first_part] = self.buffer[self.read_index : self.read_index + first_part].tobytes()
            remaining = self.size - first_part
            if remaining > 0:
                output[first_part:] = self.buffer[:remaining].tobytes()
            return bytes(output)

    def read_array(self, dtype=None) -> np.ndarray:
        """Return the stored raw samples as a NumPy array."""
        target_dtype = self.dtype if dtype is None else np.dtype(dtype)
        raw = self.read()
        if not raw:
            return np.empty(0, dtype=target_dtype)
        return np.frombuffer(raw, dtype=target_dtype).copy()

    def read_channels(self, dtype=None) -> np.ndarray:
        """Return channel-major samples for the configured signal."""
        target_dtype = self.dtype if dtype is None else np.dtype(dtype)
        expected_values = self.num_channels * self.samples_per_channel
        data = self.read_array(dtype=target_dtype)
        if data.size < expected_values:
            return np.empty((self.num_channels, 0), dtype=target_dtype)

        data = data[-expected_values:]
        return data.reshape(self.num_channels, self.samples_per_channel)

    def latest(self, dtype=None) -> np.ndarray:
        """Return the latest full buffer contents, shaped as channel-major."""
        target_dtype = self.dtype if dtype is None else np.dtype(dtype)
        block_bytes = self.capacity

        with self.lock:
            if self.size < block_bytes:
                return np.empty((self.num_channels, 0), dtype=target_dtype)
            start = (self.write_index - block_bytes) % self.capacity
            raw = bytearray(block_bytes)
            first_part = min(block_bytes, self.capacity - start)
            raw[:first_part] = self.buffer[start : start + first_part].tobytes()
            remaining = block_bytes - first_part
            if remaining > 0:
                raw[first_part:] = self.buffer[:remaining].tobytes()

        data = np.frombuffer(raw, dtype=target_dtype).copy()
        return data.reshape(self.num_channels, self.samples_per_channel)

    def clear(self) -> None:
        """Reset the buffer contents and metadata."""
        with self.lock:
            self.buffer.fill(0)
            self.write_index = 0
            self.read_index = 0
            self.size = 0
            self.step = None
            self.timestamp = None

    def status(self) -> dict:
        """Return a dictionary containing the current buffer state."""
        with self.lock:
            return {
                "signal_id": self.signal_id,
                "channel_ids": self.channel_ids,
                "num_channels": self.num_channels,
                "samples_per_channel": self.samples_per_channel,
                "dtype": str(self.dtype),
                "bytes_per_sample": self.bytes_per_sample,
                "frame_bytes": self.frame_bytes,
                "capacity_bytes": self.capacity,
                "used_bytes": self.size,
                "free_bytes": self.capacity - self.size,
                "write_index": self.write_index,
                "read_index": self.read_index,
                "full": self.size == self.capacity,
                "step": self.step,
                "timestamp": self.timestamp,
                "input_layout": self.input_layout,
            }

    def memory_usage(self) -> int:
        """Return the raw memory usage in bytes."""
        return self.buffer.nbytes
