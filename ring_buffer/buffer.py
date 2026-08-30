import threading
import weakref
from typing import Optional, Sequence

import numpy as np


class RingBuffer:
    """
    Thread-safe circular buffer for one signal.

    A signal may contain one or more channels.
    Capacity = num_channels * samples_per_channel * bytes_per_sample
    Channel IDs may be arbitrary and do not need to be sequential.
    """

    _instances: weakref.WeakSet = weakref.WeakSet()
    _instances_lock = threading.Lock()

    @staticmethod
    def _resolve_dtype(dtype, bytes_per_sample):
        mapping = {1: np.uint8, 2: np.int16, 4: np.float32, 8: np.complex64}

        if dtype is None and bytes_per_sample is None:
            return np.dtype(np.int16)

        if bytes_per_sample is not None:
            if not isinstance(bytes_per_sample, (int, np.integer)):
                raise TypeError("bytes_per_sample must be an integer")
            if bytes_per_sample <= 0:
                raise ValueError("bytes_per_sample must be greater than 0")
            if bytes_per_sample not in mapping:
                raise ValueError(
                    f"Unsupported bytes_per_sample={bytes_per_sample}. Supported values are 1, 2, 4, and 8."
                )
            if dtype is None:
                return np.dtype(mapping[bytes_per_sample])
            if isinstance(dtype, (int, np.integer)):
                if int(dtype) != int(bytes_per_sample):
                    raise ValueError("dtype and bytes_per_sample conflict")
                return np.dtype(mapping[int(dtype)])

            resolved = np.dtype(dtype)
            if resolved.itemsize != int(bytes_per_sample):
                raise ValueError(
                    f"dtype itemsize ({resolved.itemsize}) does not match bytes_per_sample ({bytes_per_sample})"
                )
            return resolved

        if isinstance(dtype, (int, np.integer)):
            value = int(dtype)
            if value not in mapping:
                raise ValueError(
                    f"Unsupported bytes_per_sample value {value}. Supported values are 1, 2, 4, and 8."
                )
            return np.dtype(mapping[value])

        if dtype is None:
            return np.dtype(np.int16)

        resolved = np.dtype(dtype)
        return resolved

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
        if not isinstance(signal_id, (int, np.integer)):
            raise TypeError("signal_id must be an integer")

        if not isinstance(num_channels, (int, np.integer)):
            raise TypeError("num_channels must be an integer")
        if num_channels <= 0:
            raise ValueError("num_channels must be greater than 0")

        if not isinstance(samples_per_channel, (int, np.integer)):
            raise TypeError("samples_per_channel must be an integer")
        if samples_per_channel <= 0:
            raise ValueError("samples_per_channel must be greater than 0")

        if input_layout not in ("channel_major", "interleaved"):
            raise ValueError("input_layout must be either 'channel_major' or 'interleaved'")

        if channel_ids is None:
            channel_ids = list(range(num_channels))
        else:
            channel_ids = list(channel_ids)
            if len(channel_ids) != num_channels:
                raise ValueError("channel_ids length must match num_channels")
            if len(set(channel_ids)) != len(channel_ids):
                raise ValueError("channel_ids must contain unique values")
            for channel_id in channel_ids:
                if not isinstance(channel_id, (int, np.integer)):
                    raise TypeError("Every channel ID must be an integer")

        self.dtype = self._resolve_dtype(dtype, bytes_per_sample)
        self.bytes_per_sample = self.dtype.itemsize

        self.signal_id = int(signal_id)
        self.num_channels = int(num_channels)
        self.samples_per_channel = int(samples_per_channel)
        self.channel_ids = tuple(int(ch) for ch in channel_ids)
        self.input_layout = input_layout
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

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a NumPy ndarray")

        if data.size == 0:
            raise ValueError("Cannot append an empty array")

        arr = np.asarray(data, dtype=self.dtype)
        total_samples = self.num_channels * self.samples_per_channel

        if arr.ndim == 1:
            if self.num_channels == 1:
                if arr.size != self.samples_per_channel:
                    raise ValueError(
                        f"For single-channel signal, expected {self.samples_per_channel} samples, got {arr.size}."
                    )
                return arr.reshape(1, self.samples_per_channel)

            if arr.size != total_samples:
                raise ValueError(
                    f"Expected {total_samples} samples total for {self.num_channels} channels, got {arr.size}."
                )

            if self.input_layout == "channel_major":
                return arr.reshape(self.num_channels, self.samples_per_channel)
            return arr.reshape(self.samples_per_channel, self.num_channels).T

        if arr.ndim == 2:
            if self.input_layout == "channel_major":
                if arr.shape == (self.num_channels, self.samples_per_channel):
                    return arr
                if arr.shape == (self.samples_per_channel, self.num_channels):
                    return arr.T
                raise ValueError(
                    f"channel_major data must have shape ({self.num_channels}, {self.samples_per_channel}), "
                    f"got {arr.shape}"
                )

            if arr.shape == (self.samples_per_channel, self.num_channels):
                return arr.T
            if arr.shape == (self.num_channels, self.samples_per_channel):
                return arr
            raise ValueError(
                f"interleaved data must have shape ({self.samples_per_channel}, {self.num_channels}) or "
                f"({self.num_channels}, {self.samples_per_channel}), got {arr.shape}"
            )

        raise ValueError("data must be 1D or 2D")

    def append(self, data: np.ndarray, step: int, timestamp: float) -> None:
        """Append complete multi-channel sample frames."""
        if not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")

        if not isinstance(timestamp, (int, float, np.integer, np.floating)):
            raise TypeError("timestamp must be numeric")
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")

        normalized = self._normalize_data(data)
        data = np.ascontiguousarray(normalized)
        data_bytes = data.view(np.uint8).reshape(-1)
        n = data_bytes.size

        if n > self.capacity:
            raise ValueError(
                f"Data size ({n} bytes) exceeds buffer capacity ({self.capacity} bytes)"
            )

        if n % self.bytes_per_sample != 0:
            raise ValueError("Data size is not aligned to sample boundaries")

        if n % self.frame_bytes != 0:
            raise ValueError("Data does not contain a complete number of multi-channel frames")

        with self.lock:
            first_part = min(n, self.capacity - self.write_index)
            self.buffer[self.write_index : self.write_index + first_part] = data_bytes[:first_part]

            remaining = n - first_part
            if remaining > 0:
                self.buffer[:remaining] = data_bytes[first_part:]

            self.write_index = (self.write_index + n) % self.capacity
            self.size = min(self.size + n, self.capacity)

            if self.size == self.capacity:
                self.read_index = self.write_index

            self.step = int(step)
            self.timestamp = float(timestamp)

    def read(self) -> bytes:
        with self.lock:
            if self.size == 0:
                return b""

            first_part = min(self.size, self.capacity - self.read_index)
            output = bytearray(self.size)

            output[:first_part] = self.buffer[
                self.read_index : self.read_index + first_part
            ].tobytes()

            remaining = self.size - first_part
            if remaining > 0:
                output[first_part:] = self.buffer[:remaining].tobytes()

            return bytes(output)

    def read_array(self, dtype=None) -> np.ndarray:
        raw = self.read()

        if not raw:
            return np.empty(0, dtype=self.dtype if dtype is None else np.dtype(dtype))

        target_dtype = self.dtype if dtype is None else np.dtype(dtype)
        return np.frombuffer(raw, dtype=target_dtype).copy()

    def read_channels(self, dtype=None) -> np.ndarray:
        target_dtype = self.dtype if dtype is None else np.dtype(dtype)
        expected_elements = self.num_channels * self.samples_per_channel
        data = self.read_array(dtype=target_dtype)

        if data.size < expected_elements:
            return np.empty((self.num_channels, 0), dtype=target_dtype)

        data = data[-expected_elements:]

        if self.input_layout == "channel_major":
            return data.reshape(self.num_channels, self.samples_per_channel)

        return data.reshape(self.samples_per_channel, self.num_channels).T

    def latest(self, dtype=None) -> np.ndarray:
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

        if self.input_layout == "channel_major":
            return data.reshape(self.num_channels, self.samples_per_channel)

        return data.reshape(self.samples_per_channel, self.num_channels).T

    def clear(self) -> None:
        with self.lock:
            self.buffer.fill(0)
            self.write_index = 0
            self.read_index = 0
            self.size = 0
            self.step = None
            self.timestamp = None

    @classmethod
    def total_buffers_created(cls) -> int:
        with cls._instances_lock:
            return len(cls._instances)

    @classmethod
    def all_buffers_summary(cls) -> list[dict]:
        with cls._instances_lock:
            buffers = list(cls._instances)

        return [
            {
                "buffer_index": index,
                "signal_id": buffer.signal_id,
                "channel_ids": buffer.channel_ids,
                "num_channels": buffer.num_channels,
                "samples_per_channel": buffer.samples_per_channel,
                "dtype": str(buffer.dtype),
                "bytes_per_sample": buffer.bytes_per_sample,
                "capacity_bytes": buffer.capacity,
            }
            for index, buffer in enumerate(buffers)
        ]

    @classmethod
    def buffer_report(cls) -> dict:
        buffers = cls.all_buffers_summary()
        return {
            "total_buffers": len(buffers),
            "total_active_buffers": len(buffers),
            "buffers": buffers,
        }

    @classmethod
    def summary(cls) -> str:
        buffers = cls.all_buffers_summary()
        if not buffers:
            return "No active buffers currently exist."

        lines = [f"Total active buffers: {len(buffers)}"]
        for item in buffers:
            lines.append(
                "- "
                f"buffer_index={item['buffer_index']}, "
                f"signal_id={item['signal_id']}, "
                f"channel_ids={item['channel_ids']}, "
                f"num_channels={item['num_channels']}, "
                f"samples_per_channel={item['samples_per_channel']}, "
                f"dtype={item['dtype']}, "
                f"capacity_bytes={item['capacity_bytes']}"
            )
        return "\n".join(lines)

    def status(self) -> dict:
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
                "capacity_kb": self.capacity / 1024,
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
        return self.buffer.nbytes
