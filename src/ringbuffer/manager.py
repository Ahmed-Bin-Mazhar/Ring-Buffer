from __future__ import annotations

import threading
from typing import Dict

from .ring_buffer import RingBuffer


class RingBufferManager:
    """Manage multiple independent ring buffers for different signals."""

    def __init__(self):
        self._lock = threading.RLock()
        self._buffers: Dict[int, RingBuffer] = {}

    @property
    def buffers(self):
        with self._lock:
            return list(self._buffers.values())

    def create_buffer(self, signal_id, num_channels, samples_per_channel, **kwargs):
        with self._lock:
            buffer = RingBuffer(
                signal_id=signal_id,
                num_channels=num_channels,
                samples_per_channel=samples_per_channel,
                **kwargs,
            )
            self._buffers[buffer.signal_id] = buffer
            return buffer

    def get_buffer(self, signal_id):
        with self._lock:
            return self._buffers.get(signal_id)

    def remove_buffer(self, signal_id):
        with self._lock:
            self._buffers.pop(signal_id, None)

    def total_buffers(self):
        with self._lock:
            return len(self._buffers)

    def summary(self):
        with self._lock:
            return [buffer.status() for buffer in self._buffers.values()]
