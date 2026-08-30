import numpy as np

from ringbuffer import RingBufferManager


def test_manager_creates_and_tracks_buffers():
    manager = RingBufferManager()
    a = manager.create_buffer(signal_id=1, num_channels=1, samples_per_channel=4, dtype=np.int16)
    b = manager.create_buffer(signal_id=2, num_channels=2, samples_per_channel=3, dtype=np.float32)

    assert len(manager.buffers) == 2
    assert a.signal_id == 1 and b.signal_id == 2
    assert manager.total_buffers() == 2
