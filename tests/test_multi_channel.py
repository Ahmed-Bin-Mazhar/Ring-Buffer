import numpy as np

from ringbuffer import RingBuffer


def test_arbitrary_channel_ids_are_preserved():
    buffer = RingBuffer(
        signal_id=5,
        num_channels=4,
        samples_per_channel=2,
        channel_ids=[101, 205, 999, 42],
    )

    payload = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ],
        dtype=np.int16,
    )
    buffer.append(payload, step=5, timestamp=2.0)

    assert buffer.channel_ids == (101, 205, 999, 42)
    assert np.array_equal(buffer.read_channels(), payload)


def test_input_layout_accepts_channel_major_and_interleaved_shapes():
    channel_major = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int16,
    )
    interleaved = np.array(
        [
            [1, 4],
            [2, 5],
            [3, 6],
        ],
        dtype=np.int16,
    )

    major_buffer = RingBuffer(10, 2, 3, dtype=np.int16)
    interleaved_buffer = RingBuffer(11, 2, 3, dtype=np.int16, input_layout="interleaved")

    major_buffer.append(channel_major, step=1, timestamp=1.0)
    interleaved_buffer.append(interleaved, step=1, timestamp=1.0)

    assert np.array_equal(major_buffer.latest(dtype=np.int16), channel_major)
    assert np.array_equal(interleaved_buffer.latest(dtype=np.int16), channel_major)
