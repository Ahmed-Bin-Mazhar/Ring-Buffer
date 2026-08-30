import numpy as np

from ringbuffer import RingBuffer


def test_single_channel_append_and_read():
    buffer = RingBuffer(
        signal_id=1,
        num_channels=1,
        samples_per_channel=4,
        dtype=np.int16,
    )

    payload = np.array([[1, 2, 3, 4]], dtype=np.int16)
    buffer.append(payload, step=1, timestamp=12.5)

    assert np.array_equal(buffer.read_channels(), payload)
    assert np.array_equal(buffer.latest(dtype=np.int16), payload)
    assert buffer.status()["signal_id"] == 1


def test_multi_channel_channel_major_input():
    buffer = RingBuffer(
        signal_id=2,
        num_channels=3,
        samples_per_channel=4,
        dtype=np.int16,
        channel_ids=[10, 20, 30],
    )

    payload = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        dtype=np.int16,
    )
    buffer.append(payload, step=2, timestamp=21.3)

    assert np.array_equal(buffer.latest(dtype=np.int16), payload)
    assert np.array_equal(buffer.read_channels(), payload)


def test_interleaved_input_and_overwrite():
    buffer = RingBuffer(
        signal_id=3,
        num_channels=2,
        samples_per_channel=3,
        dtype=np.int16,
        input_layout="interleaved",
    )

    first = np.array(
        [
            [1, 10],
            [2, 20],
            [3, 30],
        ],
        dtype=np.int16,
    )
    buffer.append(first, step=3, timestamp=33.0)

    second = np.array(
        [
            [4, 40],
            [5, 50],
            [3, 30],
        ],
        dtype=np.int16,
    )
    buffer.append(second, step=4, timestamp=44.0)

    latest = buffer.latest(dtype=np.int16)
    assert latest.shape == (2, 3)
    assert np.array_equal(latest, np.array([[4, 5, 3], [40, 50, 30]], dtype=np.int16))


def test_buffer_rejects_invalid_config():
    try:
        RingBuffer(signal_id="bad", num_channels=1, samples_per_channel=4)
        raise AssertionError("signal_id validation failed")
    except TypeError:
        pass

    try:
        RingBuffer(signal_id=1, num_channels=0, samples_per_channel=4)
        raise AssertionError("num_channels validation failed")
    except ValueError:
        pass

    try:
        RingBuffer(signal_id=1, num_channels=2, samples_per_channel=4, channel_ids=[1, 1])
        raise AssertionError("duplicate channel ids validation failed")
    except ValueError:
        pass


def test_bytes_per_sample_and_complex64_support():
    buffer = RingBuffer(
        signal_id=99,
        num_channels=1,
        samples_per_channel=4,
        bytes_per_sample=8,
    )

    payload = np.array([[1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j]], dtype=np.complex64)
    buffer.append(payload, step=99, timestamp=5.5)

    assert np.allclose(buffer.latest(dtype=np.complex64), payload)
