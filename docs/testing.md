# Testing and Continuous Stream Simulation

## Automated tests

Run the test suite with:

```bash
pytest
```

## Manual tests

The manual examples live in `test_cases/` and can be run with:

```bash
python test_cases/test_single_channel_array.py
python test_cases/test_multi_channel_array.py
python test_cases/test_interleaved_array.py
python test_cases/test_continuous_stream.py
```

## Continuous stream example

The stream simulation reads chunked data, appends each chunk to the buffer, and monitors the latest content.

```python
import numpy as np
import time
from ringbuffer import RingBuffer

buffer = RingBuffer(
    signal_id=1,
    num_channels=2,
    samples_per_channel=8,
    dtype=np.int16,
    input_layout="interleaved",
)

chunks = [
    np.array([[1, 10],[2, 20],[3, 30],[4, 40]], dtype=np.int16),
    np.array([[5, 50],[6, 60],[7, 70],[8, 80]], dtype=np.int16),
]

for step, chunk in enumerate(chunks, start=1):
    buffer.append(chunk, step=step, timestamp=time.time())
    print(buffer.latest(dtype=np.int16))
```

## Overflow detection

Use a small buffer and append more data than it can hold to confirm that old samples are overwritten first.
