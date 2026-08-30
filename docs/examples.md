# Examples

## Single-channel example

```python
import numpy as np
from ringbuffer import RingBuffer

buffer = RingBuffer(signal_id=1, num_channels=1, samples_per_channel=4, dtype=np.int16)
payload = np.array([[1, 2, 3, 4]], dtype=np.int16)
buffer.append(payload, step=1, timestamp=0.0)
print(buffer.read_channels())
```

## Multi-channel example

```python
import numpy as np
from ringbuffer import RingBuffer

buffer = RingBuffer(
    signal_id=2,
    num_channels=3,
    samples_per_channel=4,
    dtype=np.int16,
    channel_ids=[11, 12, 13],
)
payload = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
], dtype=np.int16)
buffer.append(payload, step=2, timestamp=1.0)
print(buffer.latest(dtype=np.int16))
```
