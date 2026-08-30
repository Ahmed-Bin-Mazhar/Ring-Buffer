# Usage

## Initialization

```python
from ringbuffer import RingBuffer

buffer = RingBuffer(
    signal_id=1,
    num_channels=2,
    samples_per_channel=256,
    dtype="int16",
    channel_ids=[10, 20],
    input_layout="channel_major",
)
```

## Append

```python
import numpy as np

payload = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
], dtype=np.int16)

buffer.append(payload, step=1, timestamp=0.0)
```

## Read options

```python
raw = buffer.read()
array_data = buffer.read_array()
channel_data = buffer.read_channels()
latest = buffer.latest()
```

## Status

```python
print(buffer.status())
print(buffer.memory_usage())
```
