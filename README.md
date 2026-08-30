# RingBuffer

A modular, object-oriented ring buffer package for single-channel and multi-channel signal data.

## Overview

This package models a `RingBuffer` as a single signal that may contain one or more channels. Each channel is an independent stream of samples belonging to the same signal.

### Hierarchy

- Source
  - Signal A
    - Channel 1
    - Channel 2
  - Signal B
    - Channel 3

This design keeps one `RingBuffer` focused on a single signal while allowing `num_channels` to hold several channels within that signal.

## Installation

From the project root, install the package in editable mode so the import works both for tests and for direct script execution:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you are running one of the scripts directly from the [test_cases](test_cases) folder, the project is also configured to add the local src directory to the import path automatically.

## Basic usage

```python
import numpy as np
from ringbuffer import RingBuffer

buffer = RingBuffer(
    signal_id=1,
    num_channels=1,
    samples_per_channel=4,
    dtype=np.int16,
)

payload = np.array([[1, 2, 3, 4]], dtype=np.int16)
buffer.append(payload, step=1, timestamp=0.0)
print(buffer.latest(dtype=np.int16))
```

## Multi-channel usage

```python
import numpy as np
from ringbuffer import RingBuffer

buffer = RingBuffer(
    signal_id=2,
    num_channels=3,
    samples_per_channel=4,
    dtype=np.int16,
    channel_ids=[101, 205, 999],
    input_layout="channel_major",
)

payload = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
], dtype=np.int16)

buffer.append(payload, step=2, timestamp=1.5)
print(buffer.read_channels())
```

## Interleaved input

```python
import numpy as np
from ringbuffer import RingBuffer

buffer = RingBuffer(
    signal_id=3,
    num_channels=3,
    samples_per_channel=4,
    dtype=np.int16,
    input_layout="interleaved",
)

payload = np.array([
    [1, 10, 100],
    [2, 20, 200],
    [3, 30, 300],
    [4, 40, 400],
], dtype=np.int16)

buffer.append(payload, step=3, timestamp=2.0)
print(buffer.latest(dtype=np.int16))
```

## Multiple signals

```python
from ringbuffer import RingBufferManager

manager = RingBufferManager()
first = manager.create_buffer(signal_id=1, num_channels=1, samples_per_channel=256, dtype="int16")
second = manager.create_buffer(signal_id=2, num_channels=4, samples_per_channel=128, dtype="float32")
print(manager.total_buffers())
```

## Thread safety

The buffer is protected by a lock around writes and reads of the internal ring state, so a producer and consumer can safely append and read without corrupting internal indexes.

## API summary

- `RingBuffer.append(data, step, timestamp)`
- `RingBuffer.read()`
- `RingBuffer.read_array()`
- `RingBuffer.read_channels()`
- `RingBuffer.latest()`
- `RingBuffer.clear()`
- `RingBuffer.status()`
- `RingBuffer.memory_usage()`
- `RingBufferManager.create_buffer(...)`
