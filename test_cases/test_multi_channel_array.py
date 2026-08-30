import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from ringbuffer import RingBuffer

buffer = RingBuffer(
    signal_id=2,
    num_channels=3,
    samples_per_channel=4,
    dtype=np.int16,
    channel_ids=[101, 202, 303],
)

payload = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
], dtype=np.int16)

buffer.append(payload, step=2, timestamp=1.0)
print(buffer.read_channels())
print(buffer.status())
