import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from ringbuffer import RingBuffer

buffer = RingBuffer(
    signal_id=3,
    num_channels=2,
    samples_per_channel=4,
    dtype=np.int16,
    input_layout="interleaved",
)

payload = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40],
], dtype=np.int16)

buffer.append(payload, step=3, timestamp=2.0)
print(buffer.latest(dtype=np.int16))
