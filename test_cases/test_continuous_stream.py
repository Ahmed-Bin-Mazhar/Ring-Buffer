import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from ringbuffer import RingBuffer


buffer = RingBuffer(
    signal_id=7,
    num_channels=2,
    samples_per_channel=4,
    dtype=np.int16,
    input_layout="interleaved",
)

chunks = [
    np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.int16),
    np.array([[5, 50], [6, 60], [7, 70], [8, 80]], dtype=np.int16),
    np.array([[9, 90], [10, 100], [11, 110], [12, 120]], dtype=np.int16),
]

for step, chunk in enumerate(chunks, start=1):
    buffer.append(chunk, step=step, timestamp=time.time())
    print(f"step={step}")
    print(buffer.latest(dtype=np.int16))
    print(buffer.status()["full"])
    time.sleep(0.1)
