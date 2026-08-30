import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from ringbuffer import RingBuffer


buffer = RingBuffer(signal_id=1, num_channels=1, samples_per_channel=5, dtype=np.int16)
payload = np.array([[10, 20, 30, 40, 50]], dtype=np.int16)

buffer.append(payload, step=1, timestamp=0.0)
print("latest:", buffer.latest(dtype=np.int16))
print("channels:", buffer.read_channels())
print("status:", buffer.status())
