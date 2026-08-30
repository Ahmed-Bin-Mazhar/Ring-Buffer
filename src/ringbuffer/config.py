from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RingBufferConfig:
    signal_id: int
    num_channels: int
    samples_per_channel: int
    dtype: np.dtype
    channel_ids: tuple[int, ...]
    input_layout: str = "channel_major"
    bytes_per_sample: Optional[int] = None
