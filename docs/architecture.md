# Architecture

The package intentionally separates concerns:

- `ring_buffer.py` holds the actual circular buffer logic.
- `config.py` defines a configuration dataclass for buffer setup.
- `dtype_utils.py` maps byte sizes to NumPy dtypes and validates compatibility.
- `manager.py` manages multiple independent buffers.
- `exceptions.py` centralizes invalid input and overflow errors.

This keeps validation and data handling independent from the algorithmic ring-buffer logic.

## Signal and channel model

One `RingBuffer` represents one signal. A signal may have one or more channels. Channel IDs are optional but supported, and they allow arbitrary identifiers such as `[101, 205, 999]`.

## Layout model

- `channel_major`: shape is `(num_channels, samples_per_channel)`
- `interleaved`: shape is `(samples_per_channel, num_channels)` or an equivalent frame-based view

The code accepts both forms and normalizes input internally.
