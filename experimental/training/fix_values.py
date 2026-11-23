import struct
import sys

import numpy as np

path = sys.argv[1]
d = np.dtype(
    [
        ("side", "u1"),
        ("board", "i1", (81,)),
        ("hand", "u1", (18,)),
        ("policy", "<f4", (729,)),
        ("value", "<f4"),
        ("game", "<u4"),
        ("ply", "u1"),
    ]
)
with open(path, "rb") as f:
    assert f.read(8) == b"SYSELF2\x00"
    version, count = struct.unpack("<II", f.read(8))
    a = np.fromfile(f, dtype=d, count=count)
assert len(a) == count, (len(a), count)

side = a["side"].astype(np.int64)
v = a["value"].astype(np.float64)
before = np.unique(v)
red = np.where(side == 0, v, 1.0 - v)
red = np.clip(red, 0.0, 1.0)
fixed = np.where(side == 0, red, 1.0 - red).astype("<f4")
a["value"] = fixed
after = np.unique(a["value"].astype(np.float64))
print(f"before {before} -> after {after}")
with open(path, "wb") as f:
    f.write(b"SYSELF2\x00")
    f.write(struct.pack("<II", version, count))
    a.tofile(f)
print(f"rewrote {count} samples")
