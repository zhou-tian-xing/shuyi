"""Check the two trainer fixes: blocking mask and score-head round trip."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_resnet as T

CELLS, POLICY = T.CELLS, T.POLICY


def one_sample():
    data = np.zeros(1, dtype=T.V4_DTYPE)
    board = data["board"].reshape(1, 9, 9)
    board[0, 4, 4] = -9           # Black 9 in the centre
    data["side"] = 0              # Red to move
    data["hand"] = 3
    data["game"] = 7
    return data


def check_mask():
    data = one_sample()
    loose = int(T.make_legal_mask(data, 0).sum())
    strict = int(T.make_legal_mask(data, 2).sum())
    empty = 80
    print(f"classic  {loose} (expect {empty * 9} = 80 cells x 9 numbers)")
    print(f"nopass   {strict} (expect {loose - 4 * 9} = four neighbours x 9 numbers)")
    assert loose == empty * 9, loose
    assert strict == loose - 36, strict
    # The four neighbours of the centre lose all nine numbers (the centre itself
    # is occupied, so it has no action either way).
    mask = T.make_legal_mask(data, 2).reshape(9, 9, 9).numpy()
    empty_blocked = sorted({i * 9 + j for i in range(9) for j in range(9)
                            if not mask[i, j].any() and not (i == 4 and j == 4)})
    print("blocked empty cells:", empty_blocked, "(expect [31, 39, 41, 49])")
    assert empty_blocked == [31, 39, 41, 49], empty_blocked
    # A 5 next to a 3 only forbids numbers <= 3 on that cell.
    data = np.zeros(1, dtype=T.V4_DTYPE)
    data["board"].reshape(1, 9, 9)[0, 4, 4] = -3
    data["hand"] = 3
    mask = T.make_legal_mask(data, 2).reshape(9, 9, 9).numpy()
    allowed = mask[4, 3].sum()
    print(f"next to a 3, allowed numbers on that cell: {allowed} (expect 6: 4..9)")
    assert allowed == 6, allowed


def check_score_roundtrip(tmp=Path("build/_score_roundtrip")):
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "m.bin"
    model = T.ResNet(96, 6, 37, 2, T.GLOBAL)
    model.use_score_head = True
    with torch.no_grad():          # make the head distinctive
        for name in ("score_conv.weight", "score_conv.bias", "score_bias.weight",
                     "score_bias.bias", "score_out.weight", "score_out.bias"):
            module, _, param = name.rpartition(".")
            getattr(getattr(model, module), param).add_(1.5)
    T.save_bin(model, path)
    restored = T.ResNet(96, 6, 37, 2, T.GLOBAL)
    restored.use_score_head = True
    T.load_bin(restored, path)
    for name in ("score_conv.weight", "score_conv.bias", "score_bias.weight",
                 "score_bias.bias", "score_out.weight", "score_out.bias"):
        module, _, param = name.rpartition(".")
        a = getattr(getattr(model, module), param)
        b = getattr(getattr(restored, module), param)
        assert torch.equal(a, b), f"{name} not restored"
    print("score head round trip: all six tensors identical")
    path.unlink()
    tmp.rmdir()


def check_game_namespacing():
    tmp = Path("build/_shard_ns")
    tmp.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        data = np.zeros(4, dtype=T.V4_DTYPE)
        data["game"] = [0, 0, 1, 1]      # each shard restarts numbering
        data["hand"] = 3
        with (tmp / f"s{index}.bin").open("wb") as handle:
            handle.write(T.MAGIC4)
            handle.write(np.array([4, len(data)], dtype="<u4").tobytes())
            handle.write(np.array([2], dtype=np.uint8).tobytes() + b"\0\0\0")
            handle.write(data.tobytes())
    merged = T.load_shards([tmp / "s0.bin", tmp / "s1.bin"], 0, 1, 2)
    ids = sorted(set(merged["game"].tolist()))
    print(f"merged game ids: {ids} (expect 4 distinct, was 2 before the fix)")
    assert len(ids) == 4, ids
    for path in tmp.glob("*.bin"):
        path.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    check_mask()
    check_score_roundtrip()
    check_game_namespacing()
    print("all trainer checks passed")
