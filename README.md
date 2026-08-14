# linear-probe
```bash
ruff check --select I --fix . && ruff format .
```

Expected input is `features` and `labels` columns.

| task | labels per example | example |
|---|---|---|
| `binary` | scalar `0`/`1` | `1` |
| `multilabel` | multi-hot list, length `num_labels` | `[1, 0, 1]` |
| `multiclass` | int class index `0..k-1` | `2` |

Multilabel must be multi-hot, not index lists. To convert `[0, 2]` → `[1, 0, 1]`:
```py
def to_multihot(ex, k):
    v = [0.0] * k
    for i in ex["labels"]:
        v[i] = 1.0
    return {"labels": v}

ds = ds.map(to_multihot, fn_kwargs={"k": num_labels})
```