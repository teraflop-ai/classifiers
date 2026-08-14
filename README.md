# linear-probe
```bash
ruff check --select I --fix . && ruff format .
```

Expected input is `embeddings` and `labels` columns. `embeddings` must be pre-computed. `labels` are raw values and encoded automatically.

| task | labels | example |
|---|---|---|
| `binary` | scalar | `1`, `"spam"` |
| `multiclass` | scalar | `2`, `"sports"` |
| `multilabel` | list of labels present | `["politics", "eu"]`, `[0, 2]` |

```py
# binary
{"embeddings": [0.12, -0.53, 0.88, ...], "labels": "spam"}

# multiclass
{"embeddings": [0.12, -0.53, 0.88, ...], "labels": "sports"}

# multilabel
{"embeddings": [0.12, -0.53, 0.88, ...], "labels": ["politics", "eu"]}
```