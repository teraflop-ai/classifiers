# linear-probe
```bash
ruff check --select I --fix . && ruff format .
```

Expected input is `features` and `labels` columns.

| task | labels per example | example |
|---|---|---|
| `binary` | scalar | `1`, `"spam"` |
| `multiclass` | scalar | `2`, `"sports"` |
| `multilabel` | list of labels present | `["politics", "eu"]`, `[0, 2]` |
