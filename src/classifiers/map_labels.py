import simdjson as json


def encode_labels(ds, task, path="label2id.json"):
    """String labels -> ids. Returns (ds, label2id)."""
    if task == "multilabel":
        names = sorted({l for ex in ds["labels"] for l in ex})
        label2id = {l: i for i, l in enumerate(names)}
        ds = ds.map(lambda ex: {"labels": [float(l in ex["labels"]) for l in names]})
    elif task in ("binary", "multiclass"):
        names = sorted(set(ds["labels"]))
        label2id = {l: i for i, l in enumerate(names)}
        ds = ds.map(lambda ex: {"labels": label2id[ex["labels"]]})
    else:
        raise ValueError(f"unknown task: {task!r}")
    with open(path, "w") as f:
        json.dump(label2id, f)
    return ds, label2id
