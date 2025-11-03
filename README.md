# smartcols

Utilities for reordering and grouping pandas DataFrame columns without the usual index gymnastics. `smartcols` keeps column manipulations readable and testable while exposing a single API that works with pure or in-place workflows.

## Features

- Swap, move before/after, push to front/end, or reorder multiple columns at once
- `sort_columns` with several strategies (`default`, `variance`, `std_dev`, `nan_ratio`, `correlation`, `mean`, `custom`)
- Column grouping helpers (`dtype`, NaN ratio buckets, regex, metadata maps, custom keys) and flatten helpers
- Optional `inplace=True` on every mutator to avoid extra copies without breaking method chains via `DataFrame.pipe`
- Typed API with validation and descriptive error messages for faster debugging

## Installation

```bash
pip install smartcols
```

## Quick start

```python
import pandas as pd
from smartcols import (
    swap_columns,
    move_after,
    move_before,
    move_to_front,
    move_to_end,
    sort_columns,
    group_columns,
)

# Base DataFrame
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9], "D": [10, 11, 12]})

# Swap columns
swap_df = swap_columns(df, "A", "D")

# Move multiple columns after a target
reordered = move_after(df, ["B", "C"], "A")

# Update in place
move_to_front(df, "D", inplace=True)

# Combine with pipe for readable pipelines
pipe_df = (
    df.copy()
      .pipe(move_to_front, ["C", "A"])
      .pipe(move_after, "B", "C")
)

# Sort by variance
variance_sorted = sort_columns(df, by="variance")

# Group by NaN ratio buckets
nan_groups = group_columns(
    pd.DataFrame({
        "good": [1, 2, 3],
        "ok": [1, None, 3],
        "bad": [None, None, 1],
        "terrible": [None, None, None],
    }),
    by="nan_ratio",
)
```

## API overview

| Function | Description |
| --- | --- |
| `swap_columns(df, col1, col2, *, inplace=False)` | Swap two columns by name |
| `move_after(df, cols, target, *, inplace=False)` | Move one or more columns immediately after target |
| `move_before(df, cols, target, *, inplace=False)` | Move columns immediately before target |
| `move_to_front(df, cols, *, inplace=False)` | Push column(s) to the front |
| `move_to_end(df, cols, *, inplace=False)` | Push column(s) to the end |
| `sort_columns(df, by='default', order='ascending', target='', key=None)` | Reorder columns by alphabetical order, NaN ratio, variance, std dev, correlation, mean, or custom key |
| `group_columns(df, by, ...)` | Split columns into groups based on dtype, NaN ratio buckets, regex patterns, metadata map, or custom key |
| `group_columns_flattened(...)` | Same as `group_columns` but returns a single DataFrame with grouped segments stacked |
| `save_column_order(df)` / `apply_column_order(df, order)` | Capture column order snapshots and reapply them |

Every mutating helper returns the reordered DataFrame unless `inplace=True`, in which case it returns `None` and the supplied DataFrame is updated.

## Testing

```bash
pip install -e .[tests]
pytest
```

(If pytest complains about temp directories in constrained environments, set `TMPDIR` to a writable location first.)

## Versioning

`smartcols` follows semantic versioning. Current release is `0.1.0`.

## License

MIT License. See `LICENSE` for details.
