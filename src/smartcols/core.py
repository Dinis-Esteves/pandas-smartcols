__version__ = "1.0.0"
__author__ = "Dinis Esteves"
__all__ = [
    "swap_columns", "move_after", "move_before", "move_to_end", "move_to_front",
    "sort_columns", "group_columns", "group_columns_flattened",
    "save_column_order", "apply_column_order"
]

import re
import pandas as pd
import numpy as np
from typing import Dict, Iterable, Optional, Tuple, Callable, Union
import warnings

def _validate_dataframe(df: pd.DataFrame) -> None:
    if df is None:
        raise ValueError("`df` must be a pandas DataFrame, got None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame, got {type(df).__name__}.")

def _ensure_columns_exist(df: pd.DataFrame, columns: Iterable[Tuple[str, str]]) -> None:
    missing = [
        f"{param_name}='{column_name}'"
        for param_name, column_name in columns
        if column_name not in df.columns
    ]
    if missing:
        formatted = ", ".join(missing)
        raise KeyError(
            f"Columns not found in DataFrame: {formatted}. "
            f"Available columns: {list(df.columns)}"
        )

def _normalize_cols(cols: Union[str, list[str]]) -> list[str]:
    """Ensure columns argument is always a list."""
    return [cols] if isinstance(cols, str) else list(cols)

def _apply_column_order(df: pd.DataFrame, new_order: list[str], inplace: bool) -> Optional[pd.DataFrame]:
    """
    Reorder DataFrame columns, mutating in-place when requested.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    new_order (list[str]): The desired column order.
    inplace (bool): If True, modifies df directly and returns None.

    Returns:
    pd.DataFrame | None: Reordered DataFrame or None if modified in-place.
    """
    if inplace:
        reordered = df[new_order].copy()
        df.drop(columns=df.columns, inplace=True)
        for col in reordered.columns:
            df[col] = reordered[col]
        return None

    return df[new_order]

def swap_columns(df: pd.DataFrame, col1: str, col2: str, *, inplace: bool = False) -> Optional[pd.DataFrame]:
    """
    Swap two columns in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): The name of the first column to swap.
    col2 (str): The name of the second column to swap.

    inplace (bool): If True, mutate ``df`` and return None.

    Returns:
    pd.DataFrame | None: A new DataFrame with columns swapped, or None when inplace.
    """
    _validate_dataframe(df)
    _ensure_columns_exist(df, [("col1", col1), ("col2", col2)])
    cols = list(df.columns)
    i1, i2 = cols.index(col1), cols.index(col2)
    cols[i1], cols[i2] = cols[i2], cols[i1]
    return _apply_column_order(df, cols, inplace)

def move_after(
    df: pd.DataFrame,
    cols_to_move: Union[str, list[str]],
    target_col: str,
    *,
    inplace: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Move one or more columns so they appear immediately after the target column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    cols_to_move (Union[str, list[str]]): Column name or list of column names to relocate.
    target_col (str): Column after which the selected columns will be inserted.

    inplace (bool): If True, mutate ``df`` and return None.

    Returns:
    pd.DataFrame | None: Reordered DataFrame or None if mutated in-place.
    """
    _validate_dataframe(df)
    cols_to_move = _normalize_cols(cols_to_move)
    _ensure_columns_exist(
        df,
        [(f"col_to_move[{i}]", col) for i, col in enumerate(cols_to_move)] + [("target_col", target_col)],
    )

    cols = list(df.columns)
    for col in cols_to_move:
        cols.remove(col)
    target_index = cols.index(target_col)
    for offset, col in enumerate(cols_to_move):
        cols.insert(target_index + 1 + offset, col)
    return _apply_column_order(df, cols, inplace)

def move_before(
    df: pd.DataFrame,
    cols_to_move: Union[str, list[str]],
    target_col: str,
    *,
    inplace: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Move one or more columns so they appear immediately before the target column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    cols_to_move (Union[str, list[str]]): Column name or list of column names to relocate.
    target_col (str): Column before which the selected columns will be inserted.

    inplace (bool): If True, mutate ``df`` and return None.

    Returns:
    pd.DataFrame | None: Reordered DataFrame or None if mutated in-place.
    """
    _validate_dataframe(df)
    cols_to_move = _normalize_cols(cols_to_move)
    _ensure_columns_exist(
        df,
        [(f"col_to_move[{i}]", col) for i, col in enumerate(cols_to_move)] + [("target_col", target_col)],
    )

    cols = list(df.columns)
    for col in cols_to_move:
        cols.remove(col)
    target_index = cols.index(target_col)
    for i, col in enumerate(cols_to_move):
        cols.insert(target_index + i, col)
    return _apply_column_order(df, cols, inplace)

def move_to_end(
    df: pd.DataFrame,
    cols_to_move: Union[str, list[str]],
    *,
    inplace: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Move one or more columns so they appear at the end of the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    cols_to_move (Union[str, list[str]]): Column name or list of column names to relocate.

    inplace (bool): If True, mutate ``df`` and return None.

    Returns:
    pd.DataFrame | None: Reordered DataFrame or None if mutated in-place.
    """
    _validate_dataframe(df)
    cols_to_move = _normalize_cols(cols_to_move)
    _ensure_columns_exist(df, [(f"col_to_move[{i}]", col) for i, col in enumerate(cols_to_move)])

    cols = list(df.columns)
    for col in cols_to_move:
        cols.remove(col)
    cols.extend(cols_to_move)
    return _apply_column_order(df, cols, inplace)

def move_to_front(
    df: pd.DataFrame,
    cols_to_move: Union[str, list[str]],
    *,
    inplace: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Move one or more columns so they appear at the front of the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    cols_to_move (Union[str, list[str]]): Column name or list of column names to relocate.

    inplace (bool): If True, mutate ``df`` and return None.

    Returns:
    pd.DataFrame | None: Reordered DataFrame or None if mutated in-place.
    """
    _validate_dataframe(df)
    cols_to_move = _normalize_cols(cols_to_move)
    _ensure_columns_exist(df, [(f"col_to_move[{i}]", col) for i, col in enumerate(cols_to_move)])

    cols = list(df.columns)
    for col in cols_to_move:
        cols.remove(col)
    cols = cols_to_move + cols
    return _apply_column_order(df, cols, inplace)

def sort_columns(df: pd.DataFrame, by: str = 'default', order: str = 'ascending', target: str = '', key: Optional[Callable] = None) -> pd.DataFrame:
    """
    Sort the columns of a pandas DataFrame based on a specified order.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    by (str): The criterion to sort by.
    order (str): The order of sorting, either 'ascending' or 'descending'.
    target (str): The target column to sort by (if applicable).
    key (callable): A function to apply to each column before sorting.

    Returns:
    pd.DataFrame: A new DataFrame with columns sorted as specified.
    """
    _validate_dataframe(df)

    if order not in ['ascending', 'descending']:
        raise ValueError("`order` must be either 'ascending' or 'descending'.")
    if by not in ['default', 'nan_ratio', 'variance', 'std_dev', 'correlation', 'mean', 'custom']:
        raise ValueError("Currently, only 'default', 'nan_ratio', 'variance', 'std_dev', 'correlation', 'mean', and 'custom' sorting are supported.")

    reverse = order == 'descending'

    if by == 'default':
        cols = list(df.columns)
        cols.sort(reverse=reverse)
        return df[cols]
    
    if by == 'nan_ratio':
        nan_ratios = df.isna().mean()
        sorted_cols = nan_ratios.sort_values(ascending=not reverse).index.tolist()
        return df[sorted_cols]
    
    if by == 'variance':
        variances = df.var(numeric_only=True)
        sorted_cols = variances.sort_values(ascending=not reverse).index.tolist()
        return df[sorted_cols]

    if by == 'std_dev':
        std_devs = df.std(numeric_only=True)
        sorted_cols = std_devs.sort_values(ascending=not reverse).index.tolist()
        return df[sorted_cols]

    if by == 'correlation':
        if target not in df.columns:
            raise KeyError(
                f"Target column '{target}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            correlations = df.corrwith(df[target], numeric_only=True)

        correlations = correlations.drop(target, errors="ignore").fillna(0).abs()

        # reverse=True (descending) => ascending=False so strongest correlations lead
        sorted_cols = correlations.sort_values(ascending=not reverse).index.tolist()
        return df[[target] + sorted_cols]

    if by == 'mean':
        means = df.mean(numeric_only=True)
        sorted_cols = means.sort_values(ascending=not reverse).index.tolist()
        return df[sorted_cols]

    if by == 'custom':
        if key is None:
            raise ValueError("`key` must be provided for 'custom' sorting.")
        sorted_cols = sorted(df.columns, key=key, reverse=reverse)
        return df[sorted_cols]

def group_columns(
    df: pd.DataFrame,
    by: str = "dtype",
    pattern: str = "",
    meta: Optional[Dict[str, str]] = None,
    key: Optional[Callable] = None,
    sort_within: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Group the columns of a pandas DataFrame based on a specified criterion.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    by (str): Grouping criterion. One of:
              'dtype', 'nan_ratio', 'regex', 'meta', 'custom'.
    pattern (str): Regex pattern (used if by='regex').
    meta (dict): Mapping of column name/prefix to group label (used if by='meta').
    key (callable): Function to compute a grouping key (used if by='custom').
    sort_within (bool): Sort columns alphabetically within each group.

    Returns:
    dict[str, pd.DataFrame]: A dictionary where each key is a group name and each value is a sub-DataFrame.
    """
    _validate_dataframe(df)
    
    if by not in ["dtype", "nan_ratio", "regex", "meta", "custom"]:
        raise ValueError("`by` must be one of: 'dtype', 'nan_ratio', 'regex', 'meta', or 'custom'.")

    groups: Dict[str, pd.DataFrame] = {}

    if by == "dtype":
        for dtype, cols in df.columns.to_series().groupby(df.dtypes):
            col_list = sorted(cols.tolist()) if sort_within else cols.tolist()
            groups[str(dtype)] = df[col_list]
        return groups

    if by == "regex":
        if not pattern:
            raise ValueError("`pattern` must be provided when grouping by 'regex'.")
        matches = [c for c in df.columns if re.search(pattern, c)]
        non_matches = [c for c in df.columns if not re.search(pattern, c)]
        if matches:
            groups["match"] = df[matches]
        if non_matches:
            groups["other"] = df[non_matches]
        return groups

    if by == "nan_ratio":
        nan_ratio = df.isna().mean()
        bins = {
            "0%": nan_ratio[nan_ratio == 0].index,
            "0-20%": nan_ratio[(nan_ratio > 0) & (nan_ratio <= 0.2)].index,
            "20-50%": nan_ratio[(nan_ratio > 0.2) & (nan_ratio <= 0.5)].index,
            ">50%": nan_ratio[nan_ratio > 0.5].index,
        }
        for label, cols in bins.items():
            if len(cols):
                groups[label] = df[sorted(cols) if sort_within else cols]
        return groups

    if by == "meta":
        if meta is None:
            raise ValueError("`meta` must be provided when grouping by 'meta'.")
        for label in set(meta.values()):
            cols = [c for c in df.columns if any(c.startswith(k) for k, v in meta.items() if v == label)]
            if cols:
                col_list = sorted(cols) if sort_within else cols
                groups[label] = df[col_list]
        return groups

    if by == "custom":
        if key is None:
            raise ValueError("`key` must be provided for custom grouping.")
        group_map: Dict[str, list] = {}
        for col in df.columns:
            group_key = key(col)
            group_map.setdefault(group_key, []).append(col)
        for group, cols in group_map.items():
            col_list = sorted(cols) if sort_within else cols
            groups[str(group)] = df[col_list]
        return groups

def save_column_order(df: pd.DataFrame) -> list[str]:
    """
    Save the current column order of a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    """
    _validate_dataframe(df)
    return list(df.columns)

def apply_column_order(df: pd.DataFrame, column_order: list[str]) -> pd.DataFrame:
    """
    Apply a saved column order to a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_order (list[str]): The desired column order.

    Returns:
    pd.DataFrame: A new DataFrame with columns ordered as specified.
    """
    _validate_dataframe(df)
    _ensure_columns_exist(df, [(f"column_order[{i}]", col) for i, col in enumerate(column_order)])
    return df[column_order]

def group_columns_flattened(
    df: pd.DataFrame,
    by: str = "dtype",
    pattern: str = "",
    meta: Optional[Dict[str, str]] = None,
    key: Optional[Callable] = None,
    sort_within: bool = True,
) -> pd.DataFrame:
    """
    Group the columns of a pandas DataFrame based on a specified criterion and return a flattened DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    by (str): Grouping criterion. One of:
              'dtype', 'nan_ratio', 'regex', 'meta', 'custom'.
    pattern (str): Regex pattern (used if by='regex').
    meta (dict): Mapping of column name/prefix to group label (used if by='meta').
    key (callable): Function to compute a grouping key (used if by='custom').
    sort_within (bool): Sort columns alphabetically within each group.

    Returns:
    pd.DataFrame: A new DataFrame with columns grouped as specified.
    """
    grouped = group_columns(df, by=by, pattern=pattern, meta=meta, key=key, sort_within=sort_within)
    flattened_cols = []
    for group in grouped.values():
        flattened_cols.extend(group.columns.tolist())
    return df[flattened_cols]

# Example usage:
if __name__ == "__main__":

    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }

    data2 = {
        'B': [4, 5, 6],
        'C': [7, 8, 9],
        'A': [1, 2, 3]
    }

    data3 = {
        'C': [7, np.nan, 9],
        'A': [np.nan, np.nan, np.nan],
        'B': [np.nan, 5, np.nan]
    }

    data4 = {
        'A': [1, 3, 9],
        'B': [4, 4, 6],
        'C': [7, 12, 18]
    }

    data5 = {
        'X': [10, 20, 30],
        'Yx': [40, 50, 60],
        'Zxy': [70, 80, 90]
    }

    data6 = {
        "sales":     [10, 20, 30, 40, 50],   # target
        "ads":       [1, 2, 3, 4, 5],        # perfectly correlated (+1.0)
        "returns":   [5, 4, 3, 2, 1],        # perfectly anti-correlated (−1.0)
        "discount":  [2, 2, 3, 3, 3],        # weakly correlated
        "constant":  [7, 7, 7, 7, 7]         # zero variance → NaN correlation
    }

    data7 = {
        'A': [1, 2, 3],
        'B': [True, False, True],
        'C': ['x', 'y', 'z'],
        'D': [1.5, 2.5, 3.5]
    }

    data8 = {
        'alpha_1': [1, 2, 3],
        'alpha_2': [4, 5, 6],
        'beta_1': [7, 8, 9],
        'beta_2': [10, 11, 12]
    }

    data9 = {
        'gamma_1': [13, 14, 15],
        'delta_1': [19, 20, 21],
        'gamma_2': [16, 17, 18],
        'delta_2': [22, 23, 24]
    }

    df = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)
    df5 = pd.DataFrame(data5)
    df6 = pd.DataFrame(data6)
    df7 = pd.DataFrame(data7)
    df8 = pd.DataFrame(data8)
    df9 = pd.DataFrame(data9)
    print("Original DataFrames:")
    print("\nDataFrame 1:")
    print(df)
    print("\nDataFrame 2:")
    print(df2)
    print("\nDataFrame 3:")
    print(df3)
    print("\nDataFrame 4:")
    print(df4)
    print("\nDataFrame 5:")
    print(df5)
    print("\nDataFrame 6:")
    print(df6)
    print("\nDataFrame 7:")
    print(df7)
    print("\nDataFrame 8:")
    print(df8)
    print("\nDataFrame 9:")
    print(df9)
    
    print("\n--- Column Manipulations ---")

    # Swap columns 'A' and 'B'
    df_swapped = swap_columns(df, 'A', 'B')
    print("\nDataFrame after swapping 'A' and 'B':")
    print(df_swapped)

    # Move column 'C' after column 'A'
    df_moved_after = move_after(df, 'C', 'A')
    print("\nDataFrame after moving 'C' after 'A':")
    print(df_moved_after)

    # Move column 'A' before column 'C'
    df_moved_before = move_before(df, 'A', 'C')
    print("\nDataFrame after moving 'A' before 'C':")
    print(df_moved_before)

    # Move column 'B' to the front
    df_moved_front = move_to_front(df, 'B')
    print("\nDataFrame after moving 'B' to the front:")
    print(df_moved_front)

    # Move column 'C' to the end
    df_moved_end = move_to_end(df, 'A')
    print("\nDataFrame after moving 'A' to the end:")
    print(df_moved_end)

    # Sort columns in ascending order in DataFrame 2
    df_sorted = sort_columns(df2)
    print("\nDataFrame after sorting columns in ascending order in DataFrame 2:")
    print(df_sorted)

    # Sort columns in descending order in DataFrame 2
    df_sorted_desc = sort_columns(df2, order='descending')
    print("\nDataFrame after sorting columns in descending order in DataFrame 2:")
    print(df_sorted_desc)

    # Sort columns by NaN ratio (though no NaNs in this example)
    df_sorted_nan = sort_columns(df3, by='nan_ratio')
    print("\nDataFrame after sorting columns by NaN ratio in DataFrame 3:")
    print(df_sorted_nan)

    # Sort columns by NaN ratio in descending order
    df_sorted_nan_desc = sort_columns(df3, by='nan_ratio', order='descending')
    print("\nDataFrame after sorting columns by NaN ratio in descending order in DataFrame 3:")
    print(df_sorted_nan_desc)

    # Sort columns by variance
    df_sorted_variance = sort_columns(df4, by='variance')
    print("\nDataFrame after sorting columns by variance in DataFrame 4:")
    print(df_sorted_variance)

    # Sort columns by variance in descending order
    df_sorted_variance_desc = sort_columns(df4, by='variance', order='descending')
    print("\nDataFrame after sorting columns by variance in descending order in DataFrame 4:")
    print(df_sorted_variance_desc)

    # Sort columns by standard deviation
    df_sorted_std_dev = sort_columns(df4, by='std_dev')
    print("\nDataFrame after sorting columns by standard deviation in DataFrame 4:")
    print(df_sorted_std_dev)

    # Sort columns by standard deviation in descending order
    df_sorted_std_dev_desc = sort_columns(df4, by='std_dev', order='descending')
    print("\nDataFrame after sorting columns by standard deviation in descending order in DataFrame 4:")
    print(df_sorted_std_dev_desc)

    # Sort columns using a custom key (length of column name)
    df_sorted_custom = sort_columns(df5, by='custom', key=lambda col: len(col))
    print("\nDataFrame after sorting columns by custom key (length of column name) in DataFrame 5:")
    print(df_sorted_custom)

    # Sort columns using a custom key in descending order
    df_sorted_custom_desc = sort_columns(df5, by='custom', order='descending', key=lambda col: len(col))
    print("\nDataFrame after sorting columns by custom key (length of column name) in descending order in DataFrame 5:")
    print(df_sorted_custom_desc)

    # Sort columns by correlation with 'sales' column
    df6 = pd.DataFrame(data6)
    df_sorted_correlation = sort_columns(df6, by='correlation', target='sales')
    print("\nDataFrame after sorting columns by correlation with 'sales' column in DataFrame 6:")
    print(df_sorted_correlation)

    # Sort columns by correlation with 'sales' column in descending order
    df_sorted_correlation_desc = sort_columns(df6, by='correlation', order='descending', target='sales')
    print("\nDataFrame after sorting columns by correlation with 'sales' column in descending order in DataFrame 6:")
    print(df_sorted_correlation_desc)

    # Sort columns by mean value
    df_sorted_mean = sort_columns(df6, by='mean')
    print("\nDataFrame after sorting columns by mean value in DataFrame 6:")
    print(df_sorted_mean)

    # Sort columns by mean value in descending order
    df_sorted_mean_desc = sort_columns(df6, by='mean', order='descending')
    print("\nDataFrame after sorting columns by mean value in descending order in DataFrame 6:")
    print(df_sorted_mean_desc)

    # Group columns by data type
    grouped_by_dtype = group_columns(df7, by='dtype')
    print("\nGrouped columns by data type in DataFrame 7:")
    for group, sub_df in grouped_by_dtype.items():
        print(f"\nGroup: {group}")
        print(sub_df)

    # Group columns by regex pattern
    grouped_by_regex = group_columns(df8, by='regex', pattern='^alpha')
    print("\nGrouped columns by regex pattern in DataFrame 8:")
    for group, sub_df in grouped_by_regex.items():
        print(f"\nGroup: {group}")
        print(sub_df)

    # Group columns by meta mapping
    meta_mapping = {
        'alpha': 'Group 1',
        'beta': 'Group 2'
    }
    grouped_by_meta = group_columns(df8, by='meta', meta=meta_mapping)
    print("\nGrouped columns by meta mapping in DataFrame 8:")
    for group, sub_df in grouped_by_meta.items():
        print(f"\nGroup: {group}")
        print(sub_df)

    # Group columns by custom key (column length even/odd)
    grouped_by_custom = group_columns(df6, by='custom', key=lambda col: 'even' if len(col) % 2 == 0 else 'odd')
    print("\nGrouped columns by custom key in DataFrame 6:")
    for group, sub_df in grouped_by_custom.items():
        print(f"\nGroup: {group}")
        print(sub_df)
    
    # Flatten grouped columns by regex
    flattened_by_regex = group_columns_flattened(df9, by='regex', pattern='^gamma')
    print("\nFlattened grouped columns by regex in DataFrame 9:")
    print(flattened_by_regex)

    # Group columns by nan ratio
    grouped_by_nan_ratio = group_columns(df3, by='nan_ratio')
    print("\nGrouped columns by NaN ratio in DataFrame 3:")
    for group, sub_df in grouped_by_nan_ratio.items():
        print(f"\nGroup: {group}")
        print(sub_df)

@pd.api.extensions.register_dataframe_accessor("cols")
class ColumnOps:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    def move_to_front(self, cols, **kwargs):
        return move_to_front(self._obj, cols, **kwargs)
    def move_after(self, cols, target, **kwargs):
        return move_after(self._obj, cols, target, **kwargs)
    def sort(self, **kwargs):
        return sort_columns(self._obj, **kwargs)
