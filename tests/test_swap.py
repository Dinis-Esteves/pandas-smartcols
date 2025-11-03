from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.smartcols.core import group_columns, swap_columns, move_after, move_before, move_to_end, move_to_front, sort_columns


def test_swap():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = swap_columns(df, "A", "C")
    assert list(result.columns) == ["C", "B", "A"]

def test_swap_reverse():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = swap_columns(df, "C", "A")
    assert list(result.columns) == ["C", "B", "A"]

def test_swap_nonexistent_column():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with pytest.raises(KeyError) as excinfo:
        swap_columns(df, "D", "E")
    assert excinfo.value.args[0] == (
        "Columns not found in DataFrame: col1='D', col2='E'. "
        "Available columns: ['A', 'B', 'C']"
    )

def test_swap_nonexistent_dataframe():
    with pytest.raises(ValueError) as excinfo:
        swap_columns(None, "A", "B")
    assert excinfo.value.args[0] == "`df` must be a pandas DataFrame, got None."

def test_swap_same_column():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = swap_columns(df, "A", "A")
    assert list(result.columns) == ["A", "B", "C"]

def test_swap_inplace_mutation():
    df = pd.DataFrame({"A": [1], "B": [2]})
    result = swap_columns(df, "A", "B", inplace=True)
    assert result is None
    assert list(df.columns) == ["B", "A"]

def test_move_after():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = move_after(df, "A", "C")
    assert list(result.columns) == ["B", "C", "A"]

def test_move_after_missing_column():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with pytest.raises(KeyError) as excinfo:
        move_after(df, "D", "A")
    assert excinfo.value.args[0] == (
        "Columns not found in DataFrame: col_to_move[0]='D'. "
        "Available columns: ['A', 'B', 'C']"
    )

def test_move_after_missing_target():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with pytest.raises(KeyError) as excinfo:
        move_after(df, "A", "Z")
    assert excinfo.value.args[0] == (
        "Columns not found in DataFrame: target_col='Z'. "
        "Available columns: ['A', 'B', 'C']"
    )

def test_move_after_nonexistent_dataframe():
    with pytest.raises(ValueError) as excinfo:
        move_after(None, "A", "B")
    assert excinfo.value.args[0] == "`df` must be a pandas DataFrame, got None."

def test_move_after_inplace():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = move_after(df, "A", "C", inplace=True)
    assert result is None
    assert list(df.columns) == ["B", "C", "A"]

def test_move_before():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = move_before(df, "C", "A")
    assert list(result.columns) == ["C", "A", "B"]

def test_move_before_missing_column():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with pytest.raises(KeyError) as excinfo:
        move_before(df, "Z", "A")
    assert excinfo.value.args[0] == (
        "Columns not found in DataFrame: col_to_move[0]='Z'. "
        "Available columns: ['A', 'B', 'C']"
    )

def test_move_before_missing_target():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with pytest.raises(KeyError) as excinfo:
        move_before(df, "C", "Z")
    assert excinfo.value.args[0] == (
        "Columns not found in DataFrame: target_col='Z'. "
        "Available columns: ['A', 'B', 'C']"
    )

def test_move_before_nonexistent_dataframe():
    with pytest.raises(ValueError) as excinfo:
        move_before(None, "A", "B")
    assert excinfo.value.args[0] == "`df` must be a pandas DataFrame, got None."

def test_move_before_inplace():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = move_before(df, "C", "A", inplace=True)
    assert result is None
    assert list(df.columns) == ["C", "A", "B"]

def test_move_to_end():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = move_to_end(df, "A")
    assert list(result.columns) == ["B", "C", "A"]

def test_move_to_end_missing_column():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with pytest.raises(KeyError) as excinfo:
        move_to_end(df, "Z")
    assert excinfo.value.args[0] == (
        "Columns not found in DataFrame: col_to_move[0]='Z'. "
        "Available columns: ['A', 'B', 'C']"
    )

def test_move_to_end_nonexistent_dataframe():
    with pytest.raises(ValueError) as excinfo:
        move_to_end(None, "A")
    assert excinfo.value.args[0] == "`df` must be a pandas DataFrame, got None."

def test_move_to_end_inplace():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = move_to_end(df, "A", inplace=True)
    assert result is None
    assert list(df.columns) == ["B", "C", "A"]

def test_move_to_front():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = move_to_front(df, "C")
    assert list(result.columns) == ["C", "A", "B"]

def test_move_to_front_missing_column():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with pytest.raises(KeyError) as excinfo:
        move_to_front(df, "Z")
    assert excinfo.value.args[0] == (
        "Columns not found in DataFrame: col_to_move[0]='Z'. "
        "Available columns: ['A', 'B', 'C']"
    )

def test_move_after_multiple_columns():
    df = pd.DataFrame({
        "A": [1],
        "B": [2],
        "C": [3],
        "D": [4],
        "E": [5],
    })
    result = move_after(df, ["C", "E"], "A")
    assert list(result.columns) == ["A", "C", "E", "B", "D"]

def test_move_before_multiple_columns():
    df = pd.DataFrame({
        "A": [1],
        "B": [2],
        "C": [3],
        "D": [4],
        "E": [5],
    })
    result = move_before(df, ["C", "E"], "B")
    assert list(result.columns) == ["A", "C", "E", "B", "D"]

def test_move_to_front_multiple_columns():
    df = pd.DataFrame({
        "A": [1],
        "B": [2],
        "C": [3],
        "D": [4],
        "E": [5],
    })
    result = move_to_front(df, ["D", "B"])
    assert list(result.columns) == ["D", "B", "A", "C", "E"]

def test_move_to_end_multiple_columns():
    df = pd.DataFrame({
        "A": [1],
        "B": [2],
        "C": [3],
        "D": [4],
        "E": [5],
    })
    result = move_to_end(df, ["A", "C"])
    assert list(result.columns) == ["B", "D", "E", "A", "C"]

def test_pipe_move_after_single_column():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = df.pipe(move_after, "A", "C")
    assert list(result.columns) == ["B", "C", "A"]

def test_pipe_move_before_multiple_columns():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4]})
    result = df.pipe(move_before, ["C", "D"], "B")
    assert list(result.columns) == ["A", "C", "D", "B"]

def test_pipe_move_to_front_chain():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4]})
    result = (
        df.pipe(move_to_front, "C")
          .pipe(move_to_front, "D")
    )
    assert list(result.columns) == ["D", "C", "A", "B"]

def test_pipe_move_to_end_multiple_columns():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4]})
    result = df.pipe(move_to_end, ["A", "B"])
    assert list(result.columns) == ["C", "D", "A", "B"]

def test_pipe_mixed_operations():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4]})
    result = (
        df.pipe(move_after, ["A", "B"], "C")
          .pipe(move_to_front, "D")
    )
    assert list(result.columns) == ["D", "C", "A", "B"]

def test_pipe_complex_chain():
    df = pd.DataFrame({
        "A": [1],
        "B": [2],
        "C": [3],
        "D": [4],
        "E": [5],
        "F": [6],
    })
    result = (
        df.pipe(move_to_front, ["E", "F"])
          .pipe(move_after, ["B", "C"], "E")
          .pipe(move_before, "A", "F")
          .pipe(move_to_end, "E")
    )
    assert list(result.columns) == ["B", "C", "A", "F", "D", "E"]

def test_move_after_bulk_including_target_raises():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    with pytest.raises(ValueError) as excinfo:
        move_after(df, ["A", "B", "C"], "C")
    assert excinfo.value.args[0] == "'C' is not in list"

def test_move_to_front_nonexistent_dataframe():
    with pytest.raises(ValueError) as excinfo:
        move_to_front(None, "A")
    assert excinfo.value.args[0] == "`df` must be a pandas DataFrame, got None."

def test_move_to_front_inplace():
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    result = move_to_front(df, "C", inplace=True)
    assert result is None
    assert list(df.columns) == ["C", "A", "B"]

def test_sort_columns_ascending():
    df = pd.DataFrame({"B": [2], "A": [1], "D": [4], "C": [3]})
    df_sorted_asc = sort_columns(df)
    assert list(df_sorted_asc.columns) == ["A", "B", "C", "D"]

def test_sort_columns_descending():
    df = pd.DataFrame({"B": [2], "A": [1], "D": [4], "C": [3]})
    df_sorted_desc = sort_columns(df, order='descending')
    assert list(df_sorted_desc.columns) == ["D", "C", "B", "A"]

def test_sort_columns_nonexistent_dataframe():
    with pytest.raises(ValueError) as excinfo:
        sort_columns(None)
    assert excinfo.value.args[0] == "`df` must be a pandas DataFrame, got None."

def test_sort_columns_invalid_order():
    df = pd.DataFrame({"B": [2], "A": [1]})
    with pytest.raises(ValueError) as excinfo:
        sort_columns(df, order='invalid_order')
    assert excinfo.value.args[0] == "`order` must be either 'ascending' or 'descending'."

def test_sort_columns_invalid_by():
    df = pd.DataFrame({"B": [2], "A": [1]})
    with pytest.raises(ValueError) as excinfo:
        sort_columns(df, by='invalid_by')
    assert excinfo.value.args[0] == "Currently, only 'default', 'nan_ratio', 'variance', 'std_dev', 'correlation', 'mean', and 'custom' sorting are supported."

def test_sort_columns_custom_key():
    df = pd.DataFrame({"BB": [2], "A": [1], "CCC": [3]})
    df_sorted_custom = sort_columns(df, by='custom', key=lambda col: len(col))
    assert list(df_sorted_custom.columns) == ["A", "BB", "CCC"]

def test_sort_columns_custom_key_descending():
    df = pd.DataFrame({"BB": [2], "A": [1], "CCC": [3]})
    df_sorted_custom_desc = sort_columns(df, by='custom', order='descending', key=lambda col: len(col))
    assert list(df_sorted_custom_desc.columns) == ["CCC", "BB", "A"]

def test_sort_columns_custom_key_missing_key():
    df = pd.DataFrame({"BB": [2], "A": [1], "CCC": [3]})
    with pytest.raises(ValueError) as excinfo:
        sort_columns(df, by='custom')
    assert excinfo.value.args[0] == "`key` must be provided for 'custom' sorting."

def test_sort_columns_nan_ratio():
    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": [None, None, None],
        "C": [1, 2, 3]
    })
    df_sorted_nan_ratio = sort_columns(df, by='nan_ratio', order='ascending')
    assert list(df_sorted_nan_ratio.columns) == ["C", "A", "B"]

def test_sort_columns_nan_ratio_descending():
    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": [None, None, None],
        "C": [1, 2, 3]
    })
    df_sorted_nan_ratio_desc = sort_columns(df, by='nan_ratio', order='descending')
    assert list(df_sorted_nan_ratio_desc.columns) == ["B", "A", "C"]

def test_sort_columns_std_dev():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 1, 1],
        "C": [1, 2, 4]
    })
    df_sorted_std_dev = sort_columns(df, by='std_dev', order='ascending')
    assert list(df_sorted_std_dev.columns) == ["B", "A", "C"]

def test_sort_columns_std_dev_descending():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 1, 1],
        "C": [1, 2, 4]
    })
    df_sorted_std_dev_desc = sort_columns(df, by='std_dev', order='descending')
    assert list(df_sorted_std_dev_desc.columns) == ["C", "A", "B"]

def test_sort_columns_variance():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 1, 1],
        "C": [1, 2, 4]
    })
    df_sorted_variance = sort_columns(df, by='variance')
    assert list(df_sorted_variance.columns) == ["B", "A", "C"]

def test_sort_columns_variance_descending():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1, 1, 1],
        "C": [1, 2, 4]
    })
    df_sorted_variance_desc = sort_columns(df, by='variance', order='descending')
    assert list(df_sorted_variance_desc.columns) == ["C", "A", "B"]

def test_sort_columns_correlation():
    df6 = pd.DataFrame({
        "sales": [100, 150, 200, 250, 300],
        "advertising": [10, 15, 20, 25, 30],
        "price": [5, 4.5, 4, 3.5, 3],
        "season": [1, 2, 1, 2, 1]
    })
    df_sorted_correlation = sort_columns(df6, by='correlation', target='sales')
    assert list(df_sorted_correlation.columns) == ["sales", "season", "advertising", "price"]

def test_sort_columns_correlation_descending():
    df6 = pd.DataFrame({
        "sales": [100, 150, 200, 250, 300],
        "advertising": [10, 15, 20, 25, 30],
        "price": [5, 4.5, 4, 3.5, 3],
        "season": [1, 2, 1, 2, 1]
    })
    df_sorted_correlation_desc = sort_columns(df6, by='correlation', order='descending', target='sales')
    print(df_sorted_correlation_desc)
    assert list(df_sorted_correlation_desc.columns) == ["sales", "advertising", "price", "season"]

def test_sort_columns_correlation_missing_target():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    with pytest.raises(KeyError) as excinfo:
        sort_columns(df, by='correlation', target='C')
    assert excinfo.value.args[0] == (
        "Target column 'C' not found in DataFrame. "
        "Available columns: ['A', 'B']"
    )

def test_sort_columns_mean():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [0, 0, 0]
    })
    df_sorted_mean = sort_columns(df, by='mean')
    assert list(df_sorted_mean.columns) == ["C", "A", "B"]

def test_sort_columns_mean_descending():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [0, 0, 0]
    })
    df_sorted_mean_desc = sort_columns(df, by='mean', order='descending')
    assert list(df_sorted_mean_desc.columns) == ["B", "A", "C"]

def test_group_columns_dtype():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [1.0, 2.0, 3.0],
        "C": ["a", "b", "c"],
        "D": [True, False, True]
    })
    grouped_by_dtype = group_columns(df, by='dtype')
    assert set(grouped_by_dtype.keys()) == { 'int64', 'float64', 'object', 'bool' }
    assert list(grouped_by_dtype['int64'].columns) == ['A']
    assert list(grouped_by_dtype['float64'].columns) == ['B']
    assert list(grouped_by_dtype['object'].columns) == ['C']
    assert list(grouped_by_dtype['bool'].columns) == ['D']

def test_group_columns_regex():
    df = pd.DataFrame({
        "alpha_1": [1, 2, 3],
        "alpha_2": [4, 5, 6],
        "beta_1": [7, 8, 9],
        "beta_2": [10, 11, 12]
    })
    grouped_by_regex = group_columns(df, by='regex', pattern='^alpha')
    assert set(grouped_by_regex.keys()) == { 'match', 'other' }
    assert list(grouped_by_regex['match'].columns) == ['alpha_1', 'alpha_2']
    assert list(grouped_by_regex['other'].columns) == ['beta_1', 'beta_2']

def test_group_columns_invalid_by():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    with pytest.raises(ValueError) as excinfo:
        group_columns(df, by='invalid_by')
    assert excinfo.value.args[0] == "`by` must be one of: 'dtype', 'nan_ratio', 'regex', 'meta', or 'custom'."

def test_group_columns_regex_missing_pattern():
    df = pd.DataFrame({
        "alpha_1": [1, 2, 3],
        "alpha_2": [4, 5, 6]
    })
    with pytest.raises(ValueError) as excinfo:
        group_columns(df, by='regex')
    assert excinfo.value.args[0] == "`pattern` must be provided when grouping by 'regex'."

def test_group_columns_nonexistent_dataframe():
    with pytest.raises(ValueError) as excinfo:
        group_columns(None, by='dtype')
    assert excinfo.value.args[0] == "`df` must be a pandas DataFrame, got None."

def test_group_columns_meta_missing_metadata():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    with pytest.raises(ValueError) as excinfo:
        group_columns(df, by='meta')
    assert excinfo.value.args[0] == "`meta` must be provided when grouping by 'meta'."

def test_group_columns_meta():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    metadata = {
        "A": "group1",
        "B": "group2",
        "C": "group1"
    }
    grouped_by_meta = group_columns(df, by='meta', meta=metadata)
    assert set(grouped_by_meta.keys()) == { 'group1', 'group2' }
    assert list(grouped_by_meta['group1'].columns) == ['A', 'C']
    assert list(grouped_by_meta['group2'].columns) == ['B']

def test_group_columns_nan_ratio():
    df = pd.DataFrame({
        "A": [1, None, 3],
        "B": [None, None, None],
        "C": [1, 2, 3],
        "D": [1, None, None],
    })
    grouped_by_nan_ratio = group_columns(df, by='nan_ratio')
    assert set(grouped_by_nan_ratio.keys()) == {'0%', '20-50%', '>50%'}
    assert list(grouped_by_nan_ratio['0%'].columns) == ['C']
    assert list(grouped_by_nan_ratio['20-50%'].columns) == ['A']
    assert list(grouped_by_nan_ratio['>50%'].columns) == ['B', 'D']

def test_group_columns_custom_missing_key():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
    with pytest.raises(ValueError) as excinfo:
        group_columns(df, by='custom')
    assert excinfo.value.args[0] == "`key` must be provided for custom grouping."

def test_group_columns_custom():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9],
        "D": [True, False, True],
        "ABCD": [1, 2, 3],
        'XYZ': [4, 5, 6],
        'XCVB': [7, 8, 9]
    })
    def custom_key(col):
        return 'even' if len(col) % 2 == 0 else 'odd'
    grouped_by_custom = group_columns(df, by='custom', key=custom_key)
    assert set(grouped_by_custom.keys()) == { 'odd', 'even' }
    assert list(grouped_by_custom['odd'].columns) == ['A', 'B', 'C', 'D', 'XYZ']
    assert list(grouped_by_custom['even'].columns) == ['ABCD', 'XCVB']
