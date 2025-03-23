import pandas as pd
import warnings
import numpy as np
import re


# def replace_commas(df):
#     # return df.map(lambda x: x.replace(',', '|') if isinstance(x, str) else x)
#     return df


# Code does a better default check of whether a column type can be a numeric (Int/Float)
def is_convertible_to_bool(series):
    bit_values = {0, 1, '0', '1', True, False, 'True', 'False'}
    series_as_str = series.astype(str).str.strip()
    return series_as_str.isin(bit_values).all()


def is_convertible_to_int(series):
    # First handle empty series
    if len(series) == 0:
        return True

    try:
        # Handle inf values which have is_integer() = True but aren't really integers
        if np.inf in series.values or -np.inf in series.values:
            return False

        series = series.astype(float)
        return series.apply(lambda x: x.is_integer() if pd.notna(x) else True).all()
    except (ValueError, TypeError):
        return False


        # try:
        #     series = pd.to_numeric(series, errors='raise')
        #     return is_convertible_to_int(series)
        # except (ValueError, TypeError):
        #     return False


def is_convertible_to_float(series):
    # Check if all elements in the series are float
    if len(series) == 0:
        return True

    try:
        # Handle inf values which have is_integer() = True but aren't really integers
        if np.inf in series.values or -np.inf in series.values:
            return False

        series.astype(float)
        return True

    except (ValueError, TypeError):
        return False


def normalize_whitespace(s):
    # Replace multiple whitespace characters with a single space
    return re.sub(r'\s+', ' ', str(s).strip())


def datetime_normalization(series):
    try:
        # First attempt: try to parse as-is
        print(series)
        return pd.to_datetime(series, errors='raise')
    except ValueError:
        # If it fails, normalize whitespace and try again
        normalized_series = series.apply(normalize_whitespace)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pd.to_datetime(normalized_series, errors='raise')
            if any(issubclass(warning.category, UserWarning) for warning in w):
                print("UserWarning: The datetime format could not be inferred even after whitespace normalization.")
            return result


def is_convertible_to_date(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    else:
        try:
            datetime_normalization(series)
            return True
        except (ValueError, TypeError, AttributeError):
            return False


# def is_convertible_to_date(series):
#     if pd.api.types.is_datetime64_any_dtype(series):
#         return True
#     else:
#         try:
#             with warnings.catch_warnings(record=True) as w:
#                 warnings.simplefilter("always")
#                 series = pd.to_datetime(series, errors='raise')
#
#                 # Check if any warnings were raised
#                 for warning in w:
#                     if issubclass(warning.category, UserWarning):
#                         print('The DateTime format could not be inferred, and a UserWarning was returned.')
#                         return False
#
#                 return is_convertible_to_date(series)
#         except (ValueError, TypeError):
#             return False


def int_with_nulls(series):
    return series.apply(lambda x: int(x) if pd.notna(x) else None)


def full_check(df):
    df_stack = df
    for col in df_stack:
        if is_convertible_to_int(df_stack[col]):
            df_stack[col] = pd.to_numeric(df_stack[col])
            df_stack[col] = pd.array(df_stack[col], dtype="Int64")
        elif is_convertible_to_float(df_stack[col]):
            df_stack[col] = pd.to_numeric(df_stack[col])
            df_stack[col] = pd.array(df_stack[col], dtype="float64")
        elif is_convertible_to_date(df_stack[col]):
            df_stack[col] = datetime_normalization(df_stack[col])
            df_stack[col] = df_stack[col].dt.strftime('%Y-%m-%d')

    return df_stack


def main():
    # Example usage
    # Add checks for holes, and fix any unexpected/unwanted values
    data = pd.Series([1.0, 2.0, 3.0, 'bear', 5.0])
    data2 = pd.Series([1.0, 2.0, 3.0, 4.5, 5.0])
    data3 = pd.Series([1.0, 2.0, 3.0, 4.0, '5.0'])
    data4 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    data5 = pd.Series([1.0, 2.0, 3.0, '4.5', 5.0])
    data6 = pd.Series([1.0, 2.0, 3.0, 'b', 5.0])
    data7 = pd.Series([1.0, 2.0, 3.0, '4.5', '5.0'])
    data8 = pd.Series([1.0, 2.0, 3.5, 'b', '4.5'])
    data9 = pd.Series([1.0, 2.0, 3.5, '4.0', '5.0'])
    data10 = pd.Series([1.0, 2.0, 3.0, '20/4', '5.0'])

    # Expected False, False, True, True, False, False, False, False, False, False
    # String, Expected False
    print(is_convertible_to_int(data))
    # Float, Expected False
    print(is_convertible_to_int(data2))
    # Convertible float cast as String, Expected True
    print(is_convertible_to_int(data3))
    # Float convertible to int, Expected True
    print(is_convertible_to_int(data4))
    # String is convertible to float, expected False
    print(is_convertible_to_int(data5))
    print(is_convertible_to_int(data6))
    print(is_convertible_to_int(data7))
    print(is_convertible_to_int(data8))
    print(is_convertible_to_int(data9))
    print(is_convertible_to_int(data10))

    print()

    # Expected false, true, true, true, true, false, true, false, true, false
    print(is_convertible_to_float(data))
    print(is_convertible_to_float(data2))
    print(is_convertible_to_float(data3))
    print(is_convertible_to_float(data4))
    print(is_convertible_to_float(data5))
    print(is_convertible_to_float(data6))
    print(is_convertible_to_float(data7))
    print(is_convertible_to_float(data8))
    print(is_convertible_to_float(data9))
    print(is_convertible_to_float(data10))

    df = pd.DataFrame({
        'A': [1, 0, True, False, '1', '0', 'True', 'False'],
        'B': [1, 0, True, False, '1', 'No', 'True', 'False']
    })

    # Check if columns are convertible to BIT
    print("Column 'A' is bit convertible:", is_convertible_to_bool(df['A']))
    print("Column 'B' is bit convertible:", is_convertible_to_bool(df['B']))


if __name__ == "__main__":
    main()
