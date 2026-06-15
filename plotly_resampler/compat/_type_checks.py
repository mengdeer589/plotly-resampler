"""类型检查函数，支持 polars/pandas/numpy"""

import numpy as np
from typing import Any

from ._detection import _check_pandas, _check_polars, get_pandas, get_polars


def is_datetime64_any_dtype(arr_or_dtype: Any) -> bool:
    """检查是否为 datetime64 类型"""
    # 优先使用 polars
    if _check_polars():
        try:
            pl = get_polars()
            if hasattr(arr_or_dtype, "dtype"):
                dtype = arr_or_dtype.dtype
                if dtype in (pl.Datetime, pl.Date):
                    return True
        except Exception:
            pass

    if _check_pandas():
        pd = get_pandas()
        return pd.core.dtypes.common.is_datetime64_any_dtype(arr_or_dtype)

    if isinstance(arr_or_dtype, np.ndarray):
        dtype = arr_or_dtype.dtype
    elif isinstance(arr_or_dtype, np.dtype):
        dtype = arr_or_dtype
    elif hasattr(arr_or_dtype, "dtype"):
        dtype = arr_or_dtype.dtype
    else:
        try:
            dtype = np.dtype(arr_or_dtype)
        except TypeError:
            return False

    return np.issubdtype(dtype, np.datetime64)


def is_categorical_dtype(arr_or_dtype: Any) -> bool:
    """检查是否为分类类型"""
    # 优先使用 polars
    if _check_polars():
        try:
            pl = get_polars()
            if hasattr(arr_or_dtype, "dtype"):
                dtype = arr_or_dtype.dtype
                if dtype == pl.Categorical or dtype == pl.Enum:
                    return True
        except Exception:
            pass

    if _check_pandas():
        pd = get_pandas()
        return pd.api.types.is_categorical_dtype(arr_or_dtype)

    from ._converters import Categorical

    if isinstance(arr_or_dtype, Categorical):
        return True

    if hasattr(arr_or_dtype, "dtype"):
        dtype = arr_or_dtype.dtype
        if str(dtype) == "category":
            return True

    return False


def is_timedelta64_dtype(arr_or_dtype: Any) -> bool:
    """检查是否为 timedelta64 类型"""
    # 优先使用 polars
    if _check_polars():
        try:
            pl = get_polars()
            if hasattr(arr_or_dtype, "dtype"):
                dtype = arr_or_dtype.dtype
                if dtype == pl.Duration:
                    return True
        except Exception:
            pass

    if _check_pandas():
        pd = get_pandas()
        return pd.api.types.is_timedelta64_dtype(arr_or_dtype)

    if isinstance(arr_or_dtype, np.ndarray):
        dtype = arr_or_dtype.dtype
    elif isinstance(arr_or_dtype, np.dtype):
        dtype = arr_or_dtype
    else:
        try:
            dtype = np.dtype(arr_or_dtype)
        except TypeError:
            return False

    return np.issubdtype(dtype, np.timedelta64)


def is_dtype_equal(source, target) -> bool:
    """检查两个 dtype 是否相等"""
    if _check_pandas():
        pd = get_pandas()
        return pd.api.types.is_dtype_equal(source, target)

    try:
        return np.dtype(source) == np.dtype(target)
    except TypeError:
        return str(source) == str(target)


def is_numeric_dtype(arr_or_dtype: Any) -> bool:
    """检查是否为数值类型"""
    # 优先使用 polars
    if _check_polars():
        try:
            pl = get_polars()
            if hasattr(arr_or_dtype, "dtype"):
                dtype = arr_or_dtype.dtype
                if dtype in (
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.UInt8,
                    pl.UInt16,
                    pl.UInt32,
                    pl.UInt64,
                    pl.Float32,
                    pl.Float64,
                ):
                    return True
        except Exception:
            pass

    if _check_pandas():
        pd = get_pandas()
        return pd.api.types.is_numeric_dtype(arr_or_dtype)

    if isinstance(arr_or_dtype, np.ndarray):
        dtype = arr_or_dtype.dtype
    elif isinstance(arr_or_dtype, np.dtype):
        dtype = arr_or_dtype
    else:
        try:
            dtype = np.dtype(arr_or_dtype)
        except TypeError:
            return False

    return np.issubdtype(dtype, np.number)
