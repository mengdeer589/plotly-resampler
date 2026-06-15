"""
plotly-resampler 兼容层

提供 pandas/polars 功能的替代实现，减少对 pandas 的依赖。

优先策略：polars > pandas > numpy
"""

import numpy as np

from ._detection import (
    _check_pandas,
    _check_polars,
    get_pandas,
    get_polars,
    get_preferred_library,
    reset_detection,
)

from ._range_index import RangeIndex
from ._datetime_index import DatetimeIndex
from ._timedelta import Timedelta

from ._converters import (
    to_datetime,
    to_numeric,
    Categorical,
)

from ._type_checks import (
    is_datetime64_any_dtype,
    is_categorical_dtype,
    is_timedelta64_dtype,
    is_dtype_equal,
    is_numeric_dtype,
)

from ._json import nested_to_record

__all__ = [
    "_check_pandas",
    "_check_polars",
    "get_pandas",
    "get_polars",
    "get_preferred_library",
    "reset_detection",
    "RangeIndex",
    "DatetimeIndex",
    "Timedelta",
    "to_datetime",
    "to_numeric",
    "Categorical",
    "is_datetime64_any_dtype",
    "is_categorical_dtype",
    "is_timedelta64_dtype",
    "is_dtype_equal",
    "is_numeric_dtype",
    "nested_to_record",
    "Timestamp",
    "ensure_index",
]


def Timestamp(value, tz=None):
    """创建时间戳，兼容 pandas.Timestamp"""
    # 优先使用 polars
    if _check_polars():
        try:
            pl = get_polars()
            if isinstance(value, str):
                s = pl.Series([value]).str.to_datetime()
                return s.to_list()[0]
            elif isinstance(value, (int, float)):
                return np.datetime64(int(value), "ns")
        except Exception:
            pass

    if _check_pandas():
        pd = get_pandas()
        return pd.Timestamp(value, tz=tz)

    if isinstance(value, str):
        dt = np.datetime64(value)
    elif isinstance(value, (int, float)):
        dt = np.datetime64(int(value), "ns")
    else:
        dt = np.datetime64(value)

    return dt


def ensure_index(data):
    """确保数据是索引类型"""
    # 优先使用 polars
    if _check_polars():
        try:
            pl = get_polars()
            if hasattr(data, "to_list"):
                # polars Series
                dtype = data.dtype
                if dtype in (pl.Datetime, pl.Date):
                    return DatetimeIndex(data.to_numpy())
        except Exception:
            pass

    if _check_pandas():
        pd = get_pandas()
        if isinstance(data, pd.Index):
            return data
        if is_datetime64_any_dtype(data):
            return pd.DatetimeIndex(data)
        return pd.Index(data)

    if isinstance(data, (RangeIndex, DatetimeIndex)):
        return data
    if is_datetime64_any_dtype(data):
        return DatetimeIndex(data)
    return np.array(data)
