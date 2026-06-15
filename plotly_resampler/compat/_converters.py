"""数据转换函数，支持 polars/pandas/numpy"""

import numpy as np
from typing import Union, Optional, Any
from datetime import datetime

from ._detection import (
    _check_pandas,
    _check_polars,
    get_pandas,
    get_polars,
    get_preferred_library,
)
from ._datetime_index import DatetimeIndex


def to_datetime(
    arg: Union[str, list, np.ndarray, int, float],
    utc: bool = False,
    errors: str = "raise",
    format: Optional[str] = None,
    **kwargs,
) -> Union[np.datetime64, np.ndarray, DatetimeIndex]:
    """
    将输入转换为 datetime64 类型

    支持策略：polars 优先 > pandas > numpy
    """
    lib = get_preferred_library()

    if lib == "polars":
        try:
            pl = get_polars()
            if isinstance(arg, str):
                result = pl.Series([arg]).str.to_datetime(format=format).to_list()[0]
                return np.datetime64(result)
            elif isinstance(arg, (list, tuple)):
                s = pl.Series(arg).cast(pl.Utf8)
                result = s.str.to_datetime(format=format, strict=(errors == "raise"))
                return DatetimeIndex(result.to_numpy())
            elif isinstance(arg, np.ndarray):
                if arg.dtype.kind == "M":
                    return arg
                s = pl.Series(arg.tolist()).cast(pl.Utf8)
                result = s.str.to_datetime(format=format, strict=(errors == "raise"))
                return DatetimeIndex(result.to_numpy())
            elif isinstance(arg, (int, float)):
                return np.datetime64(int(arg), "ns")
            else:
                raise TypeError(f"不支持的输入类型: {type(arg)}")
        except Exception as e:
            if errors == "raise":
                raise ValueError(f"无法转换为 datetime: {arg}") from e
            elif errors == "coerce":
                return np.datetime64("NaT")

    elif lib == "pandas":
        try:
            pd = get_pandas()
            result = pd.to_datetime(
                arg, utc=utc, errors=errors, format=format, **kwargs
            )
            if isinstance(result, pd.DatetimeIndex):
                return DatetimeIndex(
                    result.values, tz=str(result.tz) if result.tz else None
                )
            elif isinstance(result, pd.Timestamp):
                return result.to_datetime64()
            else:
                return np.array(result)
        except Exception:
            if errors == "raise":
                raise
            elif errors == "coerce":
                return np.datetime64("NaT")

    # 纯 numpy 实现
    try:
        if isinstance(arg, str):
            # 尝试解析常见的时间格式
            return _parse_datetime_string(arg)
        elif isinstance(arg, (list, tuple)):
            return np.array(
                [_parse_datetime_string(str(x)) for x in arg], dtype="datetime64[ns]"
            )
        elif isinstance(arg, np.ndarray):
            if arg.dtype.kind == "M":
                return arg
            elif arg.dtype.kind in ("U", "O"):
                return np.array(
                    [_parse_datetime_string(str(x)) for x in arg],
                    dtype="datetime64[ns]",
                )
            elif arg.dtype.kind in ("i", "f"):
                return arg.astype("datetime64[ns]")
            else:
                raise ValueError(f"不支持的数组类型: {arg.dtype}")
        elif isinstance(arg, (int, float)):
            return np.datetime64(int(arg), "ns")
        else:
            raise TypeError(f"不支持的输入类型: {type(arg)}")
    except Exception as e:
        if errors == "raise":
            raise ValueError(f"无法转换为 datetime: {arg}") from e
        elif errors == "coerce":
            return np.datetime64("NaT")


def _parse_datetime_string(s: str) -> np.datetime64:
    """使用纯 numpy 解析日期时间字符串"""
    # 常见格式列表
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]

    # 先尝试直接用 numpy 解析
    try:
        return np.datetime64(s)
    except ValueError:
        pass

    # 尝试各种格式
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return np.datetime64(dt)
        except ValueError:
            continue

    raise ValueError(f"无法解析日期时间字符串: {s}")


def to_numeric(
    arg: Union[str, list, np.ndarray],
    errors: str = "raise",
    downcast: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """
    将输入转换为数值类型

    支持策略：polars 优先 > pandas > numpy
    """
    lib = get_preferred_library()

    if lib == "polars":
        try:
            pl = get_polars()
            if isinstance(arg, str):
                return float(arg)
            elif isinstance(arg, (list, tuple)):
                s = pl.Series(arg, dtype=pl.Utf8)
                result = s.cast(pl.Float64, strict=(errors == "raise"))
                return result.to_numpy()
            elif isinstance(arg, np.ndarray):
                if arg.dtype.kind in ("i", "f", "u"):
                    return arg
                s = pl.Series(arg.tolist(), dtype=pl.Utf8)
                result = s.cast(pl.Float64, strict=(errors == "raise"))
                return result.to_numpy()
            else:
                return float(arg)
        except Exception as e:
            if errors == "raise":
                raise ValueError(f"无法转换为数值: {arg}") from e
            elif errors == "coerce":
                return np.nan

    elif lib == "pandas":
        pd = get_pandas()
        return pd.to_numeric(arg, errors=errors, downcast=downcast)

    # 纯 numpy 实现
    try:
        if isinstance(arg, str):
            return float(arg)
        elif isinstance(arg, (list, tuple)):
            return np.array([float(x) for x in arg])
        elif isinstance(arg, np.ndarray):
            if arg.dtype.kind in ("i", "f", "u"):
                return arg
            elif arg.dtype.kind in ("U", "O"):
                return np.array([float(x) for x in arg])
            else:
                raise ValueError(f"不支持的数组类型: {arg.dtype}")
        else:
            return float(arg)
    except (ValueError, TypeError):
        if errors == "raise":
            raise
        elif errors == "coerce":
            return np.nan


class Categorical:
    """分类数据类型，兼容 pandas.Categorical 和 polars Enum"""

    __slots__ = ("_codes", "_categories", "_ordered")

    def __init__(
        self,
        data: Union[list, np.ndarray, "Categorical"],
        categories: Optional[list] = None,
        ordered: bool = False,
        dtype: Any = None,
        as_codes: bool = False,
    ):
        """
        参数：
        - data: 输入数据，可以是原始数据或整数编码（如果 as_codes=True）
        - categories: 类别列表
        - ordered: 是否有序
        - dtype: 数据类型（可选）
        - as_codes: 如果 True，data 被解释为整数编码而不是原始数据
        """
        if isinstance(data, Categorical):
            self._codes = data._codes.copy()
            self._categories = data._categories.copy()
            self._ordered = data._ordered
            return

        if dtype is not None and hasattr(dtype, "categories"):
            categories = list(dtype.categories)
            if hasattr(dtype, "ordered"):
                ordered = dtype.ordered

        if as_codes:
            # data 是整数编码
            if isinstance(data, np.ndarray):
                self._codes = data.astype(np.int32)
            else:
                self._codes = np.array(data, dtype=np.int32)
            if categories is None:
                raise ValueError("as_codes=True 时必须提供 categories")
            self._categories = np.array(categories)
            self._ordered = ordered
            return

        if isinstance(data, np.ndarray):
            data_list = data.tolist()
        else:
            data_list = list(data)

        if categories is None:
            seen = set()
            categories = []
            for item in data_list:
                if item not in seen:
                    seen.add(item)
                    categories.append(item)

        self._categories = np.array(categories)
        self._ordered = ordered

        cat_to_code = {cat: i for i, cat in enumerate(categories)}

        self._codes = np.array(
            [cat_to_code.get(item, -1) for item in data_list], dtype=np.int32
        )

    @property
    def codes(self) -> np.ndarray:
        return self._codes

    @property
    def categories(self) -> np.ndarray:
        return self._categories

    @property
    def ordered(self) -> bool:
        return self._ordered

    @property
    def dtype(self):
        return "category"

    def __len__(self) -> int:
        return len(self._codes)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            code = self._codes[key]
            return self._categories[code] if code >= 0 else None
        elif isinstance(key, slice):
            new_cat = Categorical.__new__(Categorical)
            new_cat._codes = self._codes[key]
            new_cat._categories = self._categories.copy()
            new_cat._ordered = self._ordered
            return new_cat
        elif isinstance(key, np.ndarray):
            return self._codes[key]
        else:
            raise TypeError(f"不支持的索引类型: {type(key)}")

    def __iter__(self):
        for code in self._codes:
            yield self._categories[code] if code >= 0 else None

    def __repr__(self) -> str:
        items = [repr(x) for x in self[:5]]
        ellipsis = "..." if len(self) > 5 else ""
        return f"Categorical([{', '.join(items)}{ellipsis}])"

    @property
    def values(self) -> np.ndarray:
        return self.codes
