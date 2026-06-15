"""DatetimeIndex 替代方案，支持 polars/pandas/numpy"""

import numpy as np
from typing import Optional, Union, Iterator

from ._detection import _check_pandas, _check_polars, get_pandas, get_polars


class DatetimeIndex:
    """
    基于 numpy datetime64 的时间索引

    核心功能：
    - len(), 切片, 迭代
    - tz 属性（时区支持，可选）
    - freq 属性（频率）
    - is_monotonic_increasing 属性

    支持策略：polars 优先 > pandas > numpy
    """

    __slots__ = ("_data", "_tz", "_freq", "_name")

    def __init__(
        self,
        data: Union[np.ndarray, list, "DatetimeIndex"],
        tz: Optional[str] = None,
        freq: Optional[str] = None,
        name: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
    ):
        if isinstance(data, DatetimeIndex):
            self._data = data._data.copy()
            self._tz = data._tz if tz is None else tz
            self._freq = data._freq if freq is None else freq
            self._name = data._name if name is None else name
        elif isinstance(data, np.ndarray):
            if data.dtype.kind == "M":
                self._data = data.astype("datetime64[ns]")
            elif data.dtype.kind == "O":
                self._data = self._parse_object_array(data)
            else:
                self._data = np.array(data, dtype="datetime64[ns]")
            self._tz = tz
            self._freq = freq
            self._name = name
        elif isinstance(data, (list, tuple)):
            self._data = self._parse_list(data)
            self._tz = tz
            self._freq = freq
            self._name = name
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

    def _parse_list(self, data: list) -> np.ndarray:
        if len(data) == 0:
            return np.array([], dtype="datetime64[ns]")

        # 尝试用 polars 解析
        if _check_polars():
            try:
                pl = get_polars()
                s = pl.Series(data).cast(pl.Utf8)
                result = s.str.to_datetime(strict=False)
                return result.to_numpy().astype("datetime64[ns]")
            except Exception:
                pass

        # 尝试用 pandas 解析
        if _check_pandas():
            try:
                pd = get_pandas()
                return pd.to_datetime(data).values
            except Exception:
                pass

        # 纯 numpy 解析
        try:
            return np.array(data, dtype="datetime64[ns]")
        except (ValueError, TypeError):
            raise ValueError(
                "无法解析时间数据。请安装 polars 或 pandas 以支持复杂时间格式。"
            )

    def _parse_object_array(self, data: np.ndarray) -> np.ndarray:
        # 尝试用 polars 解析
        if _check_polars():
            try:
                pl = get_polars()
                s = pl.Series(data.tolist()).cast(pl.Utf8)
                result = s.str.to_datetime(strict=False)
                return result.to_numpy().astype("datetime64[ns]")
            except Exception:
                pass

        # 尝试用 pandas 解析
        if _check_pandas():
            try:
                pd = get_pandas()
                return pd.to_datetime(data).values
            except Exception:
                pass

        raise ValueError("解析 object 数组需要 polars 或 pandas")

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def tz(self) -> Optional[str]:
        return self._tz

    @property
    def freq(self) -> Optional[str]:
        return self._freq

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def is_monotonic_increasing(self) -> bool:
        if len(self._data) <= 1:
            return True
        return np.all(self._data[1:] >= self._data[:-1])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(
        self, key: Union[int, slice, np.ndarray]
    ) -> Union[np.datetime64, "DatetimeIndex", np.ndarray]:
        if isinstance(key, (int, np.integer)):
            return self._data[key]
        elif isinstance(key, slice):
            new_data = self._data[key]
            return DatetimeIndex(new_data, tz=self._tz, freq=self._freq)
        elif isinstance(key, np.ndarray):
            return self._data[key]
        else:
            raise TypeError(f"不支持的索引类型: {type(key)}")

    def __iter__(self) -> Iterator[np.datetime64]:
        return iter(self._data)

    def __eq__(self, other) -> bool:
        if isinstance(other, DatetimeIndex):
            return np.array_equal(self._data, other._data)
        elif isinstance(other, np.ndarray):
            return np.array_equal(self._data, other)
        return NotImplemented

    def __repr__(self) -> str:
        tz_str = f", tz='{self._tz}'" if self._tz else ""
        freq_str = f", freq='{self._freq}'" if self._freq else ""
        if len(self._data) > 0:
            return f"DatetimeIndex(['{self._data[0]}', ..., '{self._data[-1]}'], length={len(self)}{tz_str}{freq_str})"
        return f"DatetimeIndex([], length=0{tz_str}{freq_str})"

    def __array__(self, dtype=None) -> np.ndarray:
        return self._data if dtype is None else self._data.astype(dtype)

    @property
    def values(self) -> np.ndarray:
        return self._data

    def to_numpy(self, dtype=None) -> np.ndarray:
        return self._data if dtype is None else self._data.astype(dtype)

    def tz_localize(self, tz: str) -> "DatetimeIndex":
        """本地化时区"""
        # 优先使用 polars
        if _check_polars():
            try:
                pl = get_polars()
                s = pl.Series(self._data)
                result = s.dt.replace_time_zone(tz)
                return DatetimeIndex(result.to_numpy(), tz=tz, freq=self._freq)
            except Exception:
                pass

        # 使用 pandas
        if _check_pandas():
            pd = get_pandas()
            pd_index = pd.DatetimeIndex(self._data)
            new_data = pd_index.tz_localize(tz)
            return DatetimeIndex(new_data.values, tz=tz, freq=self._freq)

        raise ImportError("tz_localize 需要 polars 或 pandas")

    def tz_convert(self, tz: str) -> "DatetimeIndex":
        """转换时区"""
        if self._tz is None:
            raise ValueError("无法转换无时区的索引，请先使用 tz_localize")

        # 优先使用 polars
        if _check_polars():
            try:
                pl = get_polars()
                s = pl.Series(self._data).dt.replace_time_zone(self._tz)
                result = s.dt.convert_time_zone(tz)
                return DatetimeIndex(result.to_numpy(), tz=tz, freq=self._freq)
            except Exception:
                pass

        # 使用 pandas
        if _check_pandas():
            pd = get_pandas()
            pd_index = pd.DatetimeIndex(self._data, tz=self._tz)
            new_data = pd_index.tz_convert(tz)
            return DatetimeIndex(new_data.values, tz=tz, freq=self._freq)

        raise ImportError("tz_convert 需要 polars 或 pandas")
