"""Timedelta 替代方案"""

import numpy as np
from typing import Union

from ._detection import _check_pandas, get_pandas


class Timedelta:
    """
    轻量级时间间隔类

    功能：
    - 支持字符串解析（如 '1D', '1h', '1min', '1s', '1ms', '1us', '1ns'）
    - components 属性（返回 days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds）
    - 支持比较运算
    """

    __slots__ = ("_ns",)

    _UNIT_TO_NS = {
        "D": 24 * 60 * 60 * 1_000_000_000,
        "h": 60 * 60 * 1_000_000_000,
        "min": 60 * 1_000_000_000,
        "s": 1_000_000_000,
        "ms": 1_000_000,
        "us": 1_000,
        "ns": 1,
    }

    def __init__(
        self,
        value: Union[str, int, float, np.timedelta64, "Timedelta"] = None,
        unit: str = "ns",
        **kwargs,
    ):
        # 支持 pd.Timedelta 风格的关键字参数
        if value is None and kwargs:
            # 处理关键字参数如 seconds=0, days=1 等
            total_ns = 0
            for key, val in kwargs.items():
                if key in self._UNIT_TO_NS:
                    total_ns += int(val * self._UNIT_TO_NS[key])
                elif key == "seconds":
                    total_ns += int(val * self._UNIT_TO_NS["s"])
                elif key == "days":
                    total_ns += int(val * self._UNIT_TO_NS["D"])
                elif key == "hours":
                    total_ns += int(val * self._UNIT_TO_NS["h"])
                elif key == "minutes":
                    total_ns += int(val * self._UNIT_TO_NS["min"])
                elif key == "milliseconds":
                    total_ns += int(val * self._UNIT_TO_NS["ms"])
                elif key == "microseconds":
                    total_ns += int(val * self._UNIT_TO_NS["us"])
                elif key == "nanoseconds":
                    total_ns += int(val * self._UNIT_TO_NS["ns"])
                else:
                    raise TypeError(f"未知的关键字参数: {key}")
            self._ns = total_ns
            return

        if value is None:
            self._ns = 0
        elif isinstance(value, Timedelta):
            self._ns = value._ns
        elif isinstance(value, np.timedelta64):
            self._ns = int(value.astype("timedelta64[ns]").astype(int))
        elif isinstance(value, str):
            self._ns = self._parse_string(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            self._ns = int(value * self._UNIT_TO_NS.get(unit, 1))
        else:
            raise TypeError(f"不支持的类型: {type(value)}")

    def _parse_string(self, s: str) -> int:
        s = s.strip()
        if not s:
            return 0

        if _check_pandas():
            pd = get_pandas()
            return int(pd.Timedelta(s).value)

        total_ns = 0
        i = 0
        while i < len(s):
            while i < len(s) and s[i].isspace():
                i += 1
            if i >= len(s):
                break

            num_start = i
            while i < len(s) and (s[i].isdigit() or s[i] in ".-+"):
                i += 1
            if i == num_start:
                raise ValueError(f"无法解析: {s}")
            num = float(s[num_start:i])

            unit_start = i
            while i < len(s) and s[i].isalpha():
                i += 1
            unit = s[unit_start:i] if i > unit_start else "ns"

            unit_map = {
                "day": "D",
                "days": "D",
                "h": "h",
                "hour": "h",
                "hours": "h",
                "m": "min",
                "min": "min",
                "minute": "min",
                "minutes": "min",
                "s": "s",
                "second": "s",
                "seconds": "s",
                "ms": "ms",
                "millisecond": "ms",
                "milliseconds": "ms",
                "us": "us",
                "microsecond": "us",
                "microseconds": "us",
                "ns": "ns",
                "nanosecond": "ns",
                "nanoseconds": "ns",
            }
            unit = unit_map.get(unit.lower(), unit)

            if unit not in self._UNIT_TO_NS:
                raise ValueError(f"未知的时间单位: {unit}")

            total_ns += int(num * self._UNIT_TO_NS[unit])

        return total_ns

    @property
    def value(self) -> int:
        return self._ns

    @property
    def components(self):
        class Components:
            pass

        comp = Components()
        remaining = abs(self._ns)

        comp.days = remaining // self._UNIT_TO_NS["D"]
        remaining %= self._UNIT_TO_NS["D"]

        comp.hours = remaining // self._UNIT_TO_NS["h"]
        remaining %= self._UNIT_TO_NS["h"]

        comp.minutes = remaining // self._UNIT_TO_NS["min"]
        remaining %= self._UNIT_TO_NS["min"]

        comp.seconds = remaining // self._UNIT_TO_NS["s"]
        remaining %= self._UNIT_TO_NS["s"]

        comp.milliseconds = remaining // self._UNIT_TO_NS["ms"]
        remaining %= self._UNIT_TO_NS["ms"]

        comp.microseconds = remaining // self._UNIT_TO_NS["us"]
        remaining %= self._UNIT_TO_NS["us"]

        comp.nanoseconds = remaining

        return comp

    def __lt__(self, other):
        if isinstance(other, Timedelta):
            return self._ns < other._ns
        if isinstance(other, (int, float)):
            return self._ns < other
        # 支持与 pd.Timedelta 比较
        if _check_pandas():
            pd = get_pandas()
            if isinstance(other, pd.Timedelta):
                return self._ns < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Timedelta):
            return self._ns <= other._ns
        if isinstance(other, (int, float)):
            return self._ns <= other
        # 支持与 pd.Timedelta 比较
        if _check_pandas():
            pd = get_pandas()
            if isinstance(other, pd.Timedelta):
                return self._ns <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Timedelta):
            return self._ns > other._ns
        if isinstance(other, (int, float)):
            return self._ns > other
        # 支持与 pd.Timedelta 比较
        if _check_pandas():
            pd = get_pandas()
            if isinstance(other, pd.Timedelta):
                return self._ns > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Timedelta):
            return self._ns >= other._ns
        if isinstance(other, (int, float)):
            return self._ns >= other
        # 支持与 pd.Timedelta 比较
        if _check_pandas():
            pd = get_pandas()
            if isinstance(other, pd.Timedelta):
                return self._ns >= other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Timedelta):
            return self._ns == other._ns
        if isinstance(other, (int, float)):
            return self._ns == other
        # 支持与 pd.Timedelta 比较
        if _check_pandas():
            pd = get_pandas()
            if isinstance(other, pd.Timedelta):
                return self._ns == other.value
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Timedelta(int(self._ns * other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Timedelta(-self._ns)

    def __repr__(self):
        return f"Timedelta('{self}')"

    def __str__(self):
        if self._ns == 0:
            return "0 days 00:00:00"

        comp = self.components
        parts = []

        if comp.days > 0:
            parts.append(f"{comp.days} days")

        time_parts = []
        time_parts.append(f"{comp.hours:02d}")
        time_parts.append(f"{comp.minutes:02d}")
        time_parts.append(f"{comp.seconds:02d}")

        time_str = ":".join(time_parts)

        if comp.milliseconds > 0 or comp.microseconds > 0 or comp.nanoseconds > 0:
            frac = (
                comp.milliseconds * 1_000_000
                + comp.microseconds * 1_000
                + comp.nanoseconds
            )
            time_str += f".{frac:09d}".rstrip("0")

        if parts:
            return f"{parts[0]} {time_str}"
        return time_str

    def round(self, freq: str) -> "Timedelta":
        if freq not in self._UNIT_TO_NS:
            raise ValueError(f"未知的单位: {freq}")

        unit_ns = self._UNIT_TO_NS[freq]
        rounded = round(self._ns / unit_ns) * unit_ns
        return Timedelta(rounded)
