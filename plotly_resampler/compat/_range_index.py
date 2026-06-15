"""轻量级 RangeIndex 实现，不依赖 pandas"""

import numpy as np
from typing import Iterator, Union


class RangeIndex:
    """
    高效的整数范围索引，内存占用恒定（O(1)）

    核心功能：
    - start, stop, step 属性
    - len(), 切片, 迭代
    - dtype 属性
    - is_monotonic_increasing 属性

    使用场景：
    - 当用户不提供 x 数据时，自动生成 [0, 1, 2, ..., n-1] 索引
    - 在 plotly_aggregator_parser.py 中用于高效计算聚合后的 x 值
    """

    __slots__ = ("_start", "_stop", "_step", "_dtype")

    def __init__(
        self,
        start: int = 0,
        stop: int = None,
        step: int = 1,
        dtype=np.int64,
    ):
        if stop is None:
            start, stop = 0, start

        if step == 0:
            raise ValueError("step 不能为 0")

        self._start = int(start)
        self._stop = int(stop)
        self._step = int(step)
        self._dtype = np.dtype(dtype)

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    @property
    def step(self) -> int:
        return self._step

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def is_monotonic_increasing(self) -> bool:
        """RangeIndex 总是单调递增（如果 step > 0）"""
        return self._step > 0

    def __len__(self) -> int:
        if self._step > 0:
            return max(0, (self._stop - self._start + self._step - 1) // self._step)
        else:
            return max(0, (self._stop - self._start + self._step + 1) // self._step)

    def __getitem__(self, key: Union[int, slice, np.ndarray]) -> Union[int, np.ndarray]:
        if isinstance(key, (int, np.integer)):
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError(f"索引 {key} 超出范围 [0, {len(self)})")
            return self._start + key * self._step

        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            new_start = self._start + start * self._step
            new_step = self._step * step
            return RangeIndex(
                new_start,
                new_start + (stop - start) * new_step,
                new_step,
            )

        elif isinstance(key, np.ndarray):
            if key.dtype == bool:
                indices = np.where(key)[0]
            else:
                indices = key
            return self._start + indices * self._step

        else:
            raise TypeError(f"不支持的索引类型: {type(key)}")

    def __iter__(self) -> Iterator[int]:
        current = self._start
        if self._step > 0:
            while current < self._stop:
                yield current
                current += self._step
        else:
            while current > self._stop:
                yield current
                current += self._step

    def __eq__(self, other) -> bool:
        if isinstance(other, RangeIndex):
            return (
                self._start == other._start
                and self._stop == other._stop
                and self._step == other._step
            )
        return NotImplemented

    def __repr__(self) -> str:
        return f"RangeIndex(start={self._start}, stop={self._stop}, step={self._step})"

    def __array__(self, dtype=None) -> np.ndarray:
        if dtype is None:
            dtype = self._dtype
        return np.arange(self._start, self._stop, self._step, dtype=dtype)

    @property
    def values(self) -> np.ndarray:
        return np.array(self)

    def to_numpy(self, dtype=None) -> np.ndarray:
        return np.array(self, dtype=dtype)
