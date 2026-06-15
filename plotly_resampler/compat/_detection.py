"""检测 pandas/polars 是否可用，实现 polars 优先策略"""

import importlib.util
from typing import Optional

# 全局标志
_PANDAS_AVAILABLE: Optional[bool] = None
_POLARS_AVAILABLE: Optional[bool] = None


def _check_pandas() -> bool:
    """检测 pandas 是否可用"""
    global _PANDAS_AVAILABLE
    if _PANDAS_AVAILABLE is None:
        _PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
    return _PANDAS_AVAILABLE


def _check_polars() -> bool:
    """检测 polars 是否可用"""
    global _POLARS_AVAILABLE
    if _POLARS_AVAILABLE is None:
        _POLARS_AVAILABLE = importlib.util.find_spec("polars") is not None
    return _POLARS_AVAILABLE


def get_pandas():
    """延迟导入 pandas，不可用时抛出清晰的错误信息"""
    if not _check_pandas():
        raise ImportError("此功能需要 pandas。请安装：pip install pandas")
    import pandas

    return pandas


def get_polars():
    """延迟导入 polars，不可用时抛出清晰的错误信息"""
    if not _check_polars():
        raise ImportError("此功能需要 polars。请安装：pip install polars")
    import polars

    return polars


def get_preferred_library() -> str:
    """
    获取首选的数据处理库

    策略：
    1. 如果 polars 可用，优先使用 polars
    2. 如果只有 pandas 可用，使用 pandas
    3. 如果都不可用，返回 "numpy"

    返回：
    - "polars": 首选 polars
    - "pandas": 只有 pandas
    - "numpy": 都没有，使用纯 numpy
    """
    if _check_polars():
        return "polars"
    elif _check_pandas():
        return "pandas"
    else:
        return "numpy"


def reset_detection():
    """重置检测状态（主要用于测试）"""
    global _PANDAS_AVAILABLE, _POLARS_AVAILABLE
    _PANDAS_AVAILABLE = None
    _POLARS_AVAILABLE = None
