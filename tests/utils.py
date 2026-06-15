from __future__ import annotations

import base64
import sys

import numpy as np

from plotly_resampler.aggregation import MedDiffGapHandler, MinMaxLTTB
from plotly_resampler.aggregation.aggregation_interface import (
    DataAggregator,
    DataPointSelector,
)
from plotly_resampler.aggregation.gap_handler_interface import AbstractGapHandler
from plotly_resampler.aggregation.plotly_aggregator_parser import PlotlyAggregatorParser
from plotly_resampler.compat import (
    RangeIndex,
    DatetimeIndex,
    is_datetime64_any_dtype,
    _check_pandas,
)

# 可选导入 pandas
if _check_pandas():
    import pandas as pd
else:
    pd = None


def not_on_linux():
    """Return True if the current platform is not Linux.

    This is to avoid / alter test bahavior for non-Linux (as browser testing gets
    tricky on other platforms).
    """
    return not sys.platform.startswith("linux")


def construct_hf_data_dict(hf_x, hf_y, **kwargs):
    # 检查 axis_type
    if isinstance(hf_x, DatetimeIndex):
        axis_type = "date"
    elif is_datetime64_any_dtype(hf_x):
        axis_type = "date"
    elif _check_pandas() and isinstance(hf_x, pd.DatetimeIndex):
        axis_type = "date"
    elif _check_pandas() and pd.core.dtypes.common.is_datetime64_any_dtype(hf_x):
        axis_type = "date"
    else:
        axis_type = "linear"

    hf_data_dict = {
        "x": hf_x,
        "y": hf_y,
        "axis_type": axis_type,
        "downsampler": MinMaxLTTB(),
        "gap_handler": MedDiffGapHandler(),
        "max_n_samples": 1_000,
    }
    hf_data_dict.update(kwargs)
    return hf_data_dict


def wrap_aggregate(
    hf_x: np.ndarray | None = None,
    hf_y: pd.Series | np.ndarray = None,
    downsampler: DataPointSelector | DataAggregator = None,
    gap_handler: AbstractGapHandler = None,
    n_out: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hf_trace_data = construct_hf_data_dict(
        **{
            "hf_x": hf_x,
            "hf_y": hf_y,
            "downsampler": downsampler,
            "gap_handler": gap_handler,
            "max_n_samples": n_out,
        }
    )
    return PlotlyAggregatorParser.aggregate(hf_trace_data, 0, len(hf_y))


def construct_index(series, index_type: str):
    """Construct an index of the given type for the given series.

    series: np.ndarray or pd.Series
        The series to construct an index for
    index_type: str
        One of "range", "datetime", "timedelta", "float", or "int"
    """
    length = len(series)
    if index_type == "range":
        return RangeIndex(length)
    if index_type == "datetime":
        # 创建等间距的 datetime64 数组
        start = np.datetime64("2020-01-01")
        return DatetimeIndex(start + np.arange(length) * np.timedelta64(1, "ms"))
    if index_type == "timedelta":
        # 创建等间距的 timedelta64 数组
        return np.arange(length) * np.timedelta64(1, "ms")
    if index_type == "float":
        return np.arange(length, dtype=np.float64)
    if index_type == "int":
        return np.arange(length, dtype=np.int64)
    raise ValueError(f"Unknown index type: {index_type}")


def decode_trace_bdata(data: dict | list):
    """As from plotly>6.0.0, traces can be encoded as binary strings, we need to decode
    them to get the actual data.
    """
    if isinstance(data, dict) and "bdata" in data:
        bdata = data["bdata"]
        dtype = data["dtype"]

        # Decode the base64 encoded binary data
        decoded_data = base64.b64decode(bdata)
        # Convert the decoded data to a numpy array
        np_array = np.frombuffer(decoded_data, dtype=np.dtype(dtype))
        return np_array  # Return the numpy array for further use if needed
    else:
        return data
