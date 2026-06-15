"""Tests for the compat module."""

import numpy as np
import pytest

from plotly_resampler.compat import (
    RangeIndex,
    DatetimeIndex,
    Timedelta,
    to_datetime,
    to_numeric,
    Categorical,
    is_datetime64_any_dtype,
    is_categorical_dtype,
    nested_to_record,
    _check_pandas,
    _check_polars,
    get_preferred_library,
)


class TestRangeIndex:
    """Tests for the RangeIndex class."""

    def test_basic_creation(self):
        idx = RangeIndex(10)
        assert len(idx) == 10
        assert idx.start == 0
        assert idx.stop == 10
        assert idx.step == 1

    def test_creation_with_start(self):
        idx = RangeIndex(5, 15)
        assert len(idx) == 10
        assert idx.start == 5
        assert idx.stop == 15

    def test_creation_with_step(self):
        idx = RangeIndex(0, 10, 2)
        assert len(idx) == 5
        assert idx.step == 2

    def test_getitem_int(self):
        idx = RangeIndex(10)
        assert idx[0] == 0
        assert idx[5] == 5
        assert idx[9] == 9

    def test_getitem_negative(self):
        idx = RangeIndex(10)
        assert idx[-1] == 9
        assert idx[-10] == 0

    def test_getitem_slice(self):
        idx = RangeIndex(10)
        sliced = idx[2:5]
        assert isinstance(sliced, RangeIndex)
        assert len(sliced) == 3
        assert sliced[0] == 2

    def test_getitem_array(self):
        idx = RangeIndex(10)
        arr = idx[np.array([1, 3, 5])]
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, [1, 3, 5])

    def test_to_numpy(self):
        idx = RangeIndex(5)
        arr = idx.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, [0, 1, 2, 3, 4])

    def test_values(self):
        idx = RangeIndex(5)
        arr = idx.values
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, [0, 1, 2, 3, 4])

    def test_is_monotonic_increasing(self):
        idx = RangeIndex(10)
        assert bool(idx.is_monotonic_increasing) is True

    def test_dtype(self):
        idx = RangeIndex(10)
        assert idx.dtype == np.int64

    def test_repr(self):
        idx = RangeIndex(5, 10, 2)
        assert repr(idx) == "RangeIndex(start=5, stop=10, step=2)"

    def test_eq(self):
        idx1 = RangeIndex(10)
        idx2 = RangeIndex(10)
        idx3 = RangeIndex(5)
        assert idx1 == idx2
        assert idx1 != idx3

    def test_iter(self):
        idx = RangeIndex(5)
        assert list(idx) == [0, 1, 2, 3, 4]

    def test_zero_step_raises(self):
        with pytest.raises(ValueError):
            RangeIndex(0, 10, 0)


class TestDatetimeIndex:
    """Tests for the DatetimeIndex class."""

    def test_basic_creation(self):
        data = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
        idx = DatetimeIndex(data)
        assert len(idx) == 3

    def test_creation_from_list(self):
        data = ["2020-01-01", "2020-01-02"]
        idx = DatetimeIndex(data)
        assert len(idx) == 2

    def test_getitem_int(self):
        data = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
        idx = DatetimeIndex(data)
        assert idx[0] == np.datetime64("2020-01-01")

    def test_getitem_slice(self):
        data = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
        idx = DatetimeIndex(data)
        sliced = idx[0:2]
        assert isinstance(sliced, DatetimeIndex)
        assert len(sliced) == 2

    def test_is_monotonic_increasing(self):
        data = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
        idx = DatetimeIndex(data)
        assert bool(idx.is_monotonic_increasing) is True

    def test_values(self):
        data = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64")
        idx = DatetimeIndex(data)
        arr = idx.values
        assert isinstance(arr, np.ndarray)


class TestTimedelta:
    """Tests for the Timedelta class."""

    def test_creation_from_string(self):
        td = Timedelta("1D")
        assert td.value == 24 * 60 * 60 * 1_000_000_000

    def test_creation_from_int(self):
        td = Timedelta(1000, unit="s")
        assert td.value == 1000 * 1_000_000_000

    def test_components(self):
        td = Timedelta("1D 2h 3m 4s")
        c = td.components
        assert c.days == 1
        assert c.hours == 2
        assert c.minutes == 3
        assert c.seconds == 4

    def test_comparison(self):
        td1 = Timedelta("1D")
        td2 = Timedelta("2D")
        assert td1 < td2
        assert td2 > td1
        assert td1 == Timedelta("1D")

    def test_multiplication(self):
        td = Timedelta("1D")
        result = td * 2
        assert result.value == 2 * 24 * 60 * 60 * 1_000_000_000

    def test_negation(self):
        td = Timedelta("1D")
        neg_td = -td
        assert neg_td.value == -td.value

    def test_round(self):
        td = Timedelta("1h 30m")
        rounded = td.round("h")
        assert rounded.value == 2 * 60 * 60 * 1_000_000_000  # 2 hours


class TestToDatetime:
    """Tests for the to_datetime function."""

    def test_string(self):
        result = to_datetime("2020-01-01")
        assert result == np.datetime64("2020-01-01")

    def test_list(self):
        result = to_datetime(["2020-01-01", "2020-01-02"])
        assert isinstance(result, (np.ndarray, DatetimeIndex))
        assert len(result) == 2

    def test_numeric(self):
        ts = 1609459200000000000  # 2021-01-01 in nanoseconds
        result = to_datetime(ts)
        assert isinstance(result, np.datetime64)

    def test_errors_raise(self):
        with pytest.raises((ValueError, TypeError)):
            to_datetime("not a date", errors="raise")

    def test_errors_coerce(self):
        result = to_datetime("not a date", errors="coerce")
        assert str(result) == "NaT" or (hasattr(result, "__len__") and len(result) == 0)


class TestToNumeric:
    """Tests for the to_numeric function."""

    def test_string(self):
        result = to_numeric("123.45")
        assert result == 123.45

    def test_list(self):
        result = to_numeric(["1", "2", "3"])
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, [1.0, 2.0, 3.0])

    def test_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numeric(arr)
        assert np.array_equal(result, arr)

    def test_errors_raise(self):
        with pytest.raises((ValueError, TypeError)):
            to_numeric("not a number", errors="raise")

    def test_errors_coerce(self):
        result = to_numeric("not a number", errors="coerce")
        assert np.isnan(result)


class TestCategorical:
    """Tests for the Categorical class."""

    def test_basic_creation(self):
        data = ["a", "b", "c", "a", "b"]
        cat = Categorical(data)
        assert len(cat) == 5
        assert len(cat.categories) == 3

    def test_codes(self):
        data = ["a", "b", "c", "a"]
        cat = Categorical(data)
        assert np.array_equal(cat.codes, [0, 1, 2, 0])

    def test_getitem(self):
        data = ["a", "b", "c"]
        cat = Categorical(data)
        assert cat[0] == "a"
        assert cat[1] == "b"
        assert cat[2] == "c"

    def test_getitem_slice(self):
        data = ["a", "b", "c"]
        cat = Categorical(data)
        sliced = cat[0:2]
        assert len(sliced) == 2

    def test_iter(self):
        data = ["a", "b", "c"]
        cat = Categorical(data)
        assert list(cat) == ["a", "b", "c"]

    def test_with_categories(self):
        data = ["a", "b", "c"]
        cat = Categorical(data, categories=["c", "b", "a"])
        assert list(cat.categories) == ["c", "b", "a"]

    def test_as_codes(self):
        codes = np.array([0, 1, 2, 0])
        categories = ["a", "b", "c"]
        cat = Categorical(codes, categories=categories, as_codes=True)
        assert np.array_equal(cat.codes, codes)
        assert list(cat.categories) == categories


class TestTypeChecks:
    """Tests for type check functions."""

    def test_is_datetime64_any_dtype(self):
        arr = np.array(["2020-01-01"], dtype="datetime64")
        assert is_datetime64_any_dtype(arr) is True
        assert is_datetime64_any_dtype(np.array([1, 2, 3])) is False

    def test_is_categorical_dtype(self):
        cat = Categorical(["a", "b", "c"])
        assert is_categorical_dtype(cat) is True
        assert is_categorical_dtype(np.array([1, 2, 3])) is False


class TestNestedToRecord:
    """Tests for the nested_to_record function."""

    def test_flat_dict(self):
        d = {"a": 1, "b": 2}
        result = nested_to_record(d)
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        d = {"a": {"b": 1, "c": 2}}
        result = nested_to_record(d)
        assert result == {"a_b": 1, "a_c": 2}

    def test_deeply_nested(self):
        d = {"a": {"b": {"c": 1}}}
        result = nested_to_record(d)
        assert result == {"a_b_c": 1}

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        result = nested_to_record(d, sep=".")
        assert result == {"a.b": 1}


class TestLibraryDetection:
    """Tests for library detection functions."""

    def test_check_pandas(self):
        result = _check_pandas()
        assert isinstance(result, bool)

    def test_check_polars(self):
        result = _check_polars()
        assert isinstance(result, bool)

    def test_get_preferred_library(self):
        lib = get_preferred_library()
        assert lib in ("polars", "pandas", "numpy")


@pytest.mark.skipif(not _check_polars(), reason="polars not installed")
class TestPolarsSupport:
    """Tests for polars support (only run when polars is available)."""

    def test_polars_to_datetime(self):
        import polars as pl

        data = ["2020-01-01", "2020-01-02"]
        result = to_datetime(data)
        assert len(result) == 2

    def test_polars_categorical(self):
        import polars as pl

        s = pl.Series(["a", "b", "c", "a"], dtype=pl.Categorical)
        from plotly_resampler.aggregation.plotly_aggregator_parser import (
            PlotlyAggregatorParser,
        )

        parsed = PlotlyAggregatorParser.parse_hf_data(s)
        assert isinstance(parsed, Categorical)
        assert list(parsed.categories) == ["a", "b", "c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
