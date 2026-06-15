# Plotly-Resampler 修改说明

> 原项目地址: https://github.com/predict-idlab/plotly-resampler
> 修改日期: 2026-03-21

## 修改目的

将 pandas 从硬依赖改为可选依赖，同时添加 polars 支持，使项目可以：
1. 在不安装 pandas 的情况下使用基本功能
2. 优先使用 polars（如果可用）
3. 回退使用 pandas（如果 polars 不可用但 pandas 可用）
4. 都不可用时使用纯 numpy 实现

## 新增文件

### `plotly_resampler/compat/` 目录

```
plotly_resampler/compat/
├── __init__.py           # 主入口，导出所有兼容函数和类
├── _detection.py         # 检测 pandas/polars 是否可用，实现优先策略
├── _range_index.py       # RangeIndex 实现，替代 pd.RangeIndex
├── _datetime_index.py    # DatetimeIndex 实现，支持 polars/pandas/numpy
├── _timedelta.py         # Timedelta 实现，替代 pd.Timedelta
├── _converters.py        # 数据转换函数（to_datetime, to_numeric, Categorical）
├── _type_checks.py       # 类型检查函数
└── _json.py              # nested_to_record 实现
```

### 测试文件

- `tests/test_compat.py` - compat 模块的完整测试（56 个测试）

## 修改的文件

### 1. `plotly_resampler/figure_resampler/utils.py`

- 移除 `import pandas as pd`
- 改为 `from ..compat import Timedelta`
- 函数签名中的 `pd.Timedelta` 改为 `Timedelta`

### 2. `plotly_resampler/figure_resampler/figure_resampler_interface.py`

- 移除 `import pandas as pd` 和 `from pandas.io.json._normalize import nested_to_record`
- 添加 compat 模块导入
- 修改类型检查逻辑，支持 RangeIndex/DatetimeIndex 替代 pd.RangeIndex/pd.DatetimeIndex
- 修改数据转换逻辑，使用 `to_datetime`, `to_numeric`, `Categorical`

### 3. `plotly_resampler/aggregation/plotly_aggregator_parser.py`

- 移除 `import pandas as pd`
- 添加 polars 支持导入
- `parse_hf_data` 函数增加 polars Series 处理
- 修改类型检查，支持自定义 RangeIndex/DatetimeIndex

### 4. `pyproject.toml`

- pandas 从必选依赖改为可选依赖
- 添加 polars 作为可选依赖
- 更新 extras 配置

```toml
[tool.poetry.dependencies]
# 移除 pandas 必选依赖
# pandas = [...]  # 已删除

# 新增可选依赖
pandas = { version = ">=1", optional = true }
polars = { version = ">=0.20.0", optional = true }

[tool.poetry.extras]
pandas = ["pandas"]
polars = ["polars"]
full = ["pandas", "polars"]
```

### 5. `tests/conftest.py`

- 添加可选 pandas 导入
- 修改 fixtures 支持跳过需要 pandas 的测试

### 6. `tests/test_figure_resampler.py`

- 添加可选 pandas 导入

### 7. `tests/utils.py`

- 添加可选 pandas/polars 导入
- `construct_hf_data_dict` 支持自定义 DatetimeIndex
- `construct_index` 使用纯 numpy 实现

## 优先策略

```python
def get_preferred_library() -> str:
    """
    获取首选的数据处理库

    策略：
    1. 如果 polars 可用，优先使用 polars
    2. 如果只有 pandas 可用，使用 pandas
    3. 如果都不可用，返回 "numpy"
    """
    if _check_polars():
        return "polars"
    elif _check_pandas():
        return "pandas"
    else:
        return "numpy"
```

## 测试结果

### 没有 pandas/polars 的环境
- compat 模块测试: 56 passed ✓
- 基本 figure_resampler 测试: 通过 ✓
- 需要 pandas 的测试: 正确跳过

### 有 pandas 的环境
- compat 模块测试: 56 passed ✓
- figure_resampler 测试: 大部分通过

### 有 polars 的环境
- compat 模块测试: 56 passed ✓
- polars 特定测试: 通过 ✓

## 编译 whl 文件

```bash
# 安装构建工具
uv pip install build

# 构建 whl
python -m build --wheel

# 输出位置
dist/plotly_resampler-0.11.0-py3-none-any.whl
```

## 安装方式

```bash
# 基础安装（纯 numpy）
pip install plotly_resampler-0.11.0-py3-none-any.whl

# polars 支持
pip install plotly_resampler-0.11.0-py3-none-any.whl[polars]

# pandas 支持
pip install plotly_resampler-0.11.0-py3-none-any.whl[pandas]

# 完整功能
pip install plotly_resampler-0.11.0-py3-none-any.whl[full]
```

## 与原项目同步更新的注意事项

1. **检查依赖变更**: 如果原项目更新了 pandas 相关的依赖版本，需要同步更新 `pyproject.toml` 中的可选依赖版本

2. **检查新功能**: 如果原项目添加了新的使用 pandas 的功能，需要：
   - 在 `plotly_resampler/compat/` 中添加相应的兼容实现
   - 优先使用 polars，其次 pandas，最后 numpy

3. **检查 API 变更**: 如果原项目修改了函数签名或返回类型，需要确保 compat 模块的实现与之兼容

4. **运行测试**: 合并后务必在三种环境下运行测试：
   - 无 pandas/polars
   - 只有 pandas
   - 只有 polars

5. **重新编译**: 修改完成后重新编译 whl 文件

## 兼容性说明

### 完全兼容的功能
- RangeIndex（纯 numpy 实现）
- DatetimeIndex（支持时区转换需要 polars/pandas）
- Timedelta（纯 numpy 实现）
- to_datetime（基本格式支持）
- to_numeric（完全兼容）
- Categorical（完全兼容）
- nested_to_record（完全兼容）

### 部分兼容的功能
- 时区处理：需要 polars 或 pandas
- 复杂日期格式解析：纯 numpy 支持有限
- 高级 pandas 功能：需要 pandas

## 未来改进方向

1. 完善 polars 的时区支持
2. 添加更多日期格式的纯 numpy 解析
3. 考虑支持其他数据处理库（如 cuDF、Dask 等）
4. 优化纯 numpy 实现的性能
