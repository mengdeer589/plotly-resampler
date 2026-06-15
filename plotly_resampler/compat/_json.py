"""JSON 相关工具函数"""


def nested_to_record(d: dict, prefix: str = "", sep: str = "_") -> dict:
    """
    将嵌套字典扁平化

    参数：
    - d: 输入的嵌套字典
    - prefix: 键名前缀
    - sep: 分隔符

    返回：
    - 扁平化后的字典
    """
    result = {}
    for key, value in d.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(nested_to_record(value, new_key, sep))
        else:
            result[new_key] = value
    return result
