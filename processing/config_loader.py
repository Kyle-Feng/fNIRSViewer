# -*- coding: utf-8 -*-
"""
config_loader.py —— 设备 config.json 解析助手（config 驱动列选择）

从 BrainFlow config.json（如 548pd）的 ability 列表中解析各信号的列配置，
供 processing_fNIRS.py 及后续多设备 / 多脚本复用。新设备只需提供 config.json，
无需修改解析代码。

列号约定：config 中 ability.idx / ability.marker_idx / 顶层 marker_idx / position_idx
均为 0-based 列号（已按真实 data.csv 核实）。
"""

import json
import os


def load_config(config):
    """加载 config.json 配置。

    接受 config.json 路径（str/pathlib.Path）或已解析 dict；
    路径则 json.load，dict 则直接使用。返回完整 dict。
    文件不存在 / JSON 非法时抛出带清晰中文信息的异常。
    """
    if isinstance(config, dict):
        return config
    config_path = str(config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json 文件不存在: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"config.json 解析失败（JSON 非法）: {config_path}，错误: {e}"
        ) from e


def get_ability(config, signal):
    """在 config["ability"] 列表中按 ability["signal"] == signal 返回能力块 dict。

    signal 例如 "fNIRS" / "EEG"。找不到时抛出清晰中文 ValueError。
    """
    for ability in config.get("ability", []):
        if ability.get("signal") == signal:
            return ability
    raise ValueError(
        f"config.json 的 ability 列表中未找到 signal={signal!r} 的能力块"
    )


def resolve_fnirs_columns(config):
    """取 fNIRS ability 块，返回 0-based 列号相关的列配置 dict。

    返回字段：fnirs_idx / marker_idx（0-based 整数列号）、
    light_wave / flash_array / light_name / light_sensor_group / channel_pair
    （fNIRS 块内字段用 .get() 容错，缺失时返回空默认值）。
    """
    fnirs_cfg = get_ability(config, "fNIRS")
    return {
        "fnirs_idx": [int(i) for i in fnirs_cfg.get("idx", [])],
        "marker_idx": int(fnirs_cfg.get("marker_idx", 56)),
        "light_wave": fnirs_cfg.get("light_wave", []),
        "flash_array": fnirs_cfg.get("flash_array", []),
        "light_name": fnirs_cfg.get("light_name", []),
        "light_sensor_group": fnirs_cfg.get("light_sensor_group", {}),
        "channel_pair": fnirs_cfg.get("channel_pair", []),
    }


def resolve_position_column(config, n_rows=None):
    """解析 position 列的 0-based 列号。

    先取 config["position_idx"]（int），再校验 0 <= pos < n_rows
    （n_rows 为 data.shape[0]，None 时跳过校验）；
    config 值缺失或越界时回退旧默认 55。返回 int。
    """
    pos = config.get("position_idx")
    if pos is None:
        return 55
    pos = int(pos)
    if n_rows is not None and not (0 <= pos < n_rows):
        return 55
    return pos
