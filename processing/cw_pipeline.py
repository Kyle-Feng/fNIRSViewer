# -*- coding: utf-8 -*-
"""fNIRSViewer 的连续光强与 MBLL 转换。

本模块只负责查看器所需的连续信号转换：
TDM -> 780/850 nm 光强 -> I/I0 -> 光密度 -> HbO/HbR。
不做 epoch、特征提取或分类，也不默认做 TDDR/带通滤波。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SUPPORTED_WAVELENGTHS = frozenset((780, 850))
DPF_BY_WAVELENGTH = {780: 7.15, 850: 5.98}
EXTINCTION_BY_WAVELENGTH = {
    780: (0.740070, 1.166758),  # HbO, HbR; mM^-1 cm^-1
    850: (1.055422, 0.784144),
}
DEFAULT_FS = 5.0
DC_OFFSET = 2.0
MIN_INTENSITY = 1.0


def validate_wavelengths(wavelengths: Iterable[int]) -> Tuple[int, int]:
    """设备必须且只能配置 780/850 nm；出现 760 nm 等立即报错。"""
    values = tuple(int(value) for value in wavelengths)
    if len(values) != 2 or set(values) != SUPPORTED_WAVELENGTHS:
        raise ValueError(
            "本设备必须且只能使用 780 nm 和 850 nm，"
            f"当前配置为 {list(values)!r}")
    return values


def generate_light_groups(
    flash_array: Sequence[int], light_names: Sequence[str]
) -> Dict[str, List[Tuple[str, int]]]:
    """解析 16-bit TDM 点灯掩码。"""
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for group_index, flash_value in enumerate(flash_array, start=1):
        lights: List[Tuple[str, int]] = []
        for bit in range(16):
            if (int(flash_value) >> bit) & 1:
                light_index = 15 - bit
                if 0 <= light_index < len(light_names):
                    lights.append((str(light_names[light_index]), light_index))
        groups[str(group_index)] = lights
    return groups


def _channels_for_marker(
    marker: int,
    light_groups: Dict[str, List[Tuple[str, int]]],
    light_sensor_group: Dict[str, Sequence[int]],
    light_names: Sequence[str],
    sensor_names: Sequence[str],
) -> Tuple[List[Tuple[int, str]], int]:
    if 1 <= marker <= 5:
        wavelength = 780
        group = marker
    elif 30001 <= marker <= 30005:
        wavelength = 850
        group = marker - 30000
    else:
        return [], 0

    mappings: List[Tuple[int, str]] = []
    for _light_name, light_index in light_groups.get(str(group), []):
        detector_indices = light_sensor_group.get(str(light_index + 1), [])
        for detector_one_based in detector_indices:
            detector_index = int(detector_one_based) - 1
            if not (0 <= detector_index < len(sensor_names)):
                continue
            pair_name = f"{light_names[light_index]}-{sensor_names[detector_index]}"
            mappings.append((detector_index, pair_name))
    return mappings, wavelength


def _validate_timestamps(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("原始时间戳不足")
    if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("原始时间戳必须为有限且严格递增")
    return timestamps


def reconstruct_tdm_uniform(
    detector_data: np.ndarray,
    tdm_marker: np.ndarray,
    timestamps: np.ndarray,
    channel_pairs: Sequence[str],
    flash_array: Sequence[int],
    light_sensor_group: Dict[str, Sequence[int]],
    light_names: Sequence[str],
    sensor_names: Sequence[str],
    target_fs: float = DEFAULT_FS,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """按真实 TDM 更新时间重建并插值到统一时间轴。

    输出行严格为 ``pair0 780, pair0 850, pair1 780, ...``。尚未完成
    两波长首轮更新的前导区间不会进入输出。ADS1299 满量程 ±187500，
    数据为有符号输出，负值按真实负漂移处理（无需 16 位回绕）。
    """
    detector_data = np.asarray(detector_data, dtype=np.float64)
    marker = np.rint(np.asarray(tdm_marker)).astype(np.int64)
    timestamps = _validate_timestamps(timestamps)
    if detector_data.ndim != 2 or detector_data.shape[1] != timestamps.size:
        raise ValueError("探测器数据与时间戳尺寸不匹配")
    if marker.shape != timestamps.shape:
        raise ValueError("TDM marker 与时间戳尺寸不匹配")
    if target_fs <= 0:
        raise ValueError("目标采样率必须为正数")

    light_groups = generate_light_groups(flash_array, light_names)
    names = [
        f"{pair} {wavelength}"
        for pair in channel_pairs
        for wavelength in (780, 850)
    ]
    update_times: Dict[str, List[float]] = {name: [] for name in names}
    update_values: Dict[str, List[float]] = {name: [] for name in names}

    boundaries = np.flatnonzero(np.diff(marker) != 0) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, marker.size]
    for start, stop in zip(starts, stops):
        mappings, wavelength = _channels_for_marker(
            int(marker[start]), light_groups, light_sensor_group,
            light_names, sensor_names)
        if not mappings:
            continue
        width = int(stop - start)
        if width > 5:
            middle_start = start + width // 3
            middle_stop = start + 2 * width // 3
        else:
            middle_start, middle_stop = start, stop
        values = np.nanmean(detector_data[:, middle_start:middle_stop], axis=1)
        update_time = float(np.mean(timestamps[middle_start:middle_stop]))
        for detector_index, pair_name in mappings:
            if not (0 <= detector_index < detector_data.shape[0]):
                continue
            value = float(values[detector_index])
            # ADS1299 满量程 ±187500 有符号输出；负值按真实负漂移处理，
            # 由下方 MIN_INTENSITY 钳到最小光强，无需 16 位回绕。
            value += DC_OFFSET
            key = f"{pair_name} {wavelength}"
            if key in update_times and np.isfinite(value):
                update_times[key].append(update_time)
                update_values[key].append(max(MIN_INTENSITY, value))

    missing = [name for name in names if len(update_times[name]) < 2]
    if missing:
        raise ValueError(
            "TDM 解复用后通道更新点不足: " + ", ".join(missing[:8]))

    common_start = max(times[0] for times in update_times.values())
    common_stop = min(times[-1] for times in update_times.values())
    if common_stop <= common_start:
        raise ValueError("各通道不存在共同有效时间范围")
    step = 1.0 / float(target_fs)
    uniform_time = common_start + np.arange(
        int(np.floor((common_stop - common_start) / step)) + 1,
        dtype=np.float64,
    ) * step

    intensity = np.empty((len(names), uniform_time.size), dtype=np.float64)
    long_gap = np.zeros(uniform_time.size, dtype=bool)
    maximum_normal_gap = 3.0 / float(target_fs)
    for row, name in enumerate(names):
        times = np.asarray(update_times[name], dtype=np.float64)
        values = np.asarray(update_values[name], dtype=np.float64)
        unique_times, unique_indices = np.unique(times, return_index=True)
        for gap_index in np.flatnonzero(np.diff(unique_times) > maximum_normal_gap):
            long_gap |= (
                (uniform_time > unique_times[gap_index])
                & (uniform_time < unique_times[gap_index + 1]))
        intensity[row] = np.interp(
            uniform_time, unique_times, values[unique_indices])
    return intensity, names, uniform_time, long_gap


def optical_density(
    intensity: np.ndarray,
    time_seconds: np.ndarray | None = None,
    baseline_seconds: Tuple[float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 I/I0 与 ``-log10(I/I0)``。基线默认为整段记录。"""
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.ndim != 2 or intensity.shape[1] < 2:
        raise ValueError("连续光强矩阵尺寸无效")
    if np.any(~np.isfinite(intensity)) or np.any(intensity <= 0):
        raise ValueError("连续光强包含非正值或非有限值")

    baseline_mask = np.ones(intensity.shape[1], dtype=bool)
    if baseline_seconds is not None:
        if time_seconds is None:
            raise ValueError("指定秒级基线时必须提供时间轴")
        time_seconds = np.asarray(time_seconds, dtype=np.float64)
        start, stop = (float(baseline_seconds[0]), float(baseline_seconds[1]))
        baseline_mask = (time_seconds >= start) & (time_seconds < stop)
        if not np.any(baseline_mask):
            raise ValueError(f"基线区间 {baseline_seconds!r} 内没有样本")

    i0 = np.mean(intensity[:, baseline_mask], axis=1, keepdims=True)
    if np.any(~np.isfinite(i0)) or np.any(i0 <= 0):
        raise ValueError("基线光强 I0 无效")
    ratio = intensity / i0
    return ratio, -np.log10(ratio)


def mbll(
    optical_density_data: np.ndarray,
    channel_names: Sequence[str],
    configured_wavelengths: Sequence[int],
    dpf_by_wavelength: Dict[int, float] | None = None,
    source_detector_distance_cm: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """用显式波长映射将 OD 转成 HbO/HbR，输出单位为 μM。"""
    validate_wavelengths(configured_wavelengths)
    od = np.asarray(optical_density_data, dtype=np.float64)
    if od.ndim != 2 or od.shape[0] % 2 or len(channel_names) != od.shape[0]:
        raise ValueError("OD 数据或通道名尺寸无效")
    for index, name in enumerate(channel_names):
        expected = 780 if index % 2 == 0 else 850
        if not str(name).endswith(f" {expected}"):
            raise ValueError(
                f"波长行顺序错误：第 {index} 行应为 {expected} nm，实际 {name!r}")
    if np.any(~np.isfinite(od)):
        raise ValueError("OD 包含非有限值")

    mapping = dict(DPF_BY_WAVELENGTH if dpf_by_wavelength is None
                   else dpf_by_wavelength)
    if set(int(key) for key in mapping) != SUPPORTED_WAVELENGTHS:
        raise ValueError("DPF 必须显式包含且只包含 780/850 nm")
    wavelengths = (780, 850)
    extinction = np.asarray(
        [EXTINCTION_BY_WAVELENGTH[w] for w in wavelengths], dtype=np.float64)
    # 表值 mM^-1 cm^-1；除以 1000 后求解结果的数值单位为 μM。
    matrix = (extinction / 1000.0) * np.asarray(
        [float(mapping[w]) for w in wavelengths])[:, None]
    matrix *= float(source_detector_distance_cm)

    n_pairs = od.shape[0] // 2
    n_samples = od.shape[1]
    attenuation = np.stack((od[0::2], od[1::2]), axis=0)
    attenuation = attenuation.transpose(0, 2, 1).reshape(2, -1)
    concentration = np.linalg.solve(matrix, attenuation)
    hbo = concentration[0].reshape(n_samples, n_pairs).T
    hbr = concentration[1].reshape(n_samples, n_pairs).T
    # str.removesuffix 是 Python 3.9 才加入的；项目 HBCI_pyqt6 仍需兼容 3.8。
    base_names = [
        str(name)[:-4] if str(name).endswith(" 780") else str(name)
        for name in channel_names[0::2]
    ]
    return hbo, hbr, base_names


def process_continuous_signal(
    detector_data: np.ndarray,
    tdm_marker: np.ndarray,
    timestamps: np.ndarray,
    *,
    configured_wavelengths: Sequence[int],
    channel_pairs: Sequence[str],
    flash_array: Sequence[int],
    light_sensor_group: Dict[str, Sequence[int]],
    light_names: Sequence[str],
    sensor_names: Sequence[str],
    target_fs: float = DEFAULT_FS,
    dpf_by_wavelength: Dict[int, float] | None = None,
    source_detector_distance_cm: float = 3.0,
    baseline_seconds: Tuple[float, float] | None = None,
) -> dict:
    """查看器用完整连续转换，不含运动校正、滤波、epoch 与分类。"""
    configured = validate_wavelengths(configured_wavelengths)
    intensity, names, time_abs, gap_mask = reconstruct_tdm_uniform(
        detector_data, tdm_marker, timestamps, channel_pairs, flash_array,
        light_sensor_group, light_names, sensor_names, target_fs=target_fs)
    time_seconds = time_abs - float(timestamps[0])
    ratio, od = optical_density(intensity, time_seconds, baseline_seconds)
    hbo, hbr, pair_names = mbll(
        od, names, configured, dpf_by_wavelength=dpf_by_wavelength,
        source_detector_distance_cm=source_detector_distance_cm)
    return {
        "intensity": intensity,
        "relative_intensity": ratio,
        "optical_density": od,
        "hbo": hbo,
        "hbr": hbr,
        "channel_names": pair_names,
        "wavelength_channel_names": names,
        "time": time_seconds,
        "time_absolute": time_abs,
        "fs": float(target_fs),
        "interpolation_gap_mask": gap_mask,
        "processing": {
            "steps": ["TDM", "intensity", "I/I0", "optical_density", "MBLL"],
            "configured_wavelengths_nm": list(configured),
            "mbll_wavelength_order_nm": [780, 850],
            "dpf_by_wavelength": dict(
                DPF_BY_WAVELENGTH if dpf_by_wavelength is None
                else dpf_by_wavelength),
            "source_detector_distance_cm": float(source_detector_distance_cm),
            "output_unit": "uM",
            "motion_correction": False,
            "bandpass_filter": False,
            "epoching": False,
            "feature_extraction": False,
            "classification": False,
        },
    }
