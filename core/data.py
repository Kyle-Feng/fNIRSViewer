# -*- coding: utf-8 -*-
"""
core/data.py —— fNIRSViewer 数据层（纯计算，无 GUI 依赖）

从 annly.py 提取出来的数据加载 / Beer-Lambert / 试次检测 / 丢包检测逻辑，
供 Qt 版（ui/viewer.py）使用。

只依赖 numpy + os，不 import matplotlib / PySide6 / pyqtgraph。
"""

import json
import os
import numpy as np

try:  # 血氧带通滤波需要 scipy；缺失时查看器其余功能不受影响
    from scipy.signal import butter, sosfiltfilt
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - 纯 numpy 环境
    _HAS_SCIPY = False

try:  # ``python fNIRSViewer/main.py``
    from processing import cw_pipeline
except ModuleNotFoundError:  # ``python -m unittest fNIRSViewer...``
    from fNIRSViewer.processing import cw_pipeline

# ---------------------------------------------------------------
# 常量
# ---------------------------------------------------------------
SR_HZ = 1000
WINDOW_SEC = 20
PROCESSED_FS = cw_pipeline.DEFAULT_FS
FNIRS_CH_START = 1
FNIRS_CH_END = 17
POS_MARKER_IDX = 55
MARKER_IDX = 56
TS_IDX = 62
BEHAVIOR_MARKER_IDX = 63

MODE_ORIG = 'ORIG'
MODE_780 = '780nm'
MODE_850 = '850nm'
MODE_INTENSITY = '光强'
MODE_RELATIVE = '相对光强(%)'
MODE_OD = '光密度'
MODE_FNIRS = 'fNIRS'
# 780nm / 850nm 单波长视图已弃用，按钮注释保留；如需恢复，取消下方注释即可
DISPLAY_MODES = [
    MODE_ORIG,
    # MODE_780,
    # MODE_850,
    MODE_INTENSITY,
    MODE_RELATIVE,
    MODE_OD,
    MODE_FNIRS,
]

MARKER_COLORS = {
    0:      (1.0, 1.0, 1.0, 0.0),
    1:      (0.95, 0.55, 0.55, 0.38),
    2:      (0.95, 0.82, 0.45, 0.38),
    3:      (0.60, 0.90, 0.55, 0.38),
    4:      (0.55, 0.78, 0.95, 0.38),
    5:      (0.82, 0.58, 0.92, 0.38),
    30001:  (0.92, 0.25, 0.25, 0.42),
    30002:  (0.95, 0.68, 0.15, 0.42),
    30003:  (0.35, 0.82, 0.30, 0.42),
    30004:  (0.20, 0.50, 0.92, 0.42),
    30005:  (0.65, 0.25, 0.88, 0.42),
}

MARKER_LABELS = {
    0:      '0 (静息 Rest)',
    1:      '780nm Group 1',
    2:      '780nm Group 2',
    3:      '780nm Group 3',
    4:      '780nm Group 4',
    5:      '780nm Group 5',
    30001:  '850nm Group 1',
    30002:  '850nm Group 2',
    30003:  '850nm Group 3',
    30004:  '850nm Group 4',
    30005:  '850nm Group 5',
}

# 行为试次（col63）：MI 110=左手 111=右手 112=静息；MA 111=心算 112=放松
BEHAVIOR_COLORS = {
    100: (0.65, 0.65, 0.70, 0.9),   # 灰 准备
    110: (0.25, 0.55, 0.90, 0.9),   # 蓝 左手
    111: (0.90, 0.32, 0.32, 0.9),   # 红 右手/心算
    112: (0.28, 0.62, 0.34, 0.9),   # 绿 静息/放松
}
BEHAVIOR_LABELS = {
    'MI': {100: ('准备', 'Ready'), 110: ('左手', 'Left'), 111: ('右手', 'Right'), 112: ('静息', 'Rest')},
    'MA': {100: ('准备', 'Ready'), 111: ('心算', 'Mental math'), 112: ('放松', 'Relax')},
}

# UI 数组顺序明确为 [780, 850]；数值不变，只修正旧注释/隐式配对。
DPF_DEFAULT = [
    cw_pipeline.DPF_BY_WAVELENGTH[780],
    cw_pipeline.DPF_BY_WAVELENGTH[850],
]
OPDIST_DEFAULT = 3.0

# fNIRS TDM 解复用：ADS1299 满量程 ±187500 有符号输出，
# 负值按真实负漂移处理（无需 16 位回绕），加直流偏置并设下限
FNIRS_DC_OFFSET = 2.0
FNIRS_MIN_CURRENT = 1.0

# 光强 ZOH 台阶平滑：滑动平均窗口样本数（1000 Hz 采样，约 5 Hz 更新 → 每段约 200 样本）
CW_SMOOTH_WINDOW = 200

# 光强预处理：去饱和（548pd ADC 满量程约 187500，接近即视为削顶）
CW_SATURATION = 187000.0
# 光强有效性阈值：低于通道中位数该比例视为掉光/未点亮，用最近有效值前向填充
CW_INTENSITY_FRAC = 0.01
# 死通道判定：S-D 对两波长光强中位数之比低于该值，视为某波长失效，掩蔽该对
CW_PAIR_DEAD_FRAC = 0.10

# 消光系数 (mM^-1 cm^-1)，Matcher 1995；Beer-Lambert 时除以 1000
EXTINCTION_TABLE = {
    780: [0.740070, 1.166758],   # HbO, HbR at 780 nm
    850: [1.055422, 0.784144],   # HbO, HbR at 850 nm
}

# 任务 epoch 人工质检标注：独立 JSON 文件，绝不写入 config.json / data.csv
ANNOTATION_FILE_NAME = "epoch_annotations.json"
ANNOTATION_VERSION = 1
ANNOTATION_LABELS = {
    "invalid": "无效",
    "artifact": "伪迹",
    "questionable": "可疑",
}
ANNOTATION_LABEL_COLORS = {
    "invalid": (0.90, 0.32, 0.32, 0.9),      # 红
    "artifact": (0.95, 0.68, 0.15, 0.9),     # 橙
    "questionable": (0.82, 0.58, 0.92, 0.9),  # 紫
}


# ---------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------
def discover_streams(data_raw_dir: str):
    """扫描 data/raw 下所有含 data.csv 的 stream。

    :return: [(task, subject, stream_dir), ...] 按任务/被试/目录排序
    """
    results = []
    for task in ["MI", "MA"]:
        task_dir = os.path.join(data_raw_dir, task)
        if not os.path.isdir(task_dir):
            continue
        for subj in sorted(os.listdir(task_dir)):
            subj_dir = os.path.join(task_dir, subj)
            if not os.path.isdir(subj_dir):
                continue
            for stream in sorted(os.listdir(subj_dir)):
                stream_dir = os.path.join(subj_dir, stream)
                if os.path.isdir(stream_dir) and os.path.isfile(os.path.join(stream_dir, "data.csv")):
                    results.append((task, subj, stream_dir))
    return results


def load_fnirs_data(fname):
    # 只读需要的列（16 路 fNIRS + position + fNIRS marker + timestamp + 行为标记），
    # 用 np.loadtxt 的 C 实现；首次解析后缓存为 .npz，二次打开秒开。
    n_fnirs = FNIRS_CH_END - FNIRS_CH_START  # 16
    cols = list(range(FNIRS_CH_START, FNIRS_CH_END)) + [POS_MARKER_IDX, MARKER_IDX, TS_IDX, BEHAVIOR_MARKER_IDX]
    cache_path = fname + '.cache.npz'
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(fname):
        d = np.load(cache_path)
        return (d['fNIRS'], d['position'], d['marker'], d['timestamp'], d['behavior'])

    raw = np.loadtxt(fname, delimiter=',', usecols=cols, dtype=np.float64)
    if raw.ndim == 1:  # 只有一行数据的情况
        raw = raw[None, :]

    fNIRS = raw[:, :n_fnirs].T.copy()

    position_marker = raw[:, n_fnirs].copy()
    valid_pm = ~np.isnan(position_marker)
    position_marker[~valid_pm] = -1
    position_marker = position_marker.astype(np.int64)

    fNIRS_marker = raw[:, n_fnirs + 1].copy()
    valid_fm = ~np.isnan(fNIRS_marker)
    fNIRS_marker[~valid_fm] = -1
    fNIRS_marker = fNIRS_marker.astype(np.int64)

    timestamp = raw[:, n_fnirs + 2].copy()

    behavior = raw[:, n_fnirs + 3].copy()
    valid_b = ~np.isnan(behavior)
    behavior[~valid_b] = -1
    behavior = behavior.astype(np.int64)

    try:
        np.savez(cache_path, fNIRS=fNIRS, position=position_marker, marker=fNIRS_marker,
                 timestamp=timestamp, behavior=behavior)
    except Exception:
        pass

    return fNIRS, position_marker, fNIRS_marker, timestamp, behavior


def compute_packet_loss(position_marker):
    valid = position_marker >= 0
    if np.sum(valid) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    pm_v = position_marker[valid]
    diffs = np.diff(pm_v)
    jump_mask = diffs > 1
    jump_valid_idx = np.where(jump_mask)[0]
    valid_indices = np.where(valid)[0]
    jump_sample_start = valid_indices[jump_valid_idx]
    lost_per_jump = diffs[jump_mask].astype(np.int64) - 1
    return jump_sample_start, lost_per_jump


def window_packet_loss(jump_samples, lost_per_jump, win_start, win_end):
    if len(jump_samples) == 0:
        return 0, 0
    in_win = (jump_samples >= win_start) & (jump_samples < win_end)
    n_jumps = int(np.sum(in_win))
    n_lost = int(np.sum(lost_per_jump[in_win]))
    return n_jumps, n_lost


# ---------------------------------------------------------------
# 数据模型（状态 + 纯计算，不含任何 GUI 代码）
# ---------------------------------------------------------------
class FNIRSData:
    """封装一条 stream 的加载与所有纯计算，供查看器使用。"""

    def __init__(self, fname):
        if os.path.isdir(fname):
            fname = os.path.join(fname, "data.csv")
        self.fname = fname
        # 查看器使用设备时间戳与 TDM 片段中部测量；默认不再人为平移 marker。
        self.marker_shift = 0
        self.data_shift = 0
        self.dpf = list(DPF_DEFAULT)
        self.opdist = OPDIST_DEFAULT
        self.ival = None
        # 血氧滤波状态：None=未启用；启用后缓存滤波结果于 data_fnirs_filtered
        self.filter_low = None
        self.filter_high = None
        self.data_fnirs_filtered = None

        (self.fNIRS_orig_raw, self.position_marker, self.fNIRS_marker_raw,
         self.timestamp, self.behavior_marker) = load_fnirs_data(fname)
        self.task_type = self._detect_task(fname)
        self.annotation_path = os.path.join(
            os.path.dirname(os.path.abspath(fname)), ANNOTATION_FILE_NAME)
        self.epoch_annotations = self._load_annotations()
        self._load_fnirs_config(fname)
        self.raw_time = self._relative_raw_time(self.timestamp)
        self.fNIRS_orig = self._apply_data_shift(self.fNIRS_orig_raw, self.data_shift)
        self.fNIRS_marker = self._apply_marker_shift(self.fNIRS_marker_raw, self.marker_shift)
        self._rebuild_mode_data()

        self.n_samples = self.fNIRS_orig.shape[1]
        self.samples_per_window = int(WINDOW_SEC * SR_HZ)
        self.min_win_samples = int(1 * SR_HZ)
        self.max_win_samples = int(120 * SR_HZ)
        self.current_start = 0
        self.current_mode = MODE_ORIG
        self.channel_spacing = 5.0
        self.selected_channels = None  # None=显示全部；否则为已选通道基础名集合

        self.jump_samples, self.jump_lost = compute_packet_loss(self.position_marker)

        unique_markers = sorted(set(self.fNIRS_marker[self.fNIRS_marker >= 0]))
        self.unique_markers = [m for m in unique_markers if m in MARKER_COLORS]

        self.behavior_trials = self._find_behavior_trials(self.behavior_marker)

    @staticmethod
    def _relative_raw_time(timestamp):
        """将设备绝对时间戳转成从 0 开始的秒轴，并严格检查有效性。"""
        ts = np.asarray(timestamp, dtype=np.float64)
        if ts.ndim != 1 or ts.size < 2:
            raise ValueError("原始时间戳不足，无法显示 fNIRS 连续信号")
        if np.any(~np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
            raise ValueError("原始时间戳必须为有限且严格递增")
        return ts - ts[0]

    # ---- 移位 / 模式数据 ----
    def _apply_marker_shift(self, marker_raw, shift):
        n = len(marker_raw)
        s = int(shift)
        shifted = np.zeros_like(marker_raw)
        if s > 0 and n > s:
            shifted[s:] = marker_raw[:n - s]
        elif s < 0:
            s_neg = -s
            if n > s_neg:
                shifted[:n - s_neg] = marker_raw[s_neg:]
        else:
            shifted[:] = marker_raw[:]
        return shifted

    @staticmethod
    def _apply_data_shift(data_2d, shift):
        ch, n = data_2d.shape
        s = int(shift)
        out = np.zeros_like(data_2d, dtype=np.float64)
        if s == 0:
            out[:, :] = data_2d[:, :]
            return out
        med = np.nanmedian(data_2d, axis=1)
        med = np.where(np.isfinite(med), med, 0.0)
        if s > 0:
            if n > s:
                out[:, s:] = data_2d[:, :n - s]
            for c in range(ch):
                out[c, :min(s, n)] = med[c]
        else:
            s_neg = -s
            if n > s_neg:
                out[:, :n - s_neg] = data_2d[:, s_neg:]
            for c in range(ch):
                out[c, max(n - s_neg, 0):] = med[c]
        return out

    @staticmethod
    def _apply_data_shift_1d(arr_1d, shift):
        n = len(arr_1d)
        s = int(shift)
        out = np.zeros_like(arr_1d)
        if s == 0:
            out[:] = arr_1d[:]
            return out
        fill_val = 0
        if s > 0:
            if n > s:
                out[s:] = arr_1d[:n - s]
            out[:min(s, n)] = fill_val
        else:
            s_neg = -s
            if n > s_neg:
                out[:n - s_neg] = arr_1d[s_neg:]
            out[max(n - s_neg, 0):] = fill_val
        return out

    # ---- TDM 解复用 & Beer-Lambert ----
    def _load_fnirs_config(self, fname):
        """读取并验证 TDM 配置；任何非 780/850 nm 配置立即报错。"""
        self.channel_pair = []
        self.light_names = []
        self.sensor_names = []
        self.light_sensor_group = {}
        self.flash_array = []
        self.light_wave = []
        self.light_groups = {}
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(fname)), "config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"缺少 fNIRS 配置文件: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        fnirs_cfg = next(
            (ability for ability in cfg.get("ability", [])
             if ability.get("signal") == "fNIRS"), None)
        if fnirs_cfg is None:
            raise ValueError(f"config.json 中未找到 fNIRS ability: {cfg_path}")

        self.light_wave = list(cw_pipeline.validate_wavelengths(
            fnirs_cfg.get("light_wave", [])))
        self.light_names = list(fnirs_cfg.get("light_name", []))
        self.sensor_names = list(fnirs_cfg.get("name", []))
        self.light_sensor_group = fnirs_cfg.get("light_sensor_group", {})
        self.flash_array = list(fnirs_cfg.get("flash_array", []))
        self.channel_pair = list(fnirs_cfg.get("channel_pair", []))
        if not self.channel_pair or not self.flash_array:
            raise ValueError("fNIRS 配置缺少 channel_pair 或 flash_array")
        self.light_groups = self._generate_light_groups(self.flash_array, self.light_names)

    @staticmethod
    def _generate_light_groups(flash_array, light_names):
        """从 flash_array 解析每个光分组内的光源。flash_val 是 16-bit 位掩码，bit0=光源15，bit15=光源0。"""
        groups = {}
        for i, flash_val in enumerate(flash_array):
            groups[str(i + 1)] = []
            for bit in range(16):
                if (int(flash_val) >> bit) & 1:
                    light_idx = 15 - bit
                    if 0 <= light_idx < len(light_names):
                        groups[str(i + 1)].append((light_names[light_idx], light_idx))
        return groups

    def _get_channel_data_by_marker(self, marker):
        """由 TDM marker 判定波长（780/850）与当前点亮的 S-D 通道。

        :return: (fnirs_indexes, light_str)，fnirs_indexes = [(探测器索引列表, 通道名列表), ...]
        """
        if marker > 30000:
            light_str = "850"
            light_arr = self.light_groups.get(str(marker - 30000), [])
        elif marker >= 1:
            light_str = "780"
            light_arr = self.light_groups.get(str(marker), [])
        else:
            return [], "780"  # marker 0 / -1：无光源点亮

        fnirs_indexes = []
        for _, light_idx in light_arr:
            key = str(int(light_idx) + 1)
            if key not in self.light_sensor_group:
                continue
            idxs = self.light_sensor_group[key]
            names = []
            for j in idxs:
                sj = int(j) - 1
                if 0 <= light_idx < len(self.light_names) and 0 <= sj < len(self.sensor_names):
                    names.append(self.light_names[light_idx] + "-" + self.sensor_names[sj])
                else:
                    names.append(f"{light_idx}-{sj}")
            fnirs_indexes.append((idxs, names))
        return fnirs_indexes, light_str

    def _reconstruct_cw(self, fnirs_data, fnirs_marker):
        """TDM 解复用：16 路探测器 + TDM marker → 36 S-D 对 × 2 波长连续波光强（零阶保持）。

        :return: (cw, ch_names)，cw: (n_pairs*2, n_samples)，顺序 [pair0_780, pair0_850, ...]
        """
        n_samples = fnirs_data.shape[1]
        ch_names_all = []
        for name in self.channel_pair:
            ch_names_all.append(f"{name} 780")
            ch_names_all.append(f"{name} 850")
        cw = np.zeros((len(ch_names_all), n_samples), dtype=np.float64)
        if len(self.channel_pair) == 0:
            return cw, ch_names_all

        marker_col = fnirs_marker.tolist()
        n = len(marker_col)
        if n == 0:
            return cw, ch_names_all

        # 划分连续相同 marker 的片段
        segments = []
        start = 0
        for i in range(1, n):
            if marker_col[i] != marker_col[i - 1]:
                segments.append((start, i))
                start = i
        segments.append((start, n))

        last_values = {ch: 0.0 for ch in ch_names_all}

        for start_idx, end_idx in segments:
            segment = fnirs_data[:, start_idx:end_idx]
            seg_marker = fnirs_marker[start_idx:end_idx]
            if segment.shape[1] > 5:
                ms = segment.shape[1] // 3
                me = 2 * segment.shape[1] // 3
                avg = np.average(segment[:, ms:me], axis=1, keepdims=True)
                marker = int(round(np.average(seg_marker[ms:me])))
            else:
                avg = np.average(segment, axis=1, keepdims=True)
                marker = int(round(np.average(seg_marker)))

            fnirs_indexes, light_str = self._get_channel_data_by_marker(marker)
            for sensor_indices, c_names in fnirs_indexes:
                for j in range(len(sensor_indices)):
                    sensor = int(sensor_indices[j]) - 1  # 0-based 探测器列
                    if 0 <= sensor < segment.shape[0]:
                        v = float(avg[sensor][0])
                        # ADS1299 满量程 ±187500 有符号输出；负值按真实负漂移处理，
                        # 由下方 FNIRS_MIN_CURRENT 钳到最小光强，无需 16 位回绕。
                        v += FNIRS_DC_OFFSET
                        if np.isfinite(v):
                            v = max(FNIRS_MIN_CURRENT, v)
                            ch_key = f"{c_names[j]} {light_str}"
                            if ch_key in last_values:
                                last_values[ch_key] = v

            for ch_idx, ch_key in enumerate(ch_names_all):
                cw[ch_idx, start_idx:end_idx] = last_values[ch_key]

        return cw, ch_names_all

    def _fill_zero_hold(self, cw):
        """修复 TDM ZOH 中光强掉到接近 0 的无效段（用最近有效值填充）。

        未点亮/瞬时掉光的 0 或近 0 值经 Beer-Lambert 的 log10 会被放大成巨大尖峰，
        这里用左侧最近有效值前向填充，开头段用首个有效值后向填充。
        """
        out = cw.copy()
        n = out.shape[1]
        col = np.arange(n)
        for i in range(out.shape[0]):
            row = out[i]
            # 相对阈值：低于通道中位数一定比例（且不低于直流下限）视为掉光
            thr = max(FNIRS_MIN_CURRENT, CW_INTENSITY_FRAC * float(np.median(row)))
            valid = row >= thr
            if valid.all():
                continue
            nz = np.nonzero(valid)[0]
            if nz.size == 0:
                continue
            vi = np.where(valid, col, 0)
            ff = np.maximum.accumulate(vi)
            ff[:nz[0]] = nz[0]
            out[i] = np.where(valid, row, row[ff])
        return out

    def _preprocess_cw(self, cw):
        """光强预处理：去饱和。

        光强达到 ADC 满量程(约187500)时被削顶，用最近非饱和值前向填充。
        """
        out = cw.copy()
        n = out.shape[1]
        col = np.arange(n)
        for i in range(out.shape[0]):
            row = out[i]
            sat = row >= CW_SATURATION
            valid = ~sat
            if sat.any() and valid.any():
                nz = np.nonzero(valid)[0]
                vi = np.where(valid, col, 0)
                ff = np.maximum.accumulate(vi)
                ff[:nz[0]] = nz[0]
                row = np.where(valid, row, row[ff])
            out[i] = row
        return out

    def _smooth_cw(self, cw):
        """对 ZOH 台阶光强沿时间轴做滑动平均，消除方波台阶。

        :param cw: (n_ch, n_samples) 光强数组
        :return: 平滑后的同形状数组
        """
        w = int(CW_SMOOTH_WINDOW)
        if w <= 1 or cw.shape[1] == 0:
            return cw
        kernel = np.ones(w) / w
        out = np.empty_like(cw)
        for i in range(cw.shape[0]):
            out[i] = np.convolve(cw[i], kernel, mode='same')
        return out

    def _beer_lambert_cw(self, cw):
        """将交替排列的 780/850 光强 (n_pairs*2, n) 转换为 HbO/HbR 浓度。

        :return: (hbo, hbr)，各 (n_pairs, n_samples)
        """
        n_pairs = cw.shape[0] // 2
        n_samples = cw.shape[1]
        if n_pairs == 0:
            return (np.zeros((0, n_samples)), np.zeros((0, n_samples)))

        data_780 = cw[0::2, :].T  # (n_samples, n_pairs)
        data_850 = cw[1::2, :].T  # (n_samples, n_pairs)

        # 旧兼容入口也强制使用显式 [780,850] 行顺序。
        cw_pipeline.validate_wavelengths(self.light_wave)
        wavelengths = [780, 850]
        ext = np.array([EXTINCTION_TABLE[w] for w in wavelengths], dtype=float)
        epsilon = ext / 1000.0

        # 基线区间
        if self.ival is None:
            s0, s1 = 0, n_samples
        else:
            s0, s1 = int(self.ival[0]), int(self.ival[1])
        s0 = max(0, min(s0, n_samples))
        s1 = max(0, min(s1, n_samples))
        if s1 <= s0:
            s0, s1 = 0, n_samples

        eps = np.finfo(float).eps
        mean_780 = np.clip(np.mean(data_780[s0:s1, :], axis=0), eps, None)
        mean_850 = np.clip(np.mean(data_850[s0:s1, :], axis=0), eps, None)
        att_780 = np.real(-np.log10(np.clip(data_780 / mean_780, eps, None)))
        att_850 = np.real(-np.log10(np.clip(data_850 / mean_850, eps, None)))

        A = np.zeros((n_samples * n_pairs, 2))
        A[:, 0] = att_780.ravel()
        A[:, 1] = att_850.ravel()

        e2 = epsilon * np.array(self.dpf, dtype=float)[:, None] * self.opdist
        c = np.linalg.solve(e2, A.T)
        hbo = c[0, :].reshape(n_samples, n_pairs).T
        hbr = c[1, :].reshape(n_samples, n_pairs).T
        return hbo, hbr

    def _mask_dead_pairs(self, cw, out):
        """掩蔽死通道对：S-D 对两波长光强中位数之比过低时，该对 HbO/HbR 置 0。

        Beer-Lambert 需要 780/850 两个波长；某波长 LED 失效导致光强接近 0 时，
        反演出的 HbO/HbR 会是巨大尖峰，这里直接将该 S-D 对整对置 0。
        """
        n_pairs = cw.shape[0] // 2
        med = np.median(cw, axis=1)
        med_780 = med[0::2]
        med_850 = med[1::2]
        hi = np.maximum(med_780, med_850)
        ratio = np.where(hi > 0, np.minimum(med_780, med_850) / np.maximum(hi, 1e-12), 0.0)
        dead = ratio < CW_PAIR_DEAD_FRAC
        for p in np.nonzero(dead)[0]:
            out[2 * p:2 * p + 2, :] = 0.0

    def _rebuild_mode_data(self):
        dpf_by_wavelength = {780: float(self.dpf[0]), 850: float(self.dpf[1])}
        baseline_seconds = None if self.ival is None else (
            float(self.ival[0]), float(self.ival[1]))
        result = cw_pipeline.process_continuous_signal(
            self.fNIRS_orig,
            self.fNIRS_marker,
            self.timestamp,
            configured_wavelengths=self.light_wave,
            channel_pairs=self.channel_pair,
            flash_array=self.flash_array,
            light_sensor_group=self.light_sensor_group,
            light_names=self.light_names,
            sensor_names=self.sensor_names,
            target_fs=PROCESSED_FS,
            dpf_by_wavelength=dpf_by_wavelength,
            source_detector_distance_cm=float(self.opdist),
            baseline_seconds=baseline_seconds,
        )
        self.data_cw = result["intensity"]
        self.data_relative = result["relative_intensity"]
        self.data_od = result["optical_density"]
        self.cw_ch_names = result["wavelength_channel_names"]
        self.channel_pair = result["channel_names"]
        self.processed_time = result["time"]
        self.processed_fs = result["fs"]
        self.processing_info = result["processing"]
        self.interpolation_gap_mask = result["interpolation_gap_mask"]
        self.n_pairs = len(self.channel_pair)

        # 显示布局保持 [pair0 HbO, pair0 HbR, ...]。
        out = np.empty((self.n_pairs * 2, result["hbo"].shape[1]), dtype=np.float64)
        out[0::2, :] = result["hbo"]
        out[1::2, :] = result["hbr"]
        self.data_fnirs = out
        self._mode_marker_cache = {}
        # 处理参数变化后，旧滤波结果失效，需重新滤波
        self.data_fnirs_filtered = None
        self.filter_low = None
        self.filter_high = None

    def _get_mode_marker(self, mode):
        cache_key = (mode, int(self.data_shift))
        if cache_key in self._mode_marker_cache:
            return self._mode_marker_cache[cache_key]
        base = self.fNIRS_marker
        if mode == MODE_ORIG:
            out = base.copy()
        else:
            n = len(base)
            out = np.zeros_like(base)
            for i in range(n):
                m = int(base[i]) if base[i] >= 0 else -1
                if m < 0:
                    continue
                if 1 <= m <= 5:
                    if mode == MODE_780 or mode == MODE_FNIRS or mode == MODE_INTENSITY:
                        out[i] = m
                elif 30001 <= m <= 30005:
                    if mode == MODE_850 or mode == MODE_FNIRS or mode == MODE_INTENSITY:
                        out[i] = m - 30000
        if self.data_shift != 0:
            out = self._apply_data_shift_1d(out, self.data_shift)
        self._mode_marker_cache[cache_key] = out
        return out

    def get_visible_data(self, win_start, win_end):
        sl = slice(win_start, win_end)
        if self.current_mode == MODE_ORIG:
            raw = self.fNIRS_orig
            return (raw[:, sl],
                    [f'Ch{i + 1}' for i in range(raw.shape[0])],
                    self.raw_time[sl])

        # 导航仍使用原始样本索引，但连续光强/OD/Hb 使用真实 5 Hz 时间轴。
        t_start = self.raw_time[max(0, min(win_start, self.raw_time.size - 1))]
        if win_end >= self.raw_time.size:
            t_stop = self.raw_time[-1] + 1.0 / SR_HZ
        else:
            t_stop = self.raw_time[max(win_start + 1, win_end)]
        time_mask = (self.processed_time >= t_start) & (self.processed_time < t_stop)
        processed_time = self.processed_time[time_mask]

        n_pairs = self.n_pairs
        pair_names = self.channel_pair if len(self.channel_pair) == n_pairs else None

        def _label(p, suffix):
            base = pair_names[p] if pair_names is not None else f'S{p + 1}'
            return f'{base}-{suffix}'

        if self.current_mode == MODE_780:
            labels = [_label(p, '780') for p in range(n_pairs)]
            return self.data_cw[0::2, time_mask], labels, processed_time

        if self.current_mode == MODE_850:
            labels = [_label(p, '850') for p in range(n_pairs)]
            return self.data_cw[1::2, time_mask], labels, processed_time

        if self.current_mode == MODE_INTENSITY:
            labels = []
            for p in range(n_pairs):
                labels.append(_label(p, '780'))
                labels.append(_label(p, '850'))
            return self.data_cw[:, time_mask], labels, processed_time

        if self.current_mode == MODE_RELATIVE:
            labels = []
            for p in range(n_pairs):
                labels.append(_label(p, '%780'))
                labels.append(_label(p, '%850'))
            return ((self.data_relative[:, time_mask] - 1.0) * 100.0,
                    labels, processed_time)

        if self.current_mode == MODE_OD:
            labels = []
            for p in range(n_pairs):
                labels.append(_label(p, 'OD780'))
                labels.append(_label(p, 'OD850'))
            return self.data_od[:, time_mask], labels, processed_time

        labels = []
        for p in range(n_pairs):
            labels.append(_label(p, 'HbO'))
            labels.append(_label(p, 'HbR'))
        src = self.data_fnirs if self.data_fnirs_filtered is None else self.data_fnirs_filtered
        return src[:, time_mask], labels, processed_time

    def channel_options(self):
        """返回当前模式的有序通道基础名列表（供通道选择 UI 使用）。"""
        if self.current_mode == MODE_ORIG:
            return [f'Ch{i + 1}' for i in range(self.fNIRS_orig.shape[0])]
        n_pairs = self.n_pairs
        if len(self.channel_pair) == n_pairs:
            return list(self.channel_pair)
        return [f'S{p + 1}' for p in range(n_pairs)]

    @staticmethod
    def _detect_task(fname):
        """从路径判断任务类型：data/raw/MI/... 或 data/raw/MA/..."""
        p = os.path.abspath(fname).replace('\\', '/')
        parts = p.split('/')
        if 'MA' in parts:
            return 'MA'
        return 'MI'

    @staticmethod
    def _find_behavior_trials(behavior_marker):
        """从行为标记（col63）找试次：每个脉冲标志一个试次开始，下一个脉冲即结束。

        :return: [(marker, start_sample, end_sample), ...]
        """
        n = len(behavior_marker)
        pulses = []
        i = 0
        while i < n:
            m = int(behavior_marker[i]) if behavior_marker[i] >= 0 else -1
            if m in (110, 111, 112):
                j = i
                while j + 1 < n and int(behavior_marker[j + 1]) == m:
                    j += 1
                pulses.append((m, i))
                i = j + 1
            else:
                i += 1
        trials = []
        if pulses and pulses[0][1] > 0:
            trials.append((100, 0, pulses[0][1]))  # 首个试次前的准备阶段
        for k, (m, s) in enumerate(pulses):
            e = pulses[k + 1][1] if k + 1 < len(pulses) else n
            trials.append((m, s, e))
        return trials

    # ---- 任务 epoch 标注持久化 ----
    def _load_annotations(self) -> dict:
        """从独立 JSON 文件加载 epoch 标注；缺失/损坏/版本不匹配返回空 dict。"""
        if not os.path.isfile(self.annotation_path):
            return {}
        try:
            with open(self.annotation_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return {}
        if raw.get("version") != ANNOTATION_VERSION:
            return {}
        annotations = raw.get("annotations")
        if not isinstance(annotations, dict):
            return {}
        return {str(key): dict(value) for key, value in annotations.items()
                if isinstance(value, dict)}

    def _save_annotations(self) -> None:
        """写回 epoch 标注 JSON；失败静默（不影响查看器主流程）。"""
        try:
            payload = {
                "version": ANNOTATION_VERSION,
                "task_type": self.task_type,
                "annotations": self.epoch_annotations,
            }
            with open(self.annotation_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _marker_for_epoch(self, epoch_index):
        """由 epoch 标识反查行为标记（与 _trial_records_in_window 编号规则一致）。"""
        key = str(epoch_index)
        task_markers = {110, 111} if self.task_type == 'MI' else {111}
        rest_markers = {112}
        task_idx = 0
        rest_idx = 0
        for marker, _s, _e in self.behavior_trials:
            if marker in task_markers:
                task_idx += 1
                if key == str(task_idx):
                    return int(marker)
            elif marker in rest_markers:
                rest_idx += 1
                if key == f'R{rest_idx}':
                    return int(marker)
        return None

    def get_epoch_annotation(self, epoch_index):
        """返回某 epoch 的标注 dict 或 None。"""
        return self.epoch_annotations.get(str(epoch_index))

    def set_epoch_annotation(self, epoch_index, label, note):
        """写入/更新某 epoch 的标注并持久化。"""
        if label not in ANNOTATION_LABELS:
            raise ValueError(f"未知标注标签: {label}")
        key = str(epoch_index)
        marker = self._marker_for_epoch(key)
        if marker is None:
            raise ValueError(f"无效任务 epoch 编号: {epoch_index}")
        self.epoch_annotations[key] = {
            "marker": marker,
            "label": label,
            "note": str(note or ""),
        }
        self._save_annotations()

    def clear_epoch_annotation(self, epoch_index):
        """删除某 epoch 的标注并持久化。"""
        key = str(epoch_index)
        if key in self.epoch_annotations:
            del self.epoch_annotations[key]
            self._save_annotations()

    def apply_processing_params(self, data_shift, marker_shift, dpf, opdist, ival):
        """等价于旧版「Apply Settings」：更新处理参数并重建数据/试次。"""
        if len(dpf) != 2 or np.any(~np.isfinite(np.asarray(dpf, dtype=float))):
            raise ValueError("DPF 必须按 [780,850] 提供两个有限数值")
        if float(opdist) <= 0:
            raise ValueError("光源-探测器距离必须为正数")
        self.data_shift = int(data_shift)
        self.marker_shift = int(marker_shift)
        self.dpf = [float(dpf[0]), float(dpf[1])]
        self.opdist = float(opdist)
        self.ival = ival
        self.fNIRS_orig = self._apply_data_shift(self.fNIRS_orig_raw, self.data_shift)
        self.fNIRS_marker = self._apply_marker_shift(self.fNIRS_marker_raw, self.marker_shift)
        self._rebuild_mode_data()
        self.behavior_trials = self._find_behavior_trials(self.behavior_marker)

    # ---- 血氧数据滤波 ----
    @property
    def blood_oxygen_filter(self):
        """当前启用的滤波通带 (f_low, f_high) Hz；未启用返回 None。"""
        if self.data_fnirs_filtered is None:
            return None
        return (self.filter_low, self.filter_high)

    def apply_blood_oxygen_filter(self, f_low, f_high):
        """对 HbO/HbR 施加零相位带通滤波（4 阶 Butterworth + sosfiltfilt）。

        数据采样率为 processed_fs（5 Hz），频率需满足 0 < f_low < f_high < fs/2。
        滤波结果缓存在 data_fnirs_filtered，仅在 MODE_FNIRS 显示时使用。
        """
        fs = float(self.processed_fs)
        nyquist = fs / 2.0
        if not (0 < float(f_low) < float(f_high) < nyquist):
            raise ValueError(
                f"滤波频率需满足 0 < 低通 < 高通 < {nyquist:g} Hz（采样率 {fs:g} Hz）")
        if not _HAS_SCIPY:
            raise ValueError("缺少 scipy：请先 pip install scipy 后再启用滤波")
        if self.data_fnirs.shape[1] <= 32:
            raise ValueError("数据过短，无法执行滤波")
        sos = butter(4, [float(f_low), float(f_high)],
                     btype='bandpass', fs=fs, output='sos')
        filtered = np.empty_like(self.data_fnirs)
        for i in range(self.data_fnirs.shape[0]):
            row = self.data_fnirs[i]
            finite = np.isfinite(row)
            if finite.all():
                filtered[i] = sosfiltfilt(sos, row)
            else:
                # 极少数非有限段：仅对连续有限段滤波，其余保留 NaN
                out = np.full_like(row, np.nan)
                idx = np.flatnonzero(finite)
                if idx.size:
                    bounds = np.flatnonzero(np.diff(idx) > 1)
                    seg_starts = np.r_[idx[0], idx[bounds + 1]]
                    seg_stops = np.r_[idx[bounds], idx[-1]]
                    for s0, s1 in zip(seg_starts, seg_stops):
                        out[s0:s1 + 1] = sosfiltfilt(sos, row[s0:s1 + 1])
                filtered[i] = out
        self.data_fnirs_filtered = filtered
        self.filter_low = float(f_low)
        self.filter_high = float(f_high)

    def clear_blood_oxygen_filter(self):
        """关闭血氧滤波，恢复原始 HbO/HbR。"""
        self.data_fnirs_filtered = None
        self.filter_low = None
        self.filter_high = None

    # ---- 窗口计算：一次算出某个窗口的所有绘图数据（只算不画） ----
    def _series_per_row(self):
        """每个显示行的系列数：光强/fNIRS 为 2（双曲线同排），其余为 1。"""
        return 2 if self.current_mode in (
            MODE_INTENSITY, MODE_RELATIVE, MODE_OD, MODE_FNIRS) else 1

    @staticmethod
    def _split_label(label):
        """从 '通道-后缀' 标签拆出 (通道名, 种类)。无后缀（ORIG）时种类为 'Ch'。"""
        if '-' in label:
            base, kind = label.rsplit('-', 1)
            return base, kind
        return label, 'Ch'

    @staticmethod
    def _normalize_series(d):
        """单系列鲁棒归一化：中位数中心化 + 2%-98% 分位尺度（尺度统计量在 ≤5000 点子采样上算）。"""
        out = np.full_like(d, np.nan, dtype=np.float64)
        valid = ~np.isnan(d)
        if np.sum(valid) < 5:
            out[valid] = d[valid]
            return out
        dd = d
        if dd.shape[0] > 5000:
            dd = dd[np.linspace(0, dd.shape[0] - 1, 5000).astype(int)]
        dv = dd[~np.isnan(dd)]
        off = np.median(dv)
        q_lo, q_hi = np.percentile(dv, [2, 98])
        s = (q_hi - q_lo) / 2.0
        if s <= 1e-9:
            s = 1.0
        out[valid] = (d[valid] - off) / s
        return out

    def window_data(self, win_start, win_end):
        """返回当前模式下 [win_start, win_end) 窗口的完整绘图数据。

        :return: dict，键：
            rows: [{'label': str, 'series': [{'kind': str, 'y': ndarray}, ...]}, ...]
                  每行对应一个显示通道，y 保留真实 ADC/%/OD/μM 数值
            n_rows: 显示行数
            time_vec:  (n_win,) 真实时间戳相对记录起点的秒数
            marker_segments: [(t_s, t_e, rgba), ...]  TDM 底色分段
            trigger_xs: [float]  触发线 x 位置（0→灯切换）
            trials: [(t_s, t_e, rgba, label_zh), ...]  行为试次（事件条）
            trial_records: 保留完整边界、marker 和全局任务 epoch 编号的记录
            jumps, lost: 丢包统计
            n_win: 窗口样本数
        """
        data_win, ch_labels, time_vec = self.get_visible_data(win_start, win_end)
        n_win = data_win.shape[1]
        raw_n_win = win_end - win_start

        series_per_row = self._series_per_row()
        n_rows = data_win.shape[0] // series_per_row

        rows = []
        for r in range(n_rows):
            base_label = None
            series_out = []
            for k in range(series_per_row):
                si = r * series_per_row + k
                base, kind = self._split_label(ch_labels[si])
                if base_label is None:
                    base_label = base
                # 子图负责各自的 Y 轴范围；数据层必须保留真实 ADC/%/OD/μM 数值。
                y = np.asarray(data_win[si], dtype=np.float64)
                series_out.append({'kind': kind, 'y': y})
            rows.append({'label': base_label, 'series': series_out})

        # 通道选择过滤：仅保留已选通道对应的行
        if self.selected_channels is not None:
            keep = [i for i, row in enumerate(rows) if row['label'] in self.selected_channels]
            rows = [rows[i] for i in keep]
            n_rows = len(rows)

        if self.current_mode == MODE_ORIG:
            fm_win = self._get_mode_marker(self.current_mode)[win_start:win_end]
            marker_segments = self._marker_segments(fm_win, time_vec, n_win)
            trigger_xs = self._trigger_positions(fm_win, time_vec, n_win)
        else:
            marker_segments = []
            trigger_xs = []
        trials = self._trials_in_window(win_start, win_end)
        trial_records = self._trial_records_in_window(win_start, win_end)
        jumps, lost = window_packet_loss(self.jump_samples, self.jump_lost, win_start, win_end)

        return {
            'rows': rows,
            'n_rows': n_rows,
            'time_vec': time_vec,
            'marker_segments': marker_segments,
            'trigger_xs': trigger_xs,
            'trials': trials,
            'trial_records': trial_records,
            'jumps': jumps,
            'lost': lost,
            'n_win': n_win,
            'raw_n_win': raw_n_win,
            'display_fs': SR_HZ if self.current_mode == MODE_ORIG else self.processed_fs,
        }

    def _marker_segments(self, fm_win, time_vec, n_win):
        """TDM 底色分段：连续的非零 marker 段。缩放太小时（>200 段）整体跳过。"""
        if n_win == 0:
            return []
        segs = []
        seg_i = 0
        while seg_i < n_win:
            m = fm_win[seg_i]
            if m < 0 or m == 0 or m not in MARKER_COLORS:
                seg_i += 1
                continue
            seg_end = seg_i + 1
            while seg_end < n_win and fm_win[seg_end] == m:
                seg_end += 1
            t_s = time_vec[seg_i] if np.isfinite(time_vec[seg_i]) else time_vec[max(0, seg_i - 1)]
            t_ei = min(seg_end - 1, n_win - 1)
            t_e = time_vec[t_ei] if np.isfinite(time_vec[t_ei]) else t_s
            if np.isfinite(t_s) and np.isfinite(t_e) and t_e > t_s:
                segs.append((t_s, t_e, MARKER_COLORS[m][:3] + (0.16,)))
                if len(segs) > 200:
                    return []  # 太密，整体跳过
            seg_i = seg_end
        return segs

    def _trigger_positions(self, fm_win, time_vec, n_win):
        """触发线 x 位置：0 → 非0 marker 的切换点。缩放太小时降采样到 ≤100。"""
        if n_win <= 1:
            return []
        xs = []
        for i in range(1, n_win):
            m_prev = int(fm_win[i - 1]) if fm_win[i - 1] >= 0 else -1
            m_cur = int(fm_win[i]) if fm_win[i] >= 0 else -1
            if m_prev != m_cur and m_cur != 0 and m_cur != -1 and m_cur in MARKER_COLORS:
                t = time_vec[i] if np.isfinite(time_vec[i]) else time_vec[i - 1]
                if np.isfinite(t):
                    xs.append(t)
        if len(xs) > 100:
            xs = [xs[i] for i in np.linspace(0, len(xs) - 1, 100).astype(int)]
        return xs

    def behavior_label_positions(self):
        """返回行为试次标签（100/110/111/112）的起始样本索引（升序）。"""
        return [s for (_, s, _) in self.behavior_trials]

    def _trials_in_window(self, win_start, win_end):
        """行为试次与当前窗口的重叠：[(t_s, t_e, rgba, label_zh), ...]"""
        trials = []
        lbl_map = BEHAVIOR_LABELS.get(self.task_type, {})
        for m, a, b in self.behavior_trials:
            if b <= win_start or a >= win_end:
                continue
            ov_a = max(a, win_start)
            ov_b = min(b, win_end)
            ta = self.raw_time[ov_a]
            tb = self.raw_time[max(ov_a, ov_b - 1)]
            if tb < ta:
                tb = ta
            rgba = BEHAVIOR_COLORS.get(m, (0.85, 0.85, 0.85, 0.9))
            label = lbl_map.get(m, (str(m), ''))[0]
            trials.append((ta, tb, rgba, label))
        return trials

    def _trial_records_in_window(self, win_start, win_end):
        """返回可绘制/点击的行为试次记录，同时保留完整试次边界。

        任务 epoch 编号按整个 stream 的时间顺序计数：MI 为 110/111，
        MA 为 111；静息/放松（112）独立编号 R1..Rn；准备不参与编号。
        """
        if self.raw_time.size < 2:
            return []
        n = self.raw_time.size
        win_start = max(0, min(int(win_start), n - 1))
        win_end = max(win_start + 1, min(int(win_end), n))
        last_dt = float(self.raw_time[-1] - self.raw_time[-2])
        recording_end = float(self.raw_time[-1] + last_dt)
        window_start_time = float(self.raw_time[win_start])
        window_end_time = (float(self.raw_time[win_end])
                           if win_end < n else recording_end)
        task_markers = {110, 111} if self.task_type == 'MI' else {111}
        rest_markers = {112}
        label_map = BEHAVIOR_LABELS.get(self.task_type, {})

        records = []
        task_idx = 0
        rest_idx = 0
        for marker, start_sample, end_sample in self.behavior_trials:
            if marker in task_markers:
                task_idx += 1
                epoch_index = str(task_idx)
            elif marker in rest_markers:
                rest_idx += 1
                epoch_index = f'R{rest_idx}'
            else:
                epoch_index = None

            full_start = float(self.raw_time[start_sample])
            full_end = (float(self.raw_time[end_sample])
                        if end_sample < n else recording_end)
            if full_end <= window_start_time or full_start >= window_end_time:
                continue
            visible_start = max(full_start, window_start_time)
            visible_end = min(full_end, window_end_time)
            base_label = label_map.get(marker, (str(marker), ''))[0]
            if marker in task_markers:
                display_label = f'{base_label} E{task_idx}'
            elif marker in rest_markers:
                display_label = f'{base_label} R{rest_idx}'
            else:
                display_label = base_label
            annotation = (self.get_epoch_annotation(epoch_index)
                          if epoch_index is not None else None)
            records.append({
                'marker': int(marker),
                'label': base_label,
                'display_label': display_label,
                'epoch_index': epoch_index,
                'annotation': annotation,
                'start_time': full_start,
                'end_time': full_end,
                'visible_start': visible_start,
                'visible_end': visible_end,
                'color': BEHAVIOR_COLORS.get(
                    marker, (0.85, 0.85, 0.85, 0.9)),
            })
        return records
