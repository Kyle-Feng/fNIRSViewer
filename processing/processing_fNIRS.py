import math
import numpy as np
from scipy.interpolate import interp1d
from eeg_positions import get_elec_coords, plot_coords
from .procutil_get_extinctions import procutil_get_extinctions
from pyedflib import highlevel
import json
from .config_loader import (
    load_config,
    get_ability,
    resolve_fnirs_columns,
    resolve_position_column,
)

boardInfo = None
# with open('boardInfo.json', 'r') as f:
#     boardInfo = json.load(f)

# lightChannelName= boardInfo['lightName']
# sensorChannelName= boardInfo['senserName']

def _aggregate_middle(sub_segment, mode):
    """对 fNIRS 子段取中间 1/3 聚合，返回逐通道向量（shape=(n_fnirs,)）。

    mode="max" 取最大值（get_fNIRS_data 语义），mode="mean" 取均值
    （get_fNIRS_data_mean 语义）。子段样本数 < 2 或中间 1/3 为空时返回 None。
    """
    if sub_segment.shape[1] < 2:
        return None
    n = sub_segment.shape[1]
    middle_start = n // 3
    middle_end = 2 * n // 3
    if middle_end <= middle_start:
        return None
    middle = sub_segment[:, middle_start:middle_end]
    if mode == "max":
        return np.max(middle, axis=1)
    return np.mean(middle, axis=1)

def get_fNIRS_data(data, config, labels_number=None, marker=None):
    """基于 config 从完整数据中提取 fNIRS 分段特征（保留原语义，仅修正列选择与基线）。

    参数：
        data: 完整 2D numpy 数组（行 = 通道/列位，如 64 行；列 = 样本）。
        config: config.json 路径或已解析 dict（传给 load_config）。
        labels_number: 行为标签行的 0-based 行号；None 时取 config 顶层 marker_idx（548pd 为 63）。
        marker: TDM marker 行的 0-based 行号（按值分 780/850 子段：1..5=780、30001..30005=850）；
            None 时取 fNIRS ability 的 marker_idx（548pd 为 56）。

    返回：
        (data_780, data_850, [])，形状均为 (n_fnirs_channels, n_segments)；
        两列一一对应同一行为段（段内按 marker 值分 780/850 子段聚合）。
    """
    # Step 0: 解析 config 与列号
    cfg = load_config(config)
    fnirs_cfg = resolve_fnirs_columns(cfg)
    fnirs_idx = fnirs_cfg["fnirs_idx"]
    n_rows = data.shape[0]

    # 校验 fNIRS 通道列号
    if not fnirs_idx:
        raise ValueError("config 中未配置 fNIRS 通道列号（ability.fNIRS.idx 为空），无法提取 fNIRS 数据")
    if max(fnirs_idx) >= n_rows:
        raise ValueError(
            f"fNIRS 通道列号越界: max(fnirs_idx)={max(fnirs_idx)}，"
            f"而 data 仅有 {n_rows} 行（0-based 0~{n_rows - 1}）"
        )

    # labels 行号：显式传入优先，否则取 config 顶层 marker_idx
    labels_row = labels_number if labels_number is not None else cfg.get("marker_idx")
    if labels_row is None:
        raise ValueError("labels_number 未指定且 config 顶层缺少 marker_idx，无法确定行为标签行")
    labels_row = int(labels_row)
    if not (0 <= labels_row < n_rows):
        raise ValueError(
            f"行为标签行号 labels_row={labels_row} 越界，data 仅有 {n_rows} 行（0-based 0~{n_rows - 1}）"
        )

    # marker 行号：显式传入优先，否则取 fNIRS ability 的 marker_idx
    marker_row = marker if marker is not None else fnirs_cfg["marker_idx"]
    marker_row = int(marker_row)
    if not (0 <= marker_row < n_rows):
        raise ValueError(
            f"marker 行号 marker_row={marker_row} 越界，data 仅有 {n_rows} 行（0-based 0~{n_rows - 1}）"
        )

    # Step 1: 基于 labels 数组计算分段的索引范围（沿用原逻辑：标签变化处切段）
    segments_indices = []
    start = 0
    labels = data[labels_row]
    for i in range(0, len(labels) - 1):
        if labels[i] != labels[start]:  # 标签发生变化，记录一个段的结束
            segments_indices.append((start, i))
            start = i  # 更新起始索引
    # 最后一个段
    segments_indices.append((start, len(labels)))

    # Step 2: 提取 fNIRS 行子集（只读视图，不修改 data）。
    # 真实 TDM 设备（如 548pd）在一个行为段内 780/850 交替点灯（marker 行 56：
    # 1..5 = 780nm 光源组、30001..30005 = 850nm 光源组），故按行为段切段后，
    # 段内再按 marker 值分 780 / 850 子段，各自去基线并取中间 1/3 的最大值。
    # 行为段需同时包含 780 与 850 子段才产生一列，保证两侧列一一对齐。
    fnirs = data[fnirs_idx, :]
    marker_series = data[marker_row]
    averages_780 = []
    averages_850 = []

    for start_idx, end_idx in segments_indices:
        # 跳过样本数少于 2 的段（行为标签脉冲形态下会出现长度 1 的段）
        if end_idx - start_idx < 2:
            continue
        # 基线：该行为段前 5 个样本的逐通道均值（仅作用于 fNIRS 通道，形状匹配）
        if start_idx < 5:
            base_slice = fnirs[:, 0:start_idx]
            if base_slice.shape[1] > 0:
                baseline = base_slice.mean(axis=1, keepdims=True)
            else:
                baseline = np.zeros((fnirs.shape[0], 1))
        else:
            baseline = fnirs[:, start_idx - 5:start_idx].mean(axis=1, keepdims=True)
        segment = fnirs[:, start_idx:end_idx] - baseline

        # 段内按 marker 值分 780（1..5）/ 850（30001..30005）子段
        mk = marker_series[start_idx:end_idx]
        v_780 = _aggregate_middle(segment[:, (mk >= 1) & (mk <= 5)], "max")
        v_850 = _aggregate_middle(segment[:, (mk >= 30001) & (mk <= 30005)], "max")
        if v_780 is None or v_850 is None:
            continue  # 该行为段缺 780 或 850 子段 → 不成对，跳过
        averages_780.append(v_780)
        averages_850.append(v_850)

    if not averages_780:
        empty = np.zeros((len(fnirs_idx), 0))
        return empty, empty.copy(), []
    data_780 = np.array(averages_780).T
    data_850 = np.array(averages_850).T
    return data_780, data_850, []

def get_fNIRS_data_mean(data, config, labels=None):
    """基于 config 从完整数据中提取 fNIRS 分段均值特征（与 get_fNIRS_data 同结构，聚合用 np.mean）。

    参数：
        data: 完整 2D numpy 数组（行 = 通道/列位，如 64 行；列 = 样本）。
        config: config.json 路径或已解析 dict（传给 load_config）。
        labels: 行为标签行的 0-based 行号；None 时取 config 顶层 marker_idx（548pd 为 63）。

    返回：
        (data_780, data_850)，均为 list，每个含 n_segments 个子列表（每段一个，
        780/850 一一对应同一行为段），每子列表含 len(fnirs_idx) 个通道均值
        （顺序与 fnirs_idx 一致）。
    """
    # Step 0: 解析 config 与列号
    cfg = load_config(config)
    fnirs_cfg = resolve_fnirs_columns(cfg)
    fnirs_idx = fnirs_cfg["fnirs_idx"]
    n_rows = data.shape[0]

    # 校验 fNIRS 通道列号
    if not fnirs_idx:
        raise ValueError("config 中未配置 fNIRS 通道列号（ability.fNIRS.idx 为空），无法提取 fNIRS 数据")
    if max(fnirs_idx) >= n_rows:
        raise ValueError(
            f"fNIRS 通道列号越界: max(fnirs_idx)={max(fnirs_idx)}，"
            f"而 data 仅有 {n_rows} 行（0-based 0~{n_rows - 1}）"
        )

    # labels 行号：显式传入优先，否则取 config 顶层 marker_idx
    labels_row = labels if labels is not None else cfg.get("marker_idx")
    if labels_row is None:
        raise ValueError("labels 未指定且 config 顶层缺少 marker_idx，无法确定行为标签行")
    labels_row = int(labels_row)
    if not (0 <= labels_row < n_rows):
        raise ValueError(
            f"行为标签行号 labels_row={labels_row} 越界，data 仅有 {n_rows} 行（0-based 0~{n_rows - 1}）"
        )

    # marker 行号：取 fNIRS ability 的 marker_idx
    marker_row = int(fnirs_cfg["marker_idx"])
    if not (0 <= marker_row < n_rows):
        raise ValueError(
            f"marker 行号 marker_row={marker_row} 越界，data 仅有 {n_rows} 行（0-based 0~{n_rows - 1}）"
        )

    # Step 1: 基于 labels 数组计算分段的索引范围（沿用原逻辑：标签变化处切段）
    segments_indices = []
    start = 0
    label_series = data[labels_row]
    for i in range(0, len(label_series) - 1):
        if label_series[i] != label_series[start]:  # 标签发生变化，记录一个段的结束
            segments_indices.append((start, i))
            start = i  # 更新起始索引
    # 最后一个段
    segments_indices.append((start, len(label_series)))

    # Step 2: 提取 fNIRS 行子集（只读视图，不修改 data）。
    # 与 get_fNIRS_data 相同的 TDM 拆分语义：行为段内按 marker 值分 780/850
    # 子段，各自取中间 1/3 的均值；行为段需同时包含两侧子段才产生一列。
    fnirs = data[fnirs_idx, :]
    marker_series = data[marker_row]
    averages_780 = []
    averages_850 = []

    for start_idx, end_idx in segments_indices:
        # 跳过样本数少于 2 的段（同 get_fNIRS_data，避免中间 1/3 空切片）
        if end_idx - start_idx < 2:
            continue
        segment = fnirs[:, start_idx:end_idx]

        # 段内按 marker 值分 780（1..5）/ 850（30001..30005）子段
        mk = marker_series[start_idx:end_idx]
        v_780 = _aggregate_middle(segment[:, (mk >= 1) & (mk <= 5)], "mean")
        v_850 = _aggregate_middle(segment[:, (mk >= 30001) & (mk <= 30005)], "mean")
        if v_780 is None or v_850 is None:
            continue  # 该行为段缺 780 或 850 子段 → 不成对，跳过
        averages_780.append(v_780)
        averages_850.append(v_850)

    if not averages_780:
        return [], []
    data_780 = np.array(averages_780).T
    data_850 = np.array(averages_850).T
    # 返回格式保持原函数不变（np.array(x, axis=1) 的 stack 语义；
    # numpy>=2.1 移除了 np.array 的 axis 参数，故用等价的 np.stack 实现）
    return np.stack(data_780, axis=1).tolist(), np.stack(data_850, axis=1).tolist()

def process_origin_to_fNIRS(wave1, wave2, waveLength):
    wave2 , wave1 = get_min_length_origin_data(wave1, wave2)
    wave1 = np.array(wave1)
    wave2 = np.array(wave2)
    wave = np.concatenate((wave1, wave2), axis=1)
    data = dict({
        "x":wave,
        'wavelengths': waveLength,
        "clab": []
    })
    fNIRS_signal = proc_BeerLambert(data)
    channel_length = wave1.shape[1]
    return fNIRS_signal["x"][:, :channel_length].T, fNIRS_signal["x"][:,channel_length:].T
   
def proc_BeerLambert(dat, **kwargs):
    """
    PROC_BEERLAMBERT - 使用Beer-Lambert定律分析NIRS数据。
    根据总光路长度计算相对浓度。

    参数：
        dat: 包含NIRS数据的字典。需要有'x'字段，存储分割的NIRS数据。
        kwargs: 可选参数，以键值对形式传递：
            'Citation' - 消光系数的文献编号（默认值1）。
            'Epsilon' - 自定义消光系数（覆盖文献编号）。
            'Opdist' - 探头（光源-探测器）间距，单位为cm（默认值3）。
            'Ival' - 用于LB变换的基线区间（'all'或[start, end]，默认值'all'）。
            'DPF' - 差分路径长度因子（默认值[5.98, 7.15]）。
            'Verbose' - 输出详细信息的级别（默认值0）。

    返回值：
        dat: 更新后的字典，包含氧合血红蛋白和脱氧血红蛋白浓度（单位：mmol/L）。
    """
    # 默认参数设置
    props = {
        'Citation': 1,  # 消光系数的默认文献编号
        'Opdist': 3,    # 探头距离的默认值（cm）
        'Ival': 'all',  # 基线区间的默认值
        'DPF': [5.98, 7.15],  # 差分路径长度因子的默认值
        'Epsilon': None,      # 消光系数
        'Verbose': 0          # 输出详细信息的级别
    }

    # 更新默认参数
    props.update(kwargs)

    # 检查输入数据是否包含'x'字段
    if 'x' not in dat:
        raise ValueError("dat必须包含字段'x'。")

    # 设置基线区间
    if props['Ival'] == 'all':
        props['Ival'] = [0, dat['x'].shape[0]]  # 如果为'all'，则使用整个数据范围
    s1 = dat['x'].shape[0]  # 时间点数量
    s2 = dat['x'].shape[1] // 2  # 每个波长的通道数量

    # 将数据分为低波长和高波长部分
    wl1 = dat['x'][:, :s2]
    wl2 = dat['x'][:, s2:]
    # 获取或使用提供的消光系数
    if props['Epsilon'] is None:
        # 如果没有提供波长信息，则报错
        if 'wavelengths' not in dat:
            raise ValueError("dat字典中必须提供'wavelengths'字段。")
        # 根据文献编号获取消光系数
        ext, citation = procutil_get_extinctions(dat['wavelengths'], props['Citation'])
        if props['Verbose']:
            print(f"使用的文献编号: {citation}")
        epsilon = ext[:, :2] / 1000  # 将单位转换为mmol/L
    else:
        # 使用用户提供的消光系数
        epsilon = np.array(props['Epsilon'])

    # 确保较高波长的消光系数排在顶部
    max_wavelength_idx = np.argmax(dat['wavelengths'])
    if max_wavelength_idx == 1:  # 如果较高波长排在底部
        epsilon = np.flipud(epsilon)  # 调整顺序
        if props['Verbose']:
            print("已调整Epsilon矩阵，使较高波长的消光系数排在顶部。")
    eps = np.finfo(float).eps  # 浮点数的最小正值
    # 计算基线均值，用于归一化
    mean_wl2 = np.mean(wl2[props['Ival'][0]:props['Ival'][1], :], axis=0)
    mean_wl1 = np.mean(wl1[props['Ival'][0]:props['Ival'][1], :], axis=0)
        # 避免分母为零
    mean_wl2 = np.clip(mean_wl2, eps, None)
    mean_wl1 = np.clip(mean_wl1, eps, None)

    # 计算衰减值，避免对数中出现非正值
    Att_highWL = np.real(-np.log10(np.clip(wl2 / mean_wl2, eps, None)))
    Att_lowWL = np.real(-np.log10(np.clip(wl1 / mean_wl1, eps, None)))

    # 准备吸收矩阵，用于线性方程求解
    A = np.zeros((s1 * s2, 2))
    A[:, 0] = Att_highWL.ravel()
    A[:, 1] = Att_lowWL.ravel()

    # 根据DPF和探头距离对消光系数进行缩放
    e2 = epsilon * np.array(props['DPF'])[:, None] * props['Opdist']

    # 使用矩阵求解计算浓度
    c = np.linalg.inv(e2) @ A.T

    # 更新数据字段'x'，包含氧合和脱氧血红蛋白的浓度
    dat['x'] = np.hstack([
        c[0, :].reshape(s1, s2),
        c[1, :].reshape(s1, s2)
    ])
    lowWL= [label[0].replace('lowWL', 'oxy') for label in dat['clab'] if 'lowWL' in label[0]]
    highWL = [label[0].replace('highWL', 'deoxy') for label in dat['clab'] if 'highWL' in label[0]]
    dat['clab'] = np.hstack([
        lowWL, 
        highWL
    ])
    # # 更新数据的元信息
    # dat['signal'] = 'NIRS (oxy, deoxy)'  # 信号类型
    # dat['yUnit'] = 'mmol/L'  # 浓度单位

    return dat

def decimal_to_16bit_array(n):
    if not 0 <= n <= 0xFFFF:
        raise ValueError("数值超出16位二进制范围（0-65535）")
    # 转换为16位二进制字符串，去掉前缀，左侧补零
    binary_str = bin(n)[2:].zfill(16)
    # 转为整数数组
    return [int(bit) for bit in binary_str]

def max_every_10_points(arr):
    result = []  # 存储结果的列表
    n = len(arr)
    i = 0
    while i < n:
        # 取当前区块：从i开始到min(i+8, n)
        block = arr[i:i+10]
        # 求区块最大值并添加到结果
        result.append(max(block))
        i += 10  # 移动到下一个区块起始位置
    return result

def get_channel_data_by_marker(marker, config=None):
    global boardInfo
    if config is not None:
        # config 模式：从 config.json 推导 light groups，不读 boardInfo.json
        cfg = load_config(config)
        fnirs_cfg = resolve_fnirs_columns(cfg)
        light_name = fnirs_cfg.get('light_name', [])
        light_sensor_group = fnirs_cfg.get('light_sensor_group', {})
        sensor_name = get_ability(cfg, 'fNIRS').get('name', [])

        # 780/850 group 编号：780 的 group = marker，850 的 group = marker - 30000
        if marker > 29999:
            light = '850'
            group = marker - 30000
        else:
            light = '780'
            group = marker

        group_key = str(group)
        sensor_list = light_sensor_group.get(group_key)
        if not sensor_list:
            raise ValueError(
                f"marker={marker}（波长 {light}，light_sensor_group 键 {group_key!r}）"
                f"在 config.json 中不存在或为空，请检查 light_sensor_group 配置"
            )
        sensor_list = [int(j) for j in sensor_list]
        light_idx = group - 1  # 光源号（1-based）转 0-based 索引
        if not (0 <= light_idx < len(light_name)):
            raise ValueError(
                f"marker={marker}（波长 {light}，光源号 {group}）超出 config.json "
                f"light_name 的范围（共 {len(light_name)} 个光源，0-based 0~{len(light_name) - 1}）"
            )
        names = [f"{light_name[light_idx]}-{sensor_name[j - 1]}" for j in sensor_list]
        fNIRS_indexs = [[sensor_list, names]]
        return fNIRS_indexs, light

    if boardInfo == None:
        with open('boardInfo.json', 'r') as f:
            boardInfo = json.load(f)
    fNIRS_indexs =  []
    light= '780'
 
    if marker > 29999:
        light= '850'
        light_arr = boardInfo['light_flash_groups'][str(marker-30000 )]
    else:
        light_arr = boardInfo['light_flash_groups'][str(marker )]
    light_arr = [item[1] for item in light_arr]
    for k in range(len(light_arr)):
        i = light_arr[k] 
        fNIRS_indexs.append([boardInfo['lightIndex'][str((i+1 ))],
                             [boardInfo['lightName'][i] +'-' + boardInfo['senserName'][j-1] for j in boardInfo['lightIndex'][str((i+1))]] 
            ])
    return fNIRS_indexs, light

def analyze_packet_loss(data, position_col_0based=None, config=None):
    global boardInfo
    if position_col_0based is None:
        if config is not None:
            try:
                # config 模式：优先用 config.position_idx（0-based，校验后回退 55）
                cfg = load_config(config)
                position_col_0based = resolve_position_column(cfg, n_rows=data.shape[0])
            except Exception:
                # config 解析失败（路径无效/JSON 非法等）回退 boardInfo 逻辑
                position_col_0based = None
        if position_col_0based is None:
            if boardInfo is None:
                with open('boardInfo.json', 'r') as f:
                    boardInfo = json.load(f)
            position_idx_1based = boardInfo.get('positionIndex', 55)
            position_col_0based = position_idx_1based - 1
    position_col = int(position_col_0based)
    position_col_1based_val = position_col + 1
    if position_col < 0 or position_col >= data.shape[0]:
        return {
            'position_col_1based': position_col_1based_val,
            'position_col_0based': position_col,
            'total_expected': 0,
            'total_received': 0,
            'total_lost': 0,
            'loss_rate_percent': 0.0,
            'missing_ranges': [],
            'missing_indices': [],
            'error': f'position_col_0based {position_col} (1-based {position_col_1based_val}) out of bounds, data has {data.shape[0]} rows (0-based 0-{data.shape[0]-1})'
        }
    position_data = data[position_col]
    if len(position_data) < 2:
        return {
            'position_col_1based': position_col_1based_val,
            'position_col_0based': position_col,
            'total_expected': len(position_data),
            'total_received': len(position_data),
            'total_lost': 0,
            'loss_rate_percent': 0.0,
            'missing_ranges': [],
            'missing_indices': [],
            'first_seq': int(position_data[0]) if len(position_data) > 0 else None,
            'last_seq': int(position_data[-1]) if len(position_data) > 0 else None,
        }
    position_int = position_data.astype(np.int64)
    first_seq = int(position_int[0])
    last_seq = int(position_int[-1])
    total_expected = last_seq - first_seq + 1
    total_received = len(position_int)
    total_lost = total_expected - total_received
    loss_rate = (total_lost / total_expected * 100) if total_expected > 0 else 0.0

    diffs = np.diff(position_int)
    loss_locations = np.where(diffs > 1)[0]
    missing_ranges = []
    missing_indices = []
    for loc in loss_locations:
        gap_start = int(position_int[loc]) + 1
        gap_end = int(position_int[loc + 1]) - 1
        missing_count = gap_end - gap_start + 1
        missing_ranges.append({
            'gap_before_sample_idx': int(loc),
            'gap_after_sample_idx': int(loc + 1),
            'missing_seq_start': gap_start,
            'missing_seq_end': gap_end,
            'missing_count': missing_count
        })
        missing_indices.extend(list(range(gap_start, gap_end + 1)))

    return {
        'position_col_1based': position_col_1based_val,
        'position_col_0based': position_col,
        'first_seq': first_seq,
        'last_seq': last_seq,
        'total_expected': total_expected,
        'total_received': total_received,
        'total_lost': total_lost,
        'loss_rate_percent': round(loss_rate, 4),
        'missing_ranges': missing_ranges,
        'missing_indices': missing_indices,
        'duplicate_count': int(total_received - len(np.unique(position_int))),
        'non_monotonic_count': int(np.sum(diffs < 0)),
    }


def print_packet_loss_report(report):
    if 'error' in report:
        print(f"[丢包检测] 错误: {report['error']}")
        return
    print("=" * 60)
    print("[丢包检测报告] position列 (1-based):", report.get('position_col_1based'))
    print("-" * 60)
    print(f"  起始序列号:       {report['first_seq']}")
    print(f"  结束序列号:       {report['last_seq']}")
    print(f"  期望接收包数:     {report['total_expected']}")
    print(f"  实际接收包数:     {report['total_received']}")
    print(f"  丢包数量:         {report['total_lost']}")
    print(f"  丢包率:           {report['loss_rate_percent']}%")
    print(f"  重复序列号数量:   {report.get('duplicate_count', 0)}")
    print(f"  非递增跳变次数:   {report.get('non_monotonic_count', 0)}")
    if report['missing_ranges']:
        print("-" * 60)
        print(f"  丢失区间详情 (共{len(report['missing_ranges'])}处):")
        for i, rng in enumerate(report['missing_ranges'][:20], 1):
            print(f"    [{i}] 样本#{rng['gap_before_sample_idx']}->#{rng['gap_after_sample_idx']} "
                  f"| 缺失序列号: {rng['missing_seq_start']}-{rng['missing_seq_end']} "
                  f"| 丢{rng['missing_count']}个")
        if len(report['missing_ranges']) > 20:
            print(f"    ... 其余 {len(report['missing_ranges']) - 20} 处省略")
        if report['missing_indices']:
            print(f"  全部缺失的序列号列表 (前50个): {report['missing_indices'][:50]}")
            if len(report['missing_indices']) > 50:
                print(f"    ... 共{len(report['missing_indices'])}个缺失序号")
    else:
        print("-" * 60)
        tl = report.get('total_lost', 0)
        dup = report.get('duplicate_count', 0)
        if tl == 0 and dup == 0:
            print("  [OK] 无丢包，数据连续完整")
        elif tl < 0 or dup > 0:
            print(f"  [WARN] 此列不是 position 序号列！total_lost={tl}, duplicates={dup}")
        else:
            print(f"  [OK] 无区间跳变丢包，但 total_lost={tl}（需核实）")
    print("=" * 60)


def find_contiguous_segments(data, index, config=None):
    global boardInfo
    all_data = dict({})
    all_data['780'] = dict({})
    all_data['850'] = dict({})

    # config 模式：从 config.json 解析列配置；boardInfo 仅在 config 为 None 时加载
    cfg = None
    fnirs_cfg = None
    if config is not None:
        cfg = load_config(config)
        fnirs_cfg = resolve_fnirs_columns(cfg)
        if not fnirs_cfg.get('fnirs_idx'):
            raise ValueError(
                "config 中未配置 fNIRS 通道列号（ability.fNIRS.idx 为空），无法确定 fNIRS 起始列"
            )
    elif boardInfo == None:
        with open('boardInfo.json', 'r') as f:
            boardInfo = json.load(f)

    # index 参数是 marker 列的 0-based 索引，后面 for 循环会覆盖 index，
    # 所以这里先保存一份为 _marker_idx
    _marker_idx = int(index)
    # position 列：config 模式取 config.position_idx（0-based，校验后回退 55）；
    # 否则按 boardInfo.positionIndex (1-based)
    if cfg is not None:
        _position_idx = resolve_position_column(cfg, n_rows=data.shape[0])
        _position_idx_1based = _position_idx + 1
        _position_src = f"config.position_idx={cfg.get('position_idx')}"
    else:
        _position_idx_1based = boardInfo.get('positionIndex', 55)
        _position_idx = _position_idx_1based
        _position_src = f"boardInfo.positionIndex={_position_idx_1based}"


    print("\n" + "#" * 60)
    print(f"### [step 1] 按 {_position_src} 排序：positionIndex={_position_idx_1based} (1-based, 0-based={_position_idx}) ###")
    print("#" * 60)

    data_sorted = data
    _sort_info = {
        'did_sort': False,
        'did_dedup': False,
        'before_samples': data.shape[1],
        'after_samples': data.shape[1],
        'removed_duplicates': 0,
        'sort_swaps': 0,
    }

    if _position_idx >= 0 and _position_idx < data.shape[0] and data.shape[1] >= 2:
        pos_vals = data[_position_idx, :]
        pos_int = np.round(pos_vals).astype(np.int64)
        _diffs = np.diff(pos_int)
        _non_monotonic = np.sum(_diffs < 0)
        _duplicates = np.sum(_diffs == 0)
        _unsorted = _non_monotonic > 0

        if _unsorted or _duplicates > 0:
            print(f"  排序前: 非递增跳变={_non_monotonic}处, 重复序列号={_duplicates}个")

            order = np.argsort(pos_int, kind='stable')
            data_sorted = data[:, order]
            _sort_info['did_sort'] = True
            _sort_info['sort_swaps'] = int(_non_monotonic)

            pos_int_sorted = pos_int[order]
            _diffs2 = np.diff(pos_int_sorted)
            _duplicates_after = np.sum(_diffs2 == 0)
            if _duplicates_after > 0:
                _, keep = np.unique(pos_int_sorted, return_index=True)
                keep = np.sort(keep)
                data_sorted = data_sorted[:, keep]
                _sort_info['did_dedup'] = True
                _sort_info['removed_duplicates'] = int(len(pos_int_sorted) - len(keep))
                print(f"  去重前 {len(pos_int_sorted)} 样本，去重后 {len(keep)} 样本，移除重复 {_sort_info['removed_duplicates']} 个")
            if _non_monotonic > 0:
                print(f"  按 position_index 排序完成 (stable argsort)")
        else:
            print("  position_index 已经单调递增，无需排序")
        _sort_info['after_samples'] = data_sorted.shape[1]
    else:
        print(f"  [WARN] position 列索引 {_position_idx} 超出范围或样本不足，跳过排序")

    print(f"  样本数: 排序前 {_sort_info['before_samples']} -> 排序/去重后 {_sort_info['after_samples']}")

    # 之后全部使用 data_sorted 替换 data
    data = data_sorted

    print("\n" + "#" * 60)
    print(f"### [step 2] 排序后丢包检测：{_position_src} = 第{_position_idx_1based}列 (1-based, 0-based={_position_idx}) ###")
    print("#" * 60)
    loss_report = analyze_packet_loss(data, _position_idx, config=cfg)
    print_packet_loss_report(loss_report)

    find_contiguous_segments._last_loss_report = {
        'positionIndex': loss_report,
        'sort_info': _sort_info,
    }

    if cfg is not None:
        # config 模式：fNIRS 起始列 = min(fnirs_idx)（0-based）；通道名用 channel_pair 预初始化
        chunnels_start = min(fnirs_cfg['fnirs_idx'])
        for name in (fnirs_cfg.get('channel_pair') or []):
            all_data['780'][name] = []
            all_data['850'][name] = []
    else:
        for name in boardInfo['fNIRSChannels']:
            all_data['780'][name] = []
            all_data['850'][name] = []
        chunnels_start = 1
        if data.shape[0] == 64:
            chunnels_start = 33
        fNIRS_ch = boardInfo.get('fNIRS_channel')
        if isinstance(fNIRS_ch, list) and len(fNIRS_ch) > 0:
            chunnels_start = int(fNIRS_ch[0])

    col21 = data[_marker_idx].tolist()
    if not col21:
        return all_data, []
    segments_indices = []
    start = 0
    for i in range(1, len(col21)):
        if col21[i] != col21[i-1]:
            segments_indices.append([start, i, col21[i]])
            start = i
    segments_indices.append([start, len(col21)-1, col21[ len(col21)-1]])
    segments = []
    markers = []
    triggers = []

    for start_idx, end_idx, marker in segments_indices:
        segment = data[:, start_idx:end_idx]
        if segment.shape[1] > 5:

            segment_samples = segment.shape[1]
            middle_start = segment_samples // 3
            middle_end = 2 * segment_samples // 3
            triggers.append(np.max(segment[-1]))
            middle_segment = segment[:, middle_start:middle_end]
            average = np.average(middle_segment, axis=1, keepdims=True)
            markers.append(average[_marker_idx])
            segments.append(average)
            
    triggers_array = max_every_10_points(triggers)
    for _seg_idx in range(len(segments)):
        segment = segments[_seg_idx]
        marker_val = int(markers[_seg_idx][0])
        segment_samples = segment.shape[1]
        if marker_val == 0:
            continue
        fNIRSIndexs, light = get_channel_data_by_marker(marker_val, config=cfg)
        c_current = all_data[light]
        for _id in range(len(fNIRSIndexs)):
            fNIRSIndex= fNIRSIndexs[_id]
            for id in range(len(fNIRSIndex[0])):
                sensor = fNIRSIndex[0][id] - 1 + chunnels_start
                c_current.setdefault(fNIRSIndex[1][id], []).append(segment[sensor][0])
        all_data[light] = c_current
    return all_data, triggers_array

def get_min_length_origin_data(data_780, data_850):
    try:
        min_length = min(len(sublist) for sublist in data_780)  # 输出: 2
        min_length_850 = min(len(sublist) for sublist in data_850)  # 输出: 2
        min_length = min(min_length, min_length_850)
        min_length_850 = min_length
        # 2. 截取所有子列表至最小长度
        trimmed_data = [sublist[:min_length] for sublist in data_780]
        # 3. 转换为NumPy数组
        np_array_780 = np.array(trimmed_data)
        # 2. 截取所有子列表至最小长度
        trimmed_data_850 = [sublist[:min_length_850] for sublist in data_850]
        # 3. 转换为NumPy数组
        np_array_850 = np.array(trimmed_data_850)
        return np_array_850.tolist(), np_array_780.tolist()
    except Exception as e :
        return [], []
def get_position_by_channel(channels):
    coords = get_elec_coords(
        system="1005",
        dim="3d",
    )
    channelNames = coords['label'].to_list()
    index2 = [channelNames.index(name) for name in channels]
    coords = coords[coords['label'].isin(channels)]
    return {
        "x": coords['x'][index2].to_list(),
        "y": coords['y'][index2].to_list(),
        "z": coords['z'][index2].to_list()
    }

def get_processing_from_origin_data_48_ch(data, data_marker):
    segments, triggers = find_contiguous_segments(data, data_marker)
    channels = list(segments["780"].keys())
    data_780 = []
    data_850 = []
    for channel in channels:
        data_780.append(segments["780"][channel])
        data_850.append(segments["850"][channel])
    data_780, data_850 = get_min_length_origin_data(data_780, data_850)
    return segments["780"].keys(), data_780, data_850, triggers

def get_processiing_from_origin_data_48_ch_mean(data, data_marker):
    segments, triggers = find_contiguous_segments(data, data_marker)
    channels = list(segments["780"].keys())
    data_780 = []
    data_850 = []
    for channel in channels:
        data_780.append(segments["780"][channel])
        data_850.append(segments["850"][channel])
    data_780, data_850 = get_min_length_origin_data(data_780, data_850) 
    return segments["780"].keys(), np.mean(data_780, axis=1).tolist(), np.mean(data_850, axis=1).tolist()

def get_position_by_light_sensor_position(lightChannelName, sensorChannelName, channels):
    lightChannelPosition = get_position_by_channel(lightChannelName)
    sensorChannelPosition = get_position_by_channel(sensorChannelName)
    positions = []
    
    for channel in channels:
        lightName, sensorName = channel.split('-')[0],  channel.split('-')[1]
        lightChannelIndex = lightChannelName.index(lightName)
        sensorChannelIndex = sensorChannelName.index(sensorName)
        positions.append({
            "x": (lightChannelPosition['x'][lightChannelIndex] + sensorChannelPosition['x'][sensorChannelIndex])/2 ,
            "y": (lightChannelPosition['y'][lightChannelIndex] + sensorChannelPosition['y'][sensorChannelIndex]) /2,
            "z": (lightChannelPosition['z'][lightChannelIndex] + sensorChannelPosition['z'][sensorChannelIndex]) /2,
            's_x': sensorChannelPosition['x'][sensorChannelIndex],
            's_y': sensorChannelPosition['y'][sensorChannelIndex],
            's_z': sensorChannelPosition['z'][sensorChannelIndex],
            'l_x': lightChannelPosition['x'][lightChannelIndex],
            'l_y': lightChannelPosition['y'][lightChannelIndex],
            'l_z': lightChannelPosition['z'][lightChannelIndex] 
        })
    return positions


# # EEG 数据存储
# class EEGSAVEDATA(object):
#     def __init__(self):
#         super(EEGSAVEDATA, self).__init__()
#         print('inint')
#         self.name = 'name'
    
#     def saveFile(self,fileName, data, channels, sampleRate, otherInfo):
#         # try:
#             """
#             A convenience function to create an EDF header (a dictionary) that
#             can be used by pyedflib to update the main header of the EDF

#             Parameters
#             ----------
#             technician : str, optional
#                 name of the technician. The default is ''.
#             recording_additional : str, optional
#                 comments etc. The default is ''.
#             patientname : str, optional
#                 the name of the patient. The default is ''.
#             patient_additional : TYPE, optional
#                 more info about the patient. The default is ''.
#             patientcode : str, optional
#                 alphanumeric code. The default is ''.
#             equipment : str, optional
#                 which system was used. The default is ''.
#             admincode : str, optional
#                 code of the admin. The default is ''.
#             gender : str, optional
#                 gender of patient. The default is ''.
#             startdate : datetime.datetime, optional
#                 startdate of recording. The default is None.
#             birthdate : str/datetime.datetime, optional
#                 date of birth of the patient. The default is ''.
#             """
#             data = data.T
#             data = np.ascontiguousarray(np.array(data))
#             signals = []
#             for channel in range(len(data)):
#                 DataFilter.detrend(data[channel], DetrendOperations.NO_DETREND.value)
#                 signals.append(data[channel]/1)
#             signals = np.ascontiguousarray(np.array(signals))
#             signalHeaders = highlevel.make_signal_headers(
#                 list_of_labels=channels,
#                 sample_frequency=sampleRate, 
#                 sample_rate=sampleRate,
#                 physical_max=187500,
#                 physical_min=-187500,
#                 digital_max= 187500,
#                 digital_min= -187500
#             )
#             technician = ''
#             recording_additional = ''
#             patientname = ''
#             patient_additional = ''
#             patientcode = ''
#             equipment = ''
#             admincode = ''
#             gender = ''
#             # birthdate= datetime.datetime(1900, 1, 1).strftime('%d %b %Y')
#             keys = list(otherInfo.keys())
#             if 'technician' in keys:
#                 technician = otherInfo["technician"]
#             if 'recording_additional' in keys:
#                 recording_additional = otherInfo['recording_additional']
#             if 'patientname' in keys:
#                 patientname = otherInfo['patientname']
#             if 'patient_additional' in keys:
#                 patient_additional = otherInfo['patient_additional']
#             if 'patientcode' in keys:
#                 patientcode = otherInfo['patientcode']
#             if 'equipment' in keys:
#                 equipment = otherInfo['equipment']
#             if 'admincode' in keys:
#                 admincode=otherInfo['admincode']
#             # if 'birthdate' in keys:
#             #     birthdate = otherInfo['birthdate']
#             header = highlevel.make_header(technician=technician, 
#                                         recording_additional=recording_additional,
#                                         patientname=patientname,
#                                         patient_additional=patient_additional, 
#                                         patientcode=patientcode, 
#                                         equipment=equipment, 
#                                         admincode=admincode,
#                                         gender=gender)
#             print(signals.shape, len(signalHeaders))
#             res = highlevel.write_edf(fileName, signals=signals, signal_headers=signalHeaders, digital=False,file_type=3, header=header)
#         # except Exception as e :
#         #     print(e)

