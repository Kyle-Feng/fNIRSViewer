# fNIRSViewer

基于 PySide6 的 fNIRS（近红外脑功能成像）数据查看器：扫描 `data/raw` 下的采集数据，提供数据选择界面与交互式查看器。

## 功能

- **数据选择界面**：以树形结构列出 `data/raw` 下所有含 `data.csv` 的 stream（按任务/被试分组）
- **原始信号查看**：1000 Hz 原始 fNIRS 波形、位置标记、任务标记、行为标记，支持窗口缩放与平移
- **血氧计算**：TDM 解码 → 780/850 nm 光强 → 光密度 → MBLL（Beer-Lambert）→ HbO/HbR
- **试次与丢包检测**：自动检测试次记录与数据丢包率统计
- **标注**：支持对时间段进行标注并保存

## 目录结构

```
fNIRSViewer/
├── main.py                # 程序入口：扫描 data/raw → 选择界面 → 查看器循环
├── core/
│   └── data.py            # 数据层（加载/MBLL/试次/丢包，纯计算无 GUI 依赖）
├── processing/
│   ├── cw_pipeline.py     # 连续光强 → HbO/HbR 转换（TDM 解码、MBLL）
│   ├── config_loader.py   # 配置加载
│   ├── processing_fNIRS.py
│   └── procutil_get_extinctions.py
├── ui/
│   ├── selector.py        # 数据选择界面
│   ├── viewer.py          # 查看器主窗口
│   └── theme.py           # QSS 主题
└── data/
    └── raw/               # 采集数据（任务/被试/stream/data.csv）
```

## 数据格式与存放

```
data/raw/<任务>/<被试>/<stream>/data.csv
```

| 层级 | 命名 | 说明 |
|---|---|---|
| 任务 | `MI` | 运动想象范式（Motor Imagery） |
| 任务 | `MA` | 心算认知范式（Mental Arithmetic） |
| 被试 | `S01`、`S02`…`S0X` | 第 X 位被试 |
| stream | 自定义目录名 | 单次采集会话，内含 `data.csv` |

CSV 每行包含：fNIRS 通道（Ch1–Ch17）、位置标记（列 55）、任务标记（列 56）、时间戳（列 62）、行为标记（列 63）。设备波长必须为 780/850 nm。

同目录下可选 `config.json`（通道配置）与标注文件。

> **数据不入库**：被试目录（`S0X`）下为 GB 级大文件，已通过 `.gitignore` 排除，仓库仅保留
> `data/raw/MI`、`data/raw/MA` 目录结构（`.gitkeep` 占位）作为存放范例。
> 克隆或拉取后，请将采集数据按上述层级放入对应目录即可运行。

## 运行

```bash
pip install PySide6 numpy scipy
python main.py
```

环境要求：Python 3.8+（scipy 缺失时血氧带通滤波不可用，其余功能正常）。
