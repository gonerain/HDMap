# `src` 目录说明

## 子目录

- `src/io/`
  - 通用输入输出助手。
  - 例如 `demo_paths.py`、`pkl_frame_loader.py`。
- `src/inspect/`
  - 单帧检查、`pkl` 检查、调试入口。
- `src/detect/`
  - 从 rosbag 里筛候选帧。
- `src/vectorize/`
  - 道路矢量化、道路边缘融合。
- `src/visualize/`
  - 读取结果 `json` 并画预览图。

## 顶层文件

- 顶层同名脚本保留为兼容入口。
- 新开发优先直接放到对应子目录。
