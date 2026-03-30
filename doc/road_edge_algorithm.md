# Standalone Road Edge Algorithm

## Goal

保留 road edge 的主体算法，但不再考虑把它硬塞进 [`make_vector.py`](/mnt/ning_602/work/HDMap/make_vector.py)。

新的思路很直接：

1. 先把全部语义点云读完。
2. 从每一帧里提取 `road_class` 点。
3. 再按 `pose + sliding window` 做局部 road edge 搜索。
4. 把每个局部窗口的 left/right edge 变回 world。
5. 跨窗口 merge 成全局左右边界。
6. 单独输出结果，不和 `vector.pcd`、ROS 发布、`process()` 绑定。

这样保留了原来算法里真正有价值的部分：局部坐标系、窗口叠帧、BEV、左右边界搜索、全局融合；去掉的是 `make_vector.py` 那套流程约束。

## Data Preparation

输入仍然是：

- 逐帧 pickle 语义点云
- `pose.csv`

但执行顺序改成：

```text
road_frames = []
for each frame in pickle:
    roads = frame[frame[:, 3] == road_class][:, :3]
    road_frames.append(roads)
```

也就是说，先把所有 frame 对应的 road 点提出来放进内存，再开始后续处理。

这样做的好处：

- 主流程清楚，不需要挂进 `process()`
- 调试方便，可以随时回放任意窗口
- 算法和 ROS/可视化/导出逻辑解耦
- 后面要改窗口策略时，不用再碰主读取流程

## Main Pipeline

### Step 1: Sliding Window Accumulation

虽然先把所有点云读完，但 road edge 本身仍然按局部窗口处理，不建议直接把全局 road 点一次性铺平后硬找边界。

```text
for frame_idx in range(len(road_frames)):
    window.append(road_frames[frame_idx])
    if len(window) < window_size:
        continue
    if frame_idx % step != 0:
        continue
    roads_world = np.vstack(window)
```

原因：

- 单帧 road segmentation 仍然容易断
- 全局一次性处理会把局部方向性冲淡
- 以 pose 为中心的局部窗口更容易定义 forward / lateral

### Step 2: Transform To Current Local Frame

对当前窗口，使用当前 `pose` 把窗口点云变到当前 frame 的 local 坐标系：

```text
roads_local = pcd_trans(roads_world, p, rotation, True)
```

其中：

- `p` 是当前位姿平移
- `rotation` 是当前位姿四元数
- `inverse=True` 表示 world -> current local

这里仍然保留 pose 驱动的 local/world 变换，因为这是主体方案里最有用的一层约束。

### Step 3: Box Crop In Local Frame

在 local frame 下做局部裁剪，只保留当前车附近稳定区域：

```text
mask = (
    (roads_local[:, 1] >= 0.0) &
    (roads_local[:, 1] <= road_box_forward) &
    (np.abs(roads_local[:, 0]) <= road_box_lateral) &
    (roads_local[:, 2] >= road_box_z_min) &
    (roads_local[:, 2] <= road_box_z_max)
)
roads_local = roads_local[mask]
```

建议规则：

- forward: `y in [0, y_max]`
- lateral: `x in [-x_limit, x_limit]`
- z: `z in [z_min, z_max]`

### Step 4: BEV Projection And Rasterization

局部 road 点投到 BEV：

- `points_xy = roads_local[:, :2]`

再栅格化：

- `col = floor((x - xmin) / res)`
- `row = floor((y - ymin) / res)`
- `count[row, col] += 1`
- `mask[row, col] = count >= min_points_per_cell`

建议参数：

- `res = 0.15 ~ 0.2`
- `min_points_per_cell = 2 ~ 4`

然后做轻量清理：

- binary closing
- remove small component

输出：

- `road_mask`
- `origin_xy`
- `counts`

### Step 5: Direction Estimation

方向仍然优先使用当前 pose heading，再用局部 road 分布辅助修正。

推荐优先级：

1. 默认使用当前 pose heading。
2. 如有必要，再用 road local points 的 PCA 主轴修正。
3. 如果二者差异太大，就回退 pose heading 或历史方向。

简版实现也可以先只用 pose heading 对齐。

### Step 6: Edge Search On Aligned BEV

方向对齐后，沿前向轴分 slice 搜索左右边界。

每个 slice 内：

1. 找到该 slice 的 occupied pixels。
2. 在横向轴上找多个 occupied runs。
3. 不直接拿最左和最右。
4. 保留候选 run 后做左右配对打分。

评分建议包含：

- run support
- run span
- 与上一 slice 的左右连续性
- 与上一 slice 宽度的一致性
- 左右宽度是否在 `[min_width, max_width]`

这样能减少：

- 弯道内侧乱跳
- 毛刺导致误选
- 局部分叉被最外层 run 抢走

### Step 7: Local Edge Polyline Generation

把 slice 级别边界点变成当前窗口下的：

- `left_local_polyline`
- `right_local_polyline`

然后做：

- continuity filtering
- moving average / spline smoothing
- optional RDP simplification

RDP 只用于减点，不替代平滑。

### Step 8: Transform Back To World

局部边界算完后再变回 world：

```text
left_world = pcd_trans(left_local_polyline, p, rotation, False)
right_world = pcd_trans(right_local_polyline, p, rotation, False)
```

### Step 9: Merge Into Global Edge Tracks

每个局部窗口输出的 left/right edge 都要和全局轨迹融合，而不是简单 append。

merge 条件建议：

- 端点距离小于阈值
- 切线方向差小于阈值
- 重叠点跳过或平均
- 避免回折和重复追加

结果维护为：

- `left_edge_track`
- `right_edge_track`

## Standalone Pseudocode

```text
load all road frames
load poses
left_edge_track = []
right_edge_track = []

for frame_idx in range(num_frames):
    window.append(road_frames[frame_idx])
    if len(window) < window_size:
        continue
    if frame_idx % step != 0:
        continue

    roads_world = np.vstack(window)
    pose = poses[frame_idx]

    roads_local = pcd_trans(roads_world, pose[:3], pose[3:7], True)
    roads_local = crop_with_box(roads_local)
    road_mask = rasterize_bev(roads_local[:, :2])
    road_mask = cleanup_mask(road_mask)
    theta = estimate_direction(roads_local, pose_heading, history_heading)
    road_mask_aligned = rotate_to_direction(road_mask, theta)
    left_local, right_local = search_left_right_edges(road_mask_aligned)
    left_local, right_local = smooth_edges(left_local, right_local)
    left_world = pcd_trans(left_local, pose[:3], pose[3:7], False)
    right_world = pcd_trans(right_local, pose[:3], pose[3:7], False)
    left_edge_track = merge_edge(left_edge_track, left_world)
    right_edge_track = merge_edge(right_edge_track, right_world)

save left/right edge outputs
```

## Minimal First Version

如果先做一版最小可用实现，建议只上这些：

1. 先完整读入所有 frame 的 `road_class` 点。
2. 仍然按 `window + step` 做局部窗口。
3. `pcd_trans(..., inverse=True)` 变到 local。
4. local box crop。
5. BEV rasterize。
6. 只用 pose heading 做方向对齐。
7. slice-based 左右边界搜索。
8. `pcd_trans(..., inverse=False)` 变回 world。
9. merge 成全局 left/right edge。
10. 单独输出 `npy/json/preview` 结果。

先不要一开始就上 skeleton、graph search、全局优化，也不要再去迁就 `make_vector.py` 的流程。
