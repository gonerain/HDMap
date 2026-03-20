# Road Edge Algorithm For `make_vector.py`

## Goal

这份设计不是单独起一套新 pipeline，而是为了直接嵌入 [`make_vector.py`](/mnt/ning_602/work/HDMap/make_vector.py) 现有流程。

也就是说，整体执行顺序仍然保持和 `make_vector.py` 一致：

1. 逐帧读取语义点云。
2. 按当前 `pose` 把滑窗内点云放到当前 frame 下处理。
3. 在 `process()` 里完成局部道路边界提取。
4. 把局部结果变回 world。
5. 跨帧累积成全局左/右 road edge。
6. 最后和原有 `savepcd` / `vector.pcd` 导出逻辑并存。

目标是让 road edge 成为 `make_vector.py` 里和 lane / pole 并列的一类 vector element，而不是另起一个独立脚本。

## How To Embed Into `make_vector.py`

建议直接复用 `make_vector.py` 现有框架里的这些元素：

- `window` / `step`
- `poses`
- `pcd_trans()`
- `process()` 的逐帧处理时机
- `savepcd` 的全局累积
- `vecPubHandle` 的可视化发布
- 最终 `vector.pcd` 的导出

建议新增的全局变量：

- `roadpcd`: `myqueue(window)`
  用于缓存最近若干帧 road points
- `left_edge_segments`: list
  存每个局部窗口输出的左边界段
- `right_edge_segments`: list
  存每个局部窗口输出的右边界段
- `left_edge_track`: list
  存融合后的全局左边界
- `right_edge_track`: list
  存融合后的全局右边界
- `road_vectors`: list
  存 road edge 对应的可保存点云

如果要和现有配置文件对齐，建议在 `config` 里增加：

- `road_class`
- `road_edge_vector`
- `road_box_forward`
- `road_box_lateral`
- `road_box_z`
- `road_grid_resolution`
- `road_min_points_per_cell`
- `road_min_width`
- `road_max_width`

## Keep The Existing Main Flow

`make_vector.py` 现在的主流程是：

1. 读 pickle
2. 逐帧进入 `process()`
3. 在 `process()` 内利用当前 `pose` 做局部处理
4. 把局部 vector 发布到 ROS
5. 全部处理完后再导出 `result.pcd` 和 `vector.pcd`

road edge 也建议完全按这个节奏做，不要改成离线一次性全局扫描。

换句话说，road edge 的入口应放在 [`make_vector.py`](/mnt/ning_602/work/HDMap/make_vector.py) 的 `process()` 内，位置上和当前 lane vectorization 同级。

## Data Source

每一帧仍然从当前 `sempcd` 里取语义类别。

对于 road edge，需要取：

- `roads = sempcd[sempcd[:, 3] == config['road_class']]`

然后像当前 lane 一样，把 `roads` 放进滑窗缓存：

- `roadpcd.append(roads)`

当缓存长度达到 `window`，并且满足 `index % step == 0` 时，触发一次局部 road edge 提取。

这和当前 lane 的时机保持一致，便于直接嵌进现有 `process()`。

## Step 1: Sliding Window Accumulation

和 lane 一样，road edge 不只看当前一帧，而是使用滑窗内多帧叠加点云。

```text
roads = sempcd[sempcd[:, 3] == config['road_class']]
roadpcd.append(roads)
if len(roadpcd) >= window and index % step == 0:
    roads_local_window = np.vstack(roadpcd)
```

这么做的原因：

- 单帧 road segmentation 容易断
- 边界局部噪声较大
- 多帧叠加后 BEV mask 更稳定

## Step 2: Transform To Current Local Frame

保持和 lane 相同的思路，使用当前 `pose` 把滑窗内 road 点云变换到当前 frame 的 local 坐标系。

直接复用现有 [`make_vector.py`](/mnt/ning_602/work/HDMap/make_vector.py) 里的 `pcd_trans()`：

- `roads_local = pcd_trans(roads_local_window, p, rotation, True)`

其中：

- `p` 是当前位姿平移
- `rotation` 是当前四元数
- `inverse=True` 表示 world -> current local

这样可以保持和 lane vectorization 完全一致的坐标定义，不需要再引入新的变换逻辑。

## Step 3: Box Crop In Local Frame

在 local frame 下做 box 范围裁剪，只保留当前车前方一块稳定区域。

建议 box 规则：

- forward: `y in [0, y_max]`
- lateral: `x in [-x_limit, x_limit]`
- z: `z in [z_min, z_max]`

示例：

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

这样做是为了和 `make_vector.py` 现有的“当前 pose 附近局部处理”模式保持一致，而不是直接在全局地图上找 edge。

## Step 4: BEV Projection

裁剪后的 local road 点只保留平面坐标：

- `points_xy = roads_local[:, :2]`

这里仍然在 local frame 中做 BEV，不在 world frame 中做。

原因：

- 当前 frame 下的前向和横向定义稳定
- 更适合做“前向 slice + 左右边界”搜索
- 与 `process()` 的局部决策逻辑一致

## Step 5: Rasterization

将 `points_xy` 栅格化成二维占据图。

设分辨率为 `res`：

- `col = floor((x - xmin) / res)`
- `row = floor((y - ymin) / res)`

统计每个栅格的点数：

- `count[row, col] += 1`
- `mask[row, col] = count >= min_points_per_cell`

建议参数：

- `res = 0.15 ~ 0.2`
- `min_points_per_cell = 2 ~ 4`

然后做轻量清理：

- binary closing
- binary opening
- remove small component

输出：

- `road_mask`
- `origin_xy`
- `counts`

## Step 6: Direction Estimation

这一步要和 `make_vector.py` 当前的 local pose 体系兼容。

建议方向优先级如下：

1. 默认使用当前 pose 的 heading 作为主方向。
2. 再用当前 local road points 的 PCA 主轴做修正。
3. 若 PCA 与 heading 差异过大，则保留 heading 或上一帧 edge direction。

原因：

- `make_vector.py` 本来就是跟随当前 pose 在处理局部窗口
- 完全放弃 pose 方向会和当前脚本风格脱节
- 但只依赖 pose heading 又会把车轨迹误当道路中心趋势

所以推荐：

`road_direction = fuse(pose_heading, road_pca_heading, history_heading)`

然后把 local BEV 或 local points 旋转到该主方向对齐坐标系。

## Step 7: Edge Search On Aligned BEV

方向对齐之后，沿前向轴分 slice 搜索左右边界。

每个 slice 内：

1. 找到该 slice 的 road occupied pixels。
2. 在横向轴上找到多个 occupied runs。
3. 不取“最左一个”和“最右一个”。
4. 保留多个候选 run，再做左右配对打分。

配对评分建议包含：

- run support
- run span
- 与上一 slice 的左右连续性
- 与上一 slice 宽度的一致性
- 左右边界间宽度是否落在 `[min_width, max_width]`

这样才能避免你前面提到的：

- 内侧弯道容易乱
- 毛刺时选错簇
- 局部分叉时直接被最外层 run 抢走

## Step 8: Local Edge Polyline Generation

得到 slice 级别的左右边界点后，在当前 local frame 下形成：

- `left_local_polyline`
- `right_local_polyline`

然后做：

- continuity filtering
- moving average / spline smoothing
- optional RDP simplification

注意这里的 RDP 只用于减点，不应用它来承担平滑职责。

## Step 9: Transform Local Edge Back To World

局部边界算完之后，再变回 world frame。

仍然复用 `pcd_trans()`：

- `left_world = pcd_trans(left_local_polyline, p, rotation, False)`
- `right_world = pcd_trans(right_local_polyline, p, rotation, False)`

这样整个 road edge 分支与 lane 分支使用同一套 world/local 变换方式，便于直接插入 `process()`。

## Step 10: Merge Into Global Edge Tracks

这一步很关键，也是为了适配 `make_vector.py` 当前“每次 process 都追加局部 vector”的行为。

如果每次 `process()` 都直接把局部 left/right polyline append 到 `vectors`，最后一定会出现：

- 同一条边界被重复刷很多次
- 图上是一簇一簇局部线段
- 无法形成单条全局 road edge

所以建议在 `process()` 内不要直接把每个局部段都丢进最终输出，而是先做全局 track merge：

- 将 `left_world` 与 `left_edge_track` 做拼接或融合
- 将 `right_world` 与 `right_edge_track` 做拼接或融合
- 只在 merge 后再更新 `road_vectors`

merge 条件建议：

- 端点距离小于阈值
- 切线方向差小于阈值
- 新增点与已有轨迹重叠时做平均或跳过
- 避免回折和重复 append

## Step 11: ROS Publish

如果要和当前 `make_vector.py` 保持交互形式一致，可以把 road edge 也转成点云并发布到 `VectorCloud`。

建议：

- lane、pole、road edge 可以用不同颜色
- road left / right edge 分别编码不同颜色

例如：

- left edge: blue
- right edge: red
- pole: green
- lane center/vector: current existing color

发布时机：

- 每次 merge 完一轮局部 road edge 后
- 或每隔若干帧发布一次 merged track

## Step 12: Save To `vector.pcd`

在现有 [`make_vector.py`](/mnt/ning_602/work/HDMap/make_vector.py) 的保存逻辑里，目前 `vector.pcd` 是 lane + pole。

建议改成：

- `vector.pcd = lane_vectors + pole_vectors + road_edge_vectors`

其中：

- lane vectors 维持现有逻辑
- pole vectors 维持现有逻辑
- road edge vectors 使用 merge 后的全局左右边界

这样不会破坏原来文件输出形式，只是增加一种新的 vector element。

## Recommended Code Placement In `make_vector.py`

建议插入位置：

1. 全局变量区
   增加 `roadpcd`、`left_edge_track`、`right_edge_track`、`road_vectors`
2. 工具函数区
   增加：
   - box crop
   - BEV rasterize
   - morphology cleanup
   - direction estimate
   - edge candidate search
   - edge merge
3. `process()` 中 `if args.vector:` 分支内
   在 lane 处理后、pole 全局处理前插入 road edge 逻辑
4. 最终导出处
   在 `vector.pcd` 拼接时加入 road edge

## Pseudocode Aligned With `make_vector.py`

```text
init:
    roadpcd = myqueue(window)
    left_edge_track = []
    right_edge_track = []
    road_vectors = []

for each frame:
    sempcd = current semantic point cloud
    p = poses[index]
    rotation = pose quaternion

    if args.vector:
        lanes = sempcd[cls == lane_class]
        roads = sempcd[cls == road_class]

        lanepcd.append(lanes)
        roadpcd.append(roads)

        if len(roadpcd) >= window and index % step == 0:
            road_stack = np.vstack(roadpcd)
            road_local = pcd_trans(road_stack, p, rotation, True)
            road_local = crop_with_box(road_local)
            road_mask = rasterize_bev(road_local[:, :2])
            road_mask = cleanup_mask(road_mask)
            theta = estimate_direction(road_local, pose_heading, history_heading)
            road_mask_aligned = rotate_to_direction(road_mask, theta)
            left_local, right_local = search_left_right_edges(road_mask_aligned)
            left_local, right_local = smooth_edges(left_local, right_local)
            left_world = pcd_trans(left_local, p, rotation, False)
            right_world = pcd_trans(right_local, p, rotation, False)
            left_edge_track = merge_edge(left_edge_track, left_world)
            right_edge_track = merge_edge(right_edge_track, right_world)
            road_vectors = build_edge_points(left_edge_track, right_edge_track)
            publish road_vectors

    publish semantic cloud / image

finish:
    poles = extract_poles(savepcd)
    lane_vectors = existing lane vectors
    vector_all = lane_vectors + poles + road_vectors
    save vector.pcd
```

## Key Design Constraints

为了和 `make_vector.py` 一致，这里有几个约束不要破：

- 不改变现有 pickle 读取方式
- 不改变现有 `process()` 的逐帧调用节奏
- 不单独开一套离线全局建图流程
- world/local 变换尽量复用现有 `pcd_trans()`
- road edge 也应该作为 `args.vector` 的一部分
- 最终输出仍然并入 `vector.pcd`

## Why This Fits `make_vector.py`

这套设计和当前 `make_vector.py` 是兼容的，因为它保留了原脚本最核心的处理方式：

- 以当前 frame 为中心的滑窗处理
- 依赖 pose 做 local/world 变换
- 在 `process()` 中实时生成局部 vector
- 最终统一导出 point-cloud vector 结果

真正变化的只是 lane 那部分“点聚类 + 相邻帧连线”的方法，不适合直接拿来做 road edge；所以 road edge 分支内部改成了：

- box crop
- BEV
- 栅格化
- 方向约束
- 左右边界搜索
- 全局 merge

但外部流程仍然和 `make_vector.py` 保持一致。

## Suggested Minimal First Version

如果先做一版最小可用实现，建议只上这些：

1. 从 `sempcd` 里提取 `road_class`
2. `roadpcd` 滑窗叠帧
3. `pcd_trans(..., inverse=True)` 变到 local
4. local box crop
5. BEV rasterize
6. 用 pose heading 对齐方向
7. slice-based 左右边界搜索
8. `pcd_trans(..., inverse=False)` 变回 world
9. 与全局 left/right track merge
10. 把 merge 后结果写入 `vector.pcd`

先不要一开始就加太复杂的 skeleton、graph search、全局优化。先保证它能自然嵌进 `make_vector.py`，并且输出比当前局部 polyline 叠加稳定。
