# `road_process` 完整算法与 Pipeline

本文档基于当前仓库实现整理，覆盖：

1. 在线 `road_process` 单帧/滑窗 road edge 提取
2. `road_edge_records.json` 的输出结构
3. `fuse_road_edges.py` 的全局融合流程

对应代码入口：

- `/mnt/ning_602/work/HDMap/make_vector.py`
- `/mnt/ning_602/work/HDMap/core/vector_registry.py`
- `/mnt/ning_602/work/HDMap/core/vector_common.py`
- `/mnt/ning_602/work/HDMap/core/road_process.py`
- `/mnt/ning_602/work/HDMap/src/fuse_road_edges.py`

---

## 1. 总体 Pipeline

完整链路可以写成：

```text
原始语义点云序列
-> 过滤 road_class
-> 滑动窗口累计
-> 构造 front/current/last 三段上下文
-> 最大连通 road 点簇保留
-> alpha shape 外轮廓提取
-> 折线简化
-> 由 front/last 质心估计道路方向
-> 在轮廓上搜索中心两侧平行边界段
-> 输出每个 logical frame 的 road edge record
-> 对 record 做质量过滤、排序、平滑
-> 输出 fused left/right/center line
```

可记为：

```math
\mathcal{R}_{fused} = \mathcal{F}\big(\mathcal{E}(\mathcal{C}(\mathcal{W}(P)))\big)
```

其中：

- \(P\): 输入语义点云序列
- \(\mathcal{W}\): 滑窗上下文构造
- \(\mathcal{C}\): 当前窗口 road 轮廓构造
- \(\mathcal{E}\): 左右边界提取
- \(\mathcal{F}\): 全局融合

---

## 2. 输入与调度

### 2.1 输入

每帧语义点云记为：

```math
X_t = \{(x_i, y_i, z_i, c_i)\}_{i=1}^{N_t}
```

其中 \(c_i\) 是语义类别。`road_process` 仅保留：

```math
R_t = \{(x_i, y_i, z_i) \mid c_i = c_{road}\}
```

代码中默认：

```text
c_road = config["road_class"] = 13
```

对应实现：

```python
self.history.append(sempcd[sempcd[:, 3] == self.target_class])
```

### 2.2 调度参数

`VectorProcess` 默认参数：

- `window_size = 30`
- `dirc_window = 20`
- `step = 20`
- `static_dirc_thresh = 0.2`

历史队列长度：

```math
L = 2 \cdot dirc\_window + window\_size = 70
```

只有当队列装满后才允许处理：

```math
ready \iff |\text{history}| = L
```

逻辑帧号由最新物理帧号 `latest_frame_index` 转成：

```math
logical\_index = latest\_frame\_index - dirc\_window - window\_size + 1
```

仅当满足采样条件时处理：

```math
logical\_index \bmod step = 0
```

---

## 3. 上下文构造

对一个 `logical_index = k`，构造三段窗口：

- `front`: 历史前段
- `current`: 当前检测段
- `last`: 历史后段

记三段点集为：

```math
R_k^{front},\quad R_k^{current},\quad R_k^{last}
```

代码中由 `history` 切片得到：

```text
front   = history[0 : window_size]
current = history[dirc_window : dirc_window + window_size]
last    = history[2*dirc_window : 2*dirc_window + window_size]
```

三段堆叠后：

```math
P^{front}_k = \bigcup_{t \in front} R_t,\quad
P^{current}_k = \bigcup_{t \in current} R_t,\quad
P^{last}_k = \bigcup_{t \in last} R_t
```

然后计算三个 2D 质心：

```math
\mu^{front}_k = \frac{1}{|P^{front}_k|}\sum_{p \in P^{front}_k} p_{xy}
```

```math
\mu^{current}_k = \frac{1}{|P^{current}_k|}\sum_{p \in P^{current}_k} p_{xy}
```

```math
\mu^{last}_k = \frac{1}{|P^{last}_k|}\sum_{p \in P^{last}_k} p_{xy}
```

若三段任一为空，则直接跳过：

```math
|P^{current}_k| = 0 \;\lor\; |P^{front}_k| = 0 \;\lor\; |P^{last}_k| = 0
\Rightarrow \text{skip}
```

---

## 4. 当前窗口 Road 主体提取

### 4.1 最大簇保留

对当前窗口点集 \(P^{current}_k\) 做 DBSCAN 聚类：

```math
\text{labels} = \operatorname{DBSCAN}(P^{current}_k; \epsilon = 1.0,\ minPts = 20)
```

这里作为库算法可简写为：

```math
\text{labels} = F_{DBSCAN}(xyz)
```

然后仅保留最大非噪声簇：

```math
\hat{P}_k = \arg\max_{cluster\ j} |C_j|
```

作用：

- 去掉零散 road 噪点
- 保留当前窗口中最稳定的道路主体

### 4.2 Alpha Shape 外轮廓

对最大簇二维投影 \(\hat{P}_{k,xy}\) 提取外轮廓：

```math
\Gamma_k = F_{alphashape}(\hat{P}_{k,xy}; \alpha = 0.8)
```

代码中：

```python
shape = alphashape.alphashape(points_xyz[:, :2], alpha)
```

输出是一个 `Polygon` 或 `LineString`，最终统一为二维折线点序列：

```math
\Gamma_k = \{q_1, q_2, \dots, q_m\},\quad q_i \in \mathbb{R}^2
```

### 4.3 折线简化

对轮廓折线做基于斜率变化的简化：

```math
\tilde{\Gamma}_k = F_{simplify}(\Gamma_k;\ \theta_{th}=7^\circ,\ l_{min}=0.15)
```

其规则可理解为：

若相邻线段方向满足

```math
|\langle \hat{d}_{i-1}, \hat{d}_i \rangle| \ge \cos(\theta_{th})
```

则认为方向近似一致，可以合并；同时若线段长度

```math
\|q_i - q_{i-1}\| < l_{min}
```

则忽略该短段。

---

## 5. 道路方向估计

当前实现没有直接用姿态角，而是用 `front` 与 `last` 的中心位移估计道路方向：

```math
d_k = \mu^{last}_k - \mu^{front}_k
```

归一化后得到方向单位向量：

```math
\hat{t}_k = \frac{d_k}{\|d_k\|_2}
```

若

```math
\|d_k\|_2 < 0.2
```

则认为车辆近似静止或方向不稳定，直接跳过该窗口：

```math
\|d_k\|_2 < \tau_{static} \Rightarrow \text{skip}
```

其中 `tau_static = 0.2`。

---

## 6. 轮廓上的左右边界搜索

当前 `road_process.py` 使用的是：

```python
find_parallel_segments_around_center(...)
```

也就是直接在轮廓折线 \(\tilde{\Gamma}_k\) 上找一对平行线段。

### 6.1 参考坐标系

由方向向量构造法向量：

```math
\hat{n}_k = (-\hat{t}_{k,y}, \hat{t}_{k,x})
```

当前中心记为：

```math
c_k = \mu^{current}_k
```

### 6.2 枚举轮廓线段

对折线中的每条线段：

```math
s_i = (q_i, q_{i+1})
```

定义：

```math
v_i = q_{i+1} - q_i,\quad
\ell_i = \|v_i\|_2,\quad
\hat{u}_i = \frac{v_i}{\ell_i}
```

若 \(\ell_i < 0.5\) 则丢弃：

```math
\ell_i < l_{seg} \Rightarrow \text{reject}
```

其中 `l_seg = 0.5`。

### 6.3 平行性约束

要求候选线段与道路方向近似平行：

```math
|\langle \hat{u}_i, \hat{t}_k \rangle| \ge \cos(15^\circ)
```

若方向相反，则翻转线段端点顺序，使其与 \(\hat{t}_k\) 同向。

### 6.4 左右侧判定

线段中点：

```math
m_i = \frac{q_i + q_{i+1}}{2}
```

相对当前中心的横向偏移：

```math
\lambda_i = \langle m_i - c_k, \hat{n}_k \rangle
```

若

```math
|\lambda_i| < 0.5
```

则说明离中心太近，不视为边界。

代码里约定：

- \(\lambda_i < 0\): left candidate
- \(\lambda_i > 0\): right candidate

### 6.5 左右配对评分

从左集合与右集合中枚举配对 \((s_i^{L}, s_j^{R})\)，要求两条线段也互相平行：

```math
|\langle \hat{u}_i^L, \hat{u}_j^R \rangle| \ge \cos(10^\circ)
```

对每个左右候选对计算分数：

```math
Score(i, j) =
\ell_i^L + \ell_j^R
- 0.3(a_i^L + a_j^R)
- 0.1\big||\lambda_i^L| - |\lambda_j^R|\big|
- 0.05(r_i^L + r_j^R)
```

其中：

- \(a_i = |\langle m_i - c_k,\ \hat{t}_k \rangle|\)，表示沿道路方向的轴向偏移
- \(r_i = \|m_i - c_k\|_2\)，表示到中心的距离

选择得分最高的一对作为该窗口左右边界：

```math
(s_k^{left}, s_k^{right}) = \arg\max_{i,j} Score(i,j)
```

若没有合法配对，则该窗口跳过。

---

## 7. Record 输出

成功提取后，输出最小几何 record：

```json
{
  "index": k,
  "target_class": 13,
  "centroid": [cx, cy],
  "dirc": [tx, ty],
  "road_z": z_med,
  "left_edge": {"p1": [...], "p2": [...]},
  "right_edge": {"p1": [...], "p2": [...]}
}
```

其中道路高度取当前窗口 z 中位数：

```math
z_k = \operatorname{median}\{p_z \mid p \in P^{current}_k\}
```

因此，在线部分可以记为：

```math
record_k = F_{road\_process}(P^{front}_k, P^{current}_k, P^{last}_k)
```

最终得到：

```math
\mathcal{D} = \{record_k\}_{k \in \mathcal{K}}
```

其中 \(\mathcal{K}\) 是所有成功提取的逻辑帧集合。

---

## 8. 全局融合 Pipeline

`src/fuse_road_edges.py` 对 \(\mathcal{D}\) 做后处理，输出全局平滑的左右边界与中心线。

---

## 9. 单帧 record 几何重建

对单帧 record，先取左右边界中点：

```math
m_k^{L} = \frac{p_{k,1}^{L} + p_{k,2}^{L}}{2}
```

```math
m_k^{R} = \frac{p_{k,1}^{R} + p_{k,2}^{R}}{2}
```

若两侧都存在，则伪中心为：

```math
c_k = \frac{m_k^L + m_k^R}{2}
```

宽度为：

```math
w_k = \|m_k^L - m_k^R\|_2
```

### 9.1 法向和切向重估计

当前实现优先使用左右中点差值估计法向：

```math
\hat{n}_k = \frac{m_k^L - m_k^R}{\|m_k^L - m_k^R\|_2}
```

如果其方向与 `dirc` 提供的参考法向冲突，则翻转符号。随后取切向：

```math
\hat{t}_k = (-\hat{n}_{k,y}, \hat{n}_{k,x})
```

若只有单侧可用，则退化为：

```math
\hat{t}_k = normalize(dirc_k),\quad
\hat{n}_k = (-\hat{t}_{k,y}, \hat{t}_{k,x})
```

---

## 10. 质量过滤

### 10.1 缺边补偿

若只检测到一侧边界，则用默认宽度或全局宽度中位数补另一侧：

```math
m_k^L = m_k^R + w_{ref}\hat{n}_k
```

```math
m_k^R = m_k^L - w_{ref}\hat{n}_k
```

其中：

```math
w_{ref} = \operatorname{median}\{w_k\}
```

若全局无合法宽度，则退回参数 `default_width=4.0`。

### 10.2 左右合法性检查

要求左点在法向正侧、右点在法向负侧：

```math
\langle m_k^L - c_k,\ \hat{n}_k \rangle > 0
```

```math
\langle m_k^R - c_k,\ \hat{n}_k \rangle < 0
```

若数据整体左右定义反了，代码会自动检测并整体翻转判断规则。

### 10.3 宽度过滤

要求：

```math
width\_min \le w_k \le width\_max
```

默认：

- `width_min = 1.5`
- `width_max = 20.0`

另外还可约束其相对局部中值的偏差。

### 10.4 回退过滤

若当前样本相对前一合法样本在局部切向上明显后退，则剔除：

```math
\Delta c_k = c_k - c_{k-1}
```

```math
forward_k = \langle \Delta c_k,\ normalize(\hat{t}_{k-1} + \hat{t}_k) \rangle
```

若

```math
forward_k < -max\_backtrack
```

则 reject，默认 `max_backtrack = 0.3`。

---

## 11. 沿程坐标与排序

先按 `index` 排序得到 \(\{c_k\}_{k=1}^N\)。

当前实现的站点坐标 `s` 使用中心点欧氏距离累计：

```math
s_1 = 0
```

```math
s_k = s_{k-1} + \|c_k - c_{k-1}\|_2,\quad k \ge 2
```

即：

```math
s = F_{station}(center\_sequence)
```

---

## 12. 全局平滑与边界重建

### 12.1 中心线平滑

对中心序列做滑动平均：

```math
\tilde{c}_k = MA(c_k;\ window = center\_window)
```

默认 `center_window = 5`。

### 12.2 切向平滑

先用几何差分估计切向：

```math
\bar{t}_k =
\begin{cases}
\tilde{c}_{2} - \tilde{c}_{1}, & k=1 \\
\tilde{c}_{k+1} - \tilde{c}_{k-1}, & 1 < k < N \\
\tilde{c}_{N} - \tilde{c}_{N-1}, & k=N
\end{cases}
```

归一化后再做滑动平均：

```math
\tilde{t}_k = normalize(MA(normalize(\bar{t}_k);\ window = dir\_window))
```

默认 `dir_window = 9`。

### 12.3 宽度平滑

先中值滤波，再均值滤波：

```math
\tilde{w}_k = MA(MedianFilter(w_k;\ width\_window);\ width\_window)
```

默认 `width_window = 8`，代码内部会自动转成奇数窗。

### 12.4 左右边界重建

平滑法向量：

```math
\tilde{n}_k = (-\tilde{t}_{k,y}, \tilde{t}_{k,x})
```

最终左右边界点为：

```math
\tilde{m}_k^L = \tilde{c}_k + \frac{1}{2}\tilde{w}_k \tilde{n}_k
```

```math
\tilde{m}_k^R = \tilde{c}_k - \frac{1}{2}\tilde{w}_k \tilde{n}_k
```

高度同样做一维滑动平均：

```math
\tilde{z}_k = MA(z_k;\ window = center\_window)
```

所以最终输出样本：

```math
sample_k = (\tilde{m}_k^L,\ \tilde{m}_k^R,\ \tilde{c}_k,\ \tilde{t}_k,\ \tilde{w}_k,\ \tilde{z}_k)
```

---

## 13. 输出结果

融合输出 JSON 结构包含：

- `left_edge`: 全局左边界折线
- `right_edge`: 全局右边界折线
- `center_line`: 全局中心线
- `samples`: 每个站点的平滑结果
- `meta`: 过滤统计与参数

可写成：

```math
\mathcal{M}_{road} =
\{
L(s),\ R(s),\ C(s)
\}
```

其中：

- \(L(s)\): 左边界
- \(R(s)\): 右边界
- \(C(s)\): 中心线

---

## 14. 现实现的核心特点

### 14.1 优点

- 不依赖严格车道线，而是直接使用 road semantic 面
- 用滑窗累积提高 road 点云稠密度
- 用 `front/current/last` 质心差估计局部道路方向，适合在线处理
- 通过轮廓线段配对直接获得左右 road edge
- 后处理阶段再做全局平滑，在线和离线职责分离

### 14.2 当前假设

- 当前窗口内 road 语义主体大致连通
- alpha shape 外轮廓能较好逼近道路边界
- 道路边界在局部上近似与行进方向平行
- 左右边界在中心两侧具有相对稳定的横向间距

### 14.3 可能失效场景

- 大路口或分叉处，road 面不再接近双边界带状结构
- 强遮挡导致 alpha shape 轮廓破碎
- 急弯、掉头、回环场景中 `front-last` 方向估计不稳定
- 多片 road 连通区同时出现时，最大簇未必是目标道路

---

## 15. 一句话算法总结

当前 `road_process` 的本质是：

```math
\text{道路语义滑窗点云}
\rightarrow
\text{最大连通 road 区域}
\rightarrow
\text{alpha-shape 外轮廓}
\rightarrow
\text{基于局部方向的左右平行边界配对}
\rightarrow
\text{逐帧 road edge record}
\rightarrow
\text{全局宽度/方向/中心线平滑融合}
```

如果写成更紧凑的函数形式：

```math
(L, R, C) =
F_{fuse}\Big(
\big\{
F_{pair}\big(
F_{outline}(F_{cluster}(P_k^{current})),
\mu_k^{current},
\mu_k^{last} - \mu_k^{front}
\big)
\big\}_{k \in \mathcal{K}}
\Big)
```

