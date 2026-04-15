# Road Edge Fusion Pipeline（基于伪中心）

## 1. 输入数据（每帧 record）

每条记录包含：

- `dirc`：道路方向（已保证稳定，>10°异常已过滤）
- `left_edge / right_edge`：左右边界线段（p1, p2）
- `centroid`：点云质心（仅用于辅助）

```python
def make_road_edge_record(index, road_ctx, left_seg, right_seg, dirc, road_z):
    # Store the minimal per-frame geometry needed for later edge merging.
    return {
        'index': int(index),
        'centroid': road_ctx['current_center'].astype(np.float32).tolist(),
        'dirc': np.asarray(dirc, dtype=np.float32).tolist(),
        'road_z': float(road_z),
        'left_edge': {
            'p1': left_seg['p1'].astype(np.float32).tolist(),
            'p2': left_seg['p2'].astype(np.float32).tolist(),
        },
        'right_edge': {
            'p1': right_seg['p1'].astype(np.float32).tolist(),
            'p2': right_seg['p2'].astype(np.float32).tolist(),
        },
    }
```

---

## 2. 单帧几何构建

### 2.1 边界中点
```python
m_left  = 0.5 * (left.p1 + left.p2)
m_right = 0.5 * (right.p1 + right.p2)
```

### 2.2 伪中心
```python
c = 0.5 * (m_left + m_right)
```

### 2.3 方向与法向
```python
t = normalize(dirc)
n = np.array([-t[1], t[0]])
```

### 2.4 道路宽度
```python
w = np.linalg.norm(m_left - m_right)
```

---

## 3. 质量过滤（关键）

### 3.1 左右位置合法性（必须）
```python
dot(m_left - c, n)  > 0
dot(m_right - c, n) < 0
```
不满足 → 丢弃该帧

---

### 3.2 宽度约束（建议）
```python
w_min < w < w_max
abs(w - w_med) < threshold
```

---

### 3.3 质心偏差检测（可选）
```python
||centroid - c|| < threshold
```

---

## 4. 构建融合序列

```python
{
    "c": 伪中心,
    "m_left": 左边界中点,
    "m_right": 右边界中点,
    "t": 方向,
    "w": 宽度
}
```

---

## 5. 沿程坐标 s（排序）

```python
delta_s = dot(c_i - c_{i-1}, t_{i-1})
s_i = s_{i-1} + max(delta_s, 0)
```

---

## 6. 融合（核心）

### 6.1 方向平滑
```python
t_smooth = normalize(moving_average(t))
```

### 6.2 中心线平滑
```python
c_smooth = smooth(c(s))
```

### 6.3 左右边界平滑
```python
m_left_smooth  = smooth(m_left(s))
m_right_smooth = smooth(m_right(s))
```

### 6.4 宽度平滑
```python
w_smooth = smooth(median_filter(w(s)))
```

---

## 7. 重建输出

### 左右边界
```python
left_fused  = m_left_smooth
right_fused = m_right_smooth
```

### 中心线
```python
center = 0.5 * (left_fused + right_fused)
```

---

## 8. 缺失补偿

```python
right = left - w_avg * n
left  = right + w_avg * n
```

---

## 9. 核心思想

边界优先，中心由边界推导，质心仅作辅助
