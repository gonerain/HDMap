#!/usr/bin/python3
"""
几何工具模块，提供多边形、轮廓转换、几何验证等通用功能。
"""
import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def as_xy(points, dtype=np.float32):
    """
    将点集转换为 N×2 的 float32 数组。

    Args:
        points: 点集，可以是列表、元组、numpy数组
        dtype: 输出数据类型，默认为 np.float32

    Returns:
        N×2 的数组，类型为 dtype

    Raises:
        TypeError: 输入不是可转换的类型
        ValueError: 转换后无法得到二维坐标
    """
    try:
        arr = np.asarray(points, dtype=dtype)
    except Exception as e:
        raise TypeError(f"无法将输入转换为数组: {e}")

    if arr.size == 0:
        return arr.reshape(0, 2).astype(dtype)

    if arr.ndim == 1:
        # 单个点 [x, y, ...] -> [[x, y]]
        return arr[:2].reshape(1, 2).astype(dtype)

    # N×M 数组，取前两列
    return arr[:, :2].astype(dtype)


def close_outline(points_xy, dtype=np.float32, epsilon=1e-6):
    """
    确保多边形轮廓是闭合的（首尾点相同）。

    Args:
        points_xy: 轮廓点集
        dtype: 输出数据类型
        epsilon: 判断两点是否相同的距离阈值

    Returns:
        闭合的轮廓点集
    """
    points_xy = as_xy(points_xy, dtype=dtype)
    if len(points_xy) == 0:
        return points_xy

    # 检查首尾点是否已经相同
    if np.linalg.norm(points_xy[0] - points_xy[-1]) < epsilon:
        return points_xy

    # 添加首点使其闭合
    return np.vstack((points_xy, points_xy[:1])).astype(dtype)


def polygon_from_outline(outline_xy, buffer_radius=0.0, dtype=np.float32):
    """
    从轮廓点集创建 Shapely 多边形，增强健壮性。

    Args:
        outline_xy: 轮廓点集
        buffer_radius: 用于修复无效多边形的缓冲区半径
        dtype: 输入点集的数据类型

    Returns:
        (polygon, error_message) 元组，如果成功 polygon 为 Polygon 对象，
        失败时为 None，error_message 描述失败原因
    """
    outline_xy = as_xy(outline_xy, dtype=dtype)

    if len(outline_xy) < 3:
        return None, f"轮廓点数量不足 ({len(outline_xy)} < 3)"

    try:
        polygon = Polygon(outline_xy)
    except Exception as e:
        return None, f"创建多边形失败: {e}"

    if polygon.is_empty:
        return None, "多边形为空"

    # 尝试修复无效多边形
    if not polygon.is_valid:
        try:
            polygon = polygon.buffer(buffer_radius)
            if polygon.is_empty:
                return None, "缓冲区修复后多边形为空"
        except Exception as e:
            return None, f"缓冲区修复失败: {e}"

    # 如果是 MultiPolygon，取面积最大的
    if isinstance(polygon, MultiPolygon):
        if polygon.is_empty:
            return None, "MultiPolygon 为空"
        polygon = max(polygon.geoms, key=lambda geom: geom.area)

    if not isinstance(polygon, Polygon):
        return None, f"预期 Polygon，得到 {type(polygon)}"

    if polygon.area <= 1e-6:
        return None, f"多边形面积过小 ({polygon.area:.6f})"

    return polygon, None


def outline_from_polygon(polygon, dtype=np.float32):
    """
    从 Shapely 多边形提取轮廓点集。

    Args:
        polygon: Shapely Polygon 或 MultiPolygon
        dtype: 输出数据类型

    Returns:
        多边形外环的轮廓点集
    """
    if polygon is None or polygon.is_empty:
        return np.zeros((0, 2), dtype=dtype)

    if isinstance(polygon, MultiPolygon):
        # 取面积最大的多边形
        polygon = max(polygon.geoms, key=lambda geom: geom.area)

    if not isinstance(polygon, Polygon):
        return np.zeros((0, 2), dtype=dtype)

    try:
        return np.asarray(polygon.exterior.coords, dtype=dtype)
    except Exception:
        return np.zeros((0, 2), dtype=dtype)


def validate_contour_parameters(iterations=1, min_seg_len=0.1, max_passes=3, simplify_tol=0.1):
    """
    验证并钳位轮廓处理参数。

    Args:
        iterations: 平滑迭代次数，≥0
        min_seg_len: 最小线段长度，>0
        max_passes: 最大处理次数，≥1
        simplify_tol: 简化容差，≥0

    Returns:
        验证后的参数元组 (iterations, min_seg_len, max_passes, simplify_tol)
    """
    iterations = max(0, int(iterations))
    min_seg_len = max(1e-6, float(min_seg_len))
    max_passes = max(1, int(max_passes))
    simplify_tol = max(0.0, float(simplify_tol))

    return iterations, min_seg_len, max_passes, simplify_tol


def chaikin_smooth_closed(points_xy, iterations=2, dtype=np.float32):
    """
    Chaikin 曲线平滑算法（闭合轮廓）。

    Args:
        points_xy: 闭合轮廓点集
        iterations: 迭代次数，钳位到 [0, 10]
        dtype: 输出数据类型

    Returns:
        平滑后的轮廓点集
    """
    points_xy = close_outline(points_xy, dtype=dtype)
    if len(points_xy) < 4:
        return points_xy

    iterations, _, _, _ = validate_contour_parameters(
        iterations=iterations, min_seg_len=0.1, max_passes=1, simplify_tol=0.1
    )
    iterations = min(iterations, 10)  # 防止过多迭代

    work = points_xy[:-1].copy()  # 去掉重复的闭合点
    for _ in range(iterations):
        if len(work) < 3:
            break

        new_points = []
        for idx in range(len(work)):
            p0 = work[idx]
            p1 = work[(idx + 1) % len(work)]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_points.extend((q.astype(dtype), r.astype(dtype)))

        work = np.asarray(new_points, dtype=dtype)

    return close_outline(work, dtype=dtype)


def prune_short_edges_closed(points_xy, min_seg_len=0.2, max_passes=3, dtype=np.float32):
    """
    修剪闭合轮廓中的短边。

    Args:
        points_xy: 闭合轮廓点集
        min_seg_len: 最小线段长度，钳位到 [1e-4, inf]
        max_passes: 最大处理次数，钳位到 [1, 10]
        dtype: 输出数据类型

    Returns:
        修剪后的轮廓点集
    """
    points_xy = close_outline(points_xy, dtype=dtype)
    if len(points_xy) < 4:
        return points_xy

    _, min_seg_len, max_passes, _ = validate_contour_parameters(
        iterations=0, min_seg_len=min_seg_len, max_passes=max_passes, simplify_tol=0.1
    )
    max_passes = min(max_passes, 10)  # 防止过多处理

    work = points_xy[:-1].copy()  # 去掉重复的闭合点
    for _ in range(max_passes):
        if len(work) < 3:
            break

        changed = False
        keep = [work[0]]

        for idx in range(1, len(work)):
            curr = work[idx]
            if np.linalg.norm(curr - keep[-1]) < min_seg_len:
                changed = True
                continue
            keep.append(curr)

        # 检查首尾连接
        if len(keep) >= 3 and np.linalg.norm(keep[0] - keep[-1]) < min_seg_len:
            keep.pop()
            changed = True

        if len(keep) < 3 or not changed:
            work = np.asarray(keep, dtype=dtype)
            break

        work = np.asarray(keep, dtype=dtype)

    return close_outline(work, dtype=dtype)