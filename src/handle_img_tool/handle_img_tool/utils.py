from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField
import numpy as np
from std_msgs.msg import Header
import cv2

def print_red_text(text):
    """
    在终端以红色字体打印文本，便于突出显示调试信息。
    """
    print("\033[31m" + text + "\033[0m")

def rgbd2xyz(img, depth, fx, fy, cx, cy, step=4):
    """
    将RGB-D图像转换为三维点云。
    参数:
        img: 彩色图像 (H, W, 3)
        depth: 深度图 (H, W)
        fx, fy, cx, cy: 相机内参
        step: 采样步长，默认每4个像素采样一次
    返回:
        points: (N, 3) 三维点坐标
        colors: (N, 3) 对应的颜色
    """
    factor = 5000  # 深度缩放因子，通常与深度图格式有关
    points = []
    colors = []
    for v in range(depth.shape[0])[::step]:
        for u in range(depth.shape[1])[::step]:
            z = depth[v, u] / factor * 1000  # 将深度值转换为毫米
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            color = img[v, u]
            points.append((x, y, z))
            colors.append(color)
    points = np.asarray(points)
    colors = np.asarray(colors)
    return points, colors

def rgbd2xyz_from_points_list(pts, img, depth, fx, fy, cx, cy, step=1):
    """
    根据给定的像素点列表，将RGB-D图像中的点转换为三维坐标和颜色。
    参数:
        pts: [(u, v), ...] 像素点列表
        img: 彩色图像
        depth: 深度图
        fx, fy, cx, cy: 相机内参
        step: 采样步长
    返回:
        points: (N, 3) 三维点坐标
        colors: (N, 3) 对应的颜色
    """
    factor = 5000
    points = []
    colors = []
    num = len(pts)
    if num == 0: return [], []
    step = num // 500 if num > 500 else step  # 若点数较多则稀疏采样
    for u, v in pts[::step]:
        v, u = int(u), int(v)  # 注意这里的(u, v)顺序
        z = depth[u, v] / factor * 1000
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        color = img[u, v]
        points.append((x, y, z))
        colors.append(color)
    points = np.asarray(points)
    colors = np.asarray(colors)
    return points, colors

def create_pointcloud(points, num=1000):
    """
    将三维点列表转换为ROS的PointCloud2消息。
    参数:
        points: (N, 3) 三维点坐标
        num: 采样点数，默认最多1000个点
    返回:
        pc2_msg: PointCloud2 消息
    """
    l = len(points)
    step = l // num if l > 1000 else 1
    points = points[::step]
    header = Header()
    header.frame_id = 'map'
    dtype = PointField.FLOAT32
    fields = [
        PointField(name="x", offset=0, datatype=dtype, count=1),
        PointField(name="y", offset=4, datatype=dtype, count=1),
        PointField(name="z", offset=8, datatype=dtype, count=1),
    ]
    num_points = len(points)
    points_data = []
    for i in range(num_points):
        x, y, z = points[i]
        points_data.append([x, y, z])

    # 转换为PointCloud2格式
    points_data = np.array(points_data).reshape(3, -1).T
    pc2_msg = point_cloud2.create_cloud(header, fields, points_data)

    return pc2_msg

def draw_bino(limg, lpts, rimg, rpts):
    """
    在左右图像上绘制匹配的特征点，使用随机颜色区分不同的匹配对。
    参数:
        limg: 左图像
        lpts: 左图像特征点列表
        rimg: 右图像
        rpts: 右图像特征点列表
    返回:
        limg, rimg: 绘制后的左右图像
    """
    if len(lpts) == len(rpts):
        for (p1, p2) in zip(lpts, rpts):
            p1, p2 = p1.pt, p2.pt
            p1 = (int(p1[0]), int(p1[1]))
            p2 = (int(p2[0]), int(p2[1]))
            c = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(limg, p1, 5, c, -1)
            cv2.circle(rimg, p2, 5, c, -1)
    return limg, rimg