import matplotlib.pyplot as plt

class Window_3d:
    def __init__(self, width=6.4, height=4.8, points_num=1000):
        """
        初始化3D可视化窗口。
        参数:
            width: 图像宽度（英寸）
            height: 图像高度（英寸）
            points_num: 最大显示点数，超过则稀疏采样
        """
        super().__init__()
        self.point_num = points_num
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update_win(self, points, colors, lines=None):
        """
        更新窗口，显示点云和轨迹。
        参数:
            points: (N, 3) 三维点坐标数组
            colors: (N, 3) 点云颜色数组（0~255）
            lines: (M, 3) 轨迹点坐标数组，可选
        """
        num = points.shape[0]
        # 绘制轨迹线（如有）
        if lines is not None and len(lines) > 0:
            x, y, z = lines[:, 0], lines[:, 1], lines[:, 2]
            self.ax.plot(x, y, z)
            # 可选：自动缩放视角
            # self.ax.relim()
            # self.ax.autoscale_view(True, True, True)

        # 绘制点云
        if num != 0:
            step = num // 1000 if num > 1000 else 1  # 若点数过多则稀疏采样
            points, colors = points[::step], colors[::step]
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255)
        plt.show(block=False)  # 非阻塞显示
        plt.pause(0.01)        # 短暂暂停以刷新窗口
        plt.cla()              # 清空坐标轴，便于下次刷新
