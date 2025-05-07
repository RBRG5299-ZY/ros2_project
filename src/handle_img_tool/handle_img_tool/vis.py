
import matplotlib.pyplot as plt
class Window_3d:
    def __init__(self, width=6.4, height=4.8, points_num=1000):
        super().__init__()
        self.point_num = points_num
        self.fig = plt.figure(figsize=(width,height))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update_win(self,points,colors,lines=None):
        num = points.shape[0]
        if lines is not None and len(lines) > 0:
            x, y, z = lines[:, 0], lines[:, 1], lines[:, 2]
            self.ax.plot(x, y, z)
            # self.ax.relim()
            # self.ax.autoscale_view(True, True, True)

        if num != 0:
            step = num//1000 if num > 1000 else 1
            points,colors = points[::step],colors[::step]
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255)
        plt.show(block=False)
        plt.pause(0.01)
        plt.cla()
