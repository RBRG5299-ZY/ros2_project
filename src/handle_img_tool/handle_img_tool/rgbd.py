import rclpy
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from rclpy.node import Node
from cv_bridge import CvBridge
import numpy as np
import time
from .vis import Window_3d
from .utils import rgbd2xyz_from_points_list, create_pointcloud
from .odometry import Monocular_SLAM, Binocular_SLAM  # import both SLAM classes

class RGBD_subscriber(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f"start rgbd_sub:{node_name}")

        # 使用双目SLAM以支持多种优化算法
        self.bi_slam = Binocular_SLAM()
        self.bridge = CvBridge()
        self.clr = None
        self.depth = None
        self.dep_shape = None
        self.k = None
        self.dist_coeffs = None

        # 三种优化方式的轨迹发布器
        self.path_pub_no_opt = self.create_publisher(Path, 'traj_no_opt', 10)
        self.path_pub_ga = self.create_publisher(Path, 'traj_ga', 10)
        self.path_pub_pso = self.create_publisher(Path, 'traj_pso', 10)

        # 点云发布器
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'pointcloud', 10)

        # 参数订阅器
        self.k_sub = self.create_subscription(Float32MultiArray, 'K', self.callback_K, 1)
        self.dist_coeffs_sub = self.create_subscription(Float32MultiArray, 'dist_coeffs', self.callback_dist_coeffs, 1)
        self.clr_sub = self.create_subscription(Image, 'rgbd_color_topic', self.callback_color, 10)
        self.depth_ori_sub = self.create_subscription(Int32MultiArray, 'rgbd_depth_ori_topic', self.callback_depth, 10)
        self.depth_shape_sub = self.create_subscription(Int32MultiArray, 'rgbd_depth_shape_topic', self.callback_shape, 10)

    def callback_K(self, msg):
        arr = np.array(msg.data, dtype=np.float32).reshape(3, 3)
        self.k = arr

    def callback_dist_coeffs(self, msg):
        self.dist_coeffs = np.array(msg.data, dtype=np.float32)

    def callback_color(self, msg):
        self.clr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._try_process()

    def callback_depth(self, msg):
        self.depth = np.array(msg.data, dtype=np.int32)
        self._try_process()

    def callback_shape(self, msg):
        self.dep_shape = np.array(msg.data, dtype=np.int32)
        self._try_process()

    def _try_process(self):
        # 避免对 numpy 数组使用 "in" 判断，改为 any(... is None)
        if any(v is None for v in (self.clr, self.depth, self.dep_shape, self.k)):
            return

        # 重塑深度图
        depth_img = self.depth.reshape(self.dep_shape).astype(np.uint16)

        # —— 核心修改 —— #
        # 不再把 depth_img 传给双目 SLAM，防止单通道图像在内部被误当 BGR 转灰度
        self.bi_slam.update((self.clr, self.clr.copy()))
        # —— 修改结束 —— #

        # 发布三种优化方式的路径
        for method, pub, path_list in zip(
                ['none', 'ga', 'pso'],
                [self.path_pub_no_opt, self.path_pub_ga, self.path_pub_pso],
                [self.bi_slam.lpath_no_opt, self.bi_slam.lpath_ga, self.bi_slam.lpath_pso]):

            # 计算当前优化方法的三维重建
            self.bi_slam.compute_xyz_camera(method)

            # 构建并发布 Path
            path_msg = Path()
            path_msg.header.frame_id = 'map'
            for p in path_list:
                pose = PoseStamped()
                pose.pose.position = Point(x=p[0], y=p[1], z=p[2])
                pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                path_msg.poses.append(pose)
            pub.publish(path_msg)

        # 将像素点列表转换成全局点云并发布
        # pts_uv = self.bi_slam.map_pts
        # fx, fy = self.k[0,0], self.k[1,1]
        # cx, cy = self.k[0,2], self.k[1,2]
        # points, colors = rgbd2xyz_from_points_list(pts_uv, self.clr, depth_img, fx, fy, cx, cy)
        # points_global = np.array([self.bi_slam.lpose[:3,3] + pt for pt in points])
        # pc2_msg = create_pointcloud(points_global)
        # self.pointcloud_pub.publish(pc2_msg)
        # 直接将 SLAM 生成的三维地图点发布为点云
        if len(self.bi_slam.map_pts) > 0:
            points = np.array(self.bi_slam.map_pts)  # 已经是世界坐标下的 (N,3) 点
            pc2_msg = create_pointcloud(points)
            self.pointcloud_pub.publish(pc2_msg)

        # 重置以便接收下一帧
        self.clr = self.depth = self.dep_shape = None



def main(args=None):
    rclpy.init(args=args)
    node = RGBD_subscriber('rgbd_subscriber')
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
