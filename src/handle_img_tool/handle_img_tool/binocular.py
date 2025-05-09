import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image,PointCloud2
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import Point,Quaternion,PoseStamped
from .odometry import Binocular_SLAM
from .utils import draw_bino,create_pointcloud

class Binocular(Node):
    def __init__(self,node_name):
        super().__init__(node_name=node_name)
        self.get_logger().info(f"start bino_sub:{node_name}")
        
        # 初始化特征匹配器和ORB特征提取器
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ORB = cv2.ORB_create()
        self.MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bridge = CvBridge()  # ROS图像消息与OpenCV图像转换桥
        
        # 相机内参与图像缓存
        self.lk = None  # 左相机内参
        self.rk = None  # 右相机内参
        self.limg = None  # 左图像
        self.rimg = None  # 右图像

        # 订阅左、右相机内参和图像话题
        self.lk_sub = self.create_subscription(Float32MultiArray,'lK',self.callback_lK,1)
        self.rk_sub = self.create_subscription(Float32MultiArray,'rK',self.callback_rK,1)
        self.left_sub = self.create_subscription(Image,'left_img',self.callback_limg,10)
        self.right_sub = self.create_subscription(Image,'right_img',self.callback_rimg,10)
        
        # 初始化SLAM系统
        self.bino_slam = Binocular_SLAM()
        self.path = Path()
        self.path.header.frame_id = 'map'  # 路径参考坐标系

        # 发布处理结果的话题
        self.frame_match_pub = self.create_publisher(Image,'result',10)         # 匹配结果图像
        self.path_pub = self.create_publisher(Path,'lpath',10)                  # 路径
        self.map_pub = self.create_publisher(PointCloud2,'map',10)              # 点云地图
        self.path_pub_no_opt = self.create_publisher(Path, 'lpath_no_opt', 10)
        self.path_pub_ga = self.create_publisher(Path, 'lpath_ga', 10)
        self.path_pub_pso = self.create_publisher(Path, 'lpath_pso', 10)
        
    def callback_lK(self,k):
        # 左相机内参回调，将消息转换为3x3矩阵
        np_array = np.array(k.data, dtype=np.float32)
        self.lk = np.reshape(np_array,(3,3))
        # print(self.lk)

    def callback_rK(self,k):
        # 右相机内参回调，将消息转换为3x3矩阵
        np_array = np.array(k.data, dtype=np.float32)
        self.rk = np.reshape(np_array,(3,3))
        # print(self.rk)

    def callback_limg(self,img):
        # 左图像回调，将ROS图像消息转换为OpenCV格式
        cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        self.limg = cv_image
        self.process_frame()  # 处理帧

    def callback_rimg(self,img):
        # 右图像回调，将ROS图像消息转换为OpenCV格式
        cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        self.rimg = cv_image
        self.process_frame()  # 处理帧

    def process_frame(self):
        # 当左右图像和内参都准备好时，进行SLAM处理
        if self.limg is not None and self.rimg is not None and self.lk is not None and self.rk is not None:            
            self.bino_slam.update((self.limg,self.rimg))  # 更新SLAM状态

            # 计算三维点并保存三种优化方式下的轨迹
            self.bino_slam.compute_xyz_camera(method='none')
            self.bino_slam.compute_xyz_camera(method='ga')
            self.bino_slam.compute_xyz_camera(method='pso')

            # 发布三种轨迹
            self.publish_path(self.bino_slam.lpath_no_opt, self.path_pub_no_opt, 'no_opt')
            self.publish_path(self.bino_slam.lpath_ga, self.path_pub_ga, 'ga')
            self.publish_path(self.bino_slam.lpath_pso, self.path_pub_pso, 'pso')

            path1 = self.bino_slam.lpath                  # 左相机轨迹
            path2 = self.bino_slam.rpath                  # 右相机轨迹
            path1 = np.array(path1)
            path2 = np.array(path2)
            # 绘制特征点匹配结果
            l_img,r_img = draw_bino(self.limg,self.bino_slam.prev_lkps,self.rimg,self.bino_slam.prev_rkps)
            img = np.hstack((l_img,r_img))                # 拼接左右图像
            pose = PoseStamped()

            # 当前位姿
            point = Point()
            point.x, point.y, point.z = float(self.bino_slam.lpose[0,3]),float(self.bino_slam.lpose[1,3]),float(self.bino_slam.lpose[2,3])
            pose.pose.position = point
            
            # 姿态四元数（此处默认无旋转，仅作示例）
            quat = Quaternion()
            quat.x,quat.y,quat.z,quat.w = 0.0,0.0,0.0,1.0
            pose.pose.orientation = quat
            self.path.header.stamp = self.get_clock().now().to_msg()
            
            self.path.poses.append(pose)           # 添加到路径
            self.path_pub.publish(self.path)       # 发布路径
            
            pointscloud = create_pointcloud(self.bino_slam.map_pts)  # 生成点云
            self.map_pub.publish(pointscloud)      # 发布点云
            self.frame_match_pub.publish(self.bridge.cv2_to_imgmsg(img,'bgr8'))  # 发布匹配图像
            self.limg = None
            self.rimg = None

    def publish_path(self, path_list, publisher, label):
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped, Point, Quaternion
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for pt in path_list:
            pose = PoseStamped()
            pose.pose.position = Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
            pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            path_msg.poses.append(pose)
        publisher.publish(path_msg)

def main(args=None):
    # ROS2节点主入口
    rclpy.init(args=args)
    node = Binocular(node_name='Bino_slam_node')
    rclpy.spin(node) 
    rclpy.shutdown()

if __name__ == '__main__':
    main()