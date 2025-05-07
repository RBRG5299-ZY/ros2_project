import rclpy
from sensor_msgs.msg import Image,PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from rclpy.node import Node
from cv_bridge import CvBridge 
import numpy as np
import time
# if run this python script should remove .
# it is correct to add . when run in ros2
from .vis import Window_3d
from .utils import rgbd2xyz_from_points_list,create_pointcloud
from .odometry import Monocular_SLAM
class RGBD_subscriber(Node):
    def __init__(self,node_name):
        super().__init__(node_name)
        self.get_logger().info(f"start rgbd_sub:{node_name}")
        # show colored_pc_window
        # self.win = Window_3d()

        self.mo_slam = Monocular_SLAM()
            
        # cvbridge is to change format between cv2.Mat and sensor_msgs.Image
        self.bridge = CvBridge()
        self.clr = None
        self.depth = None
        self.dep_shape = None
        self.k = None
        self.dist_coeffs = None
        
        # create path
        self.path = Path()
        self.path.header.frame_id = 'map'

        # create five subscributer to get msg
        self.k_sub = self.create_subscription(Float32MultiArray,'K',self.callback_K,1)
        self.dist_coeffs_sub = self.create_subscription(Float32MultiArray,'dist_coeffs',self.callback_dist_coeffs,1)
        self.clr_sub = self.create_subscription(Image,'rgbd_color_topic',self.callback1,10)
        # self.depth_sub = self.create_subscription(Image,'rgbd_depth_topic',self.sub_callback,10)
        self.depth_ori_sub = self.create_subscription(Int32MultiArray,'rgbd_depth_ori_topic',self.callback2,10)
        self.depth_shape_sub = self.create_subscription(Int32MultiArray,'rgbd_depth_shape_topic',self.callback3,10)
        
        # create points pubilsher
        self.pointcloud1_pub = self.create_publisher(PointCloud2,'rgbd_to_xyz_points',10)
        self.pointcloud2_pub = self.create_publisher(PointCloud2,'map',10)
        
        # create path publisher
        self.path_pub = self.create_publisher(Path,'traj', 10)

    def callback_K(self,k):
        np_array = np.array(k.data, dtype=np.float32)
        self.k = np.reshape(np_array,(3,3))

    def callback_dist_coeffs(self,dist_coeffs):
        np_array = np.array(dist_coeffs.data, dtype=np.float32)
        self.dist_coeffs = np_array

    def callback1(self, clr):
        cv_image = self.bridge.imgmsg_to_cv2(clr, desired_encoding="bgr8")
        self.clr = cv_image
        self.process_messages()

    def callback2(self, depth):
        np_array = np.array(depth.data, dtype=np.int32)
        self.depth = np_array
        self.process_messages()
    
    def callback3(self, shape):
        np_array = np.array(shape.data, dtype=np.int32)
        self.dep_shape = np_array
        self.process_messages()

    def process_messages(self):
        if self.clr is not None and self.depth is not None and self.dep_shape is not None and self.k is not None:            
            depth_img = np.reshape(self.depth,self.dep_shape)
            depth_img = depth_img.astype(np.uint16)
            # print(depth_img.shape,depth_img.dtype)
            if self.mo_slam.k is None or self.mo_slam.dist_coeffs is None:
                self.mo_slam.set_k(self.k)
                self.mo_slam.set_dist_coeffs(self.dist_coeffs)

            self.mo_slam.extract_points(self.clr)
            position = self.mo_slam.pose[:,0]
            print('current pose {}'.format(position))
            fx,fy,cx,cy = self.k[0,0],self.k[1,1],self.k[0,2],self.k[1,2]
            
            # 恢复3d坐标
            t = time.time()
            points, colors = rgbd2xyz_from_points_list(self.mo_slam.map_points,self.clr, depth_img, float(fx), float(fy), float(cx), float(cy))
            print('compute 3d:{:4f}s'.format(time.time() - t))
            # print('ori pts',points)
            points = np.array([self.mo_slam.pose[:,0]+p for p in points])
            # print('new pts',points)

            t = time.time()
            pointscloud1 = create_pointcloud(points)
            pointscloud2 = create_pointcloud(self.mo_slam.map_colors)
            print('create pointscloud: {:4f} s'.format(time.time()-t))

            # 展示3d坐标
            # t = time.time()
            # self.win.update_win(np.array(points),np.array(colors),np.array(self.mo_slam.trajectory))
            # print('show 3d: {:4f}s'.format(time.time() - t))
            
            # 展示轨迹
            pose = PoseStamped()

            point = Point()
            point.x, point.y, point.z = float(position[0]),float(position[1]),float(position[2])
            pose.pose.position = point
            
            quat = Quaternion()
            quat.x,quat.y,quat.z,quat.w = 0.0,0.0,0.0,1.0
            pose.pose.orientation = quat
            
            self.path.poses.append(pose)
            self.path.header.stamp = self.get_clock().now().to_msg()
            
            self.path_pub.publish(self.path)
            
            self.pointcloud1_pub.publish(pointscloud1)
            self.pointcloud2_pub.publish(pointscloud2)  
            self.clr = None
            self.depth = None
            self.dep_shape = None


def main(args=None):
    rclpy.init(args=args)
    node = RGBD_subscriber(node_name="RGBD_subsvributer")  
    rclpy.spin(node) 
    rclpy.shutdown()

if __name__=='__main__':
    main()