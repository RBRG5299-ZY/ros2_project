import rclpy
from sensor_msgs.msg import Image,PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from rclpy.node import Node
from cv_bridge import CvBridge 
import numpy as np
import time
import cv2
# if run this python script should remove .
# it is correct to add . when run in ros2

class Mon_subscriber(Node):
    def __init__(self,node_name):
        super().__init__(node_name)
        self.get_logger().info(f"start mon_sub:{node_name}")
        self.position = (50, 100)  # 文字的起始位置
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        self.font_scale = 2  # 字体缩放比例
        self.color = (0, 255, 0)  # 文字颜色
        self.thickness = 2  # 文字粗细

            
        # cvbridge is to change format between cv2.Mat and sensor_msgs.Image
        self.bridge = CvBridge()

        self.k = None
        self.prev_frame = None
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.rotation_matrix = np.eye(3)
        self.pose = np.zeros((3,1))
        # create path
        self.path = Path()
        self.path.header.frame_id = 'map'

        # 使用 Brute-Force 匹配器进行特征点匹配
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ORB = cv2.ORB_create()
        self.MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # create path
        self.path = Path()
        self.path.header.frame_id = 'map'

        # create two subscributer to get msg
        self.k_sub = self.create_subscription(Float32MultiArray,'K',self.callback_K,1)
        self.clr_sub = self.create_subscription(Image,'mon_img_topic',self.callback1,10)
        
        # create path&match_result publisher
        self.path_pub = self.create_publisher(Path,'traj', 10)
        self.img_pub = self.create_publisher(Image,'match_img', 10)

    def callback_K(self,k):
        np_array = np.array(k.data, dtype=np.float32)
        self.k = np.reshape(np_array,(3,3))

    
    def callback1(self, clr):
        print('img received')
        cv_image = self.bridge.imgmsg_to_cv2(clr, desired_encoding="bgr8")
        curr_frame = cv_image
        curr_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)    
        curr_keypoints, curr_descriptors = self.ORB.detectAndCompute(curr_gray, None)

        if self.prev_frame is not None : 
            # print(depth_img.shape,depth_img.dtype)
            t_stamp = time.time()
            self.matches = self.MATCHER.match(self.prev_descriptors, curr_descriptors)
            print('match:{}'.format(time.time()-t_stamp))

            # 根据匹配结果计算相机运动
            t_stamp = time.time()
            src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # print(mask)
            src_pts_filtered = src_pts[mask.ravel() == 1]
            dst_pts_filtered = dst_pts[mask.ravel() == 1]
            src_pts_filtered = np.round(src_pts_filtered).astype(np.int16).squeeze()
            dst_pts_filtered = np.round(dst_pts_filtered).astype(np.int16).squeeze()
            _, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, self.k)
            print('recover pose:{}'.format(time.time()-t_stamp))

            # print(mask)
            # 更新相机位姿
            self.pose += self.rotation_matrix.dot(t)
            self.rotation_matrix = R.dot(self.rotation_matrix)
            # print(pose[:,0])
            print('publish show img')
            # 绘制匹配点
            matches_num = len(src_pts_filtered)
            print('matches num:',matches_num)
            colors = np.random.randint(0, 256, (matches_num, 3), dtype=np.uint8)
            p_f,c_f = self.prev_frame.copy(),curr_frame.copy()
            for i,(p1,p2) in enumerate(zip(src_pts_filtered,dst_pts_filtered)):
                clr = colors[i]
                clr = clr.tolist()
                cv2.circle(p_f,tuple(p1),4,color=tuple(clr),thickness=-1)
                cv2.putText(p_f, 'prev frame', self.position, self.font, self.font_scale, self.color, self.thickness)
                cv2.circle(c_f,tuple(p2),4,color=tuple(clr),thickness=-1)
                cv2.putText(c_f, 'curr frame', self.position, self.font, self.font_scale, self.color, self.thickness)
            re = np.vstack((p_f,c_f))
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(re,'bgr8'))
        # 展示轨迹
        pose = PoseStamped()

        point = Point()
        point.x, point.y, point.z = float(self.pose[0]),float(self.pose[1]),float(self.pose[2])
        pose.pose.position = point
        
        quat = Quaternion()
        quat.x,quat.y,quat.z,quat.w = 0.0,0.0,0.0,1.0
        pose.pose.orientation = quat
        
        self.path.poses.append(pose)
        self.path.header.stamp = self.get_clock().now().to_msg()
        print('update path')
        self.path_pub.publish(self.path)
            
        self.prev_frame = curr_frame
        self.prev_gray = curr_gray
        self.prev_descriptors = curr_descriptors
        self.prev_keypoints = curr_keypoints

def main(args=None):
    rclpy.init(args=args)
    node = Mon_subscriber(node_name="Mon_subscributer")  
    rclpy.spin(node) 
    rclpy.shutdown()

if __name__=='__main__':
    main()