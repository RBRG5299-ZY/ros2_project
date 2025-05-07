from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray,Float32MultiArray
from cv_bridge import CvBridge
import cv2
import rclpy
from .TUM_datasets import TUM_Bino_dataset
class Bino_publisher(Node):
    def __init__(self,node_name,dataset):
        super().__init__(node_name)
        self.get_logger().info(f'start bino_pub{node_name}')
        self.dataset = dataset
        self.imgs_num = len(self.dataset.path_list)
        self.get_logger().info(f'read {self.imgs_num} imgs')
        
        self.lK_pub = self.create_publisher(Float32MultiArray,'lK',1)
        self.rK_pub = self.create_publisher(Float32MultiArray,'rK',1)
        self.left_pub= self.create_publisher(Image,'left_img',10)
        self.right_pub= self.create_publisher(Image,'right_img',10)
        self.timer = self.create_timer(0.5,self.timer_callback)
        self.count = 0

        self.bridge = CvBridge()
    def timer_callback(self):
        lk  = self.dataset.l_K.flatten().tolist()
        # print(lk)
        k_msg = Float32MultiArray()
        k_msg.data = lk
        
        self.lK_pub.publish(k_msg)
        
        rk  = self.dataset.r_K.flatten().tolist()
        k_msg = Float32MultiArray()
        k_msg.data = rk
        self.rK_pub.publish(k_msg)

        l_path,r_path = self.dataset.path_list[self.count]
        l_img = cv2.imread(l_path)
        r_img = cv2.imread(r_path)
        self.left_pub.publish(self.bridge.cv2_to_imgmsg(l_img,'bgr8'))
        self.right_pub.publish(self.bridge.cv2_to_imgmsg(r_img,'bgr8'))
        print('number:',self.count)
        self.count +=1
        if self.count >= self.imgs_num:
            self.count = 0
def main(args=None):
    rclpy.init(args=args)
    tum_bino = TUM_Bino_dataset()
    node = Bino_publisher(node_name='bino_publisher',dataset=tum_bino)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
