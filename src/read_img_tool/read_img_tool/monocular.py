import cv2
import rclpy
from rclpy.node import Node
# if run this python script should remove .
# it is correct to add . when run in ros2
from .TUM_datasets import TUM_MON_dataset
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from std_msgs.msg import Int32MultiArray,Float32MultiArray

class Mon_publisher(Node):
    def __init__(self,node_name,dataset):
        super().__init__(node_name)
        self.get_logger().info(f"start mon_pub:{node_name}")
        
        # load dataset 
        self.dataset = dataset
        self.imgs_num = len(self.dataset.rgb_list)
        self.get_logger().info(f'read {len(self.dataset.rgb_list)} color imgs')
        
        # create camera msg
        self.camera_k_pub = self.create_publisher(Float32MultiArray, "K", 1)
        
        # create color publisher, queue length = 10
        self.clr_publisher = self.create_publisher(Image, "mon_img_topic", 10)
        
        # save read index to publish
        self.count  = 0
        
        # create timer ,delay = 0.5s
        self.timer = self.create_timer(0.5,self.timer_callback)

        # cvbridge is to change format between cv2.Mat and sensor_msgs.Image
        self.bridge = CvBridge()
    def timer_callback(self):
        k = self.dataset.K.flatten().tolist()
        k_msg = Float32MultiArray()
        k_msg.data = k
        # print(k_msg)
        self.camera_k_pub.publish(k_msg)
        img_path = self.dataset.rgb_list[self.count]
        img = cv2.imread(img_path)
        print('number:',self.count)

        print(img.shape)
        print(img.dtype)

        self.clr_publisher.publish(self.bridge.cv2_to_imgmsg(img,'bgr8'))
        self.count +=1
        if self.count >= self.imgs_num:
            self.count = 0

def main(args=None): 
    tum_dataset = TUM_MON_dataset()
    rclpy.init(args=args)
    node = Mon_publisher(node_name="mon_publisher",dataset=tum_dataset)  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
