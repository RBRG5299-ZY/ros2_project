from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField
import numpy as np
from std_msgs.msg import Header
import cv2
def print_red_text(text):
    print("\033[31m" + text + "\033[0m")

def rgbd2xyz(img,depth,fx,fy,cx,cy,step=4):
    factor = 5000
    points = []
    colors = []
    for v in range(depth.shape[0])[::step]:
        for u in range(depth.shape[1])[::step]:
            z = depth[v,u]/factor*1000
            x = (u-cx)*z/fx
            y = (v-cy)*z/fy
            color = img[v,u]
            # print((x,y,z))
            points.append((x,y,z))
            colors.append(color)
    points = np.asarray(points)
    colors = np.asarray(colors)
    return points,colors

def rgbd2xyz_from_points_list(pts,img,depth,fx,fy,cx,cy,step=1):
    factor = 5000
    points = []
    colors = []
    num = len(pts)
    # print(num)
    if num == 0: return [], []
    step = num// 500 if num >500 else step
    for u,v in pts[::step]:
        v,u = int(u), int(v)
        z = depth[u, v] / factor * 1000
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        color = img[u, v]
        # print((x,y,z))
        points.append((x, y, z))
        colors.append(color)
    points = np.asarray(points)
    colors = np.asarray(colors)
    return points, colors

def create_pointcloud(points,num = 1000):
    l = len(points)
    step = l//num if l >1000 else 1
    points= points[::step]
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
        points_data.append([x,y,z])

    # Create the PointCloud2
    points_data = np.array(points_data).reshape(3,-1).T
    pc2_msg = point_cloud2.create_cloud(header,fields,points_data)

    return pc2_msg

def draw_bino(limg,lpts,rimg,rpts):
    if len(lpts) == len(rpts):
        for (p1,p2) in zip(lpts,rpts):
            p1,p2 = p1.pt,p2.pt
            p1 = (int(p1[0]),int(p1[1]))
            p2 = (int(p2[0]),int(p2[1]))
            c = tuple(np.random.randint(0,255,3).tolist())
            cv2.circle(limg,p1,5,c,-1)
            cv2.circle(rimg,p2,5,c,-1)
    return limg,rimg