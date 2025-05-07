import cv2
import numpy as np
import time
from .utils import print_red_text
class Monocular_SLAM:
    def __init__(self,k=None,dist_coeffs=None):
        super().__init__()
        self.k = k
        self.dist_coeffs = dist_coeffs
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.orb = cv2.ORB_create()

        # 初始化相机位姿
        self.pose = np.zeros((3, 1))  # 相机位移（平移）
        self.rotation_matrix = np.identity(3)  # 相机旋转
        self.trajectory = []

        # 创建地图
        self.map_points = []  # 3D地图点
        self.map_ref_frames = []  # 每个地图点对应的参考帧索引
        self.map_colors = []  # 地图点的颜色（用于可视化）

    def set_k(self,k):
        self.k = k

    def set_dist_coeffs(self,dist_coeffs):
        self.dist_coeffs = dist_coeffs
    
    def extract_points(self,img):
        # 图像预处理（灰度化、去畸变等）
        # 去畸变
        t = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        undistorted = cv2.undistort(gray, self.k, self.dist_coeffs)
        print('undistort:{:4f}s'.format(time.time() - t))

        # 显示去畸变后的图像
        # pic = np.vstack((img,cv2.cvtColor(undistorted,cv2.COLOR_GRAY2BGR)))
        # cv2.imshow('Undistorted Image', pic)
        # cv2.waitKey(10)

        # 特征提取与匹配
        keypoints, descriptors = self.orb.detectAndCompute(undistorted, None)

        if self.prev_frame is not None:
            # 特征匹配
            t = time.time()
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(self.prev_descriptors, descriptors)
            print_red_text('match: {:4f}s'.format(time.time()-t))

            # 计算相机运动
            t = time.time()
            ref_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            E, mask = cv2.findEssentialMat(curr_pts, ref_pts, self.k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, T, mask = cv2.recoverPose(E, curr_pts, ref_pts, self.k, mask=mask)
            print_red_text('compute camera movtion: {:4f}s'.format(time.time() - t))

            # 更新相机位姿

            self.pose += self.rotation_matrix.dot(T)
            self.rotation_matrix = R.dot(self.rotation_matrix)
            # print(self.pose)
            self.trajectory.append(self.pose[:,0].copy())

            # 更新地图
            t = time.time()
            for i, match in enumerate(matches):
                if mask[i] == 1:
                    map_idx = match.trainIdx
                    if map_idx not in self.map_points:
                        self.map_points.append(keypoints[map_idx].pt)
                        self.map_ref_frames.append(len(self.trajectory) - 1)
                        self.map_colors.append(img[int(keypoints[map_idx].pt[1]), int(keypoints[map_idx].pt[0])].tolist())
            print_red_text('update map: {:4f}s'.format(time.time() - t))

            t = time.time()
            print_red_text('map points:{}'.format(len(self.map_colors)))
            # updat_3d_lines(self.ax,np.asarray(self.trajectory))
            print_red_text('show odometry:{:4f} s'.format(time.time() - t))

        # 更新前一帧的关键点和描述子
        self.prev_frame = undistorted
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
 
class Binocular_SLAM:
    def __init__(self):
        super().__init__()
        self.prev_limg = None
        self.prev_rimg = None

        self.prev_ldes = None
        self.prev_rdes = None

        self.prev_lkps = None
        self.prev_rkps = None

        self.lk = None
        self.rk = None

        self.lpose = np.eye(4)
        self.rpose = np.eye(4)

        self.lpath = []
        self.rpath = []

        # 三维重建
        self.points3D = []
        self.points2D_left = []
        self.points2D_right = []
        self.map_pts = []
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.set_k()
    def set_k(self):
        self.lk = np.array([[748.3, 0, 490.6],
                           [0, 748.3, 506.3],
                           [0, 0, 1]])

        self.rk = np.array([[743.7, 0, 495.3],
                        [0, 743.4, 514.7],
                        [0, 0, 1]])

    def update(self,frames):
        left_img,right_img = frames
        temps = time.time()
        left_img = cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
        right_img = cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)

        lkps,ldes = self.orb.detectAndCompute(left_img, None)
        rkps,rdes = self.orb.detectAndCompute(right_img, None)
        if self.prev_lkps is not None:
            lmatches = self.bf.match(self.prev_ldes,ldes)
            N = 100
            lmatches = lmatches[:N]
            src_pts = np.float32([self.prev_lkps[m.queryIdx].pt for m in lmatches]).reshape(-1, 1, 2)
            dst_pts = np.float32([lkps[m.trainIdx].pt for m in lmatches]).reshape(-1, 1, 2)
            E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.lk, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            src_pts_filtered = np.round(src_pts).astype(np.int16).squeeze()
            dst_pts_filtered = np.round(dst_pts).astype(np.int16).squeeze()

            _, R, t, mask = cv2.recoverPose(E, dst_pts_filtered, src_pts_filtered, self.lk)

            self.lpose[:3, :3] = np.dot(R, self.lpose[:3, :3])
            self.lpose[:3, 3] += self.lpose[:3, :3].dot(t.squeeze())
            self.lpath.append(self.lpose[:3, 3].copy())

            rmatches = self.bf.match(self.prev_rdes,rdes)
            rmatches = rmatches[:N]
            src_pts = np.float32([self.prev_rkps[m.queryIdx].pt for m in rmatches]).reshape(-1, 1, 2)
            dst_pts = np.float32([rkps[m.trainIdx].pt for m in rmatches]).reshape(-1, 1, 2)

            E, mask = cv2.findEssentialMat(dst_pts, src_pts, self.rk, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            src_pts_filtered = np.round(src_pts).astype(np.int16).squeeze()
            dst_pts_filtered = np.round(dst_pts).astype(np.int16).squeeze()

            _, R, t, mask = cv2.recoverPose(E, dst_pts_filtered, src_pts_filtered, self.lk)

            self.rpose[:3, :3] = np.dot(R, self.rpose[:3, :3])
            self.rpose[:3, 3] += self.rpose[:3, :3].dot(t.squeeze())
            self.rpath.append(self.rpose[:3, 3].copy())

        self.prev_lkps = lkps
        self.prev_rkps = rkps
        self.prev_ldes = ldes
        self.prev_rdes = rdes
        print('pose estimation',time.time()-temps)
        temps = time.time()
        self.compute_xyz_camera()
        print('3d recove',time.time()-temps)
        self.recove()

    def compute_xyz_camera(self):
        # 相机参数（内参矩阵）
        K_left = np.array([[748.3, 0, 490.6,0],
                            [0, 748.3, 506.3,0],
                            [0, 0, 1,0]])

        K_right = np.array([[743.7, 0, 495.3,1],
                           [0, 743.4, 514.7,0],
                           [0, 0, 1,0]])

        # 特征点匹配
        matches = self.bf.match(self.prev_ldes, self.prev_rdes)
        matches = sorted(matches, key=lambda x: x.distance)

        N = 100
        good_matches = matches[:N]
        # 三维重建
        points3D = []
        points2D_left = []
        points2D_right = []
        for match in good_matches:
            idx1 = match.queryIdx
            idx2 = match.trainIdx

            # 左右图像中的特征点
            p1_left = self.prev_lkps[idx1].pt
            p2_right = self.prev_rkps[idx2].pt

            # 对极几何约束
            # F, mask = cv2.findFundamentalMat(np.array([p1_left]), np.array([p2_right]), cv2.FM_LMEDS)
            # print(p1_left,p2_right)
            p1_left = np.hstack((p1_left, 1))
            p2_right = np.hstack((p2_right, 1))

            # 三角化
            point_4d = cv2.triangulatePoints(K_left, K_right, p1_left[:2], p2_right[:2])
            point_3d = point_4d[:3] / point_4d[3]

            points3D.append(point_3d)
            points2D_left.append(p1_left)
            points2D_right.append(p2_right)

        self.points3D = points3D
        self.points2D_left = points2D_left
        self.points2D_right = points2D_right

    def recove(self):
        for pt in self.points3D:
            P_c = np.vstack((pt,1))

            # 构造相机到世界坐标系的变换矩阵 T，假设世界坐标系原点位置为 (0, 0, 0)
            T = np.eye(4)  # 单位矩阵
            T[:3, :3] = self.lpose[:3,:3]
            T[:3, 3] = self.lpose[:3,3]

            # 将相机位姿与相机坐标系下的点相乘，得到世界坐标系下的三维坐标
            P_w = np.dot(T, P_c)
            self.map_pts.append(P_w[:-1])