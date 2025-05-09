import cv2
import numpy as np
import time
from .utils import print_red_text

class Monocular_SLAM:
    def __init__(self, k=None, dist_coeffs=None):
        """
        单目SLAM初始化。
        参数:
            k: 相机内参矩阵
            dist_coeffs: 相机畸变参数
        """
        super().__init__()
        self.k = k
        self.dist_coeffs = dist_coeffs
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.orb = cv2.ORB_create()

        # 初始化相机位姿
        self.pose = np.zeros((3, 1))  # 相机位移（平移向量）
        self.rotation_matrix = np.identity(3)  # 相机旋转矩阵
        self.trajectory = []  # 相机轨迹

        # 地图相关
        self.map_points = []      # 3D地图点
        self.map_ref_frames = []  # 每个地图点对应的参考帧索引
        self.map_colors = []      # 地图点的颜色（用于可视化）

    def set_k(self, k):
        """设置相机内参矩阵。"""
        self.k = k

    def set_dist_coeffs(self, dist_coeffs):
        """设置相机畸变参数。"""
        self.dist_coeffs = dist_coeffs

    def extract_points(self, img):
        """
        提取特征点并进行相机运动估计与地图更新。
        参数:
            img: 输入彩色图像
        """
        # 图像预处理（灰度化、去畸变等）
        t = time.time()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        undistorted = cv2.undistort(gray, self.k, self.dist_coeffs)
        print('undistort:{:4f}s'.format(time.time() - t))

        # 特征提取
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
            self.trajectory.append(self.pose[:, 0].copy())

            # 更新地图点
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
            print_red_text('show odometry:{:4f} s'.format(time.time() - t))

        # 更新前一帧的关键点和描述子
        self.prev_frame = undistorted
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

class Binocular_SLAM:
    def __init__(self):
        """
        双目SLAM初始化。
        包括特征提取器、匹配器、相机参数、轨迹等。
        """
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

        # 三维重建相关
        self.points3D = []
        self.points2D_left = []
        self.points2D_right = []
        self.map_pts = []
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.set_k()
        self.lpath_no_opt = []
        self.lpath_ga = []
        self.lpath_pso = []

    def set_k(self):
        """设置左右相机的内参矩阵。"""
        self.lk = np.array([[748.3, 0, 490.6],
                           [0, 748.3, 506.3],
                           [0, 0, 1]])
        self.rk = np.array([[743.7, 0, 495.3],
                        [0, 743.4, 514.7],
                        [0, 0, 1]])

    def update(self, frames):
        temps = time.time()  # <--- 加这一行
        left_img, right_img = frames
        # 只在内部变量中使用灰度，不要覆盖 left_img/right_img
        if left_img.ndim == 3 and left_img.shape[2] == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img.copy()
        if right_img.ndim == 3 and right_img.shape[2] == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img.copy()
        # 后续特征提取用 left_gray/right_gray，不要覆盖原始 left_img/right_img
        lkps, ldes = self.orb.detectAndCompute(left_gray, None)
        rkps, rdes = self.orb.detectAndCompute(right_gray, None)
        if self.prev_lkps is not None:
            # 左目特征匹配与位姿估计
            lmatches = self.bf.match(self.prev_ldes, ldes)
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

            # 右目特征匹配与位姿估计
            rmatches = self.bf.match(self.prev_rdes, rdes)
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

        # 更新前一帧特征
        self.prev_lkps = lkps
        self.prev_rkps = rkps
        self.prev_ldes = ldes
        self.prev_rdes = rdes
        print('pose estimation', time.time() - temps)
        temps = time.time()
        self.compute_xyz_camera()
        print('3d recove', time.time() - temps)
        self.recove()

    def compute_xyz_camera(self, method='none'):
        """
        双目三维点重建，并根据不同优化方法保存轨迹。
        参数:
            method: 'none'（无优化）、'ga'（遗传算法）、'pso'（粒子群算法）
        """
        # 相机参数（内参矩阵，带齐次列）
        K_left = np.array([[748.3, 0, 490.6, 0],
                           [0, 748.3, 506.3, 0],
                           [0, 0, 1, 0]])
        K_right = np.array([[743.7, 0, 495.3, 1],
                            [0, 743.4, 514.7, 0],
                            [0, 0, 1, 0]])

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

            p1_left = np.hstack((p1_left, [1]))
            p2_right = np.hstack((p2_right, [1]))

            # 三角化
            point_4d = cv2.triangulatePoints(K_left, K_right, p1_left[:2], p2_right[:2])
            point_3d = point_4d[:3] / point_4d[3]

            points3D.append(point_3d)
            points2D_left.append(p1_left)
            points2D_right.append(p2_right)

        # 根据优化方法保存轨迹
        if method == 'none':
            self.lpath_no_opt.append(self.lpose[:3, 3].copy())

        elif method == 'ga':
            # 使用GA优化3D点
            points3D_ga = ga_optimize_3d_points(points3D.copy(), points2D_left, points2D_right, K_left[:3, :3], K_right[:3, :3])

            # 用GA优化后的3D点+2D点，通过PnP重新估计位姿
            obj_pts = np.array(points3D_ga).reshape(-1, 3)
            img_pts = np.array(points2D_left).reshape(-1, 3)  # 这里原来是(-1,2)，应为(-1,2)或(-1,3)取决于你的2D点格式
            if obj_pts.shape[0] >= 4 and img_pts.shape[0] == obj_pts.shape[0]:
                img_pts = img_pts[:, :2]  # 只取前两个分量
                success, rvec, tvec, _ = cv2.solvePnPRansac(obj_pts, img_pts, self.lk, None)
                if success:
                    R_opt, _ = cv2.Rodrigues(rvec)
                    t_opt = tvec.squeeze()
                    self.lpath_ga.append(t_opt.copy())
                else:
                    self.lpath_ga.append(self.lpose[:3, 3].copy())
            else:
                self.lpath_ga.append(self.lpose[:3, 3].copy())

        elif method == 'pso':
            # 使用PSO优化3D点
            points3D_pso = pso_optimize_3d_points(points3D.copy(), points2D_left, points2D_right, K_left[:3, :3], K_right[:3, :3])

            # 用PSO优化后的3D点+2D点，通过PnP重新估计位姿
            obj_pts = np.array(points3D_pso).reshape(-1, 3)
            img_pts = np.array(points2D_left).reshape(-1, 3)
            if obj_pts.shape[0] >= 4 and img_pts.shape[0] == obj_pts.shape[0]:
                img_pts = img_pts[:, :2]
                success, rvec, tvec, _ = cv2.solvePnPRansac(obj_pts, img_pts, self.lk, None)
                if success:
                    R_opt, _ = cv2.Rodrigues(rvec)
                    t_opt = tvec.squeeze()
                    self.lpath_pso.append(t_opt.copy())
                else:
                    self.lpath_pso.append(self.lpose[:3, 3].copy())
            else:
                self.lpath_pso.append(self.lpose[:3, 3].copy())

        self.points2D_left = points2D_left
        self.points2D_right = points2D_right

    def recove(self):
        """
        将三维点从相机坐标系转换到世界坐标系下，并保存到map_pts。
        """
        for pt in self.points3D:
            P_c = np.vstack((pt, 1))  # 齐次坐标
            # 构造相机到世界坐标系的变换矩阵 T
            T = np.eye(4)
            T[:3, :3] = self.lpose[:3, :3]
            T[:3, 3] = self.lpose[:3, 3]
            # 坐标变换
            P_w = np.dot(T, P_c)
            self.map_pts.append(P_w[:-1])

# 遗传算法优化三维点
import random

def ga_optimize_3d_points(points3D, points2D_left, points2D_right, K_left, K_right, generations=50, pop_size=30, mutation_rate=0.1):
    """
    使用遗传算法优化三维点，使其投影误差最小。
    参数:
        points3D: 初始三维点列表
        points2D_left, points2D_right: 对应的左右图像特征点
        K_left, K_right: 左右相机内参
        generations: 迭代次数
        pop_size: 种群规模
        mutation_rate: 变异概率
    返回:
        最优三维点集
    """
    def project_point(P, K):
        # 简单投影函数，将三维点投影到像素平面
        # 直接用 3×3 内参矩阵乘以 3D 点
        P_vec = np.ravel(P)           # 确保形状 (3,)
        p = K @ P_vec                  # 3×3 × (3,) -> (3,)
        return p[:2] / p[2]

    def fitness(individual):
        # 计算个体适应度（投影误差的负数，误差越小越好）
        error = 0
        for i, P in enumerate(individual):
            proj_left = project_point(P, K_left)
            proj_right = project_point(P, K_right)
            error += np.linalg.norm(proj_left - points2D_left[i][:2])
            error += np.linalg.norm(proj_right - points2D_right[i][:2])
        return -error

    # 初始化种群
    population = [np.array(points3D) + np.random.randn(*np.array(points3D).shape)*0.01 for _ in range(pop_size)]

    for gen in range(generations):
        # 计算适应度
        fitnesses = [fitness(ind) for ind in population]
        # 选择适应度较高的一半个体
        selected = [population[i] for i in np.argsort(fitnesses)[-pop_size//2:]]
        # 交叉生成新个体
        children = []
        while len(children) < pop_size:
            p1, p2 = random.sample(selected, 2)
            cross_point = random.randint(1, len(p1)-1)
            child = np.vstack((p1[:cross_point], p2[cross_point:]))
            children.append(child)
        # 变异
        for child in children:
            if random.random() < mutation_rate:
                idx = random.randint(0, len(child)-1)
                # 生成与 child[idx] 相同形状的噪声
                noise = np.random.randn(*child[idx].shape) * 0.01
                child[idx] += noise
        population = children

    # 返回最优个体
    best_idx = np.argmax([fitness(ind) for ind in population])
    return population[best_idx]

def pso_optimize_3d_points(points3D, points2D_left, points2D_right, K_left, K_right, 
                           num_particles=30, max_iter=50, w=0.7, c1=1.5, c2=1.5):
    """
    粒子群优化三维点，使其投影误差最小。
    参数:
        points3D: 初始三维点列表
        points2D_left, points2D_right: 对应的左右图像特征点
        K_left, K_right: 左右相机内参
        num_particles: 粒子数量
        max_iter: 最大迭代次数
        w: 惯性权重
        c1, c2: 学习因子
    返回:
        最优三维点集
    """
    points3D = np.array(points3D)
    dim = points3D.shape  # (N, 3)
    N = dim[0]

    def project_point(P, K):
        # 投影函数
        P_vec = np.ravel(P)
        p = K @ P_vec
        return p[:2] / p[2]

    def fitness(candidate):
        # 适应度函数：投影误差总和
        error = 0
        for i, P in enumerate(candidate):
            proj_left = project_point(P, K_left)
            proj_right = project_point(P, K_right)
            error += np.linalg.norm(proj_left - points2D_left[i][:2])
            error += np.linalg.norm(proj_right - points2D_right[i][:2])
        return error

    # 初始化粒子群
    particles = [points3D + np.random.randn(*dim)*0.01 for _ in range(num_particles)]
    velocities = [np.random.randn(*dim)*0.001 for _ in range(num_particles)]
    personal_best = [p.copy() for p in particles]
    personal_best_scores = [fitness(p) for p in particles]
    global_best = personal_best[np.argmin(personal_best_scores)].copy()
    global_best_score = min(personal_best_scores)

    for it in range(max_iter):
        for i in range(num_particles):
            # 更新速度
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best[i] - particles[i]) +
                             c2 * r2 * (global_best - particles[i]))
            # 更新位置
            particles[i] += velocities[i]
            # 计算适应度
            score = fitness(particles[i])
            if score < personal_best_scores[i]:
                personal_best[i] = particles[i].copy()
                personal_best_scores[i] = score
                if score < global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = score
    return global_best
# filepath: /home/rbrg5299/Desktop/homework/slam_deployment/ros2_project/src/handle_img_tool/handle_img_tool/odometry.py