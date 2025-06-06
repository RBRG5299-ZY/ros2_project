import numpy as np
import os

class TUM_Bino_dataset:
    def __init__(self,data_name='skate-easy-vi_gt_data'):
        super().__init__()
  
        data_dir = '/home/rbrg5299/dataset/binocular/TUM/'
        img_dir = data_dir + data_name+'/'
        self.img_dir = img_dir
        if not os.path.exists(self.img_dir+'left.txt'):
            self.gen_imgs_txt()
        l_list_txt = open(img_dir + 'left.txt', 'r')
        self.l_list = sorted(l_list_txt.read().split('\n')[:-1],key = lambda x :int(x.split('/')[-1].split('.')[0]))

        r_list_txt = open(img_dir + 'right.txt', 'r')
        self.r_list = sorted(r_list_txt.read().split('\n')[:-1],key = lambda x :int(x.split('/')[-1].split('.')[0]))
        # 相机内参
        f = open(img_dir + 'left_camera.txt', 'r')
        # print([l.split(' ')[1] for l in f.read().split('\n')[:-1]])
        fx, fy, cx, cy, d0, d1, d2, d3 = [l.split(' ')[1] for l in f.read().split('\n')[:-1]]
        f.close()
        # 相机内参矩阵
        self.l_K = np.array([[float(fx), 0, float(cx)],
                    [0, float(fy), float(cy)],
                    [0, 0, 1]])
        # 畸变系数
        self.l_dist_coeffs = np.array([float(d0), float(d1), float(d2), float(d3), 0.0])
        
        # 相机内参
        f = open(img_dir + 'right_camera.txt', 'r')
        fx, fy, cx, cy, d0, d1, d2, d3 = [l.split(' ')[1] for l in f.read().split('\n')[:-1]]
        f.close()
        # 相机内参矩阵
        self.r_K = np.array([[float(fx), 0, float(cx)],
                    [0, float(fy), float(cy)],
                    [0, 0, 1]])
        # 畸变系数
        self.r_dist_coeffs = np.array([float(d0), float(d1), float(d2), float(d3), 0.0])
        
        self.path_list = list(zip(self.l_list, self.r_list))

    def gen_imgs_txt(self):
        left = open(self.img_dir+'left.txt','w')
        right = open(self.img_dir+'right.txt','w')

        for f_n in os.listdir(self.img_dir+'left_images/'):
            left.write(self.img_dir+'left_images/'+f_n+'\n')
        left.close()
        for f_n in os.listdir(self.img_dir+'right_images/'):
            right.write(self.img_dir+'right_images/'+f_n+'\n')
        right.close()
        
class TUM_RGBD_dataset:
    def __init__(self,data_name='rgbd_dataset_freiburg1_xyz'):
        super().__init__()
  
        data_dir = '/home/rbrg5299/dataset/rgbd/TUM/'
        img_dir = data_dir + data_name+'/'

        rgb_list_txt = open(img_dir + 'rgb.txt', 'r')
        rgb_list = rgb_list_txt.read().split('\n')[3:-1]

        depth_list_txt = open(img_dir + 'depth.txt', 'r')
        depth_list = depth_list_txt.read().split('\n')[3:-1]

        # 相机内参
        f = open(data_dir + 'camera.txt', 'r')
        fx, fy, cx, cy, d0, d1, d2, d3, d4 = f.read().split('\n')[1].split(' ')[1:]

        # 相机内参矩阵
        self.K = np.array([[float(fx), 0, float(cx)],
                    [0, float(fy), float(cy)],
                    [0, 0, 1]])

        # 畸变系数
        self.dist_coeffs = np.array([float(d0), float(d1), float(d2), float(d3), float(d4)])
        
        self.img_dir = img_dir
        self.rgb_list = [img_dir+i.split(' ')[1] for i in rgb_list]
        self.depth_list = [img_dir+i.split(' ')[1] for i in depth_list]
        self.path_list = list(zip(self.rgb_list, self.depth_list))

class TUM_MON_dataset:
    def __init__(self,data_name='seq1'):
        super().__init__()
        data_dir = '/home/rbrg5299/dataset/monocular/TUM/'+data_name
        img_dir = data_dir + '/images/'
        cam = open(data_dir+'/camera.txt','r').read().split('\n')[:-1]
        print(cam)
        cx,cy,fx,fy,m = cam[0].split('\t')
        w,h = cam[1].split(' ')
        print('w',w,'h',h)
        # k1,k2,p1,p2,k3 = cam[2].split(' ')
        self.K = np.array([[float(fx), 0, float(cx)],
                    [0, float(fy), float(cy)],
                    [0, 0, float(m)]])
        # dist_coeffs = np.array([k1, k2, p1, p2])

        img_list = open(data_dir+'/times.txt','r').read().split('\n')[:-1]
        num_frames = len(img_list)
        num_image = len(os.listdir(img_dir))
        print('num_time',num_frames,'num_image',num_image)
        assert num_frames == num_image,'times.txt length not comparable images number'

        self.img_dir = img_dir
        self.rgb_list = [img_dir+'{}.jpg'.format(i.split(' ')[0]) for i in img_list]

if __name__ == '__main__':
    pass
