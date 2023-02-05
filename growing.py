import open3d as o3d
import numpy as np
from tqdm import tqdm
from collections import deque

#计算点云的曲率并排序
class RegionGrowing1:

    def __init__(self,cloud, 
                 theta_threshold=30,            # 法向量夹角阈值    
                 curvature_threshold=0.05,      # 曲率阈值  
                 neighbour_number=30,           # 邻域搜索点数    
                 point_neighbours=[],  ):       # 近邻点集合
        
        self.cure = None                                 # 存储每个点曲率的容器
        self.pcd = cloud
        self.theta_threshold = np.deg2rad(theta_threshold)
        self.curvature_threshold = curvature_threshold
        self.neighbour_number = neighbour_number
        self.point_neighbours = point_neighbours
        self.cosine_threshold = np.cos(self.theta_threshold)

    def first_seed(self):
        num_of_pts = len(self.pcd.points)         # 点云点的个数
        self.pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(self.neighbour_number))
        cov_mat = self.pcd.covariances            # 获取每个点的协方差矩阵
        self.cure = np.zeros(num_of_pts)          # 初始化存储每个点曲率的容器
        point_curvature_index = np.zeros((num_of_pts, 2))
        # 计算每个点的曲率
        for i_n in range(num_of_pts):
            eignvalue, _ = np.linalg.eig(cov_mat[i_n])  # SVD分解求特征值
            idx = eignvalue.argsort()[::-1]
            eignvalue = eignvalue[idx]
            self.cure[i_n] = eignvalue[2] / (eignvalue[0] + eignvalue[1] + eignvalue[2])
        
        self.cure = np.array(self.cure)

        return self.cure.argmin()
    
        # ------------------------------------近邻点搜索-------------------------------------
    def find_neighbour_points(self):
        number = len(self.pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        self.point_neighbours = np.zeros((number, self.neighbour_number))
        for ik in range(number):
            [_, idx, _] = kdtree.search_knn_vector_3d(self.pcd.points[ik], self.neighbour_number)  # K近邻搜索
            self.point_neighbours[ik, :] = idx

    # -----------------------------------判意点所属分类-----------------------------------
    def validate_points(self, point, nebor):
        is_seed = True
          # 法向量夹角（平滑）阈值

        curr_seed_normal = self.pcd.normals[point]       # 当前种子点的法向量
        seed_nebor_normal = self.pcd.normals[nebor]      # 种子点邻域点的法向量
        dot_normal = np.fabs(np.dot(seed_nebor_normal, curr_seed_normal))
        # 如果小于平滑阈值
        if dot_normal < self.cosine_threshold:
            return False, is_seed
        # 如果小于曲率阈值
        if self.cure[nebor] > self.curvature_threshold:
            is_seed = False

        return True, is_seed


