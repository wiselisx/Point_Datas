'''
Author: 夜间清风 1874032283@qq.com
Date: 2023-03-04 20:05:50
LastEditors: 夜间清风 1874032283@qq.com
LastEditTime: 2023-04-01 15:45:28
FilePath: \Point_Datas\sample\readdatas.py
Description: 软件代码的算法部分。
'''

import CSF
import numpy as np
import open3d as o3d
from abc import ABC, abstractmethod


class ReadDatas:

    def __init__(self, path: str) -> None:
        self.path: str = path
        self.pcd = self.read_point_cloud()
        self.pcd_np = np.asarray(self.pcd.points)
        self.theta_threshold = 20
        self.cosine_threshold = np.cos(np.deg2rad(self.theta_threshold))
        self.curvature_threshold = 0.035

    def read_point_cloud(self) -> 'PointCloud':
        '''
        description: 读取点云文件
        '''
        pcd = o3d.io.read_point_cloud(self.path)
        return pcd


class BaseAlgorithm(ABC):

    def __init__(self) -> None:
        self.data: None = None

    def set_data(self, data) -> None:
        self.data = data

    # def process_data(self) -> None:
    #     self.preprocess_data()
    #     self._process_data()
    #     self.postprocess_data()

    @abstractmethod
    def _process_data(self) -> None:
        pass

    def fit_plane(self, points):
        '''
        description: 利用PCA拟合平面
        '''
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        u, _, _ = np.linalg.svd(centered.T, full_matrices=False, compute_uv=True)
        normal = u[:, -1]
        intercept = -np.dot(normal, centroid)
        return normal, intercept

    def np_to_o3d(self, points: 'ndarray'):
        name = ''
        name = o3d.geometry.PointCloud()
        name.points = o3d.utility.Vector3dVector(points)
        name.estimate_normals()

        return name

    def find_neighbour_points(self, clouds, neighbour_number=30):
        kdtree = o3d.geometry.KDTreeFlann(clouds)
        number = len(clouds.points)

        point_neighbours = np.zeros((number, neighbour_number))
        for ik in range(number):
            [_, idx, _] = kdtree.search_knn_vector_3d(clouds.points[ik], neighbour_number)  # K近邻搜索
            point_neighbours[ik, :] = idx

        return point_neighbours

    def colormap(self, value):
        colors = np.zeros([value.shape[0], 3])
        value_max = np.max(value)
        value_min = np.min(value)
        delta_c = abs(value_max - value_min) / (255 * 2)
        color_n = (value - value_min) / delta_c
        a = color_n <= 255
        b = color_n > 255
        colors[a, 1] = 1 - color_n[a] / 255
        colors[a, 2] = 1
        colors[b, 0] = (color_n[b] - 255) / 255
        colors[b, 2] = 1

        return colors

    def seed_select(self, cloud):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='Open3D', visible=True)
        vis.add_geometry(cloud)
        vis.run()
        seed = vis.get_picked_points()
        vis.destroy_window()

        return seed

    def find_index(self, list1, list2):
        # index = np.where((list1[:, None] == list2).all(-1))[1]
        # return index.tolist()
        index = []
        for i in list2:
            a = (list1 == i).all(axis=1)
            a = np.argwhere(a).ravel()
            if a:
                index.append(a[0])
        return index

    def paves_cutting(self, ground):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='Open3D', visible=True)
        vis.add_geometry(ground)
        vis.run()
        geometry = vis.get_cropped_geometry()
        vis.destroy_window()
        geometry = np.asarray(geometry.points)

        return geometry

    def find_no_paves(self, ground):
        a = True
        geometrys = np.zeros((1, 3))

        while a:
            geometry = self.paves_cutting(ground)
            geometrys = np.concatenate((geometrys, geometry), axis=0)
            input_1 = input('按Y或者y继续')
            if input_1 == 'y' or input_1 == 'Y':
                a = True
            else:
                a = False
        geometrys = np.delete(geometrys, [0], axis=0)

        return geometrys

    def curvature_calculation(self, clouds: 'o3d', neighbour_number=30):
        num_of_pts = len(clouds.points)  # 点云点的个数
        clouds.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(neighbour_number))
        cov_mat = clouds.covariances  # 获取每个点的协方差矩阵
        self.data.curvity = np.zeros(num_of_pts)  # 初始化存储每个点曲率的容器
        # point_curvature_index = np.zeros((num_of_pts, 2))
        # 计算每个点的曲率
        for i_n in range(num_of_pts):
            eignvalue, _ = np.linalg.eig(cov_mat[i_n])  # SVD分解求特征值
            idx = eignvalue.argsort()[::-1]
            eignvalue = eignvalue[idx]
            self.data.curvity[i_n] = eignvalue[2] / (eignvalue[0] + eignvalue[1] + eignvalue[2])

        self.data.curvity = np.array(self.data.curvity)

        return self.data.curvity

    def find_nearest_point(self, clouds1, clouds2, neighbour_number=1):
        neighbour_number = 1
        kdtree = o3d.geometry.KDTreeFlann(clouds1)
        number = len(clouds2.points)
        point_neighbours = np.zeros((number, neighbour_number))

        for ik in range(number):
            [_, idx, _] = kdtree.search_knn_vector_3d(clouds2.points[ik], neighbour_number)  # K近邻搜索
            point_neighbours[ik, :] = idx
        return point_neighbours

    def calculate_linear_density(self, clouds):
        nndist = clouds.compute_nearest_neighbor_distance()
        nndist = np.array(nndist)
        density = np.mean(nndist)
        return density

    def driving_path_generation(self, point, clouds):
        density = self.calculate_linear_density(clouds)
        driving_point = clouds.select_by_index(point)
        driving_point = np.array(driving_point.points)

        clouds = clouds.select_by_index(point, invert=True)
        driving_distance = np.array(
            [np.linalg.norm(driving_point[p] - driving_point[p + 1]) for p in range(len(driving_point) - 1)])
        driving_track = np.array([[0, 0, 0]])

        for i, d in enumerate(driving_distance):
            num = int(d / density)
            driving_track_x = np.linspace(driving_point[i, 0], driving_point[i + 1, 0], num)
            driving_track_y = np.linspace(driving_point[i, 1], driving_point[i + 1, 1], num)
            driving_track_z = np.linspace(driving_point[i, 2], driving_point[i + 1, 2], num)
            new_points = np.array([driving_track_x, driving_track_y, driving_track_z]).T
            driving_track = np.concatenate((driving_track, new_points), axis=0)

        driving_track = np.delete(driving_track, 0, axis=0)
        return driving_track

    def postprocess_data(self) -> None:
        # 公共的后处理逻辑
        pass


class GpfGroundExtractor(BaseAlgorithm):

    def _process_data(self, nlrp: int = 1000, thseeds: float = 0.1, ground_h: float = 0.15, iterations: int = 10):
        xyz = self.data.pcd_np
        xyz = xyz[np.lexsort(xyz.T)]
        elevation_nlrp_min_mean: float = np.average(xyz[:nlrp, 2])
        seed = xyz[xyz[:, 2] < thseeds + elevation_nlrp_min_mean]

        for i in range(iterations):
            normal, intercept = self.fit_plane(seed)
            h = (xyz[:, 0] * normal[0] + xyz[:, 1] * normal[1] + xyz[:, 2] * normal[2] + intercept) / np.linalg.norm(
                normal)
            seed = xyz[np.abs(h) < ground_h]

        self.data.ground = seed
        self.data.no_ground = xyz[np.abs(h) >= ground_h]


class CsfGroundExtractor(BaseAlgorithm):

    def _process_data(self, bSloopSmooth: 'bool' = False, cloth_resolution: float = 0.5) -> None:
        csf = CSF.CSF()
        csf.params.bSloopSmooth = bSloopSmooth
        csf.params.cloth_resolution = cloth_resolution
        csf.setPointCloud(self.data.pcd_np)

        # Extract the ground
        self.ground = CSF.VecInt()
        self.no_ground = CSF.VecInt()
        csf.do_filtering(self.ground, self.no_ground)

        self.ground = np.asarray(self.ground)
        self.no_ground = np.asarray(self.no_ground)
        self.data.ground = self.data.pcd_np[self.ground, :]
        self.data.no_ground = self.data.pcd_np[self.no_ground, :]


class RansacGroundExtractor(BaseAlgorithm):

    def _process_data(self, distance_threshold: float = 0.05, ransac_n: int = 3, num_iterations=1000) -> None:
        plane_model, inliers = self.data.pcd.segment_plane(distance_threshold=distance_threshold,
                                                           ransac_n=ransac_n, num_iterations=num_iterations)
        inliers = np.array(inliers)
        self.data.ground = self.data.pcd_np[inliers, :]
        self.data.no_ground = np.delete(self.data.pcd_np, inliers, axis=0)


class GroundExtractor:
    def __init__(self):
        # 初始化处理器
        pass

    def process_data(self, data, algorithm):
        self.algorithm = algorithm()

        # 使用指定算法处理数据
        self.algorithm.set_data(data)
        self.algorithm._process_data()


class ReGrowSegment(BaseAlgorithm):

    def _process_data(self) -> None:
        ground_np = self.np_to_o3d(self.data.ground)
        seed = self.seed_select(ground_np)
        no_paves = self.find_no_paves(ground_np)
        # np_clouds = np.asarray(clouds)
        no_paves = self.find_index(self.data.ground, no_paves)
        nebor_all = self.find_neighbour_points(ground_np)
        curvity = self.curvature_calculation(ground_np)
        paves = np.array(seed)

        while len(seed) > 0:
            seed_now = seed[0]
            nebor = nebor_all[seed_now]
            nebor = np.asarray(nebor)

            nebor_np = nebor[np.isin(nebor, paves, invert=True)]
            nebor_new = nebor_np[np.isin(nebor_np, seed, invert=True)]
            nebor_new = nebor_new[np.isin(nebor_new, no_paves, invert=True)]

            if len(nebor_new) > 0:
                curr_seed_normal = ground_np.normals[seed_now]  # 当前种子点的法向量
                seed_nebor_normal = [ground_np.normals[int(i)] for i in nebor_new]  # 种子点邻域点的法向量
                dot_normal = np.fabs(np.dot(seed_nebor_normal, curr_seed_normal))
                nebor_new = nebor_new.astype('int64')
                curvity_now = curvity[nebor_new]
                a = dot_normal > self.data.cosine_threshold
                b = curvity_now < self.data.curvature_threshold
                c = a & b
                paves_new = nebor_new[c]
                paves = np.append(paves, paves_new)
                seed = np.append(seed, paves_new)

            seed = np.delete(seed, [0])

        self.data.paves = paves


class DriPathSegment(BaseAlgorithm):

    def _process_data(self):
        ground = self.data.ground
        ground_o3d = self.np_to_o3d(ground)
        ground_nor = np.asarray(ground_o3d.normals)
        no_paves = np.array([0])
        index_all = np.array(range(len(ground)))

        driving_track_seed = self.seed_select(ground_o3d)
        driving_track = self.driving_path_generation(driving_track_seed, ground_o3d)
        driving_track_o3d = self.np_to_o3d(driving_track)
        driving_track = self.find_nearest_point(ground_o3d, driving_track_o3d)

        driving_track_o3d = ground_o3d.select_by_index(driving_track)
        point_neighbours = self.find_nearest_point(driving_track_o3d, ground_o3d)
        driving_track = np.asarray(driving_track_o3d.points)

        for i in range(len(driving_track)):
            slim = ground[point_neighbours[:, 0] == i]
            slim_index = index_all[point_neighbours[:, 0] == i]
            if len(slim) > 0:
                slim_nor = ground_nor[point_neighbours[:, 0] == i]
                slim_distance = np.sqrt(np.sum(np.power(slim - driving_track[i, :], 2), axis=1))
                p = slim_distance.argmin()
                p = slim_nor[p, :]
                a_jiao = np.fabs(np.dot(slim_nor, p))
                c = a_jiao <= self.data.cosine_threshold
                no_paves = np.concatenate((no_paves, slim_index[c]), axis=0)

        self.data.no_paves = np.delete(no_paves, 0, 0)

    


if __name__ == '__main__':
    pcd = ReadDatas('D:\project\Point_Datas\Point Cloud Data\Corner.ply')
    ge = GroundExtractor()
    ge.process_data(pcd, CsfGroundExtractor)
    be = DriPathSegment()
    be.set_data(pcd)
    be._process_data()
    ground_np = be.np_to_o3d(pcd.ground)
    ground = ground_np.select_by_index(pcd.no_paves)
    o3d.visualization.draw_geometries([ground])
    print(len(pcd.no_paves))
