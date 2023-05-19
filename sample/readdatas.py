'''
Author: 夜间清风 1874032283@qq.com
Date: 2023-03-04 20:05:50
LastEditors: 夜间清风 1874032283@qq.com
LastEditTime: 2023-05-09 20:54:44
FilePath: \Point_Datas\sample\readdatas.py
Description: 软件代码的算法部分。
'''

import CSF
import numpy as np
import open3d as o3d
from abc import ABC, abstractmethod
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from PyQt5.QtWidgets import QApplication, QMessageBox

class ReadDatas:

    def __init__(self, path: str) -> None:
        self.path: str = path
        self.pcd = self.read_point_cloud()
        self.pcd_np = np.asarray(self.pcd.points)
        self.pcd_hmax = self.pcd.get_max_bound()[2]
        self.pcd_hmin = self.pcd.get_min_bound()[2]
        self.pcd_num = len(self.pcd_np)
        self.theta_threshold = 20
        self.cosine_threshold = np.cos(np.deg2rad(self.theta_threshold))
        self.curvature_threshold = 0.035
        

    def read_point_cloud(self):
        '''
        description: 读取点云文件
        '''
        pcd = o3d.io.read_point_cloud(self.path)
        return pcd


class BaseAlgorithm(ABC):

    def __init__(self) -> None:
        self.data = None

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

    def np_to_o3d(self, points):
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
             # K近邻搜索
            [_, idx, _] = kdtree.search_knn_vector_3d(
                clouds.points[ik], neighbour_number) 
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
        vis.create_window(window_name='种子点选取', visible=True)
        vis.add_geometry(cloud)
        vis.run()
        seed = vis.get_picked_points()
        vis.destroy_window()
        return seed

    def find_index(self, list1, list2):
        index = []
        for i in list2:
            a = (list1 == i).all(axis=1)
            a = np.argwhere(a).ravel()
            if a:
                index.append(a[0])
        return index

    def paves_cutting(self, ground):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='区域切割', visible=True)
        vis.add_geometry(ground)
        vis.run()
        geometry = vis.get_cropped_geometry()
        vis.destroy_window()
        geometry = np.asarray(geometry.points)
        return geometry

    def find_no_paves(self, ground):
        # app = QApplication([])
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Question)
        message_box.setText("是否继续区域分割")
        message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        message_box.setDefaultButton(QMessageBox.No)
        a = True
        geometrys = np.zeros((1, 3))

        while a:
            geometry = self.paves_cutting(ground)
            geometrys = np.concatenate((geometrys, geometry), axis=0)
            result = message_box.exec_()
            if result == QMessageBox.Yes:
                a = True
            else:
                a = False
        geometrys = np.delete(geometrys, [0], axis=0)
        # app.exec_()
        return geometrys
    
    # def find_no_paves(self, ground):
      
    #     a = True
    #     geometrys = np.zeros((1, 3))

    #     while a:
    #         geometry = self.paves_cutting(ground)
    #         geometrys = np.concatenate((geometrys, geometry), axis=0)
    #         input_1 = input('按Y或者y继续')
    #         if input_1 == 'y' or input_1 == 'Y':
    #             a = True
    #         else:
    #             a = False
    #     geometrys = np.delete(geometrys, [0], axis=0)
    #     return geometrys
    
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
            # K近邻搜索
            [_, idx, _] = kdtree.search_knn_vector_3d(
                clouds2.points[ik], neighbour_number)  
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
        self.data.ground_num = len(self.data.ground)



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
        self.data.ground_num = len(self.data.ground)


class RansacGroundExtractor(BaseAlgorithm):

    def _process_data(self, distance_threshold: float = 0.05, ransac_n: int = 3, num_iterations=1000) -> None:
        plane_model, inliers = self.data.pcd.segment_plane(distance_threshold=distance_threshold,
                                                           ransac_n=ransac_n, num_iterations=num_iterations)
        inliers = np.array(inliers)
        self.data.ground = self.data.pcd_np[inliers, :]
        self.data.no_ground = np.delete(self.data.pcd_np, inliers, axis=0)
        self.data.ground_num = len(self.data.ground)


class ReGrowSegment(BaseAlgorithm):

    def __init__(self, data, theta_threshold = 20, curvature_threshold = 0.035) -> None:
        self.data = data
        self.ground_np = self.np_to_o3d(self.data.ground)
        self.seed = self.seed_select(self.ground_np)
        self.no_paves = self.find_no_paves(self.ground_np)
        # np_clouds = np.asarray(clouds)
        self.no_paves = self.find_index(self.data.ground, self.no_paves)
        self.nebor_all = self.find_neighbour_points(self.ground_np)
        self.curvity = self.curvature_calculation(self.ground_np)
        self.paves = np.array(self.seed)
        self.cosine_threshold = np.cos(np.deg2rad(theta_threshold))
        self.curvature_threshold = curvature_threshold
        

    def _process_data(self) -> None:

        while len(self.seed) > 0:
            seed_now = self.seed[0]
            nebor = self.nebor_all[seed_now]
            nebor = np.asarray(nebor)

            nebor_np = nebor[np.isin(nebor, self.paves, invert=True)]
            nebor_new = nebor_np[np.isin(nebor_np, self.seed, invert=True)]
            nebor_new = nebor_new[np.isin(nebor_new, self.no_paves, invert=True)]

            if len(nebor_new) > 0:
                curr_seed_normal = self.ground_np.normals[seed_now]  # 当前种子点的法向量
                seed_nebor_normal = [self.ground_np.normals[int(i)] for i in nebor_new]  # 种子点邻域点的法向量
                dot_normal = np.fabs(np.dot(seed_nebor_normal, curr_seed_normal))
                nebor_new = nebor_new.astype('int64')
                curvity_now = self.curvity[nebor_new]
                a = dot_normal > self.cosine_threshold
                b = curvity_now < self.curvature_threshold
                c = a & b
                paves_new = nebor_new[c]
                self.paves = np.append(self.paves, paves_new)
                self.seed = np.append(self.seed, paves_new)

            self.seed = np.delete(self.seed, [0])

        self.data.paves = self.paves
        self.data.paves_num = len(self.data.paves)


class DriPathSegment(BaseAlgorithm):
    def __init__(self, data, theta_threshold = 20,):
        self.data = data
        self.ground = self.data.ground
        self.ground_o3d = self.np_to_o3d(self.ground)
        self.ground_nor = np.asarray(self.ground_o3d.normals)
        self.no_paves = np.array([0])
        self.index_all = np.array(range(len(self.ground)))

        driving_track_seed = self.seed_select(self.ground_o3d)
        driving_track = self.driving_path_generation(driving_track_seed, self.ground_o3d)
        driving_track_o3d = self.np_to_o3d(driving_track)
        driving_track = self.find_nearest_point(self.ground_o3d, driving_track_o3d)

        driving_track_o3d = self.ground_o3d.select_by_index(driving_track)
        self.point_neighbours = self.find_nearest_point(driving_track_o3d, self.ground_o3d)
        self.driving_track = np.asarray(driving_track_o3d.points)
        self.theta_threshold = theta_threshold
        self.cosine_threshold = np.cos(np.deg2rad(self.theta_threshold))
        

    def _process_data(self):

        for i in range(len(self.driving_track)):
            slim = self.ground[self.point_neighbours[:, 0] == i]
            slim_index = self.index_all[self.point_neighbours[:, 0] == i]
            if len(slim) > 0:
                slim_nor = self.ground_nor[self.point_neighbours[:, 0] == i]
                slim_distance = np.sqrt(np.sum(np.power(slim - self.driving_track[i, :], 2), axis=1))
                p = slim_distance.argmin()
                p = slim_nor[p, :]
                a_jiao = np.fabs(np.dot(slim_nor, p))
                c = a_jiao <= self.cosine_threshold
                self.no_paves = np.concatenate((self.no_paves, slim_index[c]), axis=0)

        self.data.side = np.delete(self.no_paves, 0, 0)
        self.data.no_side= self.ground_o3d.select_by_index(self.data.side, invert = True)
        self.data.side = self.ground_o3d.select_by_index(self.data.side)
        self.data.paves = self.euclidean_cluster(self.data.no_side, tolerance=0.15)
        self.data.paves_num = len(self.data.paves)

    def euclidean_cluster(self,cloud, tolerance=0.2):
        seed = self.seed_select(cloud)
        no_paves = self.find_no_paves(cloud)
        paves = np.asarray(cloud.points)
        no_paves = self.find_index(paves, no_paves)
        kdtree = o3d.geometry.KDTreeFlann(cloud)
        paves = np.array(seed)
        while len(seed) > 0 :
            seed_now = seed[0]
            k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[seed_now], tolerance)
            if k == 1 :
                continue
            idx = np.array(idx)
            idx = idx[np.isin(idx, paves, invert=True)]
            idx = idx[np.isin(idx, no_paves, invert=True)]
            paves = np.append(paves, idx)
            seed = np.append(seed,idx)
            seed = np.delete(seed,[0])

        return paves
    
class DriPathSegment2(BaseAlgorithm):

    def __init__(self, data, theta_threshold = 20, dis_density = 2, h_density = 2):
        
        self.data = data
        self.ground = self.data.ground
        ground_o3d = self.np_to_o3d(self.ground)
        self.ground_nor = np.asarray(ground_o3d.normals)
        driving_track_seed = self.seed_select(ground_o3d)
        driving_track = self.driving_path_generation(driving_track_seed, ground_o3d)
        driving_track_o3d = self.np_to_o3d(driving_track)
        driving_track = self.find_nearest_point(ground_o3d, driving_track_o3d)
        self.density = self.calculate_linear_density(ground_o3d)
        driving_track_o3d = ground_o3d.select_by_index(driving_track)
        self.point_neighbours = self.find_nearest_point(driving_track_o3d,ground_o3d)
        self.driving_track = np.asarray(driving_track_o3d.points)
        self.theta_threshold = theta_threshold
        self.cosine_threshold = np.cos(np.deg2rad(self.theta_threshold))
        self.dis_density = dis_density
        self.h_density = h_density

    def _process_data(self):

        index_all = np.array(range(len(self.ground)))
        no_paves = np.array([0])
        paves = np.array([0])

        for i in range(len(self.driving_track)):
            slim = self.ground[self.point_neighbours[:,0] == i]
            slim_index = index_all[self.point_neighbours[:,0] == i]
            if len(slim) > 0:
               slim_nor = self.ground_nor[self.point_neighbours[:,0] == i]
               slim_distance = np.sqrt(np.sum(np.power(slim - self.driving_track[i,:],2),axis = 1))
               driving_track_point = slim_distance.argmin()
               driving_track_point = slim_nor[driving_track_point,:]
               included_angle = np.fabs(np.dot(slim_nor, driving_track_point))
               no_paves_slim = included_angle <= self.cosine_threshold
               paves_slim = included_angle > self.cosine_threshold

               slim_distance = slim_distance[paves_slim]
               if len(slim_distance) > 0:

                    paves_slim_h = slim[paves_slim,2].reshape(1,-1)
                    paves_slim = slim_index[paves_slim]
                    arr = np.argsort(paves_slim_h)[0]
                    paves_slim_h = np.sort(paves_slim_h)
                    diff_arr = np.diff(paves_slim_h)[0]
                    stop_index = np.where(diff_arr > self.h_density * self.density)
                    if len(stop_index[0]) > 0:
                         stop_index = stop_index[0]
                         stop_index +=1
                         stop_index = stop_index[0]
                    else:
                         stop_index = paves_slim_h.shape[1]

                    
                    arr = arr[:stop_index]
                    paves_slim = paves_slim[arr]
                    slim_distance = slim_distance[arr]
                    arr = np.argsort(slim_distance)
                    slim_distance = np.sort(slim_distance)
                    diff_arr = np.diff(slim_distance)
                    stop_index = np.where(diff_arr > self.dis_density * self.density)
                    if len(stop_index[0]) > 0:
                         stop_index = stop_index[0]
                         stop_index +=1
                         stop_index = stop_index[0]
                    else:
                         stop_index = slim_distance.shape[0]
                    arr = arr[:stop_index]
                    paves_slim = paves_slim[arr]
                    paves = np.concatenate((paves, paves_slim),axis=0)
               no_paves = np.concatenate((no_paves, slim_index[no_paves_slim]),axis=0)

        self.data.paves = np.delete(paves,0,0)     
        self.data.no_paves = np.delete(no_paves,0,0)
        self.data.paves_num = len(self.data.paves)


class SVM(BaseAlgorithm):

    def __init__(self, data, path):
        self.data = data;
        self.ground = data.ground
        self.ground_o3d = self.np_to_o3d(self.ground)
        self.ground_nor = np.asarray(self.ground_o3d.normals)
        self.feature_vector = self.get_feature_vector()
        self.path = path

    def _process_data(self):
        clf = joblib.load(self.path)
        y_pred_all = clf.predict(self.feature_vector)
        index = np.array(range(len(self.data.ground)))
        index = index[y_pred_all]
        # self.data.side =  self.ground_o3d.select_by_index(index, ivert = True)
        self.data.side = index
        self.data.no_side = self.ground_o3d.select_by_index(index)
        self.data.paves = self.euclidean_cluster(self.data.no_side)
        self.data.paves_num = len(self.data.paves)


    def euclidean_cluster(self,cloud, tolerance=0.2):
        seed = self.seed_select(cloud)
        no_paves = self.find_no_paves(cloud)
        paves = np.asarray(cloud.points)
        no_paves = self.find_index(paves, no_paves)
        kdtree = o3d.geometry.KDTreeFlann(cloud)
        paves = np.array(seed)
        while len(seed) > 0 :
            seed_now = seed[0]
            k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[seed_now], tolerance)
            if k == 1 :
                continue
            idx = np.array(idx)
            idx = idx[np.isin(idx, paves, invert=True)]
            idx = idx[np.isin(idx, no_paves, invert=True)]
            paves = np.append(paves, idx)
            seed = np.append(seed,idx)
            seed = np.delete(seed,[0])
        return paves

        

    def get_feature_vector(self):
        ground_tree = o3d.geometry.KDTreeFlann(self.ground_o3d)
        n = len(self.ground)
        feature_vector = np.zeros((n,2))
        for i in range(n):
            [num, idx, _] = ground_tree.search_radius_vector_3d(self.ground_o3d.points[i], 0.2)
            point_neighbour = self.ground[idx,:]
            point_neighbour_max = point_neighbour[:,2].max(axis=0)
            point_neighbour_min = point_neighbour[:,2].min(axis=0)
            point_neighbour_h = point_neighbour_max - point_neighbour_min
            point_neighbour_var = np.var(point_neighbour[:,2],axis=0)
            feature_vector[i,:] = [point_neighbour_h,point_neighbour_var]
        feature_vector = np.hstack((feature_vector,self.ground_nor.reshape(-1,3)))
        return feature_vector

if __name__ == '__main__':
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    path = 'D:\project\Point_Datas\Point Cloud Data\Corner.ply'
    pcd =  ReadDatas(path)
    gpf = GpfGroundExtractor()
    gpf.set_data(pcd)
    gpf._process_data()
    path = 'SVM_model.pkl'
    svm = SVM(pcd,path)
    svm._process_data()
    paves = svm.no_side.select_by_index(pcd.paves)
    o3d.visualization.draw_geometries([paves])

