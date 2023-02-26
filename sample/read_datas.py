import CSF
import numpy as np
import open3d as o3d


class ReadDatas:

    def __init__(self, cloud):

        self.cloud = cloud
        self.datas = o3d.io.read_point_cloud(self.cloud)
        self.xyz = np.asarray(self.datas.points)
        self.ground = None
        self.non_ground = None
        self.neighbour_number = 30
        self.curvity = []

        self.theta_threshold = 15
        self.cosine_threshold = np.cos(np.deg2rad(self.theta_threshold))
        self.curvature_threshold = 0.035

    def SVD(self, points: 'ndarray') -> list:
        # 二维，三维均适用
        # 二维直线，三维平面
        pts = points.copy()
        # 奇异值分解
        c = np.mean(pts, axis=0)
        A = pts - c  # shift the points
        A = A.T  # 3*n
        u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)  # A=u*s*vh
        normal = u[:, -1]

        # 法向量归一化
        nlen = np.sqrt(np.dot(normal, normal))
        normal = normal / nlen
        # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
        # u 每一列是一个方向
        # s 是对应的特征值
        # c >>> 点的中心
        # normal >>> 拟合的方向向量
        return u, s, c, normal

    def estimate_parameters(self, points: 'ndarray') -> list:
        # 最小二乘法估算平面模型
        # 只有三个点时，可以直接计算

        _, _, c, n = self.SVD(points)

        params = np.hstack((c.reshape(1, -1), n.reshape(1, -1)))[0, :]
        return params

    def csf_ground_extraction(self, bSloopSmooth: 'Bool' = False, cloth_resolution: float = 0.5):

        csf = CSF.CSF()
        csf.params.bSloopSmooth = bSloopSmooth
        csf.params.cloth_resolution = cloth_resolution
        csf.setPointCloud(self.xyz)

        # Extract the ground
        self.ground = CSF.VecInt()
        self.non_ground = CSF.VecInt()
        csf.do_filtering(self.ground, self.non_ground)

        self.ground = np.asarray(self.ground)
        self.non_ground = np.asarray(self.non_ground)
        self.ground = self.xyz[self.ground, :]
        self.non_ground = self.xyz[self.non_ground, :]

        return

    def gpf_ground_extraction(self, nlrp: int = 1000, thseeds: float = 0.1, ground_h: float = 0.15):

        xyz_cp = self.xyz.copy()
        xyz_cp = xyz_cp[np.lexsort(xyz_cp.T)]
        lrp = np.average(xyz_cp[:nlrp, 2])
        seed = xyz_cp[xyz_cp[:, 2] < thseeds + lrp]

        seed_cp = seed
        for i in range(10):
            params = self.estimate_parameters(seed_cp)
            h = ((xyz_cp[:, 0] - params[0]) * params[3] + (xyz_cp[:, 1] - params[1]) * params[4] +
                 (xyz_cp[:, 2] - params[2]) * params[5]) / (np.sqrt(params[3] ** 2 + params[4] ** 2 + params[5] ** 2))
            seed_cp = xyz_cp[h < ground_h]

        self.ground = seed_cp
        self.non_ground = xyz_cp[h >= ground_h]

    def ransac_ground_extraction(self, distance_threshold: float = 0.05, ransac_n: int = 3, num_iterations=1000):

        plane_model, inliers = self.datas.segment_plane(distance_threshold=distance_threshold,
                                                        ransac_n=ransac_n, num_iterations=num_iterations)
        inliers = np.array(inliers)
        self.ground = self.xyz[inliers, :]
        self.non_ground = np.delete(self.xyz, inliers, axis=0)

    def curvature_calculation(self, clouds: 'o3d'):
        num_of_pts = len(clouds.points)  # 点云点的个数
        clouds.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(self.neighbour_number))
        cov_mat = clouds.covariances  # 获取每个点的协方差矩阵
        self.curvity = np.zeros(num_of_pts)  # 初始化存储每个点曲率的容器
        # point_curvature_index = np.zeros((num_of_pts, 2))
        # 计算每个点的曲率
        for i_n in range(num_of_pts):
            eignvalue, _ = np.linalg.eig(cov_mat[i_n])  # SVD分解求特征值
            idx = eignvalue.argsort()[::-1]
            eignvalue = eignvalue[idx]
            self.curvity[i_n] = eignvalue[2] / (eignvalue[0] + eignvalue[1] + eignvalue[2])

        self.curvity = np.array(self.curvity)

        return self.curvity

    def np_to_o3d(self, clouds: 'ndarray'):
        name = ''
        name = o3d.geometry.PointCloud()
        name.points = o3d.utility.Vector3dVector(clouds)
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

    def region_growing_segmentation(self, clouds):

        seed = self.seed_select(clouds)
        no_paves = self.find_no_paves(clouds)
        # np_clouds = np.asarray(clouds)
        no_paves = self.find_index(self.ground, no_paves)
        nebor_all = self.find_neighbour_points(clouds)
        curvity = self.curvature_calculation(clouds)

        paves = np.array(seed)
        while len(seed) > 0:
            seed_now = seed[0]
            nebor = nebor_all[seed_now]
            nebor = np.asarray(nebor)

            nebor_np = nebor[np.isin(nebor, paves, invert=True)]
            nebor_new = nebor_np[np.isin(nebor_np, seed, invert=True)]
            nebor_new = nebor_new[np.isin(nebor_new, no_paves, invert=True)]

            if len(nebor_new) > 0:
                curr_seed_normal = clouds.normals[seed_now]  # 当前种子点的法向量
                seed_nebor_normal = [clouds.normals[int(i)] for i in nebor_new]  # 种子点邻域点的法向量
                dot_normal = np.fabs(np.dot(seed_nebor_normal, curr_seed_normal))
                nebor_new = nebor_new.astype('int64')
                curvity_now = curvity[nebor_new]
                a = dot_normal > self.cosine_threshold
                b = curvity_now < self.curvature_threshold
                c = a & b
                paves_new = nebor_new[c]
                paves = np.append(paves, paves_new)
                seed = np.append(seed, paves_new)

            seed = np.delete(seed, [0])

        return paves

    def seed_select(self, ground):

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='Open3D', visible=True)
        vis.add_geometry(ground)
        vis.run()
        seed = vis.get_picked_points()
        vis.destroy_window()

        return seed

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

    def find_index(self, list1, list2):
        index = []
        for i in list2:
            a = (list1 == i).all(axis=1)
            a = np.argwhere(a).ravel()
            if a:
                index.append(a[0])
        return index
    
    def calculate_linear_density(self,ground):
        nndist = ground.compute_nearest_neighbor_distance()
        nndist = np.array(nndist)
        density = np.mean(nndist)
        return density
    
    def driving_path_generation(self,point,ground):
        density = self.calculate_linear_density(ground)
        driving_point = ground.select_by_index(point)
        driving_point = np.array(driving_point.points)

        ground = ground.select_by_index(point, invert=True)
        driving_distance = np.array([np.linalg.norm(driving_point[p] - driving_point[p+1]) for p in range(len(driving_point) - 1 )])
        driving_track = np.array([[0,0,0]])

        for i,d in enumerate(driving_distance):
            num = int(d / density)

            driving_track_x = np.linspace(driving_point[i,0],driving_point[i+1,0],num)
            driving_track_y = np.linspace(driving_point[i,1],driving_point[i+1,1],num)
            driving_track_z = np.linspace(driving_point[i,2],driving_point[i+1,2],num)
            new_points = np.array([driving_track_x,driving_track_y,driving_track_z]).T

            driving_track = np.concatenate((driving_track,new_points),axis=0)
        driving_track = np.delete(driving_track,0,axis=0)
        return driving_track
    
    def find_nearest_point(self,clouds1,clouds2,neighbour_number = 1):

        neighbour_number = 1
        kdtree = o3d.geometry.KDTreeFlann(clouds1)
        number = len(clouds2.points)

        point_neighbours = np.zeros((number, neighbour_number))
        for ik in range(number):
            [_, point[0], _] = kdtree.search_knn_vector_3d(clouds2.points[ik], neighbour_number)  # K近邻搜索
            point_neighbours[ik, :] = point[0]
        return point_neighbours
    
    def driving_path_extraction(self,ground,driving_track,point_neighbours):
        ground_np = np.asarray(ground.points)
        ground_nom = ground.estimate_normals()
        ground_nom = np.asarray(ground.normals)
        theta_threshold = 30
        cosine_threshold = np.cos(np.deg2rad(theta_threshold))
        paves = np.array([[0,0,0]])
        for i,p in enumerate(driving_track):
            slim = ground_np[point_neighbours[:,0] == i]
            if len(slim) > 0:
                slim_distance = np.sqrt(np.sum(np.power(slim - driving_track[i,:],2),axis = 1))
                p = slim_distance.argmin()
                slim_nom = slim - driving_track[i,:]
                slim_distance[slim_nom[:,0]/slim_nom[:,1] < 0] = slim_distance[slim_nom[:,0]/slim_nom[:,1] < 0] * -1
                slim = slim[slim_distance.argsort()]
                slim_nor = ground_nom[point_neighbours[:,0] == i]
                p = slim_nor[p,:]
                slim_nor = slim_nor[slim_distance.argsort()]
                slim_distance.sort()
                a_jiao = np.fabs(np.dot(slim_nor, p))
                b = np.diff(slim[:,2])
                b =np.append(b,1)
                a = a_jiao > cosine_threshold
                d = slim[a]
                paves = np.concatenate((paves,d),axis=0)
        # pave = np.delete(pave,0)
        return paves
    
    def euclidean_cluster(self,cloud,point, tolerance=0.2, min_cluster_size=100, max_cluster_size=1000000000000):
        """
        欧式聚类
        :param cloud:输入点云
        :param tolerance: 设置近邻搜索的搜索半径（也即两个不同聚类团点之间的最小欧氏距离）
        :param min_cluster_size:设置一个聚类需要的最少的点数目
        :param max_cluster_size:设置一个聚类需要的最大点数目
        :return:聚类个数
        """

        kdtree = o3d.geometry.KDTreeFlann(cloud)  # 对点云建立kd树索引

        num_points = len(cloud.points)
        processed = [-1] * num_points  # 定义所需变量
        clusters = []  # 初始化聚类
        # 遍历各点
        while point:
            if processed[point[1]] == 1:  # 如果该点已经处理则跳过
                continue
            seed_queue = []  # 定义一个种子队列
            sq_idx = 0
            seed_queue.append(point[1])  # 加入一个种子点
            processed[point[1]] = 1

            while sq_idx < len(seed_queue):

                k, nn_indices, _ = kdtree.search_radius_vector_3d(cloud.points[seed_queue[sq_idx]], tolerance)

                if k == 1:  # k=1表示该种子点没有近邻点
                    sq_idx += 1
                    continue
                for j in range(k):

                    if nn_indices[j] == num_points or processed[nn_indices[j]] == 1:
                        continue  # 种子点的近邻点中如果已经处理就跳出此次循环继续
                    seed_queue.append(nn_indices[j])
                    processed[nn_indices[j]] = 1

                sq_idx += 1

            if max_cluster_size > len(seed_queue) > min_cluster_size:
                clusters.append(seed_queue)
                point = False

        return seed_queue


if __name__ == '__main__':
    datas = ReadDatas('D:\project\Point_Datas\Point Cloud Data\Corner.ply')
    datas.gpf_ground_extraction()
    ground = datas.np_to_o3d(datas.ground)
    point = datas.seed_select(ground)
    geometrys = datas.find_no_paves(ground)
    driving_track = datas.driving_path_generation(point,ground)
    driving_track_o3d = datas.np_to_o3d(driving_track)
    ground += driving_track_o3d
    point_neighbours = datas.find_nearest_point(driving_track_o3d,ground)
    pave = datas.driving_path_extraction(ground,driving_track,point_neighbours)
    seed = datas.find_index(pave,geometrys)
    paves = np.delete(pave,seed,0)
    paves = datas.np_to_o3d(paves)
    ec = datas.euclidean_cluster(paves,point, tolerance=0.1, min_cluster_size=1000, max_cluster_size=100000000)
    ec = paves.select_by_index(ec)
    o3d.visualization.draw_geometries([ec])


    

