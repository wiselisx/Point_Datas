import CSF
import numpy as np
import open3d as o3d

class ReadDatas:

    def __init__(self,cloud):
        
        self.cloud = cloud
        self.datas = o3d.io.read_point_cloud(self.cloud)
        self.xyz = np.asarray(self.datas.points)
        self.ground = None
        self.non_ground = None
        self.neighbour_number = 30
        self.curvity = []

        self.theta_threshold = 15
        self.cosine_threshold = np.cos(np.deg2rad(self.theta_threshold))
        self.curvature_threshold=0.035

    def SVD(self,points:'ndarray') -> list:
        # 二维，三维均适用
        # 二维直线，三维平面
        pts = points.copy()
        # 奇异值分解
        c = np.mean(pts, axis=0)
        A = pts - c # shift the points
        A = A.T #3*n
        u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True) # A=u*s*vh
        normal = u[:,-1]

        # 法向量归一化
        nlen = np.sqrt(np.dot(normal,normal))
        normal = normal / nlen
        # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
        # u 每一列是一个方向
        # s 是对应的特征值
        # c >>> 点的中心
        # normal >>> 拟合的方向向量
        return u,s,c,normal
    
    def estimate_parameters(self,points:'ndarray') -> list:
        # 最小二乘法估算平面模型
        # 只有三个点时，可以直接计算
        

        _,_,c,n = self.SVD(points)

        params = np.hstack((c.reshape(1,-1),n.reshape(1,-1)))[0,:]
        return params






    def csf_ground_extraction(self,bSloopSmooth:'Bool' = False,cloth_resolution :float = 0.5):
        
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
    
    def gpf_ground_extraction(self,nlrp :int = 1000,thseeds :float = 0.1,ground_h :float = 0.15):

        xyz_cp = self.xyz.copy()
        xyz_cp = xyz_cp[np.lexsort(xyz_cp.T)]
        lrp = np.average(xyz_cp[:nlrp, 2])
        seed = xyz_cp[xyz_cp[:,2] < thseeds+lrp]

        seed_cp = seed
        for i in range(10):    
            params = self.estimate_parameters(seed_cp)
            h = ((xyz_cp[:,0]-params[0]) * params[3] + (xyz_cp[:,1]-params[1])* params[4]+
                (xyz_cp[:,2]-params[2])* params[5])/(np.sqrt(params[3]**2+params[4]**2+params[5]**2))
            seed_cp = xyz_cp[h < ground_h]

        self.ground = seed_cp
        self.non_ground = xyz_cp[h >= ground_h]

    def ransac_ground_extraction(self,distance_threshold :float = 0.05,ransac_n :int = 3,num_iterations = 1000):

        plane_model,inliers =self.datas.segment_plane(distance_threshold = distance_threshold,
                                                        ransac_n = ransac_n,num_iterations = num_iterations)
        inliers =  np.array(inliers)
        self.ground = self.xyz[inliers, :]
        self.non_ground = np.delete(self.xyz, inliers, axis = 0)

    def curvature_calculation(self,clouds:'o3d'):
        num_of_pts = len(clouds.points)         # 点云点的个数
        clouds.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(self.neighbour_number))
        cov_mat = clouds.covariances            # 获取每个点的协方差矩阵
        self.curvity = np.zeros(num_of_pts)          # 初始化存储每个点曲率的容器
        # point_curvature_index = np.zeros((num_of_pts, 2))
        # 计算每个点的曲率
        for i_n in range(num_of_pts):
            eignvalue, _ = np.linalg.eig(cov_mat[i_n])  # SVD分解求特征值
            idx = eignvalue.argsort()[::-1]
            eignvalue = eignvalue[idx]
            self.curvity[i_n] = eignvalue[2] / (eignvalue[0] + eignvalue[1] + eignvalue[2])
        
        self.curvity = np.array(self.curvity)

        return self.curvity
    
    def np_to_o3d(self,clouds:'ndarray'):
        name = ''
        name = o3d.geometry.PointCloud()
        name.points = o3d.utility.Vector3dVector(clouds)
        name.estimate_normals()

        return name

    def find_neighbour_points(self,clouds,neighbour_number = 30 ):
        kdtree = o3d.geometry.KDTreeFlann(clouds)
        number = len(clouds.points)
        
        point_neighbours = np.zeros((number, neighbour_number))
        for ik in range(number):
            [_, idx, _] = kdtree.search_knn_vector_3d(clouds.points[ik], neighbour_number)  # K近邻搜索
            point_neighbours[ik, :] = idx
            
        return point_neighbours 
    
    def region_growing_segmentation(self,clouds):

        seed = self.seed_select(clouds)
        no_paves = self.find_no_paves(clouds)
        # np_clouds = np.asarray(clouds)
        no_paves = self.find_index(self.ground, no_paves)
        nebor_all = self.find_neighbour_points(clouds)
        curvity = self.curvature_calculation(clouds)

        paves = np.array(seed) 
        while len(seed)>0:
            seed_now = seed[0]
            nebor = nebor_all[seed_now]
            nebor = np.asarray(nebor)


            nebor_np = nebor[np.isin(nebor,paves,invert=True)]
            nebor_new = nebor_np[np.isin(nebor_np,seed,invert=True)]
            nebor_new = nebor_new[np.isin(nebor_new,no_paves,invert=True)]


            if len(nebor_new)>0:

                curr_seed_normal = clouds.normals[seed_now]       # 当前种子点的法向量
                seed_nebor_normal = [clouds.normals[int(i)]  for i in nebor_new]     # 种子点邻域点的法向量
                dot_normal = np.fabs(np.dot(seed_nebor_normal, curr_seed_normal))
                nebor_new = nebor_new.astype('int64')
                curvity_now= curvity[nebor_new]
                a = dot_normal > self.cosine_threshold
                b = curvity_now < self.curvature_threshold
                c = a&b
                paves_new = nebor_new[c]
                paves = np.append(paves,paves_new)
                seed = np.append(seed,paves_new)

            seed = np.delete(seed,[0])
    
        return paves


    def seed_select(self,ground):


        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='Open3D', visible=True)
        vis.add_geometry(ground)
        vis.run()
        seed = vis.get_picked_points()
        vis.destroy_window()

        return seed
    
    def paves_cutting(self,ground):    
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name='Open3D', visible=True)
        vis.add_geometry(ground)
        vis.run()
        geometry = vis.get_cropped_geometry()
        vis.destroy_window()
        geometry = np.asarray(geometry.points)

        return geometry
    
    def find_no_paves(self,ground):
        a = True
        geometrys = np.zeros((1,3))

        while a:
            geometry = self.paves_cutting(ground)
            geometrys = np.concatenate((geometrys,geometry),axis=0)
            input_1 = input('按Y或者y继续')
            if input_1 == 'y' or input_1 == 'Y':
                a = True
            else:
                a = False
        geometrys = np.delete(geometrys, [0], axis=0)

        return geometrys
    
    def find_index(self,list1,list2):
        index = []
        for i in list2:
            a = (list1 == i).all(axis=1)
            a = np.argwhere(a).ravel()
            if a:
                index.append(a[0])
        return index
                

    

        



        
        

if __name__ == '__main__':

    datas = ReadDatas('Point Cloud Data\Corner.ply')
    print(datas.datas)
    datas.gpf_ground_extraction()
    
    clouds = datas.np_to_o3d(datas.ground)
    print('-----------')
    paves = datas.region_growing_segmentation(clouds)
    print('-----------')
    print(len(paves))
    print('-----------')



    


    