import CSF
import numpy as np
import open3d as o3d
from traitlets import Bool

class ReadDatas:

    def __init__(self,cloud):
        
        self.cloud = cloud
        self.datas = o3d.io.read_point_cloud(self.cloud)
        self.xyz = np.asarray(self.datas.points)
        self.ground = None
        self.non_ground = None 

    def SVD(self,points):
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
    
    def estimate_parameters(self,pts):
        # 最小二乘法估算平面模型
        # 只有三个点时，可以直接计算

        _,_,c,n = self.SVD(pts)

        params = np.hstack((c.reshape(1,-1),n.reshape(1,-1)))[0,:]
        return params




    # def visual_3dmap(self,cloud):
    #     '''
    #     Visual 3dmap
    #     '''
    #     clouds = ['cloud{}'.format(i)= o3d.geometry.PointCloud() for i in range(len(cloud))]

    #     cloud = [o3d.utility.Vector3dVector(xyz) for xyz in cloud]

    #     o3d.visualization.draw_geometries(cloud)

    def csf_ground_extraction(self,bSloopSmooth :Bool = False,cloth_resolution :float = 0.5):
        
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
            params = self.estimate_parameters(pts=seed_cp)
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
        # self.non_ground = self.xyz[~inliers, :]


        
        











if __name__ == '__main__':

    datas = ReadDatas('Point Cloud Data\Corner.ply')
    print(datas.datas)
    datas.ransac_ground_extraction()
    print(datas.ground.shape)


    