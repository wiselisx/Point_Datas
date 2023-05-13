import sys 	
from PyQt5.QtWidgets import QApplication , QMainWindow, QFileDialog
from taptap import Ui_MainWindow
from readdatas import *
import open3d as o3d


class MainWindow(QMainWindow ):
	def __init__(self, parent=None):    
			super(MainWindow, self).__init__(parent)
			self.ui = Ui_MainWindow()
			self.ui.setupUi(self)
			self.gpf = GpfGroundExtractor()
			self.road_show_index = None

	def getpath(self):
		file_dialog = QFileDialog(self)
		file_path, _ = file_dialog.getOpenFileName(self, 'Select File')  # 打开文件选择对话框
		if file_path:
            # 在这里处理文件选择的逻辑，例如显示文件路径或执行其他操作 
			self.file_path = file_path
			self.ui.filepath_line.setText(self.file_path)
			self.file_path = self.ui.filepath_line.text()
			print("Selected file:", file_path)

	def compute(self):
		path = self.ui.filepath_line.text()
		self.pcd =  ReadDatas(path)
		
	def data_show(self):
		o3d.visualization.draw_geometries([self.pcd.pcd])

	def csf_compute(self):
		csf = CsfGroundExtractor()
		csf.set_data(self.pcd)
		csf._process_data(bSloopSmooth = bool(self.ui.csf_side_com.currentText()), cloth_resolution = float(self.ui.csf_web_com.value()))
		print('计算完成')

	def gpf_compute(self):
		gpf = GpfGroundExtractor()
		gpf.set_data(self.pcd)
		gpf._process_data(nlrp = int(self.ui.gpf_seed_num.value()), 
							thseeds= float(self.ui.gpf_start_h_2.value()), 
							ground_h= float(self.ui.gpf_later_h.value()),
							iterations= int(self.ui.gpf_num.value()))
		print('计算完成')

	def ra_compute(self):
		ran = RansacGroundExtractor()
		ran.set_data(self.pcd)
		ran._process_data(distance_threshold= float(self.ui.ra_dis.value()),
		    				ransac_n= int(self.ui.ra_num.value()),
							num_iterations= int(self.ui.ra_seed_num.value()))

	def ground_data_show(self):
		self.ground_o3d = self.gpf.np_to_o3d(self.pcd.ground)
		o3d.visualization.draw_geometries([self.ground_o3d])

	def reg(self):
		reg = ReGrowSegment(self.pcd, theta_threshold = float(self.ui.reg_nor.value()), 
		      curvature_threshold = float(self.ui.reg_cur.value()))
		reg._process_data()
		self.road_show_index = 'gpf'

	def dri1(self):
		dris= DriPathSegment(self.pcd, theta_threshold= float(self.ui.dri_nor.value()))
		dris._process_data()
		self.road_show_index = 'dri'

	def dri2(self):
		dri2s = DriPathSegment2(self.pcd, theta_threshold= float(self.ui.dri2_nor.value()), 
							   dis_density = float(self.ui.dir2_dis.value()), h_density = 	float(self.ui.dir2_h.value()))
		dri2s._process_data()
		self.road_show_index = 'gpf'
	
	def svm(self):
		path = 'D:\project\Point_Datas\sample\SVM_model.pkl'
		svm1 = SVM(self.pcd,path)
		svm1._process_data()
		self.road_show_index = 'svm1'

	def road_show(self):
		if self.road_show_index == None:
			print('无法可视化')
		elif self.road_show_index == 'gpf' :	
			paves = self.ground_o3d.select_by_index(self.pcd.paves)
			no_paves = self.ground_o3d.select_by_index(self.pcd.paves, invert=True)
			no_ground = self.gpf.np_to_o3d(self.pcd.no_ground)
			paves.paint_uniform_color([1,0,0])
			o3d.visualization.draw_geometries([no_paves,paves,no_ground])
		elif self.road_show_index == 'dri' :
			no_ground = self.gpf.np_to_o3d(self.pcd.no_ground)
			side = self.pcd.side
			paves = self.pcd.no_side.select_by_index(self.pcd.paves)
			paves.paint_uniform_color([1,0,0])
			o3d.visualization.draw_geometries([paves,no_ground,side, self.pcd.no_side])
		elif self.road_show_index == 'svm1' :
			no_ground = self.gpf.np_to_o3d(self.pcd.no_ground)
			side = self.ground_o3d.select_by_index(self.pcd.side, invert= True)
			paves = self.pcd.no_side.select_by_index(self.pcd.paves)
			paves.paint_uniform_color([1,0,0])
			o3d.visualization.draw_geometries([paves,no_ground,side, self.pcd.no_side])



if __name__=="__main__":  
	app = QApplication(sys.argv)  
	win = MainWindow()  
	win.show()
	print(win.ui.csf_side_com.currentText())
	print(win.ui.csf_side_com.currentText())
	sys.exit(app.exec_()) 