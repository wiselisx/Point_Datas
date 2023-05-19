import sys 	
from PyQt5.QtWidgets import QApplication , QMainWindow, QFileDialog
from taptap import Ui_MainWindow
from readdatas import *
import open3d as o3d
import numpy as np
import pyqtgraph as pg

class MainWindow(QMainWindow ):
	def __init__(self, parent=None):    
		super(MainWindow, self).__init__(parent)
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.gpf = GpfGroundExtractor()
		self.road_show_index = None
		background_color = (255, 255, 255)
		self.ui.data_show_image.setBackground(background_color)
		self.ui.ground_show_image.setBackground(background_color)
		self.ui.road_show_image.setBackground(background_color)

	def data_show_h(self, data, ui):
		z_min = min(data[:, 2])
		z_max = max(data[:, 2])
		num_intervals = 10
		width = (z_max - z_min) / num_intervals 
		
		elevations = [round(float(value),2) for value in data[:,2]]
		hist, bin_edges =  np.histogram(elevations, bins=num_intervals, range=(z_min, z_max))

		# 创建BarGraphItem对象并添加到plot_widget中
		bar_graph_item = pg.BarGraphItem(x=bin_edges[:num_intervals], height=hist, width=width, brush='b')
		ui.addItem(bar_graph_item)
	
		
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
		self.ui.data_total.setText(str(self.pcd.pcd_num))
		self.ui.data_maxh.setText(str(self.pcd.pcd_hmax))
		self.ui.data_minh.setText(str(self.pcd.pcd_hmin))
		self.data_show_h(self.pcd.pcd_np, self.ui.data_show_image)
		
	def data_show(self):
		o3d.visualization.draw_geometries([self.pcd.pcd])

	def csf_compute(self):
		csf = CsfGroundExtractor()
		csf.set_data(self.pcd)
		csf._process_data(bSloopSmooth = bool(self.ui.csf_side_com.currentText()), cloth_resolution = float(self.ui.csf_web_com.value()))
		self.ui.ground_total.setText(str(self.pcd.ground_num))
		print('计算完成')

	def gpf_compute(self):
		gpf = GpfGroundExtractor()
		gpf.set_data(self.pcd)
		gpf._process_data(nlrp = int(self.ui.gpf_seed_num.value()), 
							thseeds= float(self.ui.gpf_start_h_2.value()), 
							ground_h= float(self.ui.gpf_later_h.value()),
							iterations= int(self.ui.gpf_num.value()))
		print('计算完成')
		self.ui.ground_total.setText(str(self.pcd.ground_num))

	def ra_compute(self):
		ran = RansacGroundExtractor()
		ran.set_data(self.pcd)
		ran._process_data(distance_threshold= float(self.ui.ra_dis.value()),
		    				ransac_n= int(self.ui.ra_num.value()),
							num_iterations= int(self.ui.ra_seed_num.value()))
		self.ui.ground_total.setText(str(self.pcd.ground_num))

	def ground_data_show(self):
		self.ground_o3d = self.gpf.np_to_o3d(self.pcd.ground)
		o3d.visualization.draw_geometries([self.ground_o3d])
		self.ui.ground_maxh.setText(str(self.ground_o3d.get_max_bound()[2]))
		self.ui.ground_minh.setText(str(self.ground_o3d.get_min_bound()[2]))
		self.data_show_h(self.pcd.ground, self.ui.ground_show_image)


	def reg(self):
		reg = ReGrowSegment(self.pcd, theta_threshold = float(self.ui.reg_nor.value()), 
		      curvature_threshold = float(self.ui.reg_cur.value()))
		reg._process_data()
		self.road_show_index = 'gpf'
		self.ui.road_total.setText(str(self.pcd.paves_num))

	def dri1(self):
		dris= DriPathSegment(self.pcd, theta_threshold= float(self.ui.dri_nor.value()))
		dris._process_data()
		self.road_show_index = 'dri'
		self.ui.road_total.setText(str(self.pcd.paves_num))

	def dri2(self):
		dri2s = DriPathSegment2(self.pcd, theta_threshold= float(self.ui.dri2_nor.value()), 
							   dis_density = float(self.ui.dir2_dis.value()), h_density = 	float(self.ui.dir2_h.value()))
		dri2s._process_data()
		self.road_show_index = 'gpf'
		self.ui.road_total.setText(str(self.pcd.paves_num))
	
	def svm(self):
		if self.ui.model.currentIndex() == 0:
			path = 'D:\project\Point_Datas\sample\SVM_model.pkl'
		else:
			path = 'D:\project\Point_Datas\sample\net_model.pkl'

		svm1 = SVM(self.pcd,path)
		svm1._process_data()
		self.road_show_index = 'svm1'
		self.ui.road_total.setText(str(self.pcd.paves_num))

	def road_show(self):
		if self.road_show_index == None:
			print('无法可视化')
		elif self.road_show_index == 'gpf' :	
			paves = self.ground_o3d.select_by_index(self.pcd.paves)
			no_paves = self.ground_o3d.select_by_index(self.pcd.paves, invert=True)
			no_ground = self.gpf.np_to_o3d(self.pcd.no_ground)
			paves.paint_uniform_color([1,0,0])
			o3d.visualization.draw_geometries([no_paves,paves,no_ground])
			self.ui.road_maxh.setText(str(paves.get_max_bound()[2]))
			self.ui.road_minh.setText(str(paves.get_min_bound()[2]))
			paves_np = np.asarray(paves.points)
			self.data_show_h(paves_np,self.ui.road_show_image)
		elif self.road_show_index == 'dri' :
			no_ground = self.gpf.np_to_o3d(self.pcd.no_ground)
			side = self.pcd.side
			paves = self.pcd.no_side.select_by_index(self.pcd.paves)
			paves.paint_uniform_color([1,0,0])
			o3d.visualization.draw_geometries([paves,no_ground,side, self.pcd.no_side])
			self.ui.road_maxh.setText(str(paves.get_max_bound()[2]))
			self.ui.road_minh.setText(str(paves.get_min_bound()[2]))
			paves_np = np.asarray(paves.points)
			self.data_show_h(paves_np,self.ui.road_show_image)
		elif self.road_show_index == 'svm1' :
			no_ground = self.gpf.np_to_o3d(self.pcd.no_ground)
			side = self.ground_o3d.select_by_index(self.pcd.side, invert= True) 
			paves = self.pcd.no_side.select_by_index(self.pcd.paves)
			paves.paint_uniform_color([1,0,0])
			o3d.visualization.draw_geometries([paves,no_ground,side, self.pcd.no_side])
			self.ui.road_maxh.setText(str(paves.get_max_bound()[2]))
			self.ui.road_minh.setText(str(paves.get_min_bound()[2]))
			paves_np = np.asarray(paves.points)
			self.data_show_h(paves_np,self.ui.road_show_image)




		


if __name__=="__main__":  
	app = QApplication(sys.argv)  
	win = MainWindow()  
	win.show()
	sys.exit(app.exec_()) 