import sys 	
from PyQt5.QtWidgets import QApplication , QMainWindow, QFileDialog
from taptap import Ui_MainWindow
from readdatas import *
import open3d as o3d
import numpy as np

class MainWindow(QMainWindow ):
	def __init__(self, parent=None):    
			super(MainWindow, self).__init__(parent)
			self.ui = Ui_MainWindow()
			self.ui.setupUi(self)
			self.gpf = GpfGroundExtractor()

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
		ground_o3d = self.gpf.np_to_o3d(self.pcd.ground)
		o3d.visualization.draw_geometries([ground_o3d])


if __name__=="__main__":  
	app = QApplication(sys.argv)  
	win = MainWindow()  
	win.show()
	print(win.ui.csf_side_com.currentText())
	print(win.ui.csf_side_com.currentText())
	sys.exit(app.exec_()) 