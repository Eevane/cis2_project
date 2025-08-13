import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import random
import math
import numpy as np
from NetFT import Sensor

class ForceVisualization(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two Horizontal Bars")
        self.resize(3000, 3000)

        # set background figure
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        # PlotWidget of force magnitude
        plot1_range = 100

        self.plot1 = pg.PlotWidget()
        self.plot1.setBackground('w')
        self.plot1.setXRange(0, plot1_range)
        self.plot1.setYRange(-0.5, 0.5)
        self.plot1.setTitle("Force Magnitude", color="k", size="16pt")
        layout.addWidget(self.plot1)

        self.bar = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[0], brush='skyblue')
        self.plot1.addItem(self.bar)

        # set red zone
        threshold_start = 30
        threshold_end = 50
        bar_y = 0
        bar_height = 0.4
        red_zone = pg.BarGraphItem(
            x0=[threshold_start],
            y=[bar_y],
            height=bar_height,
            width=[threshold_end - threshold_start],
            brush=pg.mkBrush(255, 0, 0, 100)
        )
        self.plot1.addItem(red_zone)

        # set border
        bar_border = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[plot1_range], brush=None, pen=pg.mkPen(color='black', width=2))
        self.plot1.addItem(bar_border)

        # data initialization
        self.bar_value = 0

        ###################################################

        # PlotWidget of force angle
        self.plot2 = pg.PlotWidget()
        self.plot2.setBackground('w')
        # self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.plot2.setLabel('bottom', 'X', units='N')
        self.plot2.setLabel('left', 'Y', units='N')

        self.plot2.setXRange(-20, 20)
        self.plot2.setYRange(-20, 20)
        self.plot2.setAspectLocked(True)
        self.plot2.setTitle("Force Angle", color="k", size="16pt")
        layout.addWidget(self.plot2)

        # set red zone
        r = 12
        theta = np.linspace(0, 2*np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.red_zone_cir = self.plot2.plot(x, y, pen=None, brush=pg.mkBrush(255,0,0,50), fillLevel=0)

        # origin marker
        self.origin_marker = self.plot2.plot([0], [0], pen=None, symbol='o', symbolBrush='red', symbolSize=12)

        ###################################################
        
        # initialize force sensor
        self.sensor = Sensor('IP Address!')

        # set timer to update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)

    def update_plots(self):
        force_mag, angle_x, angle_z = self.force_processing()
        if force_mag is None:
            pass

        # force magnitude update
        self.bar.setOpts(width=[force_mag])

        # force angle update
        # delete previous vector line (keep red circle and origin)
        for item in self.plot2.listDataItems():
            if item not in [self.red_zone_cir, self.origin_marker]:
                self.plot2.removeItem(item)

        # generate random vector
        x = angle_z * math.cos(math.radians(angle_x))
        y = angle_z * math.sin(math.radians(angle_x))

        # plot vector
        self.plot2.plot([0, x], [0, y], pen=pg.mkPen('skyblue', width=6))
    
    def force_processing(self):
        force_data = self.sensor.getForce()
        if force_data:
            # compute magnitude and angle of force vector
            force_mag = np.linalg.norm(force_data)

            refer_x = [0, 0, 1]
            angle_x = self.get_vec_angle(force_data, refer_x)

            refer_z = [0, 0, 1]
            angle_z = self.get_vec_angle(force_data, refer_z)
            return force_mag, angle_x, angle_z
        else:
            return None, None, None
        
    def get_vec_angle(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)

        # # rad 2 deg
        # angle_deg = np.degrees(angle_rad)
        return angle_rad
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ForceVisualization()
    window.show()
    sys.exit(app.exec_())