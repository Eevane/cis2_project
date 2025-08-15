""" force detection visualization - for multi-teleop experiment """

import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import math
import random
import numpy as np

import argparse
import crtk
import geometry_msgs.msg
import PyKDL
import std_msgs.msg
import time

class ForceVisualization(QtWidgets.QMainWindow):
    def __init__(self, sensor, update_interval):
        super().__init__()
        self.setWindowTitle("Two Horizontal Bars")
        self.resize(3000, 3000)

        # set background figure
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        # PlotWidget of force magnitude
        self.plot1_range = 100

        self.plot1 = pg.PlotWidget()
        self.plot1.setBackground('w')
        self.plot1.setXRange(0, self.plot1_range)
        self.plot1.setYRange(-0.5, 0.5)
        self.plot1.getPlotItem().setTitle("Force Magnitude", color="k", size="16pt")
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
        bar_border = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[self.plot1_range], brush=None, pen=pg.mkPen(color='black', width=2))
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
        self.plot2.getPlotItem().setTitle("Force Vector (XY)", color="k", size="14pt")
        layout.addWidget(self.plot2)

        # set red zone
        r = 12
        theta = np.linspace(0, 2*np.pi, 100)
        cx = r * np.cos(theta)
        cy = r * np.sin(theta)
        self.plot2.plot(cx, cy, pen=None, brush=pg.mkBrush(255,0,0,50), fillLevel=0)

        # origin marker
        self.origin_marker = self.plot2.plot([0], [0], pen=None, symbol='o', symbolBrush='red', symbolSize=12)
        self.vector_line = self.plot2.plot([0, 0], [0, 0], pen=pg.mkPen('skyblue', width=4))

        ###################################################
        
        # initialize force sensor
        self.sensor = sensor
        self.update_interval = int(update_interval * 1000)

        # set timer to update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(self.update_interval)

    def update_plots_simulation(self):
        # force magnitude update
        bar_value = random.randint(0, 100)
        self.bar.setOpts(width=[bar_value])

        # force angle update
        # delete previous vector line (keep red circle and origin)
        for item in self.plot2.listDataItems():
            if item not in [self.red_zone_cir, self.origin_marker]:
                self.plot2.removeItem(item)

        # generate random vector
        r = random.uniform(0, 15)
        theta_deg = random.uniform(0, 360)
        x = r * math.cos(math.radians(theta_deg))
        y = r * math.sin(math.radians(theta_deg))

        # plot vector
        self.plot2.plot([0, x], [0, y], pen=pg.mkPen('skyblue', width=6))

    def update_plots(self):
        # query data from the force sensor
        try:
            wrench = self.sensor.measured_cf()[0]
            if wrench is None:
                return
            fx, fy, fz = wrench[0], wrench[1], wrench[2]
        except Exception as e:
            # don't let exceptions kill the timer
            print("Sensor read error:", e)
            return
        
        # force magnitude update
        force_mag = float(np.linalg.norm([fx,fy,fz]))
        self.bar.setOpts(width=[min(force_mag, self.plot1_range)])

        # force angle update
        self.vector_line.setData([0, fx], [0, fy])
    
class ForceSensor:
    def __init__(self, ral, arm_name, timeout):
        self.name = arm_name
        # self.ral = ral.create_child(arm_name)
        self.ral = ral
        self.utils = crtk.utils(self, self.ral, timeout)
        self.utils.add_measured_cf()


if __name__ == '__main__':    
    # parse arguments
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--interval', type=float, default=0.016,
                        help = 'time interval/period to read the force sensor, also update GUI')
    args = parser.parse_args()

    ral = crtk.ral('force_vis')
    sensor = ForceSensor(ral, 'sensor', 4*args.interval)
    ral.check_connections()

    ral.spin()
    # initialize GUI
    app = QtWidgets.QApplication(sys.argv)
    window = ForceVisualization(sensor, args.interval)
    window.show()
    ret = app.exec_()

    try:
        ral.shutdown()
    except Exception:
        print("ral.shutdown() failed!")
        pass
    sys.exit(ret)