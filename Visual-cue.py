""" force detection visualization - for multi-teleop experiment """

import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import math
import random
import numpy as np

import csv
import datetime
import argparse
import crtk
import geometry_msgs.msg
import PyKDL
import std_msgs.msg
import time

class ForceVisualization(QtWidgets.QMainWindow):
    def __init__(self, sensor, update_interval):
        super().__init__()
        self.setWindowTitle("Force Visualization GUI")
        self.resize(2000, 3000)

        # set background figure
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # ----- Controls (buttons) -----
        controls_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start Recording")
        self.btn_stop = QtWidgets.QPushButton("Stop and Save")
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # connect buttons
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)

        # PlotWidget of force magnitude
        self.plot1_range = 8
        self.plot1 = pg.PlotWidget()
        self.plot1.setBackground('w')
        self.plot1.setXRange(0, self.plot1_range)
        self.plot1.setYRange(-0.5, 0.5)
        self.plot1.getPlotItem().setTitle("Force Magnitude", color="k", size="16pt")
        main_layout.addWidget(self.plot1)

        self.bar = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[0], brush='skyblue')
        self.plot1.addItem(self.bar)

        # set red zone
        threshold_start = 2
        threshold_end = 4
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
        self.plot2.getPlotItem().setTitle("Force Vector (XY)", color="k", size="16pt")
        main_layout.addWidget(self.plot2)

        # set red zone
        self.radius = 6
        theta = np.linspace(0, 2*np.pi, 100)
        cx = self.radius * np.cos(theta)
        cy = self.radius * np.sin(theta)
        self.plot2.plot(cx, cy, pen=None, brush=pg.mkBrush(255,0,0,50), fillLevel=0)

        # origin marker
        self.origin_marker = self.plot2.plot([0], [0], pen=None, symbol='o', symbolBrush='red', symbolSize=12)
        self.vector_line = self.plot2.plot([0, 0], [0, 0], pen=pg.mkPen('skyblue', width=4))

        ###################################################
        
        # initialize force sensor
        self.sensor = sensor
        self.update_interval = int(update_interval * 1000)

        # ----- Recording state -----
        self.is_recording = False
        self.records = []  # list of dicts or tuples

        # update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(16)

    def start_recording(self):
        # clear old records and enable stop
        self.records = []
        self.is_recording = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._record_start_time = time.time()
        print("Recording started.")

    def stop_recording(self):
        # clear old records and enable stop
        self.is_recording = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        # save to CSV for later analysis (time)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"njiang19/dvrk_teleop_data/Aug_17/force_record/force_record_{timestamp}.csv"
        try:
            with open(fname, "w", newline='') as f:
                writer = csv.writer(f)
                # header
                writer.writerow(["time_s", "bar_value", "fx", "fy", "fz", "vector_r", "vector_theta_deg"])
                for rec in self.records:
                    writer.writerow([
                        f"{rec['t']:.6f}", rec['bar'], rec['fx'], rec['fy'], rec['fz'],
                        rec['r'], rec['theta_deg']
                    ])
            print(f"Saved {len(self.records)} records to {fname}")
        except Exception as e:
            print("Failed to save records:", e)

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
            print("Sensor read error:", e)
            sys.exit(ret)
            return
        
        # force magnitude update
        force_mag = float(np.linalg.norm([fx,fy,fz]))
        self.bar.setOpts(width=[min(force_mag, self.plot1_range)])

        # force angle update
        self.vector_line.setData([0, fx], [0, fy])

        # if recording, append a record
        if self.is_recording:
            rec_time = time.time() - getattr(self, "_record_start_time", time.time())
            self.records.append({
                "t": rec_time,
                "bar": bar_value,
                "fx": fx,
                "fy": fy,
                "fz": fz,
                "r": r,
                "theta_deg": theta_deg
            })
    
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