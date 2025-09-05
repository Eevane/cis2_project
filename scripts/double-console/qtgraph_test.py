import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import random
import math
import time
import csv
import datetime
import numpy as np

class ProgressBarGraph(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Force monitoring")
        self.resize(400, 200)   # intial size

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        self.bar = pg.BarGraphItem(x=[0], height=[0], width=0.8, brush='skyblue')
        self.plot_widget.addItem(self.bar)
        self.plot_widget.setYRange(0, 100)  # force amplitude range

        self.value = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_bar)
        self.timer.start(17)  # update once per 100 ms

    def update_bar(self):
        self.value = (self.value + 1) % 101  # simulate increasing from 1 to 100, then back to 100
        self.bar.setOpts(height=[self.value])

class HorizontalBarGraph(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Horizontal Force Monitoring")
        self.resize(500, 150)  # initial size width 500, height 150

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # bar initialization
        self.bar = pg.BarGraphItem(x0=[0], y=[0], height=0.8, width=[0], brush='skyblue')
        self.plot_widget.addItem(self.bar)

        # set y range
        self.plot_widget.setYRange(-0.5, 0.5)  # horizontal bar
        self.plot_widget.setXRange(0, 100)     # force amplitude range

        self.value = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_bar)
        self.timer.start(16)                   # fresh rate -- 60 FPS

    def update_bar(self):
        self.value = (self.value + 1) % 101
        self.bar.setOpts(width=[self.value])  # update bar width

class TwoHorizontalBars(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two Independent Horizontal Bars")
        self.resize(500, 200)

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # initialize two bars
        self.bar1 = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[0], brush='skyblue')
        self.bar2 = pg.BarGraphItem(x0=[0], y=[1], height=0.4, width=[0], brush='lightgreen')
        self.plot_widget.addItem(self.bar1)
        self.plot_widget.addItem(self.bar2)

        self.plot_widget.setYRange(-0.5, 1.5)
        self.plot_widget.setXRange(0, 100)

        # data queue
        self.value1 = 0
        self.value2 = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_bars)
        self.timer.start(50)

    def update_bars(self):
        self.value1 = random.randint(0, 100)  # simulate force magnitude
        self.value2 = random.randint(0, 100)  # simulate force angle

        self.bar1.setOpts(width=[self.value1])
        self.bar2.setOpts(width=[self.value2])

class TwoIndependentPlotWidgets(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two Horizontal Bars")
        self.resize(3000, 3000)

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
        self.plot1_range = 100
        self.plot1 = pg.PlotWidget()
        self.plot1.setBackground('w')
        self.plot1.setXRange(0, self.plot1_range)
        self.plot1.setYRange(-0.5, 0.5)
        self.plot1.getPlotItem().setTitle("Force Magnitude", color="k", size="16pt")
        main_layout.addWidget(self.plot1)

        self.bar = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[0], brush='skyblue')
        self.plot1.addItem(self.bar)

        # set green zone
        self.threshold_start = 30
        self.threshold_end = 50
        bar_y = 0
        bar_height = 0.4
        red_zone = pg.BarGraphItem(
            x0=[self.threshold_start],
            y=[bar_y],
            height=bar_height,
            width=[self.threshold_end - self.threshold_start],
            brush=pg.mkBrush(0, 255, 0, 100)
        )
        self.plot1.addItem(red_zone)

        # set border
        bar_border = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[self.plot1_range], brush=None, pen=pg.mkPen(color='black', width=2))
        self.plot1.addItem(bar_border)

        ###################################################

        # PlotWidget of force vector
        self.plot2 = pg.PlotWidget()
        self.plot2.setBackground('w')
        # self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.plot2.setLabel('bottom', 'X', units='N')
        self.plot2.setLabel('left', 'Y', units='N')

        self.plot2.setXRange(-20, 20)
        self.plot2.setYRange(-20, 20)
        self.plot2.setAspectLocked(True)
        self.plot2.getPlotItem().setTitle("Force Angle", color="k", size="16pt")
        main_layout.addWidget(self.plot2)

        # set red zone
        self.radius = 12
        theta = np.linspace(0, 2*np.pi, 100)
        cx = self.radius * np.cos(theta)
        cy = self.radius * np.sin(theta)
        self.plot2.plot(cx, cy, pen=None, brush=pg.mkBrush(255,0,0,50), fillLevel=0)

        # origin marker
        self.origin_marker = self.plot2.plot([0], [0], pen=None, symbol='o', symbolBrush='red', symbolSize=12)
        self.vector_line = self.plot2.plot([0, 0], [0, 0], pen=pg.mkPen('skyblue', width=4))

        # ----- Recording state -----
        self.is_recording = False
        self.records = []  # list of dicts or tuples

        # update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_bars)
        self.timer.start(33)

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
        fname = f"../../../Dataset/force_record/force_record_{timestamp}.csv"
        try:
            with open(fname, "w", newline='') as f:
                writer = csv.writer(f)
                # header
                writer.writerow(["time_s", "force", "fx", "fy", "fz", "vector_r", "vector_theta_deg"])
                for rec in self.records:
                    writer.writerow([
                        f"{rec['t']:.6f}", rec['bar'], rec['fx'], rec['fy'], rec['fz'],
                        rec['r'], rec['theta_deg']
                    ])
            print(f"Saved {len(self.records)} records to {fname}")
        except Exception as e:
            print("Failed to save records:", e)

    def update_bars(self):
        # simulating bars update
        force = random.randint(0, self.plot1_range)
        self.bar.setOpts(width=[force])

        # simulating force vector
        # generate random vector
        r = random.uniform(0, self.radius+5)
        theta_deg = random.uniform(0, 360)
        fx = r * math.cos(math.radians(theta_deg))
        fy = r * math.sin(math.radians(theta_deg))
        fz = 0.0
        # plot vector
        self.vector_line.setData([0, fx], [0, fy])

        # if recording, append a record
        if self.is_recording:
            rec_time = time.time() - getattr(self, "_record_start_time", time.time())
            self.records.append({
                "t": rec_time,
                "bar": force,
                "fx": fx,
                "fy": fy,
                "fz": fz,
                "r": r,
                "theta_deg": theta_deg
            })


class ForceVectorWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Force Vector")
        self.resize(400, 400)

        # PlotWidget
        self.plot = pg.PlotWidget()
        self.setCentralWidget(self.plot)

        # show grid and axis
        self.plot.setBackground('w')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('bottom', 'X', units='N')
        self.plot.setLabel('left', 'Y', units='N')

        # plot initialization
        self.plot.setXRange(-20, 20)
        self.plot.setYRange(-20, 20)
        self.plot.setAspectLocked(True)   # equal axis scale

        # set red zone
        r = 12
        theta = np.linspace(0, 2*np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.red_zone = self.plot.plot(x, y, pen=None, brush=pg.mkBrush(255,0,0,50), fillLevel=0)

        # Timer update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_vector)
        self.timer.start(100)

    def update_vector(self):
        # simulate a force vector
        r = random.uniform(0, 15)           # magnitude
        theta_deg = random.uniform(0, 360)  # shifted angle

        # transform to cartesian space
        x = r * math.cos(math.radians(theta_deg))
        y = r * math.sin(math.radians(theta_deg))

        # delete last vector
        for item in self.plot.listDataItems():
            if item != self.red_zone:
                self.plot.removeItem(item)

        # mark the origin point
        self.plot.plot([0], [0], pen=None, symbol='o', symbolBrush='red', symbolSize=6)

        # current vector plot
        self.plot.plot([0, x], [0, y], pen=pg.mkPen('skyblue', width=10))


class SimulationPlotWidgets(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two Horizontal Bars")
        self.resize(3000, 3000)

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
        self.plot1_range = 100
        self.plot1 = pg.PlotWidget()
        self.plot1.setBackground('w')
        self.plot1.setXRange(0, self.plot1_range)
        self.plot1.setYRange(-0.5, 0.5)
        self.plot1.getPlotItem().setTitle("Force Magnitude", color="k", size="16pt")
        main_layout.addWidget(self.plot1)

        self.bar = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[0], brush='skyblue')
        self.plot1.addItem(self.bar)

        # add time counting function
        self.safety_time_text = pg.TextItem(text="Safety Time: 0.00 / 0.00 s",
                                            color='black',
                                            anchor=(0,1))
        self.safety_time_text.setFont(QtGui.QFont("Arial", 25))
        self.plot1.addItem(self.safety_time_text)
        self.safety_time_text.setPos(2, 0.45)


        # set green zone
        self.threshold_start = 30
        self.threshold_end = 50
        bar_y = 0
        bar_height = 0.4
        red_zone = pg.BarGraphItem(
            x0=[self.threshold_start],
            y=[bar_y],
            height=bar_height,
            width=[self.threshold_end - self.threshold_start],
            brush=pg.mkBrush(0, 255, 0, 100)
        )
        self.plot1.addItem(red_zone)

        # set border
        bar_border = pg.BarGraphItem(x0=[0], y=[0], height=0.4, width=[self.plot1_range], brush=None, pen=pg.mkPen(color='black', width=2))
        self.plot1.addItem(bar_border)

        # ----- Recording state -----
        self.is_recording = False
        self.records = []  # list of dicts or tuples
        self.safety_count = 0
        self.refresh_interval = 0.033

        # update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_bars)
        self.timer.start(int(self.refresh_interval * 1000))

    def start_recording(self):
        # clear old records and enable stop
        self.records = []
        self.safety_count = 0
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

        safety_time = self.safety_count * self.refresh_interval
        total_time = len(self.records) * self.refresh_interval
        self.safety_time_text.setText(f"Safety Time: {safety_time:.2f} / {total_time:.2f} s")
        
        # save to CSV for later analysis (time)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"../../../Dataset/force_record/force_record_{timestamp}.csv"
        try:
            with open(fname, "w", newline='') as f:
                writer = csv.writer(f)
                # header
                writer.writerow(["time_s", "force"])
                for rec in self.records:
                    writer.writerow([
                        f"{rec['t']:.6f}", rec['bar'], 
                    ])

                writer.writerow([])
                writer.writerow(["total_time_in_safety_zone (s)", f"{safety_time:.6f}"])
                writer.writerow(["total_time (s)", f"{total_time:.6f}"])


            print(f"Saved {len(self.records)} records to {fname}")
        except Exception as e:
            print("Failed to save records:", e)

    def update_bars(self):
        # simulating bars update
        force = random.randint(0, self.plot1_range)
        self.bar.setOpts(width=[force])

        # if recording, append a record, update count
        if self.is_recording:
            rec_time = time.time() - getattr(self, "_record_start_time", time.time())
            self.records.append({
                "t": rec_time,
                "bar": force,
            })

            if self.threshold_start <= force <= self.threshold_end:
                self.safety_count += 1
        
        total_time = len(self.records) * self.refresh_interval
        safety_time = self.safety_count * self.refresh_interval
        self.safety_time_text.setText(f"Safety Time: {safety_time:.2f} / {total_time:.2f} s")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = SimulationPlotWidgets()
    window.show()
    sys.exit(app.exec_())