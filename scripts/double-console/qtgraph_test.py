import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import random
import math
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

        # data initialization
        self.bar_value = 0

        # update
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_bars)
        self.timer.start(100)

    def update_bars(self):
        # simulating bars update
        self.bar_value = random.randint(0, 100)
        self.bar.setOpts(width=[self.bar_value])

        # simulating force angle
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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = TwoIndependentPlotWidgets()
    window.show()
    sys.exit(app.exec_())