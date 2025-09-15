"""
Notes, It's working but a bit rough

1) No axis on the main plot
2) Add symbols to the main plot with the location of the extracted spectra highlihgted
3) Add the mesh or just scatter points to show what's been collected.
"""


import sys
import random
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QScrollArea, QGridLayout, QMainWindow, QSlider,
    QDialog, QFileDialog
)

import pyqtgraph as pg
import numpy as np
import time
import os
from scipy.interpolate import griddata, LinearNDInterpolator, CloughTocher2DInterpolator


import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

"""Global Qt Options"""
pg.setConfigOptions(imageAxisOrder='row-major')


"""
Class for storing Variables
"""

class Variables:
    def __init__(self):
        self._vars = {
            "ccd_target_temp": -90,
            "variables": {
                "Working Directory": "Results/2025-09-04/RM_2/",
                "Exposure Time (s)": 10,
                "Number of Samples": 3,
                "Center Wavelength (nm)": 613.4,
                "Slit Width (um)": 100,
                "X Center (um)": 100,
                "Y Center (um)": 100,
                "dx (um)": 200,
                "dy (um)": 200,
                "Wait time (s)": 0.05,
                "Adaptive Loss Condition": 0.001,
                "Sampling Wavelength (nm)": 615
            },
        }

        self.x = np.array
        self.y = np.array
        self.spectra = np.array
        self.wavelengths = np.array

    def set_variable(self, name: str, value):
        self._vars["variables"][name] = value

    def get_variable(self, name: str):
        return self._vars["variables"].get(name, None)

    def get_all_variables(self):
        return self._vars["variables"]

    def __repr__(self):
        return str(self._config)


"""
Window dialogue for selecting directories for opening data
"""

class FileBrowserDialog(QFileDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open a Dataset")
        self.resize(600,400)


"""
Main GUI Class
"""
class ExperimentGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experiment GUI")

        self.config = Variables()  

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel (buttons + controls)
        self.left_panel = QVBoxLayout()
        main_layout.addLayout(self.left_panel, 1)

        self.init_controls()

        # Right panel (image)
        map_and_spectrum_layout = QVBoxLayout()
        main_layout.addLayout(map_and_spectrum_layout, 4)
        self.init_plot_area()
        map_and_spectrum_layout.addWidget(self.map_widget, 4)
        #map_and_spectrum_layout.addWidget(self.spectrum_widget,1)
        #main_layout.addWidget(self.img,3)

        # Start image loader
        #self.start_csv_image_loader()

    def init_controls(self):
        self.open_button = QPushButton("Open File")
        self.left_panel.addWidget(self.open_button)
        self.open_button.clicked.connect(self.open_file_browser)

    def open_file_browser(self):
        """Callback function to open the filebrowser window"""
        dialog = FileBrowserDialog()
        dialog.exec()

    def closeEvent(self, event):
        # Make sure to stop the CSV image loader
        if hasattr(self, 'csv_worker'):
            self.csv_worker.stop()
            self.csv_thread.quit()
            self.csv_thread.wait()

        event.accept()

    # Callbacks for handling user interaction
    def updatePlot(self):
        #global img, roi, data, p2
        selected = self.roi.getArrayRegion(self.data, self.img)
        self.p2.plot(selected.mean(axis=0), clear=True)

    def updateIsocurve(self):
        global isoLine, iso
        self.iso.setLevel(self.isoLine.value())


    def imageHoverEvent(self,event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.data.shape[0] - 1))
        j = int(np.clip(j, 0, self.data.shape[1] - 1))
        val = self.data[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        self.p1.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %.3g" % (x, y, i, j, val))

    def init_plot_area(self):

        self.map_widget = pg.GraphicsLayoutWidget()

        self.map_widget.setWindowTitle('Data Analysis')


        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.map_widget.addPlot()

        # Item for displaying image data
        self.img = pg.ImageItem()
        self.p1.addItem(self.img)

        # Custom ROI for selecting an image region
        self.roi = pg.ROI([-8, 14], [6, 5])
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.p1.addItem(self.roi)
        self.roi.setZValue(10)  # make sure ROI is drawn above image

        # Isocurve drawing
        self.iso = pg.IsocurveItem(level=0.8, pen='g')
        self.iso.setParentItem(self.img)
        self.iso.setZValue(5)

        # Contrast/color control
        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img)
        self.map_widget.addItem(hist)

        # Draggable line for setting isocurve level
        self.isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        hist.vb.addItem(self.isoLine)
        hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
        self.isoLine.setValue(0.8)
        self.isoLine.setZValue(1000) # bring iso line above contrast controls
        
        # Another plot area for displaying ROI data
        self.map_widget.nextRow()
        self.p2 = self.map_widget.addPlot(colspan=2)
        self.p2.setMaximumHeight(250)
        self.map_widget.resize(800,800)
        
        
        # Generate image data
        self.data = np.random.normal(size=(200, 100))
        self.data[20:80, 20:80] += 2.
        self.data = pg.gaussianFilter(self.data, (3, 3))
        self.data += np.random.normal(size=(200, 100)) * 0.1
        self.img.setImage(self.data)
        hist.setLevels(self.data.min(), self.data.max())

        # build isocurves from smoothed data
        self.iso.setData(pg.gaussianFilter(self.data, (2, 2)))

        # set position and scale of image
        tr = QtGui.QTransform()
        self.img.setTransform(tr.scale(0.2, 0.2).translate(-50, 0))

        # zoom to fit imageo
        self.p1.autoRange()  

        

        # Connect to callbacks
        self.roi.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot()
        self.isoLine.sigDragged.connect(self.updateIsocurve)
        # Monkey-patch the image to use our custom hover function. 
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this. 
        self.img.hoverEvent = self.imageHoverEvent

        

 



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ExperimentGUI()
    gui.resize(1200, 800)
    gui.show()
    sys.exit(app.exec())
