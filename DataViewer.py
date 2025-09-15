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
from PySide6.QtCore import Qt, QThread, Signal, QObject

from PySide6.QtGui import QTransform
import pyqtgraph as pg
import numpy as np
import time
import os
from scipy.interpolate import griddata, LinearNDInterpolator, CloughTocher2DInterpolator


def closest(lst, K):
    item = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
    return np.where(lst == item)[0][0]

# 🔧 Placeholder for actual temperature reading function
def get_ccd_temperature():
    # Simulate changing temperature (you'll replace this)
    return round(random.uniform(-85.0, -75.0), 2)

class ConfigStore:
    def __init__(self):
        self._config = {
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

    def set_ccd_target_temp(self, temp: float):
        self._config["ccd_target_temp"] = temp

    def get_ccd_target_temp(self) -> float:
        return self._config["ccd_target_temp"]

    def set_variable(self, name: str, value):
        self._config["variables"][name] = value

    def get_variable(self, name: str):
        return self._config["variables"].get(name, None)

    def get_all_variables(self):
        return self._config["variables"]

    def __repr__(self):
        return str(self._config)

class CSVImageLoaderWorker(QObject):
    image_updated = Signal(np.ndarray)
    coordinates_updated = Signal(np.ndarray)
    data_updated = Signal(dict)
    finished = Signal()

    def __init__(self, config: ConfigStore, poll_interval=1.0):
        super().__init__()
        self.config = config
        self.poll_interval = poll_interval
        self._running = True
        self.last_modified_time = None

    def load_file(self,workingDir):

        data = np.loadtxt(workingDir + "spectra.csv", delimiter=",")
        wavelengths = np.genfromtxt(workingDir + "wavelenths.csv")*1e9 # load and convert to nm
        
        x = np.array(data[:,0],dtype=float)
        y = np.array(data[:,1],dtype=float)
        spectra = np.array(data[:,3:],dtype=float)

        data_dict = {
            "x":x,
            "y":y,
            "wavelengths":wavelengths,
            "spectra":spectra,
        }

        self.data_updated.emit(data_dict)

        sampling_wavelength = self.config.get_variable("Sampling Wavelength (nm)")
        idx2plot = closest(wavelengths,sampling_wavelength)
        
        
        intensity = spectra[:,idx2plot]

        
        interp_i   = LinearNDInterpolator(list(zip(x, y)), intensity,rescale=True)

        """
        Check for the smallest triangle vertex in-order to decide the interpolation resolution
        and then interpolate the unstructured dataset to a regular grid.
        """

        points = []
        for i in range(0,len(x)):
            points.append((x[i],y[i]))

        def euclideanDistance(coordinate1, coordinate2):
            return pow(pow(coordinate1[0] - coordinate2[0], 2) + pow(coordinate1[1] - coordinate2[1], 2), .5)

        distances = []
        for i in range(len(points)-1):
            for j in range(i+1, len(points)):
                distances += [euclideanDistance(points[i],points[j])]

        dx = min(distances)
        
        if dx>5:
            dx = 5

        dy = dx

        x1 = min(x)
        x2 = max(x)
        y1 = min(y)
        y2 = max(y)

        xnew = np.arange(x1,x2+dx,dx)
        ynew = np.arange(y1,y2+dy,dy)
        X,Y = np.meshgrid(xnew,ynew)
        
        
        
        #grid_z1 = interp_i(X,Y)
        
        return np.array([dx,dy,xnew[0],ynew[0]]),interp_i(X,Y)
        

    def run(self):
        while self._running:
            try:
                working_dir = self.config.get_variable("Working Directory")
                file_path = os.path.join(working_dir, "spectra.csv")

                if os.path.exists(file_path):
                    modified_time = os.path.getmtime(file_path)
                    if self.last_modified_time is None or modified_time > self.last_modified_time:
                        self.last_modified_time = modified_time

                        # Load the new image
                        #data = np.loadtxt(file_path, delimiter=",")
                        
                        data = self.load_file(working_dir)
                        
                        if data[1].ndim == 2:
                            self.image_updated.emit(data[1])
                            self.coordinates_updated.emit(data[0])
            except Exception as e:
                print(f"[CSV Loader] Error loading CSV: {e}")

            time.sleep(self.poll_interval)

    def stop(self):
        self._running = False


class FileBrowserDialog(QFileDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open a Dataset")
        self.resize(600,400)




class LineProfileDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Profile Viewer")
        self.resize(700, 500)

        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
        self.color_index = 0
        self.plots = []
        self.annotations = []

        # Layout
        layout = QVBoxLayout(self)

        # Plot widget
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Wavelength, nm')
        self.plot.setLabel('left', 'Intensity')
        self.plot.setTitle("Spectra")
        self.plot.addLegend()
        layout.addWidget(self.plot)

        # Button layout
        btn_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all)
        btn_layout.addStretch()
        btn_layout.addWidget(self.clear_button)
        layout.addLayout(btn_layout)

    def add_line(self, x,y,wavelength,intensity):
        

        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1

        # Plot the line
        plot_item = self.plot.plot(
            wavelength,
            intensity,
            pen=pg.mkPen(color=color, width=2),
            name=f"X= {x}, Y = {y}"
        )
        self.plots.append(plot_item)

        # Add text annotation
        #text = pg.TextItem(f"Y = {y}", color=color, anchor=(1, 0))
        #self.plot.addItem(text)
        #text.setPos(wavelength[-1], intensity[-1])  # Place at end of line
        #self.annotations.append(text)

    def clear_all(self):
        for item in self.plots:
            self.plot.removeItem(item)
        self.plots.clear()

        for annotation in self.annotations:
            self.plot.removeItem(annotation)
        self.annotations.clear()

        self.color_index = 0





class ExperimentGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experiment GUI")

        self.config = ConfigStore()  

        self.line_profile_window = None
        self.fileBrowser_window = None


        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel (buttons + controls)
        self.left_panel = QVBoxLayout()
        main_layout.addLayout(self.left_panel, 1)

        self.init_controls()

        # Right panel (image)
        self.init_plot_area()
        main_layout.addWidget(self.plot_widget, 4)

        # Start image loader
        self.start_csv_image_loader()

    def init_controls(self):
        self.open_button = QPushButton("Open File")
        self.left_panel.addWidget(self.open_button)

        self.open_button.clicked.connect(self.open_file_browser)

    def open_file_browser(self):
        dialog = FileBrowserDialog()
        dialog.exec()

    def open_line_profile_window(self, x, y):
        dialog = LineProfileDialog(x, y, self.image_data)
        dialog.exec()

    def closeEvent(self, event):
        # Stop CSV image loader
        if hasattr(self, 'csv_worker'):
            self.csv_worker.stop()
            self.csv_thread.quit()
            self.csv_thread.wait()

        event.accept()



    def init_plot_area(self):

        # Set a custom color map
        cmap = pg.colormap.getFromMatplotlib('jet')

        #plot = pg.PlotItem()
        #plot.setLabel(axis='left', text='X-axis')
        #plot.setLabel(axis='bottom', text='Y-axis')


        self.plot_widget = pg.ImageView()

        self.image_data = np.random.rand(512, 512)
        self.plot_widget.setImage(self.image_data)


        tr = QTransform()
        tr.translate(-256,-256)

        # Scale the image
        self.plot_widget.setImage(self.image_data,transform=tr)

        self.plot_widget.setColorMap(cmap)
        # Enable mouse interaction
        #self.plot_widget.scene.mousePressEvent = self.image_mouse_press_event

    


    def image_mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            
            pos = event.scenePos()
            vb = self.plot_widget.getView()
            mouse_point = vb.mapSceneToView(pos)

            x = int(mouse_point.x())
            y = int(mouse_point.y())

            print("QtWindow: ",x,y)
            self.add_spectra(x, y)

    def add_spectra(self, x, y):
        if self.line_profile_window is None or not self.line_profile_window.isVisible():
            self.line_profile_window = LineProfileDialog()
            self.line_profile_window.show()

        coordinate_pick = np.array((y,x))
        coordinates = []
        for i in range(0,len(self.xpos)):
            coordinates.append((self.xpos[i],self.ypos[i]))
    

        distances = np.linalg.norm(coordinates-coordinate_pick, axis=1)
        min_index = np.argmin(distances)
        print(f"the closest point is {coordinates[min_index]}, at a distance of {distances[min_index]}")

        self.line_profile_window.add_line(coordinates[min_index][0],coordinates[min_index][1],self.wavelengths,self.spect[min_index,:])

    def start_csv_image_loader(self):
        self.csv_thread = QThread()
        self.csv_worker = CSVImageLoaderWorker(self.config, poll_interval=1.0)

        self.csv_worker.moveToThread(self.csv_thread)
        self.csv_thread.started.connect(self.csv_worker.run)
        self.csv_worker.image_updated.connect(self.update_image_data)
        self.csv_worker.coordinates_updated.connect(self.update_image_coordinates)
        self.csv_worker.data_updated.connect(self.update_data)
        self.csv_worker.finished.connect(self.csv_thread.quit)

        self.csv_thread.start()

    def update_data(self,data_dict):
        self.xpos = data_dict['x']
        self.ypos = data_dict['y']
        self.wavelengths = data_dict['wavelengths']
        self.spect = data_dict['spectra']

        #self.plot_widget.plot(
        #    self.xpos,
        #    self.ypos,
        #    pen=None,
        #    symbol='o',
        #    symbolSize=1,
        #    #symbolBrush=pg.mkColor((256, g, b, 150))
        #    )


    def update_image_data(self, new_image):
        self.image_data = new_image
        self.plot_widget.setImage(self.image_data)

    def update_image_coordinates(self,coordinates):
        dx = coordinates[0]
        dy = coordinates[1]
        tx = coordinates[2]
        ty = coordinates[3]

        print(coordinates)
        tr = QTransform()
        tr.scale(dy, dx)       # scale horizontal and vertical axes
        tr.translate(ty/dy, tx/dx) # move 3x3 image to locate center at axis origin

        self.plot_widget.imageItem.setTransform(tr)
        self.plot_widget.adjustSize()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ExperimentGUI()
    gui.resize(1200, 800)
    gui.show()
    sys.exit(app.exec())
