"""
Notes, It's working but a bit rough

1) No axis on the main plot
2) Add symbols to the main plot with the location of the extracted spectra highlihgted
3) Add the mesh or just scatter points to show what's been collected.
"""

import matplotlib.pyplot as plt
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


import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

"""Global Qt Options"""
pg.setConfigOptions(imageAxisOrder='row-major')


"""Global Functions"""
def closest(lst, K):
    item = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
    return np.where(lst == item)[0][0]


"""
Class for storing Variables
"""

class Variables:
    def __init__(self):
        self._vars = {
            #Experiment Settings
            "Working Directory": "Results/2025-09-04/RM_2/",
            
            #Spectrometer Camera Settings
            "CCD Temp (C)": -90,
            "Exposure Time (s)": 10,
            "Number of Samples": 3,
            
            #Spectrometer Settings
            "Center Wavelength (nm)": 613.4,
            "Slit Width (um)": 100,
            
            #Map settings
            "X Center (um)": 100,
            "Y Center (um)": 100,
            "dx (um)": 100,
            "dy (um)": 100,
            "Wait time (s)": 0.05,
            "Adaptive Loss Condition": 0.001,
            "Sampling Wavelength (nm)": 615

        }

        self.x = np.array
        self.y = np.array
        self.spectra = np.array
        self.wavelengths = np.array

    def set_variable(self, name: str, value):
        self._vars[name] = value

    def get_variable(self, name: str):
        return self._vars.get(name, None)

    def get_all_variables(self):
        return self._vars


"""
Window dialogue for selecting directories for opening data
"""

class FileBrowserDialog(QFileDialog):
    def __init__(self,variables: Variables):
        super().__init__()
        self.variables = variables
        self.setWindowTitle("Open a Dataset")
        self.resize(600,400)

        self.fileSelected.connect(self.loadData_and_setPath)
        

    def loadData_and_setPath(self,path):
        #print("Load in the data from a file")
        #print(path)
        #print(os.path.dirname(os.path.abspath(path)))
        wD = os.path.dirname(os.path.abspath(path))
        self.variables.set_variable("Working Directory",wD)

"""
Window dialogue for changing variables
"""

class VariableDialog(QDialog):
    def __init__(self, Variables: Variables):
        super().__init__()
        self.setWindowTitle("Set Variables")
        self.setMinimumWidth(450)
        self.Variables = Variables

        layout = QVBoxLayout()
        self.input_fields = {}

        variable_names = [
            #Experiment Settings
            #"Working Directory",
            
            #Spectrometer Camera Settings
            "CCD Temp (C)",
            "Exposure Time (s)",
            "Number of Samples",
            
            #Spectrometer Settings
            "Center Wavelength (nm)",
            "Slit Width (um)",
            
            #Map settings
            "X Center (um)",
            "Y Center (um)",
            "dx (um)",
            "dy (um)",
            "Wait time (s)",
            "Adaptive Loss Condition",
            "Sampling Wavelength (nm)"
        ]

        for name in variable_names:
            row = QHBoxLayout()
            label = QLabel(name)
            input_line = QLineEdit()
            input_line.setText(str(self.Variables.get_variable(name) or ""))
            self.input_fields[name] = input_line
            row.addWidget(label)
            row.addWidget(input_line)
            layout.addLayout(row)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.set_button = QPushButton("Set")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.set_button)
        button_layout.addWidget(self.cancel_button)

        layout.addSpacing(10)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.cancel_button.clicked.connect(self.close)
        self.set_button.clicked.connect(self.set_variables)

    def set_variables(self):
        for name, field in self.input_fields.items():
            self.Variables.set_variable(name, field.text())
        self.accept()

"""
Window for controlling the MCL stage manually
"""

class MCL_Directional_CTRL(QDialog):
    def __init__(self, Variables: Variables):
        super().__init__()
        self.setWindowTitle("MCL Stage Control")
        self.setMinimumWidth(450)
        self.Variables = Variables

        self.up_button = QPushButton("-x")
        self.down_button = QPushButton("+x")
        self.left_button = QPushButton("-y")
        self.right_button = QPushButton("+y")
        self.z_up_button = QPushButton("Z+")
        self.z_down_button = QPushButton("Z-")

        self.x_pos_label = QLabel("X: 0.0")
        self.y_pos_label = QLabel("Y: 0.0")
        self.z_pos_label = QLabel("Z: 0.0")

        grid = QGridLayout()
        
        grid.addWidget(self.up_button, 0, 1)
        grid.addWidget(self.left_button, 1, 0)
        grid.addWidget(self.right_button, 1, 2)
        grid.addWidget(self.down_button, 2, 1)
        #grid.addWidget(self.z_up_button, 1, 1)
        #grid.addWidget(self.z_down_button, 3, 1)

        pos_layout = QVBoxLayout()
        pos_layout.addWidget(self.x_pos_label)
        pos_layout.addWidget(self.y_pos_label)
        pos_layout.addWidget(self.z_pos_label)

        container = QHBoxLayout()
        button_widget = QWidget()
        button_widget.setLayout(grid)

        container.addWidget(button_widget)
        container.addLayout(pos_layout)

        self.setLayout(container)



"""
Class that runs in a seperate thread watching the spectra file and updating the plot as it's updated
"""

class CSVImageLoaderWorker(QObject):
    data_updated = Signal(dict)
    #coordinates_updated = Signal(np.ndarray)
    #spectra_updated = Signal(dict) #containing the raw data, i.e. for the spectra plot
    finished = Signal()

    def __init__(self, vars: Variables, poll_interval=1.0):
        super().__init__()
        self.vars = vars
        self.poll_interval = poll_interval
        self._running = True
        self.last_modified_time = None

    def load_file(self,workingDir):

        data = np.loadtxt(workingDir + "/spectra.csv", delimiter=",")
        wavelengths = np.genfromtxt(workingDir + "/wavelenths.csv")*1e9 # load and convert to nm
        
        x = np.array(data[:,0],dtype=float)
        y = np.array(data[:,1],dtype=float)
        spectra = np.array(data[:,3:],dtype=float)

        sampling_wavelength = self.vars.get_variable("Sampling Wavelength (nm)")
        
        idx2plot = closest(wavelengths,sampling_wavelength)
        
        interp_i   = LinearNDInterpolator(list(zip(x, y)), spectra[:,idx2plot],rescale=True,fill_value=0)

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
        elif dx<0.1:
            dx = 0.1

        dy = dx

        x1 = min(x) + dx
        x2 = max(x) - dx
        y1 = min(y) + dy
        y2 = max(y) - dy

        xnew = np.arange(x1,x2+dx,dx)
        ynew = np.arange(y1,y2+dy,dy)
        X,Y = np.meshgrid(xnew,ynew)
        
        data = interp_i(X,Y)
        

        """Add everything to a dict for easy data transport"""
        data_dict = {
            "x":x, # irregularly distrobuted x coords
            "y":y, # irregularly distrobuted y coords
            "wavelengths":wavelengths, # spectra wavelengths
            "spectra":spectra,         # irregularly distrobuted intensity counts
            "img":data,       # re-interpolated intensity map at sampling wavelegnth
            "x_new":xnew,              # re-interpolated x coordinates 
            "y_new":ynew,              # re-interpolated y coordinates
        }
        
        return data_dict


    def run(self):
        while self._running:
            try:
                working_dir = self.vars.get_variable("Working Directory")
                file_path = os.path.join(working_dir, "spectra.csv")

                if os.path.exists(file_path):
                    modified_time = os.path.getmtime(file_path)
                    if self.last_modified_time is None or modified_time > self.last_modified_time:
                        self.last_modified_time = modified_time
                        data_dict = self.load_file(working_dir)
            
                        self.data_updated.emit(data_dict)
            except Exception as e:
                print(f"[CSV Loader] Error loading CSV: {e}")

            time.sleep(self.poll_interval)

    def stop(self):
        self._running = False


"""
Main GUI Class
"""
class ExperimentGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Viewer")

        self.vars = Variables()  

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel (buttons + controls)
        self.left_panel = QVBoxLayout()
        main_layout.addLayout(self.left_panel, 1)

        self.init_controls()
        
        #Add table to record positions
        self.table = pg.TableWidget()
        self.left_panel.addWidget(self.table)
        
        self.table_data =  [
                    ["X (um)", "Y (um)","Color"],
                    #[1, 100, 100, 'r']
                ]
    
        self.table.setData(self.table_data)

        # Right panel (image)
        map_and_spectrum_layout = QVBoxLayout()
        main_layout.addLayout(map_and_spectrum_layout, 4)
        self.init_plot_area()
        map_and_spectrum_layout.addWidget(self.map_widget, 4)
        #map_and_spectrum_layout.addWidget(self.spectrum_widget,1)
        #main_layout.addWidget(self.img,3)

        # Start image loader
        self.start_csv_image_loader()
        
        # Initialise list of spectra points
        self.positions = []
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
        self.color_index = 0
        self.pens = []

    def init_controls(self):
        self.open_button = QPushButton("Open File")
        self.left_panel.addWidget(self.open_button)

        self.parameter_button = QPushButton("Parameters")
        self.left_panel.addWidget(self.parameter_button)

        self.MCL_button = QPushButton("Manual Stage Control")
        self.left_panel.addWidget(self.MCL_button)
        
        self.clear_button = QPushButton("Clear Points")
        self.left_panel.addWidget(self.clear_button)
        
        self.open_button.clicked.connect(self.open_file_browser)
        self.clear_button.clicked.connect(self.clear_plots)
        self.parameter_button.clicked.connect(self.open_parameter_window)
        self.MCL_button.clicked.connect(self.open_MCL_stage_ctrl)

    def open_file_browser(self):
        """Callback function to open the filebrowser window"""
        dialog = FileBrowserDialog(self.vars)
        dialog.exec()

    def open_parameter_window(self):
        """Callback to open the parameters window"""
        dialog = VariableDialog(self.vars)
        dialog.exec()

    def open_MCL_stage_ctrl(self):
        """Callback to open the MCL stage control window"""
        dialog = MCL_Directional_CTRL(self.vars)
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
        #selected = self.roi.getArrayRegion(self.data, self.img)
        #self.p2.plot(selected.mean(axis=0), clear=True)
        pass

    def updateIsocurve(self):
        global isoLine, iso
        self.iso.setLevel(self.isoLine.value())
        
    # creating a mouse double click event
    def mouseDblClick(self, event):
        
        
        #Calculate coordinates that were double clicked"""
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.data.shape[0] - 1))
        j = int(np.clip(j, 0, self.data.shape[1] - 1))
        val = self.data[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        
        colour = self.colors[self.color_index % len(self.colors)]
        #create a new pen for the new point (pen is essentually a scatter point theme)"""
        self.pens.append(pg.mkPen(width=5, color=colour))
        
        #iterate the color index so a new color is chosen"""
        self.color_index += 1
        
        """Removed the scatter point where the user has clicked, instead 
            we can just keep the scatter point that shows the closest datapoint
        """
        #Add the scatter point to the figure"""
        #self.scatter_1.addPoints(x=[x],y=[y])
        #update the list of pens"""
        #self.scatter_1.setPen(self.pens) 
        
        #maybe we need this idk
        #self.positions.append((x,y))
        # print the message
        print("Mouse Double Click Event: " + "pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %.3g" % (x, y, i, j, val))
        
        # grab the nearest datapoint, plot an arrow to it and plot the spectra in the new panel
        coordinate_pick = np.array((y,x))
        distances = np.linalg.norm(self.coordinates-coordinate_pick, axis=1)
        min_index = np.argmin(distances)
        closest_coord = self.coordinates[min_index]
        
        print(f"the closest point is {closest_coord}, at a distance of {distances[min_index]}")
        
        #Add the scatter point to the figure"""
        self.scatter_2.addPoints(x=[closest_coord[1]],y=[closest_coord[0]])
        
        #update the list of pens"""
        self.scatter_2.setPen(self.pens)
        
        
        self.table_data.append([closest_coord[0], closest_coord[1], colour ])
    
        self.table.setData(self.table_data)
        
        
        #add a line to the spectra plot
        plot_item = self.p2.plot(
            self.wavelengths,
            self.spect[min_index],
            pen=self.pens[-1],
            name=f"X= {x}, Y = {y}"
        )
        #self.p2.append(plot_item)
        
        
    def clear_plots(self,event):
        self.scatter_1.setData(x=[],y=[])
        self.scatter_2.setData(x=[],y=[])
        self.p2.clear()
        self.pens = []
        self.color_index = 0
        self.table_data =  [
                    ["X (um)", "Y (um)","Color"],
                    #[1, 100, 100, 'r']
                ]
    
        self.table.setData(self.table_data)

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
        self.p1.setTitle("pos (X,Y): (%0.3f, %0.3f) um,  value: %.3g" % (y, x, val))

    def init_plot_area(self):

        

        # Set a custom color map
        
        self.map_widget = pg.GraphicsLayoutWidget()
        self.map_widget.setWindowTitle('Data Analysis')


        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.map_widget.addPlot()
        
        self.p1.setLabel(axis='left', text='X-axis, um')
        self.p1.setLabel(axis='bottom', text='Y-axis, um')

        # Item for displaying image data
        self.img = pg.ImageItem()

        

        self.p1.addItem(self.img)

        # item for adding scatter points
        self.scatter_1 = pg.ScatterPlotItem(symbol='o', size=5)
        self.p1.addItem(self.scatter_1)
        self.scatter_2 = pg.ScatterPlotItem(symbol='x', size=5)
        self.p1.addItem(self.scatter_2)
        self.scatter_1.setZValue(10)
        self.scatter_2.setZValue(11)
        
        # Custom ROI for selecting an image region
        #self.roi = pg.ROI([20, 10], [1, 1])
        #self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        #self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        #self.p1.addItem(self.roi)
        #self.roi.setZValue(10)  # make sure ROI is drawn above image

        # Isocurve drawing
        #self.iso = pg.IsocurveItem(level=0.8, pen='g')
        #self.iso.setParentItem(self.img)
        #self.iso.setZValue(5)

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.map_widget.addItem(self.hist)

        

        # Draggable line for setting isocurve level
        #self.isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        #self.hist.vb.addItem(self.isoLine)
        #self.hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
        #self.isoLine.setValue(1300)
        #self.isoLine.setZValue(1000) # bring iso line above contrast controls
        
        # Another plot area for displaying ROI data
        self.map_widget.nextRow()
        self.p2 = self.map_widget.addPlot(colspan=2)
        self.p2.setMaximumHeight(250)
        self.map_widget.resize(800,800)
        
        #self.line_plt = pg.PlotItem()
        #self.p2.addItem(self.line_plt)
        
        
        # Generate image data
        #self.data = np.random.normal(size=(200, 100))
        #self.data[20:80, 20:80] += 2.
        #self.data = pg.gaussianFilter(self.data, (3, 3))
        #self.data += np.random.normal(size=(200, 100)) * 0.1
        #self.img.setImage(self.data)
        #self.hist.setLevels(self.data.min(), self.data.max())

        # build isocurves from smoothed data
        #self.iso.setData(pg.gaussianFilter(self.data, (2, 2)))

        # set position and scale of image
        #tr = QtGui.QTransform()
        #self.img.setTransform(tr.scale(0.2, 0.2).translate(-50, 0))


        # Connect to callbacks
        #self.roi.sigRegionChanged.connect(self.updatePlot)
        #self.updatePlot()
        #self.isoLine.sigDragged.connect(self.updateIsocurve)
        
        #self.map_widget.sigSceneMouseMoved
                
        self.img.mouseDoubleClickEvent = self.mouseDblClick
        # Monkey-patch the image to use our custom hover function. 
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this. 
        self.img.hoverEvent = self.imageHoverEvent

    def start_csv_image_loader(self):
        self.csv_thread = QThread()
        self.csv_worker = CSVImageLoaderWorker(self.vars, poll_interval=1.0)

        self.csv_worker.moveToThread(self.csv_thread)
        self.csv_thread.started.connect(self.csv_worker.run)
        
        self.csv_worker.data_updated.connect(self.update_data)
        
        self.csv_worker.finished.connect(self.csv_thread.quit)

        self.csv_thread.start()

    def update_data(self,data_dict):
        self.xpos = data_dict['x']
        self.ypos = data_dict['y']
        self.wavelengths = data_dict['wavelengths']
        self.spect = data_dict['spectra']
        self.xnew = data_dict['x_new']
        self.ynew = data_dict['y_new']
        
        self.coordinates = []
        for i in range(0,len(self.xpos)):
            self.coordinates.append((self.xpos[i],self.ypos[i]))
            
        self.data = np.array(data_dict['img']).T
        
        #print(np.shape(self.data))

        self.img.setImage(self.data)
        
        #print(np.isnan(self.data))
        
        minimum = np.min(self.data)
        maximum = np.max(self.data)
        
        #print(minimum,maximum)
        
        self.hist.setLevels(minimum,maximum)
        

        dx = self.xnew[1]-self.xnew[0]
        dy = self.ynew[1]-self.ynew[0]
        
        tr = QTransform()
        tr.scale(dy, dx)       # scale horizontal and vertical axes
        tr.translate(self.ynew[0]/dy, self.xnew[0]/dx) # move 3x3 image to locate center at axis origin

        self.img.setTransform(tr)
        self.map_widget.adjustSize()
        
        self.p1.autoRange()
        self.p1.invertY()
        
        #change cmaps
        #cmap = pg.colormap.getFromMatplotlib('jet')
        #self.img.setColorMap(cmap)
 



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ExperimentGUI()
    gui.resize(1200, 800)
    gui.show()
    sys.exit(app.exec())
