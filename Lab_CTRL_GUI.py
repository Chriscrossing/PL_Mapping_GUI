import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

import sys, csv, os, time
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QGridLayout, QMainWindow, QDialog, QFileDialog, QPlainTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QTransform
import pyqtgraph as pg

from LabCTRL import ExperimentCTRL
from pathlib import Path

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
            #Dataviewer Settings
            "Dataviewer Directory": "Results/temp/",
            #Experiment Settings
            "Working Directory": "Results/temp/",
            
            #Spectrometer Camera Settings
            "CCD Temp (C)": -90,
            "Exposure Time (s)": 1,
            "Number of Samples": 1,
            
            #Spectrometer Settings
            "Center Wavelength (nm)": 613.4,
            "Slit Width (um)": 100,
            
            #Map settings
            "X Center (um)": 100,
            "Y Center (um)": 100,
            "dx (um)": 100,
            "dy (um)": 100,
            "Wait time (s)": 0.1,
            "AS Loss Condition": 0.001,
            "AS Wavelength (nm)": 615

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

    def saveMetadata(self):
        Path(self._vars['Working Directory']).mkdir(parents=True, exist_ok=True)

        w = csv.writer(open(self._vars['Working Directory'] + "metadata.csv", "w",newline=''))
        for key, val in self._vars.items():
            w.writerow([key,val])
        np.savetxt(self._vars['Working Directory'] + "wavelenths.csv",self.wavelengths,delimiter=',')


"""
Window dialogue for selecting directories for opening data
"""

class FileBrowserDialog(QFileDialog):
    def __init__(self,variables: Variables,expPath):
        super().__init__()
        self.variables = variables
        self.setWindowTitle("Select a Folder")
        self.resize(600,400)
        
        # Set the file dialog to select directories only
        self.setFileMode(QFileDialog.Directory)
        self.setOption(QFileDialog.ShowDirsOnly, True)

        if expPath:
            self.fileSelected.connect(self.set_experiment_Path)
        else:
            self.fileSelected.connect(self.set_dataviewer_Path)
        
    def set_dataviewer_Path(self,path):
        self.variables.set_variable("Dataviewer Directory",path + "\\")
        
    def set_experiment_Path(self,path):
        self.variables.set_variable("Working Directory",path + "\\")


"""
Window for controlling the MCL stage manually
"""

class MCL_Directional_CTRL(QWidget):
    def __init__(self, Variables: Variables, MCL_go_xy,MCL_go_z,MCL_get_position):
        super().__init__()
        self.setWindowTitle("MCL Stage Control")
        self.setMinimumWidth(450)
        self.Variables = Variables
        self.MCL_go_xy = MCL_go_xy
        self.MCL_go_z = MCL_go_z
        self.MCL_get_position = MCL_get_position

        self.up_button = QPushButton("-x")
        self.down_button = QPushButton("+x")
        self.left_button = QPushButton("-y")
        self.right_button = QPushButton("+y")
        self.z_up_button = QPushButton("Z+")
        self.z_down_button = QPushButton("Z-")
        


        self.x_pos_label = QLabel("X: 0.0")
        self.y_pos_label = QLabel("Y: 0.0")
        self.z_pos_label = QLabel("Z: 0.0")
        
        self.x_step = QLineEdit("5")
        self.y_step = QLineEdit("5")
        self.z_step = QLineEdit("5")

        self.x_goto = QLineEdit("100")
        self.y_goto = QLineEdit("100")
        self.z_goto = QLineEdit("100")

        grid = QGridLayout()
        
        grid.addWidget(self.up_button, 0, 1)
        grid.addWidget(self.left_button, 1, 0)
        grid.addWidget(self.right_button, 1, 2)
        grid.addWidget(self.down_button, 2, 1)
        
        grid.addWidget(self.z_up_button, 0, 3)
        grid.addWidget(self.z_down_button, 2, 3)

        pos_layout = QHBoxLayout()
        pos_layout.addWidget(self.x_pos_label)
        pos_layout.addWidget(self.y_pos_label)
        pos_layout.addWidget(self.z_pos_label)
        self.refresh_position = QPushButton("Refresh")
        pos_layout.addWidget(self.refresh_position)

        absolute_layout = QHBoxLayout()
        label = QLabel("X: ")
        label.setToolTip("um")
        absolute_layout.addWidget(label)
        absolute_layout.addWidget(self.x_goto)
        label = QLabel("Y: ")
        label.setToolTip("um")
        absolute_layout.addWidget(label)
        absolute_layout.addWidget(self.y_goto)
        label = QLabel("Z: ")
        label.setToolTip("um")
        absolute_layout.addWidget(label)
        absolute_layout.addWidget(self.z_goto)
        self.goto_button = QPushButton("Go")
        absolute_layout.addWidget(self.goto_button)
        
        step_layout = QHBoxLayout()
        label = QLabel("X step: ")
        label.setToolTip("um")
        step_layout.addWidget(label)
        step_layout.addWidget(self.x_step)
        label = QLabel("Y step: ")
        label.setToolTip("um")
        step_layout.addWidget(label)
        step_layout.addWidget(self.y_step)
        label = QLabel("Z step: ")
        label.setToolTip("um")
        step_layout.addWidget(label)
        step_layout.addWidget(self.z_step)

        container = QVBoxLayout()
        button_widget = QWidget()
        button_widget.setLayout(grid)
        
        label = QLabel("Current Stage Position (um): ")
        container.addWidget(label)
        container.addLayout(pos_layout)
        label = QLabel("Absolute Stage Positioning (um): ")
        container.addWidget(label)
        container.addLayout(absolute_layout)
        label = QLabel("Relative Stage Positioning (um): ")
        container.addWidget(label)
        container.addWidget(button_widget)
        label = QLabel("Relative Positioning Steps (um): ")
        container.addWidget(label)
        container.addLayout(step_layout)

        self.setLayout(container)
        
        self.goto_button.clicked.connect(self.goto_xyz)
        self.refresh_position.clicked.connect(self.ask_for_position)
        self.up_button.clicked.connect(self.step_minux_x)
        self.down_button.clicked.connect(self.step_plus_x)
        self.right_button.clicked.connect(self.step_plus_y)
        self.left_button.clicked.connect(self.step_minus_y)
        self.z_down_button.clicked.connect(self.step_minus_z)
        self.z_up_button.clicked.connect(self.step_plus_z)

        self.current_xpos = 0
        self.current_ypos = 0
        self.current_zpos = 0
        self.position_asked = False

        
    def step_plus_x(self):
        try:
            dx = float(self.x_step.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")
        
        self.step_xyz(+dx,0,0)
        
    def step_minux_x(self):
        try:
            dx = float(self.x_step.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")
        
        self.step_xyz(-dx,0,0)
        
    def step_plus_y(self):
        try:
            dy = float(self.y_step.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")
        
        self.step_xyz(0,+dy,0)
        
    def step_minus_y(self):
        try:
            dy = float(self.y_step.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")
        
        self.step_xyz(0,-dy,0)
            
    def step_plus_z(self):
        try:
            dz = float(self.z_step.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")
        
        self.step_xyz(0,0,+dz)
        
    def step_minus_z(self):
        try:
            dz = float(self.z_step.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")
        
        self.step_xyz(0,0,-dz)

    def step_xyz(self,x,y,z):
        self.ask_for_position()

        # Wait here until you hear an answer of the current position
        while self.position_asked != True:
            time.sleep(self.Variables.get_variable("Wait time (s)"))

        x0 = self.current_xpos
        y0 = self.current_ypos
        z0 = self.current_zpos

        if z == 0:
            self.MCL_go_xy.emit((x0 + x,y0 + y))
        else:
            self.MCL_go_xy.emit((x0 + x,y0 + y))
            self.MCL_go_z.emit(z0+z)

        time.sleep(self.Variables.get_variable("Wait time (s)"))

    
    def goto_xyz(self):
        
        try:
            x = float(self.x_goto.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")
        try:
            y = float(self.y_goto.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")
        try:
            z = float(self.z_goto.text())
        except Exception as e:
            print(f"[MCL_CTRL] x value error: {e}")    
            
        
        self.MCL_go_xy.emit((x,y))
        self.MCL_go_z.emit(z)
        time.sleep(self.Variables.get_variable("Wait time (s)"))

        self.ask_for_position()
        
    def ask_for_position(self):
        self.position_asked = True
        self.MCL_get_position.emit()

    
    def update_position(self,actual_pos):
        self.position_asked = False
        self.current_xpos = actual_pos[0]
        self.current_ypos = actual_pos[1]
        self.current_zpos = actual_pos[2]

        self.x_pos_label.setText(str(np.round(actual_pos[0],3)))
        self.y_pos_label.setText(str(np.round(actual_pos[1],3)))
        self.z_pos_label.setText(str(np.round(actual_pos[2],3)))

"""
Class that runs in a seperate thread watching the spectra file and updating the plot as it's updated
"""

class CSVImageLoaderWorker(QObject):
    data_updated = Signal(dict)
    finished = Signal()

    def __init__(self, vars: Variables):
        super().__init__()
        self.vars = vars
        self._running = True
        self.last_modified_time = None

    def load_file(self,workingDir):

        data = np.genfromtxt(workingDir + "/spectra.csv", delimiter=",")
        wavelengths = np.genfromtxt(workingDir + "/wavelenths.csv")*1e9 # load and convert to nm
        
        x = np.array(data[:,0],dtype=float)
        y = np.array(data[:,1],dtype=float)
        spectra = np.array(data[:,3:],dtype=float)

        sampling_wavelength = self.vars.get_variable("AS Wavelength (nm)")
        
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
        try:
            working_dir = self.vars.get_variable("Dataviewer Directory")
            file_path = os.path.join(working_dir, "spectra.csv")
            if os.path.exists(file_path):
                data_dict = self.load_file(working_dir)    
                self.data_updated.emit(data_dict)
                
        except Exception as e:
            print(f"[CSV Loader] Error loading CSV: {e}")
        self.finished.emit()
    
    def stop(self):
        self._running = False


"""
Main Window, For DataAnalysis
"""
class DataAnalysisGUI(QMainWindow):
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
        #self.start_csv_image_loader()
        
        # Initialise list of spectra points
        self.positions = []
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
        self.color_index = 0
        self.pens = []
        
        #
        self.loading = False
        self.run_csv_image_loader()

    def init_controls(self):
        self.open_button = QPushButton("Open Experiment")
        self.open_button.setToolTip("Select a experiment folder to open")
        self.left_panel.addWidget(self.open_button)
        
        self.load_data_button = QPushButton("Reload Maping Data")
        self.load_data_button.setToolTip("Refresh data from file and replot")
        self.left_panel.addWidget(self.load_data_button)

        self.experiment_control = QPushButton("Experiment Control")
        self.experiment_control.setToolTip("Open the window for experimental control")
        self.left_panel.addWidget(self.experiment_control)
        
        self.clear_button = QPushButton("Clear Spectra Points")
        self.experiment_control.setToolTip("Remove the points from the map and clear spectra")
        self.left_panel.addWidget(self.clear_button)
        
        self.open_button.clicked.connect(self.open_file_browser)
        self.load_data_button.clicked.connect(self.run_csv_image_loader)
        self.clear_button.clicked.connect(self.clear_plots)
        self.experiment_control.clicked.connect(self.open_experiment_control)
        
    def open_file_browser(self):
        """Callback function to open the filebrowser window"""
        dialog = FileBrowserDialog(self.vars,expPath=False)
        dialog.exec()
        self.run_csv_image_loader()

    def open_experiment_control(self):
        """Callback to open the experiment control window"""
        self.Experimental_window = ExperimentGUI(self.vars)
        self.Experimental_window.show()
        
    def run_csv_image_loader(self):
        
        if self.loading:
            pass
        else:
            self.load_data_button.setText("Loading ...")
            self.loading = True
            
            self.csv_thread = QThread()
            self.csv_worker = CSVImageLoaderWorker(self.vars)

            self.csv_worker.moveToThread(self.csv_thread)
            self.csv_thread.started.connect(self.csv_worker.run)
            
            self.csv_worker.data_updated.connect(self.update_data)
            
            self.csv_worker.finished.connect(self.csv_thread.quit)
            self.csv_worker.finished.connect(self.loading_finished)

            self.csv_thread.start()
        
    def loading_finished(self):
        self.load_data_button.setText("Reload Maping Data")
        self.loading = False

    def closeEvent(self, event):
        # Make sure to stop the CSV image loader
        if hasattr(self, 'csv_worker'):
            self.csv_worker.stop()
            self.csv_thread.quit()
            self.csv_thread.wait()

        event.accept()
        
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
        #print("Mouse Double Click Event: " + "pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %.3g" % (x, y, i, j, val))
        
        # grab the nearest datapoint, plot an arrow to it and plot the spectra in the new panel
        coordinate_pick = np.array((y,x))
        distances = np.linalg.norm(self.coordinates-coordinate_pick, axis=1)
        min_index = np.argmin(distances)
        closest_coord = self.coordinates[min_index]
        
        #print(f"the closest point is {closest_coord}, at a distance of {distances[min_index]}")
        
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

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.map_widget.addItem(self.hist)

        # Another plot area for displaying ROI data
        self.map_widget.nextRow()
        self.p2 = self.map_widget.addPlot(colspan=2)
        self.p2.setMaximumHeight(250)
        self.map_widget.resize(800,800)
        
                
        self.img.mouseDoubleClickEvent = self.mouseDblClick
        # Monkey-patch the image to use our custom hover function. 
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this. 
        self.img.hoverEvent = self.imageHoverEvent

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


class SpectrumWindow(QWidget):
    closed = Signal()
    def __init__(self, Variables: Variables, Exp):
        super().__init__()
        self.setWindowTitle("Spectrum Viewer")
        self.resize(700, 500)
        
        self.vars = Variables
        self.Exp = Exp 

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
        
        # Save Curves
        self.save_curves_button = QPushButton("Save")
        self.save_curves_button.clicked.connect(self.save_curves)

        # Clear Curves
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_curves_button)
        btn_layout.addWidget(self.clear_button)
        layout.addLayout(btn_layout)

        # Monkey-patch the image to use our custom hover function. 
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this. 
        #self.plot.hoverEvent = self.imageHoverEvent

        self.cont_plot_first_plot = False
        self.continous_is_running = False
        self.stop_continous = False

    def save_curves(self):
        working_dir = self.vars.get_variable("Working Directory")

        """The curves will be saved to seperate csv files incase they have different wavelength ranges"""
        # make a new folder:
        try:
            Path(working_dir + "/Single_Spectra").mkdir(parents=True, exist_ok=False)
            curve_idx = 0
            if len(self.plots) > 0:
                for curve in self.plots:
                    x = curve.x
                    y = curve.y
                    np.savetxt(working_dir + "/Single_Spectra/Curve_" + str(curve_idx), (x,y))
                    curve_idx += 1

        except:
            print("ERR, the directory already exists")


    def closeEvent(self,event):
        self.cont_plot_first_plot = False
        self.continous_is_running = False
        self.stop_continous = False
        self.closed.emit()
        super().closeEvent(event)

    def run_single(self):
        #("Running")
        """Emit signal to take a spectra"""
        self.Exp.emit(True)

    def run_continous(self):
        #print("Continous Scan Starting")
        """Emit signal to take a spectra"""
        self.stop_continous = False
        self.continous_is_running = True
        self.Exp.emit(False)

    def update_line(self,data):
        #print("[Spectrum Widget] Trying to update line",hasattr(self, 'continous_plot_item'))
        wavelength,intensity = data

        color = self.colors[0]

        if hasattr(self, 'continous_plot_item'):
            self.continous_plot_item.setData(wavelength,intensity)
        else:
            self.continous_plot_item = self.plot.plot(
                wavelength,
                intensity,
                pen=pg.mkPen(color=color, width=2),
                #name=f"X= {x}, Y = {y}"
            )


        if self.stop_continous == False:
            self.run_continous()
        else:
            #self.reset_continous_label()
            self.continous_is_running = False

    def add_line(self, data):
        
        wavelength,intensity = data
        
        color = self.colors[self.color_index % len(self.colors)]
        

        # Plot the line
        plot_item = self.plot.plot(
            wavelength,
            intensity,
            pen=pg.mkPen(color=color, width=2),
            name=f"Curve {self.color_index}"
        )
        # Add legend
        plot_item.addLegend()
        self.plots.append(plot_item)
        self.color_index += 1

    def clear_all(self):
        for item in self.plots:
            self.plot.removeItem(item)
        self.plots.clear()

        for annotation in self.annotations:
            self.plot.removeItem(annotation)
        self.annotations.clear()

        self.color_index = 0

    def imageHoverEvent(self,event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            self.p1.setTitle("")
            return
        pos = event.pos()
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        self.p1.setTitle("Pos: %.3g" % (x, yvalvalvalval))

class ExperimentGUI(QWidget):
    # Signals to run methods in experiment control thread
    connect_2_instruments = Signal(object)
    disconnect_instruments = Signal()
    initialise_instruments = Signal(object)
    run_scan = Signal(bool)
    run_adaptive_mapping = Signal()
    stop_adaptive_mapping = Signal()
    ask_for_ccd_temperature = Signal()

    MCL_go_xy = Signal(tuple)
    MCL_go_z = Signal(float)
    MCL_get_position = Signal()

    def __init__(self,Variables:Variables):
        super().__init__()
        self.setWindowTitle("Experiment Control")
        
        
        self.vars = Variables  
        
        # Left panel (buttons + controls)
        self.left_panel = QVBoxLayout()
        self.init_controls()
        
        self.setLayout(self.left_panel)
        
        self.single_spectra_windows = {}
        self.single_spectra_idx = 0

        """Initialise the experiment control thread"""

        self.experiment_thread = QThread()
        self.experiment_worker = ExperimentCTRL()

        #Move ExperimentCTRL class to a sepearate thread
        self.experiment_worker.moveToThread(self.experiment_thread)

        """Connect Signals and Slots"""
        """Slots: to get stuff back from thread"""
        # 
        self.experiment_worker.instruments_connected.connect(self.instruments_are_connected)
        self.experiment_worker.finished.connect(self.experiment_thread.quit)
        self.experiment_worker.CCD_temp_updated.connect(self.temperature_recieved)


        """Signals to run functions in thread from here"""
        self.connect_2_instruments.connect(self.experiment_worker.connect)
        self.disconnect_instruments.connect(self.experiment_worker.disconnect_instruments)
        self.initialise_instruments.connect(self.experiment_worker.initialise_instruments)
        self.run_scan.connect(self.experiment_worker.getSpectra)

        self.run_adaptive_mapping.connect(self.experiment_worker.runAdaptiveMapping)
        self.stop_adaptive_mapping.connect(self.experiment_worker.stop_adaptive_mapping)
        self.ask_for_ccd_temperature.connect(self.experiment_worker.get_CCD_temperature)


        self.experiment_thread.start()

        self.instruments_connected = False
        self.single_spectra_window = None
        self.continous_spectra_window = None

        self.adaptive_mapping_running = False

    def init_controls(self):
        
        
        self.set_folder_button = QPushButton("Set Experiment Folder")
        self.set_folder_button.setToolTip("Choose where to save the experiment")
        self.left_panel.addWidget(self.set_folder_button)
        
        row = QHBoxLayout()
        label = QLabel("Instrument Status:")
        row.addWidget(label)
        self.connect_instruments_status = QLabel("Disconnected")
        row.addWidget(self.connect_instruments_status)
        self.connect_instruments_button = QPushButton("Connect")
        self.connect_instruments_button.setToolTip("Connect to all instruments and set default values")
        row.addWidget(self.connect_instruments_button)
        
        self.left_panel.addLayout(row)

        self.MCL_button = QPushButton("Manual Stage Control")
        self.set_folder_button.setToolTip("Open a new window to manually control the MCL Stage")
        self.left_panel.addWidget(self.MCL_button)
        
        label = QLabel("Experiment Variables")
        self.left_panel.addWidget(label)
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
            "AS Loss Condition",
            "AS Wavelength (nm)"
        ]
        
        variable_tooltips = [
            "Set temperature of the spectrometer camera",
            "Exposure time for each specturm in seconds",
            "Number of spectra to take the median over (mainly to remove cosmic rays, but also reduces noise)",
            "Center wavelength of the spectrometer",
            "Size of the slit at the enterence to the spectrometer",
            "Center X coordinate for the mapping",
            "Center Y coordinate for the mapping",
            "Size of the mapping area in the X direction",
            "Size of the mapping area in the Y direction",
            "Time to wait between moving the MCL stage and taking a spectra",
            "Adaptive Sampling Loss Condition (related to the minimum feature size)",
            "The wavelength that the Adaptive Sampling Alogrithm uses to measure intensity"
            
        ]

        for name,tooltip in zip(variable_names,variable_tooltips):
            row = QHBoxLayout()
            label = QLabel(name)
            label.setToolTip(tooltip)
            input_line = QLineEdit()
            input_line.setText(str(self.vars.get_variable(name) or ""))
            input_line.setFixedWidth(100)
            self.input_fields[name] = input_line
            row.addWidget(label)
            #row.addSpacing(10)
            row.addWidget(input_line)
            self.left_panel.addLayout(row)

        self.set_button = QPushButton("Set Variables")
        self.set_button.setToolTip("Save and Apply Variables")
        self.left_panel.addWidget(self.set_button)
        
        label = QLabel("Experiment Control")
        self.left_panel.addWidget(label)
        
        row = QHBoxLayout()
        self.single_spectra_button = QPushButton("Single Spectra")
        self.single_spectra_button.setToolTip("Take a single spectrum")
        row.addWidget(self.single_spectra_button)
        self.continous_spectra_button = QPushButton("Continous Spectra")
        self.continous_spectra_button.setToolTip("Continously take spectra")
        row.addWidget(self.continous_spectra_button)
        self.left_panel.addLayout(row)
        
        
        self.run_adaptive_mapping_button = QPushButton("Run Adaptive Mapping")
        self.run_adaptive_mapping_button.setToolTip("Start Adaptive Mapping Run")
        self.left_panel.addWidget(self.run_adaptive_mapping_button)
        
        
        self.Experiment_Status_Label = QLabel("Status:")
        self.left_panel.addWidget(self.Experiment_Status_Label)
        
        row = QHBoxLayout()
        label = QLabel("Current CCD Temp:")
        row.addWidget(label)
        self.current_CCD_temp_label = QLabel("20 ℃")
        row.addWidget(self.current_CCD_temp_label)
        row.addSpacing(10)
        self.left_panel.addLayout(row)
        
        
        self.set_folder_button.clicked.connect(self.open_file_browser)
        self.connect_instruments_button.clicked.connect(self.connect_disconnect_instruments)
        
        self.set_button.clicked.connect(self.set_variables)
        self.MCL_button.clicked.connect(self.open_MCL_stage_ctrl)
        
        self.single_spectra_button.clicked.connect(self.single_spectra)
        self.continous_spectra_button.clicked.connect(self.continous_spectra)

        self.run_adaptive_mapping_button.clicked.connect(self.adaptive_mapping)

    def open_file_browser(self):
        """Callback function to open the filebrowser window
            expPath is just to define which path to save in the variables dict
        """
        dialog = FileBrowserDialog(self.vars,expPath=True)
        dialog.exec()
        
    def connect_disconnect_instruments(self):
        if self.instruments_connected == False:
            self.connect_instruments_status.setText("Connecting ...")
            self.connect_2_instruments.emit(self.vars)
        elif self.instruments_connected == True:
            self.disconnect_instruments.emit()
            self.connect_instruments_button.setText("Connect")
            self.connect_instruments_status.setText("Disconnected")
            self.instruments_connected = False

    def instruments_are_connected(self):
        self.connect_instruments_button.setText("Disconnect")
        self.connect_instruments_status.setText("Connected")
        self.instruments_connected = True
        """Now we can initialise the slots"""
        self.MCL_go_xy.connect(self.experiment_worker.MCL_goxy)
        self.MCL_go_z.connect(self.experiment_worker.MCL_goz)
        self.MCL_get_position.connect(self.experiment_worker.MCL_get_position)
        
        """Start monitoring the CCD temperature"""
        self.start_CCD_temperature_timer()
        
    def start_CCD_temperature_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_CCD_tempearture)
        self.timer.start(2000)  # Interval in milliseconds
        
    def poll_CCD_tempearture(self):
        self.ask_for_ccd_temperature.emit()
    
    def temperature_recieved(self,temp):
        self.current_CCD_temp_label.setText(str(int(temp)) + " ℃") 

    def open_MCL_stage_ctrl(self):
        """Callback to open the MCL stage control window"""
        self.MCL_ctrl_window = MCL_Directional_CTRL(self.vars,self.MCL_go_xy,self.MCL_go_z,self.MCL_get_position)
        self.experiment_worker.MCL_position_updated.connect(self.MCL_ctrl_window.update_position)
        self.MCL_ctrl_window.show()
        
    def set_variables(self):
        """Grab all variables from the user inputs and update the vars dict"""
        for name, field in self.input_fields.items():
            self.vars.set_variable(name, float(field.text()))
        self.initialise_instruments.emit(self.vars)

    def init_single_plotting_window(self):
        #print("Making New Window")
        self.single_spectra_window = SpectrumWindow(self.vars,self.run_scan)
        self.single_spectra_window.show()
        self.experiment_worker.single_spectra_updated.connect(self.single_spectra_window.add_line)
        self.experiment_worker.single_spectra_updated.connect(self.updateSingleSpectraButtonTxt)
        self.single_spectra_window.run_single()

    def single_spectra(self):
        """Callback to gather a single spectra and open a window to plot the results"""
        self.single_spectra_button.setText("Running ...")

        if self.single_spectra_window is None:
            self.init_single_plotting_window()
        elif self.single_spectra_window.isVisible():
                #print("Running Single")
                self.single_spectra_window.run_single()
        else:
            self.init_single_plotting_window()

    def updateSingleSpectraButtonTxt(self,data):
        """When scan is complete, run this"""
        self.single_spectra_button.setText("Single Spectra")

    def init_continous_plotting_window(self):
        self.continous_spectra_window = SpectrumWindow(self.vars,self.run_scan)
        self.continous_spectra_window.show()
        self.continous_spectra_window.cont_plot_first_plot = False
        self.experiment_worker.continous_spectra_updated.connect(self.continous_spectra_window.update_line)

        #self.continous_spectra_window.closeEvent.connect(self.updateContinousSpectraButtonTxt)

    def continous_spectra(self):
        
        if self.continous_spectra_window is None:
            self.init_continous_plotting_window()
            self.continous_spectra_window.run_continous()
            self.continous_spectra_button.setText("Running ... (stop)")
        elif self.continous_spectra_window.isVisible():
            
            if self.continous_spectra_window.continous_is_running == True:
                # if the system is in a continous running state, stop running.
                self.continous_spectra_window.stop_continous = True
                self.reset_continous_label()
            else:
                # start running 
                self.continous_spectra_window.run_continous()
                self.continous_spectra_button.setText("Running ... (stop)")
            #stop scan here somehow
        else:
            # make a window
            self.init_continous_plotting_window()
            self.continous_spectra_window.run_continous()
            self.continous_spectra_button.setText("Running ... (stop)")
            
    def reset_continous_label(self):
        self.continous_spectra_button.setText("Continous Spectra")
    
    def adaptive_mapping(self):
        
        if self.adaptive_mapping_running == False:
            self.run_adaptive_mapping_button.setText("Running ... (stop)")
            self.adaptive_mapping_running = True
            self.run_adaptive_mapping.emit()
        else:
            self.stop_adaptive_mapping.emit()
            self.run_adaptive_mapping_button.setText("Run Adaptive Mapping")
        
    
    def closeEvent(self, event):
        print("This Event Happens")
        # Make sure to stop the CSV image loader
        self.disconnect_instruments.emit()
        time.sleep(1) #cheating here for now
        #wait until completed
        self.experiment_thread.quit()
        event.accept() 
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DataAnalysisGUI()
    gui.resize(1200, 800)
    gui.show()
    sys.exit(app.exec())