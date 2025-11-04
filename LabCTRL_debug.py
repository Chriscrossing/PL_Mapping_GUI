import numpy as np
from scipy import integrate
from ctypes import cdll, c_int, c_uint, c_double
import atexit
from time import sleep
import adaptive
import csv
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject, QTimer
import time

#from pylablib.devices import Andor


class Madpiezo():
    def __init__(self):
		# provide valid path to Madlib.dll. Madlib.h and Madlib.lib should also be in the same folder
        pass
    def goxy(self,pos):
        print("Going to xy pos: ", pos)

    def goz(self,z_position):
        print("Going to z pos: ", z_position)

    def get_position(self):
        
        return (np.random.rand(), np.random.rand(), np.random.rand())


class ExperimentCTRL(QObject):
    """Slot Signals"""
    finished = Signal()
    logger_info = Signal(str)
    logger_err = Signal(str)
    instruments_connected = Signal()
    MCL_position_updated = Signal(tuple)
    single_spectra_updated = Signal(object)
    continous_spectra_updated = Signal(object)
    new_adaptive_spectra_added = Signal()
    adaptive_sampling_done = Signal()
    regular_mapping_done = Signal()
    CCD_temp_updated = Signal(float)



    def __init__(self):
        super().__init__()
        self.connected = False
        self.adaptive_mapping_active = False
    
    @Slot(object)
    def connect(self,ExpCfg):
        """Connect to Instruments"""
        self.connect2instruments()
        self.initialise_instruments(ExpCfg)
        
        self.instruments_connected.emit()
        
        
    def connect2instruments(self):
        """Connect to Piezo Stage"""
        try:
            self.MCL = Madpiezo()
        except Exception as e:
            err_string = f"[LabCTRL] MCL connect error: {e}"
            print(err_string)
            self.logger_err.emit(err_string)

        self.logger_info.emit("[LabCTRL] MCL Connected")
        """Connect to cam and then spectrometer"""
        #self.cam = Andor.AndorSDK2Camera(fan_mode="full")  # camera should be connected first
        #print("Connected to CCD")
        #self.spec = Andor.ShamrockSpectrograph() # then the spectrometer
        #print("Connected to Spectrometer")

    @Slot(object)
    def initialise_instruments(self,ExpCfg):
        self.ExpCfg = ExpCfg
        #self.ExpCfg.calculate() 

        self.ExpCfg.wavelengths = np.linspace(500,600,5)

        for k,v in self.ExpCfg.get_all_variables().items():
            print(k,v) 

    """Used for single and continous spectra"""        
    def getSpectra(self,single=True):

        spectrum = np.random.rand(5)
        time.sleep(0.2)
        return spectrum
    

    """This is used for both single and continous spectra"""        
    @Slot()
    def getSingleSpectra(self):

        spectrum = self.getSpectra()
        
        self.logger_info.emit("[LabCTRL] Single Spectra Updated")
        self.single_spectra_updated.emit((self.ExpCfg.wavelengths, spectrum))
        
        #    #self.logger_info.emit("[LabCTRL] Continous Spectra Updated")
        #    self.continous_spectra_updated.emit((self.ExpCfg.wavelengths, spectrum))
    
    def getContinousSpectra(self):
        
        spectrum = self.getSpectra()
        
        self.logger_info.emit("[LabCTRL] Continous Spectra Updated")
        self.continous_spectra_updated.emit((self.ExpCfg.wavelengths, spectrum))


    @Slot()
    def startContinousSpectra(self):
        self.logger_info.emit("[LabCTRL] Starting Continous Scan QTimer")
        self.continous_timer = QTimer()
        self.continous_timer.setInterval(250)
        self.continous_timer.timeout.connect(self.getContinousSpectra)
        self.continous_timer.start()
        self.logger_info.emit("[LabCTRL] Emitting continous_is_running signal")
        self.continous_is_running.emit()

    @Slot()
    def stopContinousSpectra(self):
        self.logger_info.emit("[LabCTRL] stop_continous called, stopping continous timer")
        self.continous_timer.stop()
        # next, emit a signal to change the label of the button
        self.logger_info.emit("[LabCTRL] Emitting continous_stopped_running signal")
        self.continous_stopped_running.emit()

    @Slot()
    def runAdaptiveMapping(self):
		
        self.setupAdaptiveSampling()

        self.adaptive_pt_idx = 1

        self.runner_timer = QTimer()
        self.runner_timer.setInterval(250)
        self.runner_timer.timeout.connect(self.stepAdaptive)
        self.runner_timer.start()


    @Slot()
    def stepAdaptive(self):

        print("[LabCTRL] Stepping Adaptive", self.adaptive_pt_idx)

        self.runner = adaptive.runner.simple(
                    self.learner,
                    npoints_goal=self.adaptive_pt_idx
                    )
        self.adaptive_pt_idx += 1

        loss = self.learner.loss()
        if loss < 0.001:
            self.stop_adaptive_mapping()

        time.sleep(0.1)

    def setupAdaptiveSampling(self):
        def getSpectra(xy):
            x,y = xy

            x, y = xy
            a = 0.2
            intensity = x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)


            spectrum = np.ones(5) * intensity

            """Here we need to save the spectrum to a file as we go"""

            # Append to CSV
            #xyz = np.array(self.MCL.get_position())
            xyz = np.array([x,y,1])
            with open(self.ExpCfg._vars['Working Directory'] + 'spectra.csv', mode='a',newline='') as file:
                spamwriter = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(np.append(xyz,spectrum))
            logging_string = "X: " +str(np.round(x,3)) + " Y: " +str(np.round(y,3))  +   " Intensity: " + str(intensity)
            self.logger_info.emit(logging_string)
            
            return intensity
        

        print("[LabCTRL] Setting up Adaptive")
        """Setup the adaptive learner"""
        
        x1 = self.ExpCfg._vars['X Center (um)'] - self.ExpCfg._vars['dx (um)']/2
        x2 = self.ExpCfg._vars['X Center (um)'] + self.ExpCfg._vars['dx (um)']/2
        y1 = self.ExpCfg._vars['Y Center (um)'] - self.ExpCfg._vars['dy (um)']/2
        y2 = self.ExpCfg._vars['Y Center (um)'] + self.ExpCfg._vars['dy (um)']/2
        
        self.learner = adaptive.Learner2D(
                getSpectra, 
                bounds=[(-1, 1), (-1, 1)]
                )

        self.ExpCfg.saveMetadata()

        
        #self.runner.live_info()

    @Slot()
    def stop_adaptive_mapping(self):
        print("[LabCTRL Thread] Trying to stop")
        self.runner_timer.stop()
        self.adaptive_sampling_done.emit()


    @Slot(object)
    def runRegularMapping(self,coordinates):

        self.ExpCfg.saveMetadata()

        self.Xpts = coordinates[0]
        self.Ypts = coordinates[1]

        self.regular_npts = len(self.Xpts.flatten())
        
        self.regular_pt_idx = 0


        self.regular_mapping_timer = QTimer()
        self.regular_mapping_timer.setInterval(250)
        self.regular_mapping_timer.timeout.connect(self.stepRegularMapping)
        self.regular_mapping_timer.start()

    def stepRegularMapping(self):
        def getSpectra(xy):
            x,y = xy

            x, y = xy
            a = 0.2
            intensity = x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)


            spectrum = np.ones(5) * intensity

            """Here we need to save the spectrum to a file as we go"""

            # Append to CSV
            #xyz = np.array(self.MCL.get_position())
            xyz = np.array([x,y,1])

            with open(self.ExpCfg._vars['Working Directory'] + 'spectra.csv', mode='a',newline='') as file:
                spamwriter = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(np.append(xyz,spectrum))
            
            
            
            logging_string =  str(self.regular_pt_idx) + "/" + str(self.regular_npts) + ", X: " +str(np.round(x,3)) + " Y: " +str(np.round(y,3))  +   " Intensity: " + str(intensity)
            self.logger_info.emit(logging_string)
            
        
        if self.regular_pt_idx <= self.regular_npts:
            getSpectra((self.Xpts.flatten()[self.regular_pt_idx],self.Ypts.flatten()[self.regular_pt_idx]))
            self.regular_pt_idx +=1
        else:
            self.stopRegularMapping()
        
    @Slot()
    def stopRegularMapping(self):
        print("[LabCTRL Thread] Trying to stop")
        self.regular_mapping_timer.stop()
        self.regular_mapping_done.emit()

    @Slot()
    def get_CCD_temperature(self):
        temp = np.random.rand()
        self.CCD_temp_updated.emit(temp)
    
    @Slot(tuple)
    def MCL_goxy(self,pos):
        self.MCL.goxy(pos)
        sleep(self.ExpCfg._vars['Wait time (s)'])
        self.MCL_get_position()
    
    @Slot(float)
    def MCL_goz(self,pos):
        self.MCL.goz(pos)
        sleep(self.ExpCfg._vars['Wait time (s)'])
        self.MCL_get_position()
    
    @Slot()
    def MCL_get_position(self):
        self.MCL_position_updated.emit(self.MCL.get_position())

    @Slot()    
    def disconnect_instruments(self):
        print("Disconnecting_instruments")
        self.MCL.mcl_close()
        self.cam.close()
        self.spec.close()     