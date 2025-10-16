import numpy as np
from scipy import integrate
from ctypes import cdll, c_int, c_uint, c_double
import atexit
from time import sleep
import adaptive
import csv
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
import time
from pylablib.devices import Andor


class Madpiezo():
    """https://github.com/yurmor/mclpiezo"""
    def __init__(self):
		# provide valid path to Madlib.dll. Madlib.h and Madlib.lib should also be in the same folder
        path_to_dll = 'C:/Program Files/Mad City Labs/NanoDrive/Madlib.dll'
        self.madlib = cdll.LoadLibrary(path_to_dll)
        self.handler = self.mcl_start()
        atexit.register(self.mcl_close)
    def mcl_start(self):
        """
        Requests control of a single Mad City Labs Nano-Drive.
        Return Value:
            Returns a valid handle or returns 0 to indicate failure.
        """
        mcl_init_handle = self.madlib['MCL_InitHandle']

        mcl_init_handle.restype = c_int
        handler = mcl_init_handle()
        if(handler==0):
            print("MCL init error")
            return -1
        return 	handler
    def mcl_read(self,axis_number):
        """
        Read the current position of the specified axis.

        Parameters:
            axis [IN] Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            handle [IN] Specifies which Nano-Drive to communicate with.
        Return Value:
            Returns a position value or the appropriate error code.
        """
        mcl_single_read_n = self.madlib['MCL_SingleReadN']
        mcl_single_read_n.restype = c_double
        return  mcl_single_read_n(c_uint(axis_number), c_int(self.handler))
    def mcl_write(self,position, axis_number):
        """
        Commands the Nano-Drive to move the specified axis to a position.

        Parameters:
            position [IN] Commanded position in microns.
            axis [IN] Which axis to move. (X=1,Y=2,Z=3,AUX=4)
            handle [IN] Specifies which Nano-Drive to communicate with.
        Return Value:
            Returns MCL_SUCCESS or the appropriate error code.
        """
        mcl_single_write_n = self.madlib['MCL_SingleWriteN']
        mcl_single_write_n.restype = c_int
        error_code = mcl_single_write_n(c_double(position), c_uint(axis_number), c_int(self.handler))

        if(error_code !=0):
            print("MCL write error = ", error_code)
        return error_code
    def goxy(self,pos):
        x_position,y_position = pos
        self.mcl_write(x_position,1)
        self.mcl_write(y_position,2)
    def goz(self,z_position):
        self.mcl_write(z_position,3)
    def get_position(self):
        return (self.mcl_read(1), self.mcl_read(2), self.mcl_read(3))
    def mcl_close(self):
        """
        Releases control of all Nano-Drives controlled by this instance of the DLL.
        """
        mcl_release_all = self.madlib['MCL_ReleaseAllHandles']
        mcl_release_all()


class ExperimentCTRL(QObject):
    """Slot Signals"""
    finished = Signal()
    instruments_connected = Signal()
    MCL_position_updated = Signal(tuple)
    single_spectra_updated = Signal(object)
    continous_spectra_updated = Signal(object)
    new_adaptive_spectra_added = Signal()


    def __init__(self):
        super().__init__()
        self.connected = False
        self.stop_adaptive = False
    
    @Slot(object)
    def connect(self,ExpCfg):
        """Connect to Instruments"""
        self.connect2instruments()
        self.initialise_instruments(ExpCfg)

        self.instruments_connected.emit()
        
        
    def connect2instruments(self):
        """Connect to Piezo Stage"""
        self.MCL = Madpiezo()
        print("Connected to MCL")
        """Connect to cam and then spectrometer"""
        self.cam = Andor.AndorSDK2Camera(fan_mode="full")  # camera should be connected first
        print("Connected to CCD")
        self.spec = Andor.ShamrockSpectrograph() # then the spectrometer
        print("Connected to Spectrometer")

    @Slot(object)
    def initialise_instruments(self,ExpCfg):
        self.ExpCfg = ExpCfg
        #self.ExpCfg.calculate() 

        for k,v in self.ExpCfg.get_all_variables().items():
            print(k,v) 

        """Set camera variables"""
        self.cam.set_temperature(float(self.ExpCfg._vars['CCD Temp (C)']))
        self.cam.set_read_mode(0)
        self.cam.set_exposure(float(self.ExpCfg._vars['Exposure Time (s)']))

        """Set spectrometer variables"""
        self.spec.set_wavelength(float(self.ExpCfg._vars['Center Wavelength (nm)'])*1e-9) # set 600nm center wavelength
        self.spec.setup_pixels_from_camera(self.cam) # setup camera sensor parameters (number and size of pixels) for wavelength calibration
        self.spec.set_slit_width("input_side",float(self.ExpCfg._vars['Slit Width (um)'])*1e-6)
        
        """Grab the pixel calibrated wavelengths"""
        self.ExpCfg.wavelengths = self.spec.get_calibration()  # return array of wavelength corresponding to each pixel

        
    @Slot(bool)
    def getSpectra(self,single=True):

        #print("[ExperimentCTRL thread] Running scan", single)

        timeout = self.cam.get_exposure()*1.3

        #timeout should be at min 1s right?
        if timeout < 1:
            timeout = 1

        """Take some spectra and calculate the median of them"""
        spectra = np.zeros((int(self.ExpCfg._vars['Number of Samples']),len(self.ExpCfg.wavelengths)))
        for i in range(0,int(self.ExpCfg._vars['Number of Samples'])):
            spectra[i,:] = self.cam.snap(timeout=timeout)[0]
            
        spectrum = np.median(spectra,axis=0)
        if single:
            self.single_spectra_updated.emit((self.ExpCfg.wavelengths, spectrum))
        else:
            self.continous_spectra_updated.emit((self.ExpCfg.wavelengths, spectrum))
    @Slot()
    def runAdaptiveMapping(self):

        def closest(lst, K):
            item = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
            return np.where(lst == item)[0][0]
		
        def getSpectra(xy):
            x,y = xy
            
            """z-axis focus interpolation"""
            #zmax,xmax = self.ExpCfg.V['z_x_max']
            #zmin,xmin = self.ExpCfg.V['z_x_min']

            #zpos = np.interp(x,[xmin,xmax],[zmin,zmax])

            """Move the stage to the correct position"""
            #self.piezo.goz(zpos)
            self.MCL_goxy((x,y))
            sleep(self.ExpCfg._vars['Wait time (s)'])

            """Take some spectra and calculate the median of them"""
            spectra = np.zeros((int(self.ExpCfg._vars['Number of Samples']),len(self.ExpCfg.wavelengths)))
            for i in range(0,int(self.ExpCfg._vars['Number of Samples'])):
                spectra[i,:] = self.cam.snap(timeout=self.cam.get_exposure()*1.3)[0]
                
            spectrum = np.median(spectra,axis=0)

            """Here we need to save the spectrum to a file as we go"""

            # Append to CSV
            xyz = np.array(self.MCL.get_position())
            
            with open(self.ExpCfg._vars['Working Directory'] + 'spectra.csv', mode='a',newline='') as file:
                spamwriter = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(np.append(xyz,spectrum))

            
            """Integrate the spectrum^2 and use this with the adaptive function"""
            #min_idx = 600
            #max_idx = -500
            #energy = integrate.simpson(spectrum[min_idx:max_idx]) #-1049667.9166666665
            #print(xy,energy)

            idx = closest(self.ExpCfg.wavelengths,self.ExpCfg._vars['AS Wavelength (nm)'])

            print(xy,spectrum[idx])

            return spectrum[idx]
        

        """Setup the adaptive learner"""
        
        x1 = self.ExpCfg._vars['X Center (um)'] - self.ExpCfg._vars['dx (um)']/2
        x2 = self.ExpCfg._vars['X Center (um)'] + self.ExpCfg._vars['dx (um)']/2
        y1 = self.ExpCfg._vars['Y Center (um)'] - self.ExpCfg._vars['dy (um)']/2
        y2 = self.ExpCfg._vars['Y Center (um)'] + self.ExpCfg._vars['dy (um)']/2
        
        self.learner = adaptive.Learner2D(
                getSpectra, 
                bounds=[(x1, x2), (y1, y2)]
                )

        self.ExpCfg.saveMetadata()

        self.runner = adaptive.runner.simple(
            self.learner, 
            loss_goal=self.ExpCfg._vars['AS Loss Condition']
            )
        
        
        #self.runner.live_info()
    
    @Slot()
    def stop_adaptive_mapping(self):
        self.runner.stop()
    
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
        self.MCL.mcl_close()
        self.cam.close()
        self.spec.close()     