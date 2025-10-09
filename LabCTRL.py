import numpy as np
from scipy import integrate
from ctypes import cdll, c_int, c_uint, c_double
import atexit
from time import sleep
import adaptive
import csv
from pathlib import Path

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
	def goxy(self,x_position,y_position):
		self.mcl_write(x_position,1)
		self.mcl_write(y_position,2)
	def goz(self,z_position):
		self.mcl_write(z_position,3)
	def get_position(self):
		return self.mcl_read(1), self.mcl_read(2), self.mcl_read(3)
	def mcl_close(self):
		"""
		Releases control of all Nano-Drives controlled by this instance of the DLL.
		"""
		mcl_release_all = self.madlib['MCL_ReleaseAllHandles']
		mcl_release_all()


class Experiment_Config:
    def __init__(self):
        
        self.V = {
            #Experiment Settings
            'ExpDir': "Results/temp/",

            #Camera Settings
            'set_CCD_temp': -90 , # degC
            'fan_mode': "full",
            'exposure_t': 1, #seconds
            'nSpectra':3, #number of exposures per position
            
            #Spectrometer Settings
            'center_wavelength':635e-9, #in meters
            'slit_width':100e-6, 
            
            # Stage Settings
            # units in um for this module for convenience.
            'center_x':100,
            'center_y':100,
            'dx':10,
            'dy':10,
            #'step_size': 10, #will use this for debugging
            #'npoints': npoints,   #will use this for the adaptive algorythm
            'adaptive_loss_goal':0.1,
            'wait_time': 0.01, # in seconds
            'adaptive_sampling_wavelength':613.3e-9, #in meters
            #'z_x_max':z_x_max, #should be an array of (zpos,xpos) at max xpos
            #'z_x_min':z_x_min  #should be an array of (zpos,xpos) at min xpos
        }
        self.wavelengths = None
        
    #def calculate(self):
    #    self.nrows = int(self.V['dx']/self.V['step_size']) 
    #    self.ncols = int(self.V['dx']/self.V['step_size'])
    #    
    #    self.xpos = np.linspace(self.V['center_x'] - self.V['dx']/2, self.V['center_x'] + self.V['dx']/2,self.nrows)
    #    self.ypos = np.linspace(self.V['center_y'] - self.V['dy']/2, self.V['center_y'] + self.V['dy']/2,self.ncols)

    def saveMetadata(self):
        Path(self.V['ExpDir']).mkdir(parents=True, exist_ok=True)

        w = csv.writer(open(self.V['ExpDir'] + "metadata.csv", "w",newline=''))
        for key, val in self.V.items():
            w.writerow([key,val])
        np.savetxt(self.V['ExpDir'] + "wavelenths.csv",self.wavelengths,delimiter=',')


class MappingClass:
    def __init__(self):
        """Imedietly Connect to Instruments"""
        self.connect2instruments()
        
        
    def connect2instruments(self):
        """Connect to Piezo Stage"""
        self.piezo = Madpiezo()
        
        """Connect to cam and then spectrometer"""
        self.cam = Andor.AndorSDK2Camera(fan_mode="full")  # camera should be connected first
        self.spec = Andor.ShamrockSpectrograph() # then the spectrometer
    
    def initialise_instruments(self,ExpCfg):
        self.ExpCfg = ExpCfg
        #self.ExpCfg.calculate()  

        """Set camera variables"""
        self.cam.set_temperature(self.ExpCfg.V['set_CCD_temp'])
        self.cam.set_read_mode(0)
        self.cam.set_exposure(self.ExpCfg.V['exposure_t'])

        """Set spectrometer variables"""
        print(self.ExpCfg.V['center_wavelength'])
        self.spec.set_wavelength(self.ExpCfg.V['center_wavelength']) # set 600nm center wavelength
        self.spec.setup_pixels_from_camera(self.cam) # setup camera sensor parameters (number and size of pixels) for wavelength calibration
        self.spec.set_slit_width("input_side",self.ExpCfg.V['slit_width'])
        
        """Grab the pixel calibrated wavelengths"""
        self.ExpCfg.wavelengths = self.spec.get_calibration()  # return array of wavelength corresponding to each pixel

        

    def getSpectra(self,xy):

        x,y = xy
        
        """Move the stage to the correct position"""
        #self.piezo.goz(z)
        self.piezo.goxy(x,y)
        sleep(self.ExpCfg.V['wait_time'])

        """Take some spectra and calculate the median of them"""
        spectra = np.zeros((self.ExpCfg.V['nSpectra'],len(self.ExpCfg.wavelengths)))
        for i in range(0,self.ExpCfg.V['nSpectra']):
            spectra[i,:] = self.cam.snap(timeout=self.cam.get_exposure()*1.3)[0]
            
        spectrum = np.median(spectra,axis=0)

        return self.ExpCfg.wavelengths, spectrum
        
    
    
    
    def runMapping(self):

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
            self.piezo.goxy(x,y)
            sleep(self.ExpCfg.V['wait_time'])

            """Take some spectra and calculate the median of them"""
            spectra = np.zeros((self.ExpCfg.V['nSpectra'],len(self.ExpCfg.wavelengths)))
            for i in range(0,self.ExpCfg.V['nSpectra']):
                spectra[i,:] = self.cam.snap(timeout=self.cam.get_exposure()*1.3)[0]
                
            spectrum = np.median(spectra,axis=0)

            """Here we need to save the spectrum to a file as we go"""

            # Append to CSV
            xyz = np.array(self.piezo.get_position())
            
            with open(self.ExpCfg.V['ExpDir'] + 'spectra.csv', mode='a',newline='') as file:
                spamwriter = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(np.append(xyz,spectrum))

            
            """Integrate the spectrum^2 and use this with the adaptive function"""
            #min_idx = 600
            #max_idx = -500
            #energy = integrate.simpson(spectrum[min_idx:max_idx]) #-1049667.9166666665
            #print(xy,energy)

            idx = closest(self.ExpCfg.wavelengths,self.ExpCfg.V['adaptive_sampling_wavelength'])

            print(xy,spectrum[idx])

            return spectrum[idx]
        

        """Setup the adaptive learner"""
        
        x1 = self.ExpCfg.V['center_x'] - self.ExpCfg.V['dx']/2
        x2 = self.ExpCfg.V['center_x'] + self.ExpCfg.V['dx']/2
        y1 = self.ExpCfg.V['center_y'] - self.ExpCfg.V['dy']/2
        y2 = self.ExpCfg.V['center_y'] + self.ExpCfg.V['dy']/2
        
        self.learner = adaptive.Learner2D(
              getSpectra, 
              bounds=[(x1, x2), (y1, y2)])

        self.runner = adaptive.runner.simple(
            self.learner, 
            loss_goal=self.ExpCfg.V['adaptive_loss_goal']
            )
        
        self.ExpCfg.saveMetadata()
        #self.runner.live_info()
        
        
    def disconnect_instruments(self):
        self.piezo.mcl_close()
        self.cam.close()
        self.spec.close()
        
        