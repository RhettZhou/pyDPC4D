import os
import numpy as np
import copy
from bs4 import BeautifulSoup

import cv2
import py4DSTEM
import hyperspy.api as hs
import pixstem.api as ps

from pylab import cm
from matplotlib.colors import hsv_to_rgb
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from numpy import arange
from scipy import interpolate
from scipy import optimize
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max

import shutil

import h5py

global points

## basic functions #######################################################################
def cal_wavelength_nm(voltage_kV):
    e_m0 = 9.109383E-31                                             ## mass of electroon, unit kg
    e = 1.6022E-19                                                  ## charge of an electron, unit C
    c = 2.998E8                                                     ## speed of light, unit m/s
    h = 6.62E-34                                                    ## Planck's constant, Unit Js
    V = voltage_kV * 1E3                                     
    numerator = (h ** 2) * (c ** 2)
    denominator = (e * V) * ((2 * e_m0 * (c ** 2)) + (e * V))
    wavelength_nm = (10 ** 9) * ((numerator / denominator) ** 0.5)  ## wavelength in nm
    return wavelength_nm

def cal_speed_electron_m1s(voltage_kV):
    c = 2.998E8                                                     ## speed of light, unit m/s
    V = voltage_kV * 1E3                                            ## voltage of electron, unit V
    v_m1s = c*(1-((1+1.96E-6*V)**-2))**0.5                          ## speed of electron, unit m/s
    return v_m1s

def cal_sigma_1vm(voltage_kV):
    e_m0 = 9.109383E-31                                             ## mass of electroon, unit kg
    e = 1.6022E-19                                                  ## charge of an electron, unit C
    c = 2.998E8                                                     ## speed of light, unit m/s
    h = 6.62E-34                                                    ## Planck's constant, Unit Js
    v = cal_speed_electron_m1s(voltage_kV)                          ## speed of electron, unit m/s
    wavelength = 1E-9*cal_wavelength_nm(voltage_kV)                 ## speed of electron, unit m/s
    e_m = e_m0/((1-(v/c)**2)**0.5)                                  ## relativistic mass    
    sigma_rad1vm = 2*np.pi*e_m*e*wavelength/(h**2)                  ## interaction parameter, unit rad/Vm
    # print('Interaction parameter: ' + '%.2f' % sigma_rad1vm + ' rad/Vm')
    return sigma_rad1vm

def get_real_scale_pm(a_pm,dataset):
    real_scale_pm = a_pm/(dataset.data.shape[1] - 1)                ## calculated the scale in the real space, unit pm / pixel
    print('Scale in real space: ' + '%.2f' % real_scale_pm + ' pm/pixel')
    return real_scale_pm

def get_recip_scale_1nm_from_probe(aperture_mrad,voltage_kV,probe_semiangle):
    aperture  = aperture_mrad*1E-3                                  ## Radians, unit rad
    wavelength_nm = cal_wavelength_nm(voltage_kV)
    recip_scale_1nm_from_probe = aperture / (probe_semiangle * wavelength_nm)
    print('Reciprocal scale (measured from the probe):' + '%.2f' % recip_scale_1nm_from_probe + ' (1/nm) / pixel')
    return recip_scale_1nm_from_probe 

def get_recip_scale_mrad_from_probe(aperture_mrad,probe_semiangle):
    recip_scale_mrad_from_probe = aperture_mrad / probe_semiangle
    print('Reciprocal scale (measured from the probe):' + '%.2f' % recip_scale_mrad_from_probe + ' (mrad) / pixel')
    return recip_scale_mrad_from_probe 
###########################################################################################




## 1.0 ##
## io #####################################################################################
## for file input and out, experiments
def scan_infor(fold_name):
    path = os.path.normpath(fold_name)
    file_inform = path.split(os.sep)[-1] + '.xml'
    inform_path_input = os.path.join(fold_name, file_inform)
    with open(inform_path_input , 'r') as f:
        inform = f.read()
    Bs_data = BeautifulSoup(inform, "xml")
    n_cl = Bs_data.find('nominal_camera_length')
    n_cl_mm = float(n_cl.text)*1000
    pixel = Bs_data.find("scan_parameters",{'mode':"acquire"})
    pixel = pixel.find('scan_resolution_x')
    pixel = float(pixel.text)
    size = Bs_data.find('x')
    size = float(size.text)*1e12
    real_scale_pm = size/pixel
    scale_f = Bs_data.find('scale_factor')
    scale_f = float(scale_f.text)
    voltage = Bs_data.find('high_voltage')
    voltage_kV = float(voltage.text)/1000
    return n_cl, pixel, size, real_scale_pm, scale_f, voltage_kV

def input_file(fold_name,file_in,data_size):
    file_path_input = os.path.join(fold_name, file_in)
    dataset = py4DSTEM.io.DataCube(data=np.reshape(np.fromfile(file_path_input, dtype='16384float32'),data_size))
    n_cl_mm, pixel, size, real_scale_pm, scale_f,voltage_kV = scan_infor(fold_name)
    return dataset,real_scale_pm, scale_f,n_cl_mm,voltage_kV

def region_of_interest_crop(fold_name,dataset,apply_crop_image,h0:int=0,h1:int=0,w0:int=0,w1:int=0):
    [h11,w11] = dataset.data.shape[0:2]
    if apply_crop_image == 0:
        export_name = 'Overview'
        dataset.crop_data_real(h0,h1,w0,w1)
    else:    
        export_name = 'ROI_' + str(apply_crop_image)
        dataset.crop_data_real(h0,h1,w0,w1)
    path = os.path.join(fold_name,export_name)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    return dataset, path

def initial_data_Process(fold_name,file_in,data_size):
    dataset_raw, real_scale_pm, scale_f, n_cl_mm, voltage_kV = input_file(fold_name,file_in,data_size)
    dataset_raw.crop_data_diffraction(0,128,0,128)
    dataset = copy.copy(dataset_raw)
    [h1,w1] = dataset.data.shape[0:2] # "Overall"
    dataset,path = region_of_interest_crop(fold_name,dataset,0,0,h1,0,w1)
    return dataset,path,voltage_kV,real_scale_pm

def add_name_extra_hf(fft_clean,fft_clean_CoM,drift_correction,path,thickness_nm,aperture_mrad,flip_key,rotation):
    ## Depending on the data processing steps, the file names would be added some prefix
    name_extra = ''
    name_extra_CoM = ''
    if drift_correction == 1:
        name_extra = name_extra + '_Drift'
        name_extra_CoM = name_extra_CoM + '_Drift'
    if fft_clean == 1:
        name_extra = name_extra + '_FFT'
    if fft_clean_CoM == 1:
        name_extra_CoM = name_extra_CoM + '_FFT'

    process_file = os.path.join(os.path.split(path)[0],'process.h5')
    if os.path.exists(process_file) == False:
        hf = h5py.File(process_file, 'w')
    else:
        hf = h5py.File(process_file, 'r+')
    if 'thickness_nm' in hf.keys():
            del hf['thickness_nm']
    hf.create_dataset('thickness_nm', data=np.array([thickness_nm]))
    if 'aperture_mrad' in hf.keys():
            del hf['aperture_mrad']
    hf.create_dataset('aperture_mrad', data=np.array([aperture_mrad]))
    if 'flip_key' in hf.keys():
            del hf['flip_key']
    hf.create_dataset('flip_key', data=np.array([flip_key]))
    if 'rotation' in hf.keys():
            del hf['rotation']
    hf.create_dataset('rotation', data=np.array([rotation]))
    hf.close()
    return name_extra, name_extra_CoM

## some extra functions for simulations
def get_muSTEM_sim_data(file_path_input,flip_key = 0):
    fold_name = os.path.split(file_path_input)[0]
    dataset = py4DSTEM.io.read(file_path_input, data_id='datacube_0')
    [h1,w1] = dataset.data.shape[0:2] # "Overall"
    dataset,path = region_of_interest_crop(fold_name,dataset,0,0,h1,0,w1)
    if flip_key == 1:
        dataset.data = np.flip(dataset.data,2)
        dataset.data = np.flip(dataset.data,3)
    return dataset,path,fold_name
###########################################################################################




## 2.0 ##
## imaging and fft clean (optional)   ######################################################
## imaging by py4dstem
def plot_diffraction_mean(dataset,path,scale_type = 'unknown',recip_scale = 0):
    # Plot the mean CBED
    diffraction_pattern_mean = np.mean(dataset.data, axis=(0,1))
    
    if scale_type == '1nm':
        fig = py4DSTEM.visualize.show(
            diffraction_pattern_mean,
            scaling='power',
            power=0.5,
            cmap='inferno',
            # cmap=cm.nipy_spectral,
            returnfig = True,
            pixelsize = recip_scale,
            pixelunits='1/nm',
            scalebar={'position':'br','label':True},
            )
        fig[0].savefig(os.path.join(path, 'CBED_1nm.png'))
    elif scale_type == 'mrad':
        fig = py4DSTEM.visualize.show(
            diffraction_pattern_mean,
            scaling='power',
            power=0.5,
            cmap='inferno',
            # cmap=cm.nipy_spectral,
            returnfig = True,
            pixelsize = recip_scale,
            pixelunits='mrad',
            scalebar={'position':'br','label':True},
            )
        fig[0].savefig(os.path.join(path, 'CBED_mrad.png'))
        plt.close()
    else:
        fig = py4DSTEM.visualize.show(
            diffraction_pattern_mean,
            scaling='power',
            power=0.5,
            cmap='inferno',
            # cmap=cm.nipy_spectral,
            returnfig = True,
            pixelsize = 1,
            pixelunits= 'pixel',
            scalebar= None,
            figax=None
            )
        plt.axis('off')
        fig[0].savefig(os.path.join(path, 'CBED.png'))
        plt.close()
    return diffraction_pattern_mean

def estimate_BF_radius(path,diffraction_pattern_mean):
    # Estimate the radius of the BF disk, and the center coordinates
    probe_semiangle, qx0, qy0 = py4DSTEM.process.calibration.get_probe_size(
        diffraction_pattern_mean)
    # plot the mean diffraction pattern, with the estimated probe radius overlaid as a circle
    fig = py4DSTEM.visualize.show_circles(
        diffraction_pattern_mean,
        (qx0, qy0), 
        probe_semiangle,
        returnfig = True)
    # Print the estimate probe radius
    fig[0].savefig(os.path.join(path, 'CBED_mean.png'))
    print('Estimated probe radius =', '%.2f' % probe_semiangle, 'pixels')
    return probe_semiangle, qx0, qy0

def show_masks(path,diffraction_pattern_mean,probe_semiangle, qx0, qy0, expand_BF, det_DF):
    # Next, create a BF virtual detector using the the center beam position (qxy0, qy0), and a virtual DF detector.
    # The DF virtual detector will be set to high scattering angles, picking up just a hint of the coherent diffraction.
    # Show selected virtual detectors
    fig = py4DSTEM.visualize.show(
            diffraction_pattern_mean,
            scaling='log',
            cmap='gray',
            circle={'R':probe_semiangle + expand_BF,
                    'center':(qx0,qy0),
                    'fill':True,
                    'color':'r',
                    'alpha':0.35},
            annulus={'Ri':det_DF[0],
                    'Ro':det_DF[1],
                    'center':(qx0,qy0),
                    'fill':True,
                    'color':'y',
                    'alpha':0.35},
            returnfig = True
       )
    fig[0].savefig(os.path.join(path,'BF&DF_Mask.png'))

def show_BF_mask(path,dataset,diffraction_pattern_mean,probe_semiangle, qx0, qy0, expand_BF):
    # Next, create a BF virtual detector using the the center beam position (qxy0, qy0), and a virtual DF detector.
    # Show selected virtual detectors
    fig = py4DSTEM.visualize.show_circles(diffraction_pattern_mean, (qx0, qy0), probe_semiangle + expand_BF, returnfig = True)
    mask = py4DSTEM.process.utils.make_circular_mask(shape = (dataset.Q_Nx,dataset.Q_Ny),
                                                qxy0 = (qx0, qy0),
                                                radius = probe_semiangle + expand_BF)
    ## Note： This must will apply to raw images. 
    fig[0].savefig(os.path.join(path,'BF_Mask.png'))
    return mask

def show_diffraction_patterns(dataset,path,aperture,voltage_kV,expand_BF,det_DF = 0,dataset_raw = 0):
    # Diffraction space imaging
    diffraction_pattern_mean = plot_diffraction_mean(dataset,path)
    plt.close('all')
    # Estimate the radius of bright field disk    
    probe_semiangle, qx0, qy0 = estimate_BF_radius(path,diffraction_pattern_mean)
    recip_scale_1nm = get_recip_scale_1nm_from_probe(aperture,voltage_kV,probe_semiangle)
    plot_diffraction_mean(dataset,path,'1nm',recip_scale_1nm)
    recip_scale_mrad = get_recip_scale_mrad_from_probe(aperture,probe_semiangle)
    plot_diffraction_mean(dataset,path,'mrad',recip_scale_mrad)
    if det_DF == 0:
        mask = show_BF_mask(path,dataset_raw,diffraction_pattern_mean,probe_semiangle,qx0,qy0,expand_BF) 
    elif det_DF == 1:
        det_DF = (probe_semiangle + expand_BF,qx0)
        show_masks(path,diffraction_pattern_mean,probe_semiangle,qx0,qy0,expand_BF,det_DF)
        mask = []
    else:
        show_masks(path,diffraction_pattern_mean,probe_semiangle,qx0,qy0,expand_BF,det_DF)
        mask = []    
    return probe_semiangle,qx0,qy0,mask,recip_scale_1nm,recip_scale_mrad,det_DF

def show_raw_BF_DF_images(dataset,path,qx0, qy0,probe_semiangle,expand_BF,det_DF,scale):
    # Compute BF and DF virtual images

    image_BF = py4DSTEM.process.virtualimage.get_virtualimage(
        dataset, 
        ((qx0, qy0), 
        probe_semiangle + expand_BF))
    image_DF = py4DSTEM.process.virtualimage.get_virtualimage(
        dataset, 
        ((qx0, qy0), 
        (det_DF[0],
        det_DF[1])))
    # Show the BF and DF images
    fig = py4DSTEM.visualize.show_image_grid(
        lambda i:[image_BF, image_DF][i],H=1,W=2,
        cmap='gray',
        returnfig = True,
        pixelsize=scale,
        pixelunits='pm', 
        scalebar={'color':'red'}
        )
    file_out = 'BF_E' + str(expand_BF) + '&DF_I' + str(det_DF[0]) + '_O' + str(det_DF[1]) + '.png'
    file_path_out = os.path.join(path, file_out)
    fig[0].savefig(file_path_out)
    return image_BF, image_DF


## for removing high frequency noise
def fft_clean(image,type,ratio,plot=0,intensity = 0.001):
    image_min = 0
    if image.min() < 0:
        image_min = image.min()
        image = image - image.min()
    FFT = np.fft.fft2(image)
    x, y = image.shape
    x_c = int(x/2)
    y_c = int(y/2)
    r = min(x_c,y_c)
    r_filter = int(r*ratio)
    FFT_shift = np.fft.fftshift(FFT)
    if plot == 1:
        fig = plt.figure()    
        plt.imshow(abs(FFT_shift),vmax=np.amax(intensity*abs(FFT_shift)))
        plt.show()
    FFT_shift_new = 0*FFT_shift
    if type == 'circle':
        for i in range(0,x):
            for j in range(0,y):
                if (x_c - i)**2 + (y_c - i)**2  <= r_filter**2:
                    FFT_shift_new[i,j] = FFT_shift[i,j] 
    if type == 'rectangular':
        for i in range(x_c-r_filter,x_c+r_filter):
            for j in range(y_c-r_filter,y_c+r_filter):
                    FFT_shift_new[i,j] = FFT_shift[i,j]
    if type == 'gaussian':
        G_blur = ratio
        FFT_shift_new = cv2.GaussianBlur(abs(FFT_shift),(G_blur,G_blur),0)
    if plot == 1:
        fig = plt.figure()
        plt.imshow(abs(FFT_shift_new),vmax=np.amax(intensity*abs(FFT_shift_new)))
        plt.show() 
    FFT_shift_ishift = np.fft.ifftshift(FFT_shift_new)
    FFT_shift_ishift_iFFT = np.fft.ifft2(FFT_shift_ishift)
    FFT_shift_ishift_iFFT = abs(FFT_shift_ishift_iFFT)
    if image_min < 0:
        FFT_shift_ishift_iFFT = FFT_shift_ishift_iFFT + image_min
    return FFT_shift_ishift_iFFT

def BF_DF_fft(path,image_BF,image_DF,fft_type,fft_ratio,scale,fft_plot = 0,intensity = 0.001):
    ## Apply FFT filter to the bright field and dark field images. 
    image_BF_clean = fft_clean(image_BF,fft_type,fft_ratio,fft_plot,intensity)
    image_DF_clean = fft_clean(image_DF,fft_type,fft_ratio,fft_plot)
    # Show the BF and DF images
    fig = py4DSTEM.visualize.show_image_grid(
        lambda i:[image_BF_clean, image_DF_clean][i],H=1,W=2,
        cmap='gray',
        returnfig = True,
        pixelsize=scale,
        pixelunits='pm', 
        scalebar={'color':'red'}
        )
    file_out = 'BF&DF_FFT_' + fft_type + '_' + str(fft_ratio) + '.png'
    file_path_out = os.path.join(path, file_out)
    fig[0].savefig(file_path_out)
    return image_BF_clean, image_DF_clean
###########################################################################################




## 3.0 (optional) ##
## manually correction of scan distortion #################################################
###  image correction
def mouse_event(event):
   global points
   # print('x: {} and y: {}'.format(event.xdata, event.ydata))
   plt.scatter(event.xdata, event.ydata)
   plt.pause(0.0001)
   plt.show()
   points = np.append(points,np.array([[event.xdata,event.ydata]]),axis = 0)

def objective_line(x, a, b):
	return a * x + b

def gaussianx2(xdata_tuple,bg,height, center_x, center_y, width_x, width_y):

    (x, y) = xdata_tuple
    width_x = float(width_x)
    width_y = float(width_y)
    g = height*np.exp(
                 -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)+bg
    return g.ravel()

def fitGaussianx2(data, guess, bounds):
    x = np.linspace(0, data.shape[1]-1, data.shape[1])
    y = np.linspace(0, data.shape[0]-1, data.shape[0])
    x, y = np.meshgrid(x, y)
    xdata_tuple = (x,y)
    popt, pcov = optimize.curve_fit(gaussianx2, xdata_tuple, data.ravel(), p0=guess,bounds=bounds,method='trf',verbose=0,maxfev=100000)
    return popt

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    
def IntegratedInt(average,plot,min_distance=3,test_pixel=3,tolorence=0.5,max_factor=3):
    # min_distance: pixel, the minimum distance to find peaks
    # test_pixel: pixel, the first round to estimate x, y spacing
    # tolorence: ratio, the maximum ratio to search for the spots in the same line or column.
    # max_factor: ratio, spacing*tolorence*max_factor, spacing*tolorence/max_factor, define the
    #             maximum and minimum to find next spot in the same line or column

    xDim = average.shape[1]
    yDim = average.shape[0]
    
    coordinates = peak_local_max(average, min_distance=min_distance)

    x_max = average.shape[1]
    y_max = average.shape[0]

    peaks_x_fit = []
    peaks_y_fit = []

    ps = min_distance  #  Patch size must be Even

    for i in range(coordinates.shape[0]):

        x = int(coordinates[i][1])
        y = int(coordinates[i][0])

        if x >=ps+1 and y >=ps+1 and x <=x_max-ps-1 and y <=y_max-ps-1:
            patch = average[y-ps:y+ps+1,x-ps:x+ps+1]
            patch = patch - np.amin(patch)
            bg = 0

            # parameters are in the order of : background, height, cx, cy, wx, wy
            guess = (bg,patch[ps,ps]-bg,ps, ps, ps/2, ps/2)
            bounds = ([0,0,0,0,1,1],[np.inf,np.inf,2*(ps-1),2*(ps-1),2*(ps-1),2*(ps-1)])
            param = fitGaussianx2(patch, guess, bounds)

            peaks_x_fit.append(param[2]+x-ps)
            peaks_y_fit.append(param[3]+y-ps)

    # a = np.asarray([peaks_x_fit])
    # b = np.asanyarray([peaks_y_fit])
    # co_txt = np.concatenate((a.T,b.T),axis = 1)
    
    int_list = []
    for i in range(len(peaks_x_fit)):
        mask = create_circular_mask(average.shape[0], average.shape[1], center=(peaks_x_fit[i],peaks_y_fit[i]), radius=min_distance)
        int_list.append(np.sum(mask * average))

    if xDim <= yDim:
        DimMax = yDim
    else:
        DimMax = xDim    

    pad_inch_set = DimMax/8192
    if pad_inch_set < 0.01:
        pad_inch_set = 0.01
        
    plt.ion()
    average_position = copy.deepcopy(average)
    average_position[average_position>0] = 0
    plt.scatter(peaks_x_fit, peaks_y_fit,s=min_distance,c='r',alpha=0.4)

def fit_line(x,y,x_add):
    popt, _ = curve_fit(objective_line,x,y)
    a, b = popt
    # print('y = %.5f * x + %.5f' % (a, b))
    y_add = objective_line(x_add, a, b)
    return y_add

def extroplate_end(x_line,y_line,x_shape):
    if x_line[0] > 0:
        x_start = x_line[0:6]
        y_start = y_line[0:6]
        x_start_add = arange(0, x_line[0], 1)
        y_start_add = fit_line(x_start,y_start,x_start_add)
        x_line = np.concatenate([x_start_add,x_line])
        y_line = np.concatenate([y_start_add,y_line])
    if x_line[-1] < x_shape:
        x_end = x_line[-7:-1]
        y_end = y_line[-7:-1]
        x_end_add = arange(x_line[-1]+1,x_shape, 1)
        y_end_add = fit_line(x_end,y_end,x_end_add)
        x_line = np.concatenate([x_line,x_end_add])
        y_line = np.concatenate([y_line,y_end_add])
    return x_line, y_line

def swip(image):
   image = np.swapaxes(image,0,1)
   return image

def plot_image_points(image,swip_key = 0):
    global points
    if swip_key == 1:
        image = swip(image) # swip x and y axis
    fig = plt.figure()
    plt.ion()
    points = np.empty((0,2),np.single)
    cid = fig.canvas.mpl_connect('button_press_event', mouse_event)
    plt.imshow(image)
    IntegratedInt(image,1)
    plt.show()
    return image

def points2line(image,fold_name,axis = 'x',plot = 0):
    global points
    x = points[:,0]
    y = points[:,1]
    f = interpolate.interp1d(x, y)
    x_line = arange(np.ceil(x.min()), np.floor(x.max()), 1)
    y_line = f(x_line)
    x_kernel = int((x.max() - x.min())/2)
    if (x_kernel % 2) == 0:
        x_kernel = x_kernel + 1
    y_line_fit = savgol_filter(y_line, x_kernel, 1)
    x_line, y_line_fit_extro = extroplate_end(x_line, y_line_fit, image.shape[1] + 1)
    line_txt = np.array([x_line,y_line_fit_extro]).T
    save_txt = os.path.join(fold_name, 'Image_correction_' + axis + '.txt')
    np.savetxt(save_txt,line_txt,fmt='%.8f')
    if plot == 1:
        plt.imshow(image)
        IntegratedInt(image,1)
        plt.plot(x_line,y_line_fit_extro)
        plt.axis([0, image.shape[1]-1, image.shape[0]-1,0])
        plt.show
    return x_line, y_line_fit_extro

def image_deform(image,x_line,y_line,scale_ratio,plot = 0):
    width = int(image.shape[1]) 
    height = int(image.shape[0])
    y_line_delta = scale_ratio*(y_line - y_line[0])
    f = interpolate.interp1d(x_line, y_line_delta)
    x_line_new = arange(0, image.shape[1], 1/scale_ratio)
    y_line_delta_new = f(x_line_new)
    y_line_delta_new = y_line_delta_new.astype(int)
    width_up = int(width * scale_ratio)
    height_up = int(height * scale_ratio)
    dim_up = (width_up, height_up)
    image_resized = cv2.resize(image, dim_up, interpolation = cv2.INTER_AREA) # resize image
    image_resized_shifted = np.zeros(image_resized.shape,dtype=float)
    for i in range (0,width_up):
        shift = y_line_delta_new[i]
        for j in range (0,height_up):
            destinate_cell = j+shift
            if destinate_cell >= 0 and destinate_cell < height_up:
                image_resized_shifted[j,i] = image_resized[destinate_cell,i]
    dim_raw = (width, height)
    image = cv2.resize(image_resized_shifted, dim_raw, interpolation = cv2.INTER_AREA) # resize image back
    if plot == 1:
        fig = plt.figure()
        plt.imshow(image)
        plt.show()
    return image

def apply_image_deformation(image,fold_name,axis,scale_ratio,plot = 1):
    x_line,y_line = points2line(image,fold_name,axis)
    image = image_deform(image,x_line,y_line,scale_ratio,plot)
    return image

def image_correction(image,fold_name,scale_ratio,plot1,plot2):
    image_correction_name = os.path.join(fold_name, 'Image_correction_x.txt')
    x_corr = np.loadtxt(image_correction_name, dtype=float)
    image_correction_name = os.path.join(fold_name, 'Image_correction_y.txt')
    y_corr = np.loadtxt(image_correction_name, dtype=float)
    image = swip(image)
    image = image_deform(image,x_corr[:,0],x_corr[:,1],scale_ratio,plot1)
    image = swip(image)
    image = image_deform(image,y_corr[:,0],y_corr[:,1],scale_ratio,plot2)
    return image

def show_BF_DF_drift_corrected_images(path,fold_name,image_BF,image_DF,name_extra,scale,scale_ratio):
    image_BF_deformed = image_correction(image_BF,fold_name,scale_ratio,0,0)
    image_DF_deformed = image_correction(image_DF,fold_name,scale_ratio,0,0)
    fig = py4DSTEM.visualize.show_image_grid(
            lambda i:[image_BF_deformed, image_DF_deformed][i],H=1,W=2,
            cmap='gray',
            returnfig = True,
            pixelsize=scale,
            pixelunits='pm', 
            scalebar={'color':'red'}
            )
    file_out = 'BF&DF' + name_extra + '.png'
    file_path_out = os.path.join(path, file_out)
    fig[0].savefig(file_path_out)
    return image_BF_deformed, image_DF_deformed
###########################################################################################




## 4.0 (optional) ##
## Manually crop #########################################################################
###  Crop for region of interest
def show_image_for_crop(image,crop_type = 0):
    im = hs.signals.Signal2D(image)
    h, w = image.shape   # might not right
    im.plot(cmap="viridis",colorbar=False, scalebar=False,axes_ticks=False,axes_off = True,title = '')
    if crop_type == 1:
        roi_rect = hs.roi.RectangularROI(left=int(0.25*w), top=int(0.25*h), right=int(0.75*w), bottom=int(0.75*h))
    elif crop_type == 2:
        roi_rect = hs.roi.RectangularROI(left=int(0.25*w), top=int(0.05*h), right=int(0.75*w), bottom=int(0.95*h))
    elif crop_type == 3:
        roi_rect = hs.roi.RectangularROI(left=int(0.05*w), top=int(0.05*h), right=int(0.95*w), bottom=int(0.95*h))
    elif crop_type == 4:
        roi_rect = hs.roi.RectangularROI(left=int(0.10*w), top=int(0.10*h), right=int(0.90*w), bottom=int(0.90*h))
    elif crop_type == 5:
        roi_rect = hs.roi.RectangularROI(left=int(0.05*w), top=int(0.25*h), right=int(0.95*w), bottom=int(0.75*h))
    elif crop_type == 6:
        roi_rect = hs.roi.RectangularROI(left=int(0.10*w), top=int(0.15*h), right=int(0.90*w), bottom=int(0.85*h))   
    elif crop_type == 7:
        roi_rect = hs.roi.RectangularROI(left=int(0.10*w), top=int(0.15*h), right=int((0.10 + 75.0/128.0)*w), bottom=int((0.15+50.0/128.0)*h))  
    else:
        roi_rect = hs.roi.RectangularROI(left=int(0.25*w), top=int(0.25*h), right=int(0.75*w), bottom=int(0.75*h))
    roi_rect.interactive(im,color="red")
    return roi_rect

def crop_txt(fold_name, roi_rect, apply_crop_image):
    left, right, top, bottom  = roi_rect
    boundary_txt = np.array([int(top), int(bottom), int(left), int(right)])
    save_txt = os.path.join(fold_name, 'ROI_' + str(apply_crop_image) + '.txt') 
    np.savetxt(save_txt,boundary_txt,fmt='%d')

def load_crop(fold_name,dataset,path,apply_crop_image):
    dataset_crop = copy.copy(dataset)
    if apply_crop_image != 0:
        crop_name = os.path.join(fold_name, 'ROI_' + str(apply_crop_image) + '.txt') 
        top, bottom, left, right = np.loadtxt(crop_name, dtype=int)
        dataset_crop, path = region_of_interest_crop(fold_name, dataset_crop, apply_crop_image, top, bottom, left, right)
    else:
        top = 0
        left = 0
        bottom, right = dataset.data.shape[0:2]
    crop_matrix = [top, bottom, left, right]
    return dataset_crop, path, crop_matrix

def save_BF_DF(path,image_BF,image_DF,crop,name_extra,scale,expand_BF,det_DF):
    image_BF = image_BF[crop[0]:crop[1], crop[2]:crop[3]]
    image_DF = image_DF[crop[0]:crop[1], crop[2]:crop[3]]
    fig = py4DSTEM.visualize.show_image_grid(
            lambda i:[image_BF, image_DF][i],H=1,W=2,
            cmap='gray',
            returnfig = True,
            pixelsize=scale,
            pixelunits='pm', 
            scalebar={'color':'red'}
            )
    file_out = 'BF_E' + str(expand_BF) + '&DF_I' + str(det_DF[0]) + '_O' + str(det_DF[1]) + name_extra + '.png'
    file_path_out = os.path.join(path, file_out)
    fig[0].savefig(file_path_out)
    return image_BF, image_DF

def plot_processed_BF_DF_images(dataset,path,fold_name,name_extra,qx0_crop,qy0_crop,probe_semiangle_crop,expand_BF,det_DF,fft_clean,fft_type,fft_ratio,apply_drift_correction,scale,scale_ratio,crop):
    # Show bright field and dark filed images
    image_BF, image_DF = show_raw_BF_DF_images(dataset,path,qx0_crop,qy0_crop,probe_semiangle_crop,expand_BF,det_DF,scale)
    # Use fft to clean the images
    if fft_clean == 1:
        image_BF, image_DF = BF_DF_fft(path,image_BF,image_DF,fft_type,fft_ratio,scale)
    # Load drift corrected file for replotting the BF and DF images
    if apply_drift_correction == 1:
        image_BF, image_DF = show_BF_DF_drift_corrected_images(path,fold_name,image_BF,image_DF,name_extra,scale,scale_ratio) 
    plt.close('all')
    image_BF, image_DF = save_BF_DF(path,image_BF,image_DF,crop,name_extra,scale,expand_BF,det_DF)
    return image_BF, image_DF
###########################################################################################




## 5.0 ##
## Get Center of mass (CoM) ###############################################################
###  Get center of mass for further processing
def plot_processed_CoM(dataset,path,fold_name,scale,recip_scale,apply_drift_correction,scale_ratio,crop,mask):
    # Get the raw CoM from py4dstem
    CoMx, CoMy = py4DSTEM.process.dpc.get_CoM_images(dataset, mask=mask)
    name_extra = ''
    # Load drift corrected file for replotting the BF and DF images
    if apply_drift_correction == 1:
        CoMx = image_correction(CoMx,fold_name,scale_ratio,0,0)
        CoMy = image_correction(CoMy,fold_name,scale_ratio,0,0)
        name_extra = '_Drift'
    CoMx = CoMx[crop[0]:crop[1], crop[2]:crop[3]]
    CoMy = CoMy[crop[0]:crop[1], crop[2]:crop[3]]
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].imshow(recip_scale*CoMx,cmap='RdBu')
    ff = ax[1].imshow(recip_scale*CoMy,cmap='RdBu')
    fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
    plt.setp(ax, xticks=[],yticks=[])
    fig.savefig(os.path.join(path, 'CoM_raw' + name_extra + '.png'), bbox_inches='tight')
    plt.close()
    return CoMx, CoMy

def GetPLRotation(dpcx: np.ndarray, dpcy: np.ndarray, order: int = 3, outputall: bool = False):
    "This code is from GetDPC by Jordan Adam Hachtel. Oak Ridge National Laboratory"
    """Find Rotation from PL Lenses by minimizing curl/maximizing divergence of DPC data
    :param dpcx: X-Component of DPC Data (2D numpy array)
    :param dpcy: Y-Component of DPC Data (2D numpy array)
    :param order: Number of times to iterated calculation (int)
    :param outputall: Output Curl and Divergence curves for all guesses in separate array (bool)
    :return: The true PL Rotation value (Note: Can potentially be off by 180 degrees, determine by checking signs of charge/field/potential)
    """
    def DPC_ACD(dpcx,dpcy,tlow,thigh):        
        A,C,D=[],[],[]
        for t in np.linspace(tlow,thigh,10,endpoint=False):            
            rdpcx,rdpcy=dpcx*np.cos(t)-dpcy*np.sin(t),dpcx*np.sin(t)+dpcy*np.cos(t)        
            gXY,gXX=np.gradient(rdpcx);gYY,gYX=np.gradient(rdpcy)        
            C.append(np.std(gXY-gYX));D.append(np.std(gXX+gYY));A.append(t)
        R=np.average([A[np.argmin(C)],A[np.argmax(D)]])
        return R,A,C,D
    RotCalcs=[]
    RotCalcs.append(DPC_ACD(dpcx,dpcy,0,np.pi))
    for i in range(1,order): 
        RotCalcs.append(DPC_ACD(dpcx,dpcy,RotCalcs[i-1][0]-np.pi/(10**i),RotCalcs[i-1][0]+np.pi/(10**i)))
    if outputall: return RotCalcs
    else: return RotCalcs[-1][0]

def plot_processed_CoM_with_PL_rotation(path,scale,recip_scale,CoMx,CoMy,apply_drift_correction,fft_clean_CoM,fft_type_CoM,fft_ratio_CoM,flip_key):
    ### Calculate PL Rotation
    RotationCalcs = GetPLRotation(CoMx,CoMy,order=4,outputall=True)
    f,a=plt.subplots(dpi=200,figsize=(4,4))
    for i in range(len(RotationCalcs)):
        print('Pass '+str(i+1)+': Angle='+str(round(RotationCalcs[i][0]*180./np.pi,1)))
        a.plot(RotationCalcs[i][1],RotationCalcs[i][2],lw=i+1,color='r',label='Curl')
        a.plot(RotationCalcs[i][1],RotationCalcs[i][3],lw=i+1,color='b',label='Divergence')
        if i==0: a.legend()
    PLRotation=RotationCalcs[-1][0]
    f.savefig(os.path.join(path, 'PL_rotation.png'))
    plt.close(f)
    rCoMx, rCoMy = CoMx * np.cos(PLRotation) - CoMy * np.sin(PLRotation), CoMx * np.sin(PLRotation) + CoMy * np.cos(PLRotation)
    if flip_key == 1:
        rCoMy = -rCoMy   # ## For FeO_04_01CS not sure why?

    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].imshow(recip_scale*rCoMx,cmap='RdBu')
    ff = ax[1].imshow(recip_scale*rCoMy,cmap='RdBu')
    fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
    plt.setp(ax, xticks=[],yticks=[])
    name_extra = ''
    if apply_drift_correction == 1:
        name_extra = '_Drift'
    fig.savefig(os.path.join(path, 'CoM' + name_extra + '.png'),bbox_inches='tight')
    
    if fft_clean_CoM == 1:
        plt.close()
        rCoMx_clean = fft_clean(rCoMx,fft_type_CoM,fft_ratio_CoM)
        rCoMy_clean = fft_clean(rCoMy,fft_type_CoM,fft_ratio_CoM)

        fig,ax=plt.subplots(1,2,figsize=(12,4))
        ax[0].imshow(recip_scale*rCoMx_clean,cmap='RdBu')
        ff = ax[1].imshow(recip_scale*rCoMy_clean,cmap='RdBu')
        fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
        plt.setp(ax, xticks=[],yticks=[])

        fig.savefig(os.path.join(path, 'CoM' + name_extra + '_' + fft_type_CoM + '_' + str(fft_ratio_CoM) + '.png'),bbox_inches='tight')

    return rCoMx, rCoMy

def set_PL_rotation(path,scale,recip_scale,CoMx,CoMy,apply_drift_correction,fft_clean_CoM,fft_type_CoM,fft_ratio_CoM,PLRotation,flip_key=0):
    PLRotation = (PLRotation*np.pi) / 180
    rCoMx, rCoMy = CoMx * np.cos(PLRotation) - CoMy * np.sin(PLRotation), CoMx * np.sin(PLRotation) + CoMy * np.cos(PLRotation)
    if flip_key == 1:
        rCoMy = -rCoMy

    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].imshow(recip_scale*rCoMx,cmap='RdBu')
    ff = ax[1].imshow(recip_scale*rCoMy,cmap='RdBu')
    fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
    plt.setp(ax, xticks=[],yticks=[])
    name_extra = ''
    if apply_drift_correction == 1:
        name_extra = '_Drift'
    fig.savefig(os.path.join(path, 'CoM' + name_extra + '.png'),bbox_inches='tight')

    if fft_clean_CoM == 1:
        plt.close()
        rCoMx_clean = fft_clean(rCoMx,fft_type_CoM,fft_ratio_CoM)
        rCoMy_clean = fft_clean(rCoMy,fft_type_CoM,fft_ratio_CoM)
        fig,ax=plt.subplots(1,2,figsize=(12,4))
        ax[0].imshow(recip_scale*rCoMx_clean,cmap='RdBu')
        ff = ax[1].imshow(recip_scale*rCoMy_clean,cmap='RdBu')
        fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
        plt.setp(ax, xticks=[],yticks=[])
        fig.savefig(os.path.join(path, 'CoM' + name_extra + '_' + fft_type_CoM + '_' + str(fft_ratio_CoM) + '.png'),bbox_inches='tight')
    
    return rCoMx, rCoMy
###########################################################################################




## 6.0 ##
## Electric field map #####################################################################
###  Get electric field map
def vector_to_rgb(angle, absolute):
    """Get the rgb value for the given `angle` and the `absolute` value
    Parameters
    ----------
    angle : float
        The angle in radians
    absolute : float
        The absolute value of the gradient
    Returns
    -------
    array_like
        The rgb value as a tuple with values [0..1]
    """
    global max_abs
    # normalize angle
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi
    return colors.hsv_to_rgb((angle / 2 / np.pi, 
                                absolute / max_abs, 
                                absolute / max_abs))

def angle_vector(CoMx,CoMy,E_f,h0,h1,w0,w1):
    global max_abs
    x,y = np.meshgrid(np.linspace(w0,w1,w1-w0),np.linspace(h1,h0,h1-h0))  # exchanged h0 and h1 20220213
    u = -CoMx*E_f
    v = -CoMy*E_f
    angles = np.arctan2(v, u)
    lengths = np.sqrt(np.square(u) + np.square(v))
    max_abs = np.max(lengths)
    c = np.array(list(map(vector_to_rgb, angles.flatten(), lengths.flatten())))
    return x, y, u, v, c

def cal_electric_fields(CoMx, CoMy, E_f, LegPix=301, LegRad=0.85):
    "This code is from GetDPC by Jordan Adam Hachtel. Oak Ridge National Laboratory"
    """Convert dpcx and dpcy maps to to a color map where the color corresponds to the angle
    :param LegPix: Number of Pixels in Color Wheel Legend
    :param LegRad: Radius of Color Wheel in Legend (0-1)
    :return: The electric fields as a 2D numpy array
    """
    EX = -CoMx*E_f
    EY = -CoMy*E_f
    EMag = np.sqrt(EX ** 2 + EY ** 2)
    XY = np.zeros(EX.shape + (3,), dtype=float)
    M_max = np.amax(EMag)
    EMagScale = EMag / M_max
    for i in range(EX.shape[0]):
        for j in range(EX.shape[1]):
            XY[i, j] = np.angle(np.complex(EX[i, j], EY[i, j])) / (2 * np.pi) % 1, 1, EMagScale[i, j]
    EDir = hsv_to_rgb(XY)
    x, y = np.meshgrid(np.linspace(-1, 1, LegPix, endpoint=True), np.linspace(-1, 1, LegPix, endpoint=True))
    X, Y = x * (x ** 2 + y ** 2 < LegRad ** 2), y * (x ** 2 + y ** 2 < LegRad ** 2)           ## Rhett changed here add "-" before y, 20220208
    XYLeg = np.zeros(X.shape + (3,), dtype=float)
    RI = np.sqrt(X ** 2 + Y ** 2) / np.amax(np.sqrt(X ** 2 + Y ** 2))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            XYLeg[i, j] = np.angle(np.complex(X[i, j], -Y[i, j])) / (2 * np.pi) % 1, 1, RI[i, j] 
    EDirLeg = hsv_to_rgb(XYLeg)
    return EMag, EDir, EDirLeg

def pre_factor_E(voltage_kV, thickness_nm,recip_scale_1nm,print_value = 0):
    v = cal_speed_electron_m1s(voltage_kV)                                                     ## speed of electron, unit m/s
    t = thickness_nm * 1E-9                                                                    ## thickness of the specimen, unit m
    recip_scale = recip_scale_1nm * 1E9                                                        ## scale for each pixel in recip space, unit 1/m / pixel
    e = 1.6022E-19                                                                             ## charge of an electron, unit C
    h = 6.62E-34                                                                               ## Planck's constant, unit Js   
    ## way1
    E_f = (1E-12*(recip_scale*h*v))/(e*t)                                                      ## pre-factor for calulating electric field, unit V/pm    
    ## way2
    sigma_1vm = cal_sigma_1vm(voltage_kV)/(2*np.pi)
    E_f2 = (1E-12*(recip_scale)) / (sigma_1vm*t)
    if print_value == 1:
        print('Pre-factor for calculating electric field: ' + '%.2f' % E_f + ' V/pm (based on Ehrenfest theorem)')
        print('Pre-factor for calculating electric field: ' + '%.2f' % E_f2 + ' V/pm (based on multislice algorithm)')
    return E_f

def plot_eletric_field(path,rCoMx,rCoMy,name_extra_CoM,image_DF,real_scale_pm,voltage_kV,thickness_nm,recip_scale_1nm,G_blur):
    E_f = pre_factor_E(voltage_kV, thickness_nm, recip_scale_1nm,1)                            ## prefactor for calulating electric field, unit V/pm
    EImR,EDirImR,EDirLegR=cal_electric_fields(rCoMx,rCoMy,E_f)                                 ## from Rotated CoM Shifts
    fig,ax = plt.subplots(1,4,frameon=False,dpi=200,figsize=(15,4))
    py4DSTEM.visualize.show(
        image_DF,cmap='gray',
        returnfig = True,
        pixelsize = real_scale_pm,
        pixelunits='pm',
        scalebar={'position':'bl','label':False,'color':'white'},
        figax=(fig,ax[0]))
    ax[0].set_title('Dark Field',fontsize=10)
    ax[1].imshow(EImR)
    ax[1].set_title('Electric field Magnitude',fontsize=10)
    ax[2].imshow(EDirImR)
    ax[2].set_title('Electric field Directions',fontsize=10)
    ax[3].imshow(EDirLegR)
    ax[3].set_title('Electric field Legend',fontsize=10)
    plt.setp(ax, xticks=[],yticks=[])
    plt.tight_layout()
    fig.savefig(os.path.join(path, 'Electric Field' + name_extra_CoM + '.png'))


    fig,ax=plt.subplots(dpi=200)
    ff = ax.imshow(EImR,vmin=np.amin(EImR),vmax=np.amax(EImR))
    fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
    ax.set_title('Electric field Magnitude (V/pm)',fontsize=8)
    plt.setp(ax, xticks=[],yticks=[])
    fig.savefig(os.path.join(path, 'Electric Field Magnitude' + name_extra_CoM + '.png'))
    if G_blur != 0:
        plt.close()
        EImR_b = cv2.GaussianBlur(EImR,(G_blur,G_blur),0)
        fig,ax=plt.subplots(dpi=200)
        ff = ax.imshow(EImR_b,vmin=np.amin(EImR_b),vmax=np.amax(EImR_b))
        fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
        ax.set_title('Electric field Magnitude (V/pm)',fontsize=8)
        plt.setp(ax, xticks=[],yticks=[])
        fig.savefig(os.path.join(path, 'Electric Field Magnitude' + name_extra_CoM + '_G' + str(G_blur) +  '.png'))

def plot_vector(path,rCoMx,rCoMy,name_extra_CoM,voltage_kV,thickness_nm,recip_scale_1nm,crop,lkey,scalef):
    E_f = pre_factor_E(voltage_kV, thickness_nm, recip_scale_1nm,0)                             ## prefactor for calulating electric field, unit V/pm
    x, y, u, v, c = angle_vector(rCoMx,rCoMy,E_f,crop[0],crop[1],crop[2],crop[3])
    fig, ax = plt.subplots(figsize=(12, 12))
    q = ax.quiver(x, y, u, v, angles='xy', scale_units='xy',color=c, scale=scalef)
    # q = ax.quiver(x, y, u, v, color=c, units='dots')
    ql = ax.quiverkey(q, X=0.95, Y=0.95, U=lkey, label='', labelpos='E',color = 'black')
    plt.axis('equal')
    plt.xlim([crop[2],crop[3]])
    plt.ylim([crop[0],crop[1]])
    plt.axis('off')
    plt.show()
    fig.savefig(os.path.join(path, 'Angle_Vector'+ name_extra_CoM + '_l' + str(lkey) + '_s' + str(scalef) + '.png'))
###########################################################################################




## 7.0 ##
## Charge density map #####################################################################
###  Get charge density map
def cal_gradient(dpcx: np.ndarray, dpcy: np.ndarray):
    "This code is from GetDPC by Jordan Adam Hachtel. Oak Ridge National Laboratory"
    """Calculate Charge Density from the Divergence of the Ronchigram Shifts
    :param dpcx: X-Component of DPC Data (2D numpy array)
    :param dpcy: Y-Component of DPC Data (2D numpy array)
    :return: The charge density as a 2D numpy array
    """
    gxx, gyy = np.gradient(dpcx)[1], np.gradient(dpcy)[0]
    return - gxx + gyy   #### Original -gyy, due to different co system, I have to change it to +gyy

def pre_factor_Div_Rho(voltage_kV, thickness_nm, recip_scale_1nm,real_scale_pm):
    v = cal_speed_electron_m1s(voltage_kV)                                                               ## speed of electron, unit m/s
    t = thickness_nm * 1E-9                                                                              ## thickness of the specimen, unit m
    recip_scale = recip_scale_1nm * 1E9                                                                  ## scale for each pixel in recip space, unit 1/m / pixel
    real_scale = real_scale_pm * 1E-12                                                                   ## scale for each pixel in real space, unit, m / pixel
    e = 1.6022E-19                                                                                       ## charge of an electron, unit C
    h = 6.62E-34                                                                                         ## Planck's constant, unit Js
    epsilon = 8.8542E-12                                                                                 ## dielectric permittivity of free space, Unit C/(Vm)
    Div_f = (recip_scale*h)/(real_scale)                                                                 ## Divergency, unit Js/m-2
    C_f_1A2 =  (1E-20*Div_f*epsilon*v)/(e*e)                                                             ## Charge density, unit 1/A2
    C_f_1A3 =  (1E-30*Div_f*epsilon*v)/(e*e*t)                                                           ## Charge density, unit 1/A3
    return Div_f, C_f_1A2, C_f_1A3

def plot_divergence_vector(path,rCoMx,rCoMy,name_extra_CoM,voltage_kV,thickness_nm,recip_scale_1nm,real_scale_pm,G_blur):
    GraIm = cal_gradient(rCoMx,rCoMy)    
    DivIm = GraIm * pre_factor_Div_Rho(voltage_kV,thickness_nm,recip_scale_1nm,real_scale_pm)[0]
    fig,ax=plt.subplots(dpi=200)
    ff = ax.imshow(DivIm,cmap=cm.seismic,vmin=np.amin(DivIm),vmax=np.amax(DivIm))
    fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=1.0)
    ax.set_title('Divergence of the vector field (Js/$m^{-2}$)',fontsize=8)
    plt.setp(ax, xticks=[],yticks=[])
    fig.savefig(os.path.join(path, 'Divergence of the vector field' + name_extra_CoM + '.png'))
    if G_blur != 0:
        plt.close()
        DivIm_b = cv2.GaussianBlur(DivIm,(G_blur,G_blur),0)
        fig,ax=plt.subplots(dpi=200)
        ff = ax.imshow(DivIm_b,cmap=cm.seismic,vmin=np.amin(DivIm_b),vmax=np.amax(DivIm_b))
        fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=1.0)
        ax.set_title('Divergence of the vector field (Js/$m^{-2}$)',fontsize=8)
        plt.setp(ax, xticks=[],yticks=[])
        fig.savefig(os.path.join(path, 'Divergence of the vector field' + name_extra_CoM + '_G' + str(G_blur) +  '.png'))
    return DivIm

def plot_charge_density(path,rCoMx,rCoMy,name_extra_CoM,voltage_kV,thickness_nm,recip_scale_1nm,real_scale_pm,G_blur):
    GraIm = cal_gradient(rCoMx,rCoMy)    
    RhoIm = GraIm * pre_factor_Div_Rho(voltage_kV,thickness_nm,recip_scale_1nm,real_scale_pm,)[1]
    fig,ax=plt.subplots(dpi=200)
    ff = ax.imshow(RhoIm,cmap=cm.seismic,vmin=np.amin(RhoIm),vmax=np.amax(RhoIm))                       ## Rhett changed here 2001102 RdBu_r
    fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
    ax.set_title('Charge Density (1/$\AA^2$)',fontsize=8)
    plt.setp(ax, xticks=[],yticks=[])
    fig.savefig(os.path.join(path, 'Charge Density' + name_extra_CoM + '.png'))
    if G_blur != 0:
        plt.close()
        RhoIm_b = cv2.GaussianBlur(RhoIm,(G_blur,G_blur),0)
        fig,ax=plt.subplots(dpi=200)
        ff = ax.imshow(RhoIm_b,cmap=cm.seismic,vmin=np.amin(RhoIm_b),vmax=np.amax(RhoIm_b))             ## Rhett changed here 2001102 RdBu_r
        fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
        ax.set_title('Charge Density (1/$\AA^2$)',fontsize=8)
        plt.setp(ax, xticks=[],yticks=[])
        fig.savefig(os.path.join(path, 'Charge Density' + name_extra_CoM + '_G' + str(G_blur) +  '.png'))
    return RhoIm
###########################################################################################




## 8.0 ##
## Potential map ##########################################################################
###  Get Ppotential map
def pre_factor_V(voltage_kV,thickness_nm,recip_scale_1nm,real_scale_pm):
    v = cal_speed_electron_m1s(voltage_kV)                                                              ## speed of electron, unit m/s
    t = thickness_nm * 1E-9                                                                             ## thickness of the specimen, unit m
    recip_scale = recip_scale_1nm * 1E9                                                                 ## scale for each pixel in recip space, unit 1/m / pixel
    real_scale = real_scale_pm * 1E-12                                                                  ## scale for each pixel in real space, unit, m / pixel
    e = 1.6022E-19                                                                                      ## charge of an electron, unit C
    h = 6.62E-34                                                                                        ## Planck's constant, unit Js
    V_f = (recip_scale*real_scale*h*v)/(e*t)                                                            ## V, unit V
    return V_f

def cal_integral(rCoMx,rCoMy,use_pass_filter,lP=0.1,hP=1000):
    qx = np.fft.fftfreq(rCoMx.shape[0])
    qy = np.fft.fftfreq(rCoMx.shape[1])
    qr2 = qx[:, None] ** 2 + qy[None, :] ** 2
    denominator = qr2
    denominator_filter = qr2 + hP + ((qr2 ** 2) * lP)
    _ = np.seterr(divide="ignore")
    denominator = 1.0 / denominator
    denominator[0, 0] = 0
    denominator_filter = 1.0 / denominator_filter
    denominator_filter[0, 0] = 0
    _ = np.seterr(divide="warn")
    f = -1j * 0.25 
    x_op  = f * qx[:, None] * denominator
    y_op  = f * qy[None, :] * denominator
    x_op_filter  = f * qx[:, None] * denominator_filter
    y_op_filter  = f * qy[None, :] * denominator_filter
    im_pot = np.fft.ifft2(np.fft.fft2(-rCoMy)*x_op).real + np.fft.ifft2(np.fft.fft2(rCoMx)*y_op).real
    im_pot_filter = np.fft.ifft2(np.fft.fft2(-rCoMy)*x_op_filter).real + np.fft.ifft2(np.fft.fft2(rCoMx)*y_op_filter).real
    im_pot = im_pot - im_pot.min()                          ## Make all values positive
    im_pot_filter = im_pot_filter - im_pot_filter.min()     ## Make all values positive
    rescale = im_pot.max()/im_pot_filter.max()
    if use_pass_filter == 1:
        im_pot_filter = im_pot_filter*rescale
        im_pot = im_pot_filter
    return im_pot

def plot_potential(path,rCoMx,rCoMy,name_extra_CoM,voltage_kV,thickness_nm,recip_scale_1nm,real_scale_pm,use_pass_filter,lP=0,hP=0.01,G_blur=0):
    ## Calculate Potential from Inverse Gradient of CoM Shifts
    VIm = cal_integral(rCoMx,rCoMy,use_pass_filter,lP,hP)
    VIm = VIm*pre_factor_V(voltage_kV,thickness_nm,recip_scale_1nm,real_scale_pm)
    if use_pass_filter == 1:
        name_extra_CoM = name_extra_CoM + '_lP' + str(lP) + '_hP' + str(hP) 
        # print(name_extra_CoM)
    fig,ax=plt.subplots(dpi=200)
    ff = ax.imshow(VIm,cmap=cm.hot,vmin=np.amin(VIm),vmax=np.amax(VIm))
    fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
    ax.set_title('Potential (V)',fontsize=8)
    plt.setp(ax, xticks=[],yticks=[])
    fig.savefig(os.path.join(path, 'Potential' + name_extra_CoM + '.png'))
    if G_blur != 0:
        plt.close()
        VIm_b = cv2.GaussianBlur(VIm,(G_blur,G_blur),0)
        fig,ax=plt.subplots(dpi=200)
        ff = ax.imshow(VIm_b,cmap=cm.hot,vmin=np.amin(VIm_b),vmax=np.amax(VIm_b))
        fig.colorbar(ff, ax = ax,location='right', anchor=(0, 0.5), shrink=0.8)
        ax.set_title('Potential (V)',fontsize=8)
        plt.setp(ax, xticks=[],yticks=[])
        fig.savefig(os.path.join(path, 'Potential' + name_extra_CoM + '_G' + str(G_blur) +  '.png'))
    return VIm












## Appendix
###################################################################################################
def bin_diff(file_path_input,dataset,bin_diff = 8):
    ## Binning of the diffraction image, I normally use it for the simulated 4DSTEM data set
    fold_name = os.path.split(file_path_input)[0]
    file_name = os.path.split(file_path_input)[1]
    binned_size = int(dataset.data[0,0].shape[0]/bin_diff)
    dataset.bin_data_diffraction(bin_diff)
    fold_extra = 'DiffBinned_' +str(binned_size)
    fold_name_new = os.path.join(fold_name,fold_extra)
    if not os.path.exists(fold_name_new):
        os.makedirs(fold_name_new)
    file_name_output_h5 = file_name[:-3] + '_DiffBinned_' +str(binned_size) + '.h5'
    file_path_output_h5 = os.path.join(fold_name_new,file_name_output_h5)
    py4DSTEM.io.save(file_path_output_h5,dataset,overwrite=True)
###################################################################################################
def check_DP(path,dataset,DP_h,DP_w,recip_scale_1nm,plot = 1):
    ## Check individual diffraction pattern
    fig, ax = py4DSTEM.visualize.show(
            dataset.data[DP_h,DP_w],
            scaling='power',
            power=0.5,
            cmap='inferno',
            returnfig = True,
            pixelsize = recip_scale_1nm,
            pixelunits='1/nm',
            scalebar={'position':'br','label':True,'alpha':0},
            )
    ax.set_title('Pattern_h' + str(DP_h) + 'w' + str(DP_w),fontsize=18)
    fig.savefig(os.path.join(path, 'Pattern_h' + str(DP_h) + 'w' + str(DP_w) + '.png'))
    if plot == 0:
        plt.close()
###################################################################################################
def plot_4D(data_4D,real_scale_pm,recip_scale_mrad):
        ## This code was learnt from Dr. Christoph Hofer, University of Antwerp
        ## It can plot the change of diffraction patterns as a function of probe position
        s = ps.PixelatedSTEM(data_4D)
        s.axes_manager[0].name = 'x'
        s.axes_manager[0].units = 'pm'
        s.axes_manager[0].scale = real_scale_pm
        s.axes_manager[1].name = 'y'
        s.axes_manager[1].units = 'pm'
        s.axes_manager[1].scale = real_scale_pm
        s.axes_manager[2].name = 'angle x'
        s.axes_manager[2].units = 'mrad'
        s.axes_manager[2].scale = recip_scale_mrad
        s.axes_manager[3].name = 'angle y'
        s.axes_manager[3].units = 'mrad'
        s.axes_manager[3].scale = recip_scale_mrad
        s.metadata.General.title = '4D data'
        s.plot(cmap='viridis')
        return s
####################################################################################################
def export_video(folder):
    ## Export a video.
    video_name = os.path.join(folder,'video.avi')
    image_folder = os.path.join(folder,'video')
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 30, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()
    shutil.rmtree(image_folder)
####################################################################################################
def generate_DP_video(path,dataset,recip_scale_1nm):
    ## This code is used to export a video showing the change in diffraction patterns as a function of probe position.
    DP_h_all,DP_w_all,_,_ = dataset.data.shape
    image_folder = os.path.join(path,'video')
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    os.makedirs(image_folder)
    for DP_h in range(int(DP_h_all/5)):
        for DP_w in range(int(DP_w_all/5)):
            check_DP(image_folder,dataset,DP_h,DP_w,recip_scale_1nm,0)   # check individual diffraction pattern
    export_video(path)
####################################################################################################
