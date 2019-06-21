#!/usr/bin/env python
# 
# Based on demo/tracker_comparison.ipynb (from commit 92d79d3)
# - this generates reference data
# 
import os
import numpy as np
import pandas as pd

from samwich.dataloaders import PlanarData
from samwich.waketrackers import track
from samwich.gaussian_functions import PorteAgel
trackerlist = track()


datadir = '../examples/MWE-data/'

#kind = 'mean'  # sanity check, not very interesting
kind = 'instantaneous'

D = 27.0  # to define the search range, and the reference area for the contour methods
zhub = 32.1  # hub height [m], for estimating the freestream reference velocity
aref = 0.3  # induction, for estimating the momentum theory mass/momentum flux
verbose = True

tol = 1e-13  # for checking reference data


## Preliminary calculations
ref_area = np.pi*D**2/4
ref_CT = 4*aref*(1-aref)  # thrust coefficient
ref_CP = 4*aref*(1-aref)**2  # power coefficient
if verbose:
    print('reference area/C_T/C_P :',ref_area,ref_CT,ref_CP)


## Read in test data
varlist = ['x','y','z','u','v','w']
sample = PlanarData({v: np.loadtxt(datadir+'3D_{}_{}_WFoR.txt'.format(kind,v)) for v in varlist})


## Calculate freestream
data = {v: np.loadtxt(datadir+'freestream_mean_{}_WFoR.txt'.format(v)) for v in varlist}
free_z = data['z'][0,:]
free_Uprofile = np.mean(data['u'],axis=0)
jhub = np.argmin(np.abs(data['y'][:,0]-np.mean(data['y'][:,0])))
khub = np.argmin(np.abs(data['z'][0,:]-zhub))
ref_velocity = data['u'][jhub,khub]
#free_const = free_Uprofile*0.0 + ref_velocity
ref_thrust = ref_CT * 0.5*ref_velocity**2 * ref_area  # force / density
if verbose:
    print('ref velocity (at z={z}) : {Uref}'.format(z=data['z'][0,khub],Uref=ref_velocity))
    print('ref thrust (momentum deficit) :',ref_thrust,'N/(kg/m^3)')


## Perform wake identification
wake,yc,zc = {},{},{}

### - Constant area contours
tracker = 'const area'
wake[tracker] = track(sample.sliceI(),method='ConstantArea',verbose=verbose) 
wake[tracker].remove_shear(wind_profile=free_Uprofile)
yc[tracker],zc[tracker] = wake[tracker].find_centers(ref_area,tol=0.01)

### - Momentum deficit contours
tracker = 'momentum deficit'
func = lambda u,u_tot: -u*u_tot
wake[tracker] = track(sample.sliceI(),method='ConstantFlux',verbose=verbose) 
wake[tracker].remove_shear(wind_profile=free_Uprofile)
yc[tracker],zc[tracker] = wake[tracker].find_centers(ref_thrust,
                                                     flux_function=func,
                                                     field_names=('u','u_tot'),
                                                     tol=0.01)

### - Gaussian Fits
tracker = 'gaussian (R)'
wake[tracker] = track(sample.sliceI(),method='Gaussian',verbose=verbose) 
wake[tracker].remove_shear(wind_profile=free_Uprofile)
yc[tracker],zc[tracker] = wake[tracker].find_centers(umin=None,sigma=D/2)

tracker = 'gaussian (Porte-Agel)'
#xref = np.mean(sample.x) - x0  # need to know turbine location...
xloc = 3*D  # samples 3D downstream
# model depends on case-dependent wake growth rate, kstar
fernando = PorteAgel(CT=ref_CT,d0=D,kstar=0.03) # ad-hoc value taken from Bastankhah & Porte-Agel 2014 (the smallest of the full-scale cases)
wake[tracker] = track(sample.sliceI(),method='Gaussian',verbose=verbose) 
wake[tracker].remove_shear(wind_profile=free_Uprofile)
yc[tracker],zc[tracker] = wake[tracker].find_centers(umin=fernando.amplitude(xloc,-ref_velocity),
                                                     sigma=fernando.sigma(xloc))

tracker = 'gaussian (ideal)' # sigma estimated from momentum deficit
wake[tracker] = track(sample.sliceI(),method='Gaussian',verbose=verbose) 
wake[tracker].remove_shear(wind_profile=free_Uprofile)
max_VD = -np.min(wake[tracker].u,axis=(1,2))  # max velocity deficit, u.shape == (Ntimes,Nh,Nv)
sigma_opt = np.sqrt(ref_thrust / (np.pi*max_VD*(2*ref_velocity - max_VD)))
yc[tracker],zc[tracker] = wake[tracker].find_centers(umin=None,
                                                     sigma=sigma_opt)

### - 2D Gaussian
tracker = 'elliptical'
wake[tracker] = track(sample.sliceI(),method='Gaussian2D',verbose=verbose) 
wake[tracker].remove_shear(wind_profile=free_Uprofile)
yc[tracker],zc[tracker] = wake[tracker].find_centers(umin=None,
                                                     A_ref=ref_area,
                                                     A_min=ref_area/5.0,  # ad hoc value
                                                     A_max=ref_area*2.0,  # ad hoc value
                                                     AR_max=10.0,  # ad hoc value
                                                    )

### - Test Region
tracker = 'min power'
wake[tracker] = track(sample.sliceI(),method='CircularTestRegion',verbose=verbose) 
wake[tracker].remove_shear(wind_profile=free_Uprofile)
yc[tracker],zc[tracker] = wake[tracker].find_centers(test_radius=D/2,  # following Vollmer 2016
                                                     test_function=lambda u: u**3,
                                                     test_field='u_tot',
                                                    )


## Output centers
itime = 0
all_yc = [ yc[tracker][itime] for tracker in wake.keys() ]
all_zc = [ zc[tracker][itime] for tracker in wake.keys() ]
df = pd.DataFrame(index=wake.keys(), data={'y':all_yc, 'z':all_zc})

fname = '{}_detected_centers.csv'.format(kind)
refpath = 'ref-data/'+fname
writecsv = True
if os.path.isfile(refpath):
    ref = pd.read_csv(refpath,index_col=0)
    #ymatch = (ref['y'] == df['y'])
    #zmatch = (ref['z'] == df['z'])
    yerr = np.abs(ref['y'] - df['y'])
    zerr = np.abs(ref['z'] - df['z'])
    if np.all(yerr < tol) and np.all(zerr < tol):
        writecsv = False
        print('\nAll values match! (tol={:g})'.format(tol))
    else:
        # mismatch
        print(pd.DataFrame(index=wake.keys(), data={'y':yerr, 'z':zerr}))

if writecsv:
    df.to_csv(fname)
    if verbose:
        print('Wrote',fname)

