import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia

import numpy as np
import sys

from tqdm import tqdm

%matplotlib inline
import matplotlib.pyplot as plt
import csv
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def chi2(array, parallax, parallax_err):
    for i in range(0, len(parallax)):
        array.append([])
        for j in range(i+1, len(parallax)):
            chi2 = (parallax[i]-parallax[j])**2/(parallax_err[i]**2+parallax_err[j]**2)
            if chi2 <=9:
                w = (parallax[i]/parallax_err[i]**2+parallax[j]/parallax_err[j]**2)/(1/parallax_err[i]**2+1/parallax_err[j]**2)
        
                array[i].append(j)
            else:
                continue
                
def match(array, parallax, parallax_err, ra, ra_err, dec, dec_err, pmra, pmdec, ID, x_id, y_id, x_pmra, x_pmdec, 
            x_ra, x_dec, chi2_mu_arr, tnew_arr, told_arr, matches):
    for i in tqdm(range(0, len(parallax))):
        array.append([])
        for j in range(i+1, len(parallax)):
            chi2 = (parallax[i]-parallax[j])**2/(parallax_err[i]**2+parallax_err[j]**2)
            
            if chi2 <=9:
                x_parallax= parallax[i]*1e-3
                y_parallax= parallax[j]*1e-3
                
                x_parallax_err = parallax_err[i]*1e-3
                y_parallax_err = parallax_err[j]*1e-3
                  
                w = ((x_parallax**2)/(x_parallax_err)**2+(y_parallax)**2/(y_parallax_err)**2)/(1/(x_parallax_err)**2+1/(y_parallax_err)**2)
                delta_ra = (ra[i]-ra[j])*np.pi/180*np.cos(dec[i]*np.pi/180)*206265
                delta_dec = (dec[i]-dec[j])*np.pi/180*206265
                delta_pmra = (pmra[i]-pmra[j])*1e-3
                delta_pmdec = (pmdec[i]-pmdec[j])*1e-3
                
                def sigmara_2(ra_err,ti,i,j):
                    return (ra_err[i]*1e-3)**2+(ra_err[j]*1e-3)**2+(10*w)**2/ti**2
                
                def sigmadec_2(dec_err,ti,i,j):
                    return (dec_err[i]*1e-3)**2+(dec_err[j]*1e-3)**2+(10*w)**2/ti**2 
                
                def t_closest(sigma_2, sigma):
                    return -((delta_ra)**2*(sigma)+(delta_dec)**2*sigma_2)/((delta_ra)*delta_pmra*(sigma)+(delta_dec)*delta_pmdec*(sigma_2))
       
                t_old = 1e4
                t_new = t_closest(sigmara_2(ra_err, t_old,i,j), sigmadec_2(dec_err, t_old,i,j))
                
                while abs(t_new - t_old) > 10**-4:
                    t_old = t_new
                    t_new = t_closest(sigmara_2(ra_err, t_old,i,j), sigmadec_2(dec_err, t_old,i,j))
                    chi2_mu = (delta_ra/t_new - delta_pmra)**2/sigmara_2(ra_err, t_new,i,j) + \
                    (delta_dec/t_new - delta_pmdec)**2/sigmadec_2(dec_err, t_new,i,j)
                    
                if chi2_mu <= 3: 
                    x_id = np.concatenate((x_id, ID[i]), axis = None)
                    y_id = np.concatenate((y_id, ID[j]), axis = None)
                    
                    x_pmra = np.concatenate((x_pmra, pmra[i]), axis = None)
                    x_pmdec = np.concatenate((x_pmdec, pmdec[i]), axis = None)
                    
                    x_ra = np.concatenate((x_ra, ra[i]), axis = None)
                    x_dec = np.concatenate((x_dec, dec[i]*np.pi/180), axis = None)
                    
                    chi2_mu_arr = np.concatenate((chi2_mu_arr, chi2_mu), axis = None)
                    tnew_arr = np.concatenate((tnew_arr, t_new), axis = None)
                    told_arr = np.concatenate((told_arr, t_old), axis = None)
                    
                    matches = np.stack((x_id, y_id, x_pmra, x_pmdec, x_ra, x_dec, tnew_arr, told_arr), axis = 1)
               
    return matches  
