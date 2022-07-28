# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:24:02 2022

@author: rahul
"""

import astroalign
import os
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from skimage.transform import SimilarityTransform
import numpy as np
from skimage.transform import warp
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy.wcs import WCS
from astroquery.vizier import Vizier
import astropy.coordinates as coord
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from astropy.time import Time
import pandas as pd
from datetime import timedelta, date
import math as mt
from astropy.visualization import simple_norm
from astropy.timeseries import LombScargle
from astroquery.astrometry_net import AstrometryNet

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

def dates(start_dt,end_dt,pt5m_loc):
    date_list=[]
    for dt in daterange(start_dt, end_dt):
        date_list.append(dt.strftime("%Y-%m-%d"))
    files=os.listdir(pt5m_loc)
    final_date_list=[]
    for i in date_list:
        if i in files:
            final_date_list.append(i)
    return final_date_list

def object_log_table(pt5m_loc,object_name,final_date_list):
    table={}
    date_list=[]
    for i in final_date_list:
        meow=[]
        try:
            file_path=pt5m_loc+'/'+i+'/'+i+'.html'
            tables = pd.read_html(file_path)
            tables=tables[0]
            table_values=(tables[tables[1]==object_name])
            if len(table_values) != 0:
                table[str(i)]=table_values
                date_list.append(str(i))
        except:
            pass
    return table,date_list

def reference_image(files_loc,dates,object_logs,filter,ref_date):
    locals().update(object_logs)
    stack_image=0
    ref_img_data=fits.open(files_loc+ref_date+'/'+np.array(object_logs[ref_date][0])[0]+'.fits')[0].data
    ref_img_data = image_calibration(files_loc,ref_date,filter,ref_img_data)
    norm = simple_norm(ref_img_data, 'sqrt', percent=99)
    #plt.imshow(ref_img_data, norm=norm,cmap='Greys', interpolation='nearest',origin='lower')
    #plt.xlabel('X pixel')
    #plt.ylabel('Y pixel')
    #plt.show()
    for i in dates:
        value=np.array(object_logs[i][0])[0]
        image_array=fits.open(files_loc+i+'/'+value+'.fits')[0].data
        image_array = image_calibration(files_loc,i,filter,image_array)
        try:
            transf, xy_coord = astroalign.find_transform(ref_img_data,image_array)
            transformation_matrix = transf.params
            inv_mat = (transformation_matrix)
            inv_mat=SimilarityTransform(inv_mat)
            transformed_image_array = warp(np.float32(image_array), inv_mat)
            stack_image=stack_image+transformed_image_array
        except:
            pass
    #norm = simple_norm(stack_image, 'sqrt', percent=99)
    #plt.imshow(stack_image,norm=norm,cmap='Greys', interpolation='nearest',origin='lower')
    #plt.xlabel('X pixel')
    #plt.ylabel('Y pixel')
    #plt.show()
    return stack_image

def image_transform(files_loc,dates,object_logs,filter,centre_x,centre_y,radius,source_positions,object_pos,ref_date,FWHM):
    locals().update(object_logs)
    ref_img_data=fits.open(files_loc+ref_date+'/'+np.array(object_logs[ref_date][0])[0]+'.fits')[0].data
    ref_img_data=image_calibration(files_loc,ref_date,filter,ref_img_data)
    object={}
    ap_table={}
    array=[]
    for i in dates:
        value=np.array(object_logs[i][0])
        for run in value:
            image_array=fits.open(files_loc+i+'/'+run+'.fits')[0].data
            image_array=image_calibration(files_loc,i,filter,image_array)
            try:
                run_time=(fits.open(files_loc+i+'/'+run+'.fits')[0].header)['EXPTIME']
                time=(fits.open(files_loc+i+'/'+run+'.fits')[0].header)['DATE-OBS']
                transf, xy_coord = astroalign.find_transform(ref_img_data,image_array)
                transformation_matrix = transf.params
                inv_mat = (transformation_matrix)
                inv_mat=SimilarityTransform(inv_mat)
                transformed_image_array = warp(np.float32(image_array), inv_mat)
                table=ap_less_phot(transformed_image_array,source_positions,run_time,FWHM)
                test=[]
                for t in table:
                    test.append(not(mt.isnan(t['aper_mag']) or mt.isinf(t['aper_mag'])))
                table=table[test]
                ap_table[str(time)]=table
                object_pos=(float(object_pos[0]),float(object_pos[1]))
                table=ap_less_phot(transformed_image_array,object_pos,run_time,FWHM)
                test=[]
                for t in table:
                    test.append(not(mt.isnan(t['aper_mag']) or mt.isinf(t['aper_mag'])))
                table=table[test]
                time=Time(time)
                table['time']=time.jd
                array.append(table)
                object['table']=array
            except:
                pass
    return ap_table, object

def nan_removal(matching_sources):
    sources={}
    locals().update(matching_sources)
    for i in matching_sources:
        t=locals()[i]
        list=[]
        for d in t:
            test=mt.isnan(d['aper_mag']) or mt.isinf(d['aper_mag'])
            if test == False:
                list.append(d)
        sources[i]=list
    w={}
    locals().update(sources)
    for i in sources:
        d=locals()[i]
        if len(d) > 50:
            w['final_'+i]=d
    return w

def image_calibration(pt5m_loc,dates,filter,image):
    try:
        bias=fits.open(pt5m_loc+dates+'/'+'calibs/bias2x2.fits')[0].data
        dark=fits.open(pt5m_loc+dates+'/'+'calibs/dark2x2.fits')[0].data
        try:
            flat=fits.open(pt5m_loc+dates+'/'+'calibs/'+filter+'flatnorm2x2.fits')[0].data
            image=image-bias-dark-flat
        except:
            image=image-bias-dark
    except:
        pass
    return image

def checking_and_finding_sources(ref_img_data,radius,centre_x, centre_y):
    sources_coordinates=[]
    mean, median, std = sigma_clipped_stats(ref_img_data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=(std))
    sources = daofind(ref_img_data - median)
    sources_ref=sources[np.sqrt((sources['xcentroid']-centre_x)**2+(sources['ycentroid']-centre_y)**2)<radius]
    #sources_ref,FWHM=PSF(sources_ref,ref_img_data)
    FWHM=5
    sources_ref=clean_up(sources_ref,FWHM)
    for i in sources_ref:
         sources_coordinates.append((i['xcentroid'],i['ycentroid']))
    return sources_ref,sources_coordinates,FWHM



def gaia_query(ra_deg, dec_deg, rad_deg, maxmag=20,
               maxsources=10000):
    """
    Query Gaia DR1 @ VizieR using astroquery.vizier
    :param ra_deg: RA in degrees
    :param dec_deg: Declination in degrees
    :param rad_deg: field radius in degrees
    :param maxmag: upper limit G magnitude (optional)
    :param maxsources: maximum number of sources
    :return: astropy.table object
    """
    vquery = Vizier(columns=['Source', 'RA_ICRS', 'DE_ICRS',
                             'phot_g_mean_mag'],
                    column_filters={"phot_g_mean_mag":
                                    ("<%f" % maxmag)},
                    row_limit = maxsources)

    field = coord.SkyCoord(ra=ra_deg, dec=dec_deg,
                           unit=(u.deg, u.deg),
                           frame='icrs')
    return vquery.query_region(field,
                               width=("%fd" % rad_deg),
                               catalog="I/337/gaia")[0]

def w_c_s(files_loc,object_name,ref_date,object_logs):
    ref_image_file=files_loc+ref_date+'/'+np.array(object_logs[ref_date][0])[0]+'.fits'
    ast = AstrometryNet()
    ast.api_key = 'vbzlospxxqaentba'
    wcs = ast.solve_from_image(ref_image_file)                        # works even if the job fails
    return wcs

def coordinate_transform(sources_ref,wcs):
    x,y=sources_ref['xcentroid'],sources_ref['ycentroid']
    w= WCS(wcs)
    sky=w.pixel_to_world(x,y)
    return sky

def source_matching(trans_image_arrays, sources_ref):
    match_sources={}
    obs_list=[]
    locals().update(trans_image_arrays)
    for i in sources_ref:
        match=[]
        for d in trans_image_arrays:
            t=locals()[d]
            diff=np.min(np.sqrt((t['xcenter'].value-i['xcentroid'])**2+((t['ycenter']).value-i['ycentroid'])**2))<5
            if diff == True:
                #t['real_mag']=i['real_mag']
                m=t[np.sqrt((t['xcenter'].value-i['xcentroid'])**2+((t['ycenter']).value-i['ycentroid'])**2)==np.min(np.sqrt((t['xcenter'].value-i['xcentroid'])**2+((t['ycenter']).value-i['ycentroid'])**2))]
                time=Time(d)
                jd=time.jd
                obs_list.append(jd)
                m['time']=jd
                match.append(m)
        match_sources['match_to_'+str(i['id'])]=match
    w={}
    locals().update(match_sources)
    for i in match_sources:
        d=locals()[i]
        if len(d) > 1:
            w['final_'+i]=d
    locals().update(w)
    return w,obs_list

def stables(matching_sources,image_sources,stable_limit):
    locals().update(image_sources)
    offset_list=[]
    std=[]
    record=[]
    for i in image_sources:
        time=Time(i)
        jd=time.jd
        record.append(jd)
    counter= 15
    while counter!=0:
        locals().update(matching_sources)
        mag=[]
        median_list=[]
        for i in matching_sources:
            d=locals()[i]
            for t in d:
                mag.append(t['aper_mag'])
            median_list.append(np.median(mag))
        length=len(matching_sources)
        offset_list=[]
        std=[]
        time=[]
        for m in record:
            mag=[]
            count=0
            for i in matching_sources:
                d=locals()[i]
                for t in d:
                    if m==t['time']:
                        mag.append(t['aper_mag']-median_list[count])
                count=count+1
            offset_list.append(np.median(mag))
            std.append(np.std(mag))
            time.append(m)
        upper_limit=np.array(offset_list)+(stable_limit*np.array(std))
        lower_limit=np.array(offset_list)-(stable_limit*np.array(std))
        #plt.plot(time,upper_limit)
        #plt.plot(time,lower_limit)
        #plt.xlabel('Time (JD)')
        #plt.ylabel('Magnitude offset')
        for i in list(matching_sources):
            mag=[]
            upper=[]
            lower=[]
            r=[]
            for t in matching_sources[i]:
                mag.append(float(t['aper_mag']))
                upper.append(upper_limit[time.index(t['time'])])
                lower.append(lower_limit[time.index(t['time'])])
                r.append(t['time'])
            offset=mag-np.median(mag)
            #plt.plot(r,offset)
            test=[]
            k=np.array(upper-offset)
            for g in k:
                if g < 0:
                    test.append(0)
                else:
                    test.append(1)
            k=np.array(offset-lower)
            for g in k:
                if g < 0:
                    test.append(0)
                else:
                    test.append(1)
            if (0 in test) == True:
                del(matching_sources[i])
        #plt.show()
        if len(matching_sources) <11:
            counter=1
        print(len(matching_sources))
        if len(matching_sources)==length:
            counter=1
        counter=counter-1
        #plt.show()
    locals().update(matching_sources)
    return matching_sources

def ap_less_phot(ref_img_array,sources_ref,exposure_time,FWHM):
    data=ref_img_array
    positions=sources_ref
    aperture = CircularAperture(positions, r=FWHM*4)
    annulus_aperture = CircularAnnulus(positions, r_in=5*FWHM, r_out=(5*FWHM)+10)
    annulus_masks = annulus_aperture.to_mask(method='center')
    bkg_median = []
    try:
        for mask in annulus_masks:
            annulus_data = mask.multiply(data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
    except:
        annulus_data = annulus_masks.multiply(data)
        annulus_data_1d = annulus_data[annulus_masks.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)
    phot = aperture_photometry(data, aperture)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * aperture.area
    phot['aper_sum_bkgsub'] = (phot['aperture_sum'] - phot['aper_bkg'])/exposure_time
    phot['aper_mag']=-2.5 * np.log10(phot['aper_sum_bkgsub'])
    phot['aper_sum_bkgsub_err']=np.sqrt(phot['aper_sum_bkgsub'])
    phot['aper_mag_err']=abs(-2.5*(1/(np.log(10)*phot['aper_sum_bkgsub']))*(np.sqrt(phot['aper_sum_bkgsub'])))
    return phot

def magnitude_matcher(sky_coords,gaia_list,sources_ref):
    x2=gaia_list['RA_ICRS']
    y2=gaia_list['DE_ICRS']
    l=[]
    for i in sky_coords:
        l.append(gaia_list[(np.sqrt((i.ra-x2)**2+(i.dec-y2)**2))==min(np.sqrt((i.ra-x2)**2+(i.dec-y2)**2))])
    real_mag=[]
    for i in l:
        real_mag.append(i[0]['__Gmag_'])
    sources_ref['real_mag']=real_mag
    return sources_ref

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def PSF(sources_ref,ref_image_array):
    check=[]
    FWHM_list=[]
    for i in sources_ref:
        torf=False
        try:
            x=list(range(int(np.round((sources_ref[sources_ref['id']==i['id']]['xcentroid'])-20)),int(np.round((sources_ref[sources_ref['id']==i['id']]['xcentroid'])+20))))
            y=int(np.round(sources_ref[sources_ref['id']==i['id']]['ycentroid']))
            popt,pcov=curve_fit(gaus,np.arange(1,len(ref_image_array[y,x])+1),ref_image_array[y,x])
            torf=(abs(popt[2])*2.355) < 10
            if (abs(popt[2])*2.355) < 10:
                FWHM_list.append(abs(popt[2])*2.355)
        except:
            pass
        check.append(torf)
    print(check)
    sources_ref=sources_ref[check]
    FWHM=np.median(FWHM_list)
    return sources_ref,FWHM

def object_finder(wcs,pos):
    w=WCS(wcs)
    coords=pos
    c = SkyCoord(coords, frame=ICRS, unit=(u.hourangle, u.deg))
    pos=w.world_to_pixel(c)
    return pos

def object_records(object_pos,sources):
    value=np.min(np.sqrt((sources['xcentroid']-object_pos[0])**2+((sources['ycentroid']-object_pos[1])**2)))
    match=sources[np.sqrt((sources['xcentroid']-object_pos[0])**2+((sources['ycentroid']-object_pos[1])**2))==value]
    return match

def clean_up(array,FWHM):
    egg=[]
    for i in array:
        for element in array:
            t=0<abs(i['xcentroid']-element['xcentroid']) <4*FWHM and 0<abs(i['ycentroid']-element['ycentroid']) <4*FWHM
            if t == True:
                egg.append(element)
    for i in egg:
        torf=array['xcentroid']!=i['xcentroid']
        array=array[torf]
    return array

def function(x,fract_sin):
    return ((1/x)*((0.49*x**(2/3))/((0.6*x**(2/3))+np.log(1+(x**(1/3)))))**(3))-fract_sin

def sine(x,a,b,e):
    return (a*np.cos(4*np.pi*x+b)+e)

def sine_1(x,a,b,c):
    return(a*np.cos(2*np.pi*x)+b*np.sin(2*np.pi*x)+c)

def sine_2(x,a,b,c,d,e):
    return(a*np.cos(2*np.pi*x)+b*np.sin(2*np.pi*x)+c*np.sin(4*np.pi*x)+d*np.cos(4*np.pi*x)+e)

def sine_3(x,a,b,c,d,e):
    return(a*np.cos(2*np.pi*x)+b*np.sin(2*np.pi*x)+c*np.sin(4*np.pi*x)+d*np.cos(4*np.pi*x)+e)

def sine_4(x,a,b,c,d,e,f,g):
    return(a*np.cos(2*np.pi*x)+b*np.sin(2*np.pi*x)+c*np.sin(4*np.pi*x)+d*np.cos(4*np.pi*x)+e*np.cos(6*np.pi*x)+f*np.sin(6*np.pi*x)+g)

def sine_5(x,a,b,c,d,e,f,g,h,i):
    return(a*np.cos(2*np.pi*x)+b*np.sin(2*np.pi*x)+c*np.cos(4*np.pi*x)+d*np.sin(4*np.pi*x)+e*np.sin(6*np.pi*x)+f*np.sin(6*np.pi*x)+g*np.sin(8*np.pi*x)+h*np.sin(8*np.pi*x)+i)

def light_curve(stable,object, PEpoch, Porb,inclination_angle):
    locals().update(stable)
    flux=[]
    mag=[]
    time=[]
    time_2=[]
    mag_error=[]
    flux_error=[]
    time_unedited=[]
    for i in object['table']:
        diff=[]
        diff_2=[]
        mag_err=[]
        flux_err=[]
        for m in stable:
            d=locals()[m]
            for t in d:
                if i['time']==t['time']:
                    diff.append((i['aper_mag']-t['aper_mag']))#+t['real_mag'])
                    mag_err.append(np.sqrt(i['aper_mag_err']**2+t['aper_mag_err']**2))
                    diff_2.append(i['aper_sum_bkgsub']-t['aper_sum_bkgsub'])#+60*(10**(t['real_mag']/(-2.5))))
                    flux_err.append(np.sqrt(i['aper_sum_bkgsub_err']**2+t['aper_sum_bkgsub_err']**2))
        if mt.isnan(np.std(diff)) == False:
            mag.append(np.median(diff))
            flux.append(np.median(diff_2))
            flux_error.append(np.std(flux_err, ddof=1) / np.sqrt(np.size(flux_err)))
            mag_error.append(np.std(mag_err, ddof=1) / np.sqrt(np.size(mag_err)))
            time.append((((float(i['time'])- 2400000.5)-PEpoch)/Porb)%1)
            time_2.append((((float(i['time'])- 2400000.5)-PEpoch)/Porb)%1)
            time_unedited.append((float(i['time'])- 2400000.5)-PEpoch)
    flare_count=10
    plt.plot(time,flux,'x')
    plt.show()
    while flare_count!=0:
        std=np.std(flux)
        popt,pcov= curve_fit(sine,time,flux)
        count=0
        y_max=[]
        y_min=[]
        x=[]
        for i in time:
            if sine(i,*popt)+std < flux[count] or flux[count] <sine(i,*popt)-std:
                flux.remove(flux[count])
                time.remove(i)
            count=count+1
        flare_count=flare_count-1
    ti_me=np.linspace(0,1,num=10000)
    popt,pcov= curve_fit(sine_5,time,flux)
    mean_a=np.mean(sine_5(ti_me,*popt))
    flux=flux/np.mean(flux)
    plt.plot(time,flux,'x')
    #popt,pcov= curve_fit(sine_1,time,flux)
    #plt.plot(ti_me,sine_1(ti_me,*popt))
    #popt,pcov= curve_fit(sine_2,time,flux)
    #plt.plot(ti_me,sine_2(ti_me,*popt))
    #popt,pcov= curve_fit(sine_3,time,flux)
    #plt.plot(ti_me,sine_3(ti_me,*popt))
    #popt,pcov= curve_fit(sine_4,time,flux)
    #plt.plot(ti_me,sine_4(ti_me,*popt))
    popt,pcov= curve_fit(sine_5,time,flux)
    plt.plot(ti_me,sine_5(ti_me,*popt))
    plt.ylabel('Flux (Photon count)')
    plt.xlabel('Orbital phase')
    plt.show()
    ti_me=np.linspace(0.25,0.75,num=10000)
    flux_sec=min(sine_5(ti_me,*popt))
    ti_me=np.linspace(-0.25,0.25,num=10000)
    flux_prim=min(sine_5(ti_me,*popt))
    print(flux_sec/flux_prim)
    ti_me=np.linspace(0,1,num=10000)
    A_2_max=max(sine_5(ti_me,*popt))
    A_2_min=min(sine_5(ti_me,*popt))
    T_diff=(flux_sec/flux_prim)**(0.25)
    A_2=(A_2_max-A_2_min)/(2*np.mean(sine_5(ti_me,*popt)))
    print(A_2)
    peaks_2 = (find_peaks(sine_5(np.linspace(0,1,num=10000),*popt)))
    phi_2=ti_me[peaks_2[0][1]]
    print(phi_2)
    fract_sin=(A_2*-np.cos(((phi_2-0.5)/0.5)*2*np.pi))
    plt.plot(time_2,mag,'x')
    plt.show()
    time_2=np.array(time_2)
    time_unedited=np.array(time_unedited)
    time_2=time_2.tolist()
    time_unedited=time_unedited.tolist()
    flare_count=5
    while flare_count!=0:
        std=np.std(mag)
        popt,pcov= curve_fit(sine,time_2,mag)
        count=0
        y_max=[]
        y_min=[]
        x=[]
        for i in time_2:
            if sine(i,*popt)+std < mag[count] or mag[count] <sine(i,*popt)-std:
                mag.remove(mag[count])
                time_2.remove(i)
                time_unedited.remove(time_unedited[count])
                mag_error.remove(mag_error[count])
            count=count+1
        flare_count=flare_count-1
    plt.plot(time_2,mag,'x')
    popt,pcov= curve_fit(sine_5,time_2,mag)
    ti_me=np.linspace(0,1,num=10000)
    plt.plot(ti_me,sine_5(ti_me,*popt))
    popt,pcov= curve_fit(sine_1,time_2,mag)
    plt.plot(ti_me,sine_1(ti_me,*popt))
    popt,pcov= curve_fit(sine_2,time_2,mag)
    plt.plot(ti_me,sine_2(ti_me,*popt))
    popt,pcov= curve_fit(sine_3,time_2,mag)
    plt.plot(ti_me,sine_3(ti_me,*popt))
    popt,pcov= curve_fit(sine_4,time_2,mag)
    plt.plot(ti_me,sine_4(ti_me,*popt))
    plt.ylabel('Mag')
    plt.xlabel('Orbital phase')
    plt.gca().invert_yaxis()
    plt.show()
    popt,pcov= curve_fit(sine_5,time_unedited,mag)
    ti_me=np.linspace(time_unedited[0],time_unedited[len(time_unedited)-1],num=10000)
    frequency, power = LombScargle(time_unedited, mag).autopower(minimum_frequency=0.1,maximum_frequency=10)
    peaks = find_peaks(power, height = 0.1)
    peak_pos = frequency[peaks[0]]
    height = peaks[1]['peak_heights']
    plt.plot(frequency, power)
    plt.scatter(peak_pos, height, color = 'r', s = 15, marker = 'D', label = 'Maxima')
    plt.xlabel('Frequency $d^-1$')
    plt.ylabel('Power')
    plt.show()
    max_height = np.max(height)
    counter=0
    for i in height:
        if i != max_height:
            counter=counter+1
        if i == max_height:
            print(counter)
            phot_frequency=(peaks[0])[counter]
    phot_freq=(frequency[phot_frequency])
    x=np.linspace(0.000001,2,num=1000)
    count=0
    for i in function(x,fract_sin):
        if abs(i)!=min(abs(function(x,fract_sin))):
            count=count+1
        if abs(i)==min(abs(function(x,fract_sin))):
                q=(x[count])
    print('Period of orbit: {} d'.format((1/(phot_freq))*2))
    print('Fraction amplitude A2:{}'.format(fract_sin))
    print('The mass ratio q: {}'.format(q))
    print('The ratio of the night side temperature to the day side is: {}'.format(T_diff))
    return peak_pos[counter]