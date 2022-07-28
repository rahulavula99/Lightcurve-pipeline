# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:27:25 2022

@author: rahul
"""

from astropy.io import fits
import Pipeline_code as cu
from datetime import timedelta, date
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import numpy as np


#variable section
start_dt = date(2016, 4, 5)
end_dt = date(2022,6 , 1)
pt5m_loc='/scratch/nas_spiders/pt5m/'
filter='R'


object_name='3FGL J0212+5320'
#object_name='PSR J1023+0038'
#object_name='PSR J2129-0429'

object_pos=['02 12 10.4774318712 +53 21 38.778250464']#3std,1 std above med, 1 ref
#object_pos=['10 23 47.68720 +00 38 40.8455']#3.5std, 1 std above med, 15 ref
#object_pos=['21 29 45.039 -04 29 05.59 '] #3 std, 1 std above med, 8 ref

PEpoch= 57408.539
#PEpoch= 55000
#PEpoch= 55196

Porb= 0.86955
#Porb=0.1980963569
#Porb=0.6352274131

inclination_angle=90

#file log section

dates=cu.dates(start_dt,end_dt,pt5m_loc)
logs,date_list=cu.object_log_table(pt5m_loc,object_name,dates)
ref_date=date_list[15]
print(logs)


#real position finder
stacked_image=cu.reference_image(pt5m_loc,date_list,logs,'R',ref_date)
wcs=cu.w_c_s(pt5m_loc,object_name,ref_date,logs)
object_pos=cu.object_finder(wcs,object_pos)
norm = simple_norm(stacked_image, 'sqrt', percent=99)
plt.imshow(stacked_image, norm=norm,cmap='Greys', interpolation='nearest',origin='lower')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.plot(object_pos[0],object_pos[1],'x',c='blue')
plt.show()
radius = int(input("Extraction Radius: "))
sources,source_coords,FWHM=cu.checking_and_finding_sources(stacked_image,object_pos[0],object_pos[1], radius)
print(sources)
sky_coords=cu.coordinate_transform(sources,wcs)
t=cu.gaia_query(wcs['CRVAL1'],wcs['CRVAL2'],0.6)
sources=cu.magnitude_matcher(sky_coords,t,sources)


#image reader and matching
image_sources,object=cu.image_transform(pt5m_loc,date_list,logs,'R',object_pos[0],object_pos[1], radius,source_coords,object_pos,ref_date,FWHM)
matching_sources,obs_list=cu.source_matching(image_sources, sources)
matching_sources=cu.nan_removal(matching_sources)
locals().update(matching_sources)



#variable and stable sources seperator
stables_limit=3
comparison_stars=cu.stables(matching_sources,image_sources,stables_limit)
locals().update(comparison_stars)
for i in comparison_stars:
    d=locals()[i]
    flux=[]
    time=[]
    for t in d:
        flux.append(t['aper_mag'])
    plt.plot(flux)
plt.xlabel('Image number')
plt.ylabel('Instrumental mag')
plt.show()


#light curve plotting
light_curve_params=cu.light_curve(comparison_stars,object,PEpoch, Porb,inclination_angle)