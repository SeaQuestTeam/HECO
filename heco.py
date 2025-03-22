'''
This module contains the functions develped for the HECO Proof of Concept
author: Gianfranco Di Pietro ~ PhD student at University of Catania
contributors: Martina Stagnitti, Massimiliano Marino, Elisa Castro, Sofia Nasca
supervisor: Rosaria Ester Musumeci
'''

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcb
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import xarray as xr
import ssl
import ipywidgets as widgets
from IPython.display import display
import pyproj
from glob import glob
from geopandas import GeoDataFrame
from shapely.geometry import Point, LineString
import yaml

import folium
from folium.plugins import TimestampedGeoJson
import json
import requests
from PIL import Image
from io import BytesIO
import geopandas as gpd
import math

def plot_simple_map (inputdata):
    '''
    Function that plots a simple map with the position of the oil spill
    @param inputdata: dictionary with the input data
    @return: plot with the map
    '''
    
    # open xarray

    LocalDS = xr.open_dataset(inputdata['dataset_file_name'])

    # Set the map projection
    proj = ccrs.Mercator() #pseudo mercator

    # convert lat,lon into x,y using pyproj function
    def latlon_to_mercator(lat, lon):
        crs_wgs = pyproj.CRS("EPSG:4326")
        crs_mercator = pyproj.CRS("EPSG:3857")
        project = pyproj.Transformer.from_crs(crs_wgs, crs_mercator, always_xy=True).transform
        x, y = project(lon, lat)
        return x, y

    # get lat,lon extent from LocalDS
    latmin = LocalDS.latitude.min()
    latmax = LocalDS.latitude.max()
    lonmin = LocalDS.longitude.min()
    lonmax = LocalDS.longitude.max()

    # close dataset
    LocalDS.close()

    # prevent SSL error
    ssl._create_default_https_context = ssl._create_unverified_context

    #retrive lat0,long0,time0 from inputdata
    lat0 = inputdata['lat0']
    lon0 = inputdata['lon0']
    time0 = inputdata['time0']

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([lonmin,lonmax,latmin,latmax]) #

    ax.gridlines(draw_labels=True)

    ax.add_image(cimgt.GoogleTiles(style='satellite'), 6)

    time0 = pd.to_datetime(time0).strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.5, 1, 'HECO: HEre Comes the Oil', ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.965, f'Origin of spill event:{time0} ', ha='center', va='center', fontsize=8)

    

    plt.scatter(x=lon0, y=lat0, c='red', s=10, transform=ccrs.PlateCarree())

    plt.show()
    return plt

def inputwidgets():
    '''
    Function that creates the widgets for the input of the oil spill simulation
    @return: dictionary with the input data
    '''

    # defining the widgets
    step1_descr = widgets.HTML(
        value="<h2>Origin inputs for the oil spill model</h2> \n"
        "<p>Copy the cmems_mod_med .nc file downloaded before in the same folder of this notebook.<br>\n"
        "Copy the <b>name of the file (with .nc extension)</b> and paste in the cell 'nc file name' below.</p>\n"
        )

    # lat lon widget input
    nc_file_path = widgets.Text(
        description='nc file path',
        value='HECO_TEST.nc',
        disabled=False,
    )

    step2_descrp = widgets.HTML(value="Insert the <b>latitude</b> and <b>longitude</b> of the point of interest.<br>\n")
    
    lat = widgets.FloatText(
        value=40.5,
        description='Latitude:',
        disabled=False
    )

    lon = widgets.FloatText(
        value=10.5,
        description='Longitude:',
        disabled=False
    )

    step3_descrp = widgets.HTML(value="Insert the <b>time</b> of the spill origin in format 'YYYY-MM-DDTHH:MM:SS'.<br>\n")

    timedate = widgets.NaiveDatetimePicker(
    value=pd.Timestamp('2025-03-08 00:00:00'),
    description='Time',
    disabled=False
    )

    display(
        step1_descr, nc_file_path, # nc_file_name and description HTML
        step2_descrp, lat,lon, # lat lon and description HTML
        step3_descrp, timedate, # time and description HTML
            )
    
    
    
    return nc_file_path, lat, lon, timedate


def check_lat_lon(lat,lon,nc_file_path):
    '''
    Function that checks if the latitude and longitude are inside the domain of the dataset
    @param lat: latitude of the point of interest
    @param lon: longitude of the point of interest
    @param nc_file_path: string path of Copernicus Marine Services file
    @return: True if the point is inside the domain, False otherwise
    '''
    LocalDS = xr.open_dataset(nc_file_path)

    latmin = LocalDS.latitude.min()
    latmax = LocalDS.latitude.max()
    lonmin = LocalDS.longitude.min()
    lonmax = LocalDS.longitude.max()
    LocalDS.close()
    if latmin <= lat <= latmax and lonmin <= lon <= lonmax:
        return True
    else:
        return False
    
def check_timedateformatcorrect(timedate, nc_file_path):
    '''
    Function that checks if the time format is correct
    @param timedate: time of the data in format "YYYY-MM-DDTHH:MM:SS"
    @return: True if the format is correct, False otherwise
    '''
    try:
        LocalDS = xr.open_dataset(nc_file_path)
        pd.to_datetime(timedate, format='%Y-%m-%dT%H:%M:%S')
        timemin = pd.to_datetime(LocalDS.time.min().values)
        timemax = pd.to_datetime(LocalDS.time.max().values)
        LocalDS.close()
        if timemin <= pd.to_datetime(timedate) <= timemax:
            return True
        else:
            return False
    except:
        return False
    

def check_vo_uo(nc_file_path):
    '''
    Function that checks if the dataset has the variables 'vo' and 'uo'
    @param LocalDS: dataset (xarray) from Copernicus Marine Services
    @return: True if the dataset has the variables, False otherwise
    '''
    LocalDS = xr.open_dataset(nc_file_path)
    if 'vo' in LocalDS and 'uo' in LocalDS:
        LocalDS.close()
        return True
        
    else:
        LocalDS.close()
        return False
        
    
    

def display_check_button(inputdata):
    '''
    Function that creates a button to check the validity of the input data
    @param nc_file_name: widget with the name of the dataset file
    @param lat: widget with the latitude of the point of interest
    @param lon: widget with the longitude of the point of interest
    @param timedate: widget with the time of the spill origin
    @return: button widget
    '''
    nc_file_path = inputdata['dataset_file_name']
    lat = inputdata['lat0']
    lon = inputdata['lon0']
    timedate = inputdata['time0']

    b = widgets.Button(
        description='Finally click here to check values',
        layout=widgets.Layout(width='50%', height='30px')
        )

    def on_checkbutton_clicked(b):
        '''
        Function that checks if the latitude and longitude are inside the domain of the dataset
        and plot a somple map
        @param b: button widget
        '''
        

        if check_lat_lon(lat,lon,nc_file_path) and check_timedateformatcorrect(timedate, nc_file_path) and check_vo_uo(nc_file_path):
            b.description = 'lat/lon, time are valid, dataset has vo and uo'
            b.style.button_color = 'lightgreen'
            plot_simple_map(inputdata)
            
        else:
            b.description = 'Not valid point/time or dataset'
            b.style.button_color = 'red'
    
    b.on_click(on_checkbutton_clicked)

    display(b)

    return


# Simulation setting Widget

def sim_setting_widgets(inputdata):

    # defining the widgets
    descr = widgets.HTML(
        value="<h2>Insert  simulation settings parameters</h2> \n"
        "<p>please read the documentation </p>"
        )
    volume_spilled_m3 = widgets.FloatText(
        value=1000,
        description='Vol.[m^3]',
        disabled=False
    )
    
    spill_release_duration_h = widgets.FloatText(
        value=6,
        description='Event [h]',
        disabled=False
    )
    sim_timedelta_s = widgets.FloatText(
        value=3600,
        description='Sim step[s]',
        disabled=False
    )
    sim_particles = widgets.FloatText(
        value=100,
        description='Particles',
        disabled=False
    )
    sim_diffusion_coeff = widgets.FloatText(
        value=10,
        description='Dh',
        disabled=False
    )
    sim_duration = widgets.FloatText(
        value=72,
        description='iterations',
        disabled=False
    )

    ###
    
    

    display(
        descr, 
        volume_spilled_m3,

        widgets.HTML(value="<p>Insert the duration of the spill event in hours</p>"),
        spill_release_duration_h,

        widgets.HTML(value="<p>Insert the simulation time-step [s] as the same of the CMEMS dataset</p>"),
        sim_timedelta_s, 
        
        widgets.HTML(value="<p>Insert the number of particles for the lagrangian simulation</p>"),
        sim_particles,

        widgets.HTML(value="<p>Insert the diffusion coefficient Dh</p>"),
        sim_diffusion_coeff, 

        widgets.HTML(value="<p>Insert the duration of the simulation in number of iteration</p>"),
        sim_duration, )

    
    return volume_spilled_m3, spill_release_duration_h, sim_timedelta_s, sim_particles, sim_diffusion_coeff, sim_duration

def check_sim_settings(inputdata, simsettings):

    out = widgets.Output(
        layout={'border': '1px solid black'},
    )


    b = widgets.Button(
        description='Click here to check values',
        layout=widgets.Layout(width='50%', height='30px'),
        disabled=False,
        tooltip='Click here to check values',
        icon='check')
    
    
    def get_time_delta_DS(inputdata):
        '''
        Function that returns the time delta of the dataset
        @param LocalDS: dataset (xarray) from Copernicus Marine Services
        @return: time delta in hours
        '''
        ds = xr.open_dataset(inputdata['dataset_file_name'])
        time0 = ds.time[0].values
        time1 = ds.time[1].values
        ds.close()

        return ((time1-time0).astype('timedelta64[h]'))
        

    

    def on_button_clicked(b):
        '''
        Function that reads the values of the widgets and print them in output
        @param b: button widget
        '''
        #global simsettings
        
        with out:
            out.clear_output()
            print("The values of the simulation settings are:")
            print(f"Volume of spilled fluid: {simsettings['volume_spilled_m3']}", "[m^3]")
            print(f"Duration of the spill event: {simsettings['spill_release_duration_h']}", "[hours]")
            print(f"Time step of the simulation: {simsettings['sim_timedelta_s']}", "[s]")
            print(f"Time delta of the dataset: {get_time_delta_DS(inputdata)}", "[hours]")
            print(f"Number of particles: {simsettings['sim_particles']}")
            print(f"Diffusion coefficient: {simsettings['sim_diffusion_coeff']}")
            print(f"Duration of the simulation: {simsettings['sim_duration_h']}", "iterations of time step")
            
        return 
    b.on_click(on_button_clicked)
    display(b, out)

    return

def save_yaml(inputdata,simsettings, yaml_file_path):
    '''
    Function that saves the input data and simulation settings in a yaml file
    @param inputdata: dictionary with the input data
    @param simsettings: dictionary with the simulation settings
    @param yaml_file_path: path to the yaml file
    ---
    Usage: save_yaml(inputdata,simsettings, 'input.yaml')
    '''
    # merge the two dictionaries in one
    data = inputdata.copy()
    data.update(simsettings)

    # fix time format for time0 value
    data['time0'] = pd.to_datetime(data['time0']).strftime('%Y-%m-%d %H:%M:%S')

    yamlfiledata = {'input': data}

    with open(yaml_file_path, 'w') as f:
        yaml.dump(yamlfiledata, f)
    print(f"Data saved in {yaml_file_path}")
    return

def open_yaml(yaml_file):
    ''' Function that opens a yaml file and check the data
    @param yaml_file: path to the yaml file
    @return: data from the yaml file
    '''
    
    
    with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
    inputdata = data['input']
    # convert timestamp to datetime 
    inputdata['time0'] = pd.to_datetime(inputdata['time0']) # some datetime format are not supported in yaml


    DS = xr.open_dataset(inputdata['dataset_file_name'])
    print(f"Dataset {inputdata['dataset_file_name']} opened")

    # check lat lon
    latmin = DS.latitude.min()
    latmax = DS.latitude.max()
    lonmin = DS.longitude.min()
    lonmax = DS.longitude.max()
    if latmin <= inputdata['lat0'] <= latmax and lonmin <= inputdata['lon0'] <= lonmax:
           pass #print(f"Latitude and longitude of origin spill are IN the dataset domain")
    else:
            print(f"WARNING: Latitude and longitude of origin spill are OUT of the dataset domain")

    #check time0
    pd.to_datetime(inputdata['time0'], format='%Y-%m-%dT%H:%M:%S')
    timemin = pd.to_datetime(DS.time.min().values)
    timemax = pd.to_datetime(DS.time.max().values)
    if timemin <= pd.to_datetime(inputdata['time0']) <= timemax:
        pass #print (f"Time of origin spill is IN the dataset domain")
    else: print(f"WARNING: Time of origin spill is OUT of the dataset domain")

    if 'vo' in DS and 'uo' in DS: 
        pass #print("Found 'v_o' and 'u_o' variables in dataset") 
    else: 
        print("WARNING: vo and wo varaibles not found in dataset")

    return inputdata, DS

def lagrangian_iteration(DS, lat,lon,timedate, D,dt):
        '''
        Function that simulates the oil spill particle for one time step
        @param dataset: dataset (xarray) from Copernicus Marine Services
        @param lat: latitude of the point of interest
        @param lon: longitude of the point of interest
        @param timedate: time of the data in format "YYYY-MM-DDTHH:MM:SS"
        @param D: diffusion coefficient (adimensional)
        @param dt: time step in seconds
        ---
        @return: New position of the particle
        '''
        

        
        # prepare conversion methods using pyproj

        crs_wgs = pyproj.CRS("EPSG:4326")
        crs_mercator = pyproj.CRS("EPSG:3857")
        from_lonlat_to_xy = pyproj.Transformer.from_crs(crs_wgs, crs_mercator, always_xy=True).transform
        from_xy_to_lonlat = pyproj.Transformer.from_crs(crs_mercator, crs_wgs, always_xy=True).transform

        # retrive x,y value from lat,lon
        x, y = from_lonlat_to_xy(lon, lat)
        
        # retrive wave velocity from dataset
        u = DS.uo.sel(latitude=lat, longitude=lon, time=timedate, method='nearest')
        v = DS.vo.sel(latitude=lat, longitude=lon, time=timedate, method='nearest')
        # if u is NaN value
        
        if u is None or v is None:
            print(f"WARNING: No data available for wave velocity at {lat}, {lon} - {timedate}")
            u = 0
            v = 0
            pass
        
        # Compute the new position of the oil spill
        x_new = x + u * dt + np.random.normal(0,1)* np.sqrt(2 * D * dt)
        y_new = y + v * dt + np.random.normal(0,1)* np.sqrt(2 * D * dt)

        # retrive distance value from x,y
        #dist = (np.sqrt((x_new - x)**2 + (y_new - y)**2)).values
        
        # Convert the new position to latitude and longitude

        lon_new, lat_new = from_xy_to_lonlat(x_new, y_new)
        position = Point(lon_new, lat_new)
        return position

def get_sim_info(inputdata):
    ''' Function that retrives the simulation info from the input data
    @param inputdata: dictionary with the input data
    @return: time step, discrete spill steps, volume per particle, number of particles for each spill step
    '''
     
    dt = pd.Timedelta(inputdata['sim_timedelta_s'], unit='s')
  
    discrete_spill_steps = (inputdata['spill_release_duration_h'] * 3600)/dt.total_seconds()
    if discrete_spill_steps<1:
        print("WARNING: The duration of the spill is less than the time step")
        discrete_spill_steps = 1
    else:
        discrete_spill_steps = int(discrete_spill_steps)
    
    volume_per_particle = inputdata['volume_spilled_m3'] / inputdata['sim_particles']

    #how many particles for each spill step
    num_part_i = (inputdata['sim_particles']/discrete_spill_steps)
    if num_part_i < 1:
        print("WARNING: The number of particles is less than the number of spill steps")
        num_part_i = 1
    else:
        num_part_i = int(num_part_i)

    return  dt, discrete_spill_steps, volume_per_particle, num_part_i


def multiple_spill_release_sim (yaml_file):
    '''
    Function that simulates the oil spill for multiple release steps
    usage: output = multiple_spill_release_sim('input.yaml')
    @param yaml_file: path to the yaml file
    @return: dataframe with the output of the simulation

    '''

    def single_spill_step(release_step, sim_duration_h, num_part_i, DS, lat0, lon0, time0, D, dt):
        '''
        Function that simulates the oil spill for one release step.
        usage: output = single_spill_step(release_step, sim_duration_h, num_part_i, DS, lat0, lon0, time0, D, dt)
        @param release_step: number of the release step
        @param sim_duration_h: duration of the simulation in hours
        @param num_part_i: number of particles for each spill step
        @param DS: dataset (xarray) from Copernicus Marine Services
        @param lat0: latitude of the point of interest
        @param lon0: longitude of the point of interest
        @param time0: time of the data in format "YYYY-MM-DDTHH:MM:SS"
        @param D: diffusion coefficient (adimensional)
        @param dt: time step in seconds
        ---
        @return: dataframe with the output of the simulation

        '''
    
        latitude = lat0
        longitude = lon0
        time = time0
        for i in range(num_part_i): #for all particles
            for j in range(sim_duration_h): 
                #print(latitude, longitude, time)
                position = lagrangian_iteration(DS, latitude, longitude, time, D, dt.total_seconds())
                output.loc[len(output)] = [release_step, j, i, time, position.y, position.x]
                latitude = position.y
                longitude = position.x
                time = time + dt
            #reset for next particle iteration
            latitude = lat0
            longitude = lon0
            time = time0
        return output

    # open DS and retrive info from yaml
    inputdata, DS = open_yaml(yaml_file)

    dt, discrete_spill_steps, volume_per_particle, num_part_i = get_sim_info(inputdata)
    
    print(f"Volume per particle considered: {volume_per_particle} m3")

    #Prepare output dataframe
    columns = ['release_step','sim_iteration', 'particle_id', 'time', 'lat', 'lon']
    output = pd.DataFrame(columns=columns)

    sim_duration_h = inputdata['sim_duration_h']
    
    # iteration across multiple istantaneous spills
    for i in range(discrete_spill_steps):
        release_step = i
        time_i = pd.to_datetime(inputdata['time0']) + i*dt
        lat0 = inputdata['lat0']
        lon0 = inputdata['lon0']
        residual_sim_time_h = int(pd.Timedelta(sim_duration_h*3600 -  i*dt.total_seconds(), unit='s').total_seconds()/3600)
        output = single_spill_step(release_step, residual_sim_time_h, num_part_i, DS, lat0, lon0, time_i, inputdata['sim_diffusion_coeff'], dt)
        print(f"discrete spill step {i} , release time {time_i}")

    return output

def output_points_toconvexhull_polygons(gdf):
    ''' Function that converts points to convex hull polygons
    @param gdf: GeoDataFrame with the data to be converted
    @return: GeoDataFrame with the convex hull polygons
    '''
    gdf['time'] = gdf['time'].astype(str)
    points_time = gdf.dissolve(by='time', as_index=True)
    convex_hull = points_time.convex_hull
    return convex_hull

def create_webmap(HECOpoint_output_gdf_path, EMODnetLayers = True, settingsFile_path = 'settings.yaml', output_path = 'heco_map.html', savepolygons = True):
    ''' Function that creates a webmap using Folium
    @param gdf: GeoDataFrame with the data to be displayed
    @param EMODnetLayers: boolean to decide if EMODnet layers should be added to the map
    @param settingsFile: path to the yaml file with the settings used for simulation
    @return: path to the html file with the map
    '''
    # from yaml file extract html table with data
    with open(settingsFile_path, 'r') as f:
        data = yaml.safe_load(f)
    inputdata = data['input']
    
    # open GeoDataFrame
    gdf = gpd.read_file(HECOpoint_output_gdf_path) # must contain Points from HECO simulation

    # convert points to convex hull polygons
    convex_hull = output_points_toconvexhull_polygons(gdf)

    
    datajson = convex_hull.to_json()

   

    # PREPARE MAP

    # get gdf extent
    bounds = gdf.bounds

    #retrive max lat and max lon from bounds
    max_lat = bounds.maxy.max()
    max_lon = bounds.maxx.max()
    #retrieve min lat and min lon from bounds
    min_lat = bounds.miny.min()
    min_lon = bounds.minx.min()

    # get distance between max and min lat and lon
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon

    #create map
    location = [min_lat + lat_diff/2, min_lon + lon_diff/2]

    # get map bounds
    mapbounds = [[min_lat - lat_diff, min_lon - lon_diff], [max_lat + lat_diff, max_lon + lon_diff]]

    # calculate zoom level from lat and lon diff
    def calculate_zoom_level(max_diff):
        zoom = math.ceil(8 - math.log2(max_diff))
        return max(2, min(18, zoom))
    zoom_start=calculate_zoom_level(max(lat_diff*4, lon_diff*4))

    m = folium.Map(location=location, 
                   zoom_start=zoom_start,
                   tiles='openstreetmap',
                   max_bounds=True, 
                   max_bounds_visbility=True, 
                   bounds=mapbounds, 
                   control_scale=True)
    folium.TileLayer('Esri.WorldImagery').add_to(m)

 


    if EMODnetLayers == True:
        # import WMS route density
        folium.raster_layers.WmsTileLayer(url = 'https://ows.emodnet-humanactivities.eu/wms?',
                                        layers = 'routedensity_all',
                                        transparent = True, 
                                        control = True,
                                        fmt="image/png",
                                        name = 'EmodNet - Route Density (EMSA) Monthly Totals - All types',
                                        overlay = True,
                                        show = False,
                                        ).add_to(m)

        # import WMS of vesseldensity_all

        folium.raster_layers.WmsTileLayer(url = 'https://ows.emodnet-humanactivities.eu/wms?',
                                        layers = 'vesseldensity_all',
                                        transparent = True, 
                                        control = True,
                                        fmt="image/png",
                                        name = 'EmodNet - Vessel Density Monthly totals - All types',
                                        overlay = True,
                                        show = True,
                                        ).add_to(m)
        
        
        # import WMS of natura2000 areas

        folium.raster_layers.WmsTileLayer(url = 'https://ows.emodnet-humanactivities.eu/wms?',
                                        layers = 'natura2000areas',
                                        transparent = True, 
                                        control = True,
                                        fmt="image/png",
                                        name = 'EmodNet - EU-Natura 2000 areas',
                                        overlay = True,
                                        show = True,
                                        queryable = True
                                        ).add_to(m)
        
        
        # import WMS of bathing waters areas

        folium.raster_layers.WmsTileLayer(url = 'https://ows.emodnet-humanactivities.eu/wms?',
                                        layers = 'bathingwaters',
                                        transparent = True, 
                                        control = True,
                                        fmt="image/png",
                                        name = 'EmodNet - Bathing waters',
                                        overlay = True,
                                        show = False,
                                        ).add_to(m)
        
        # import WMS of oil-gas platforms waters areas

        folium.raster_layers.WmsTileLayer(url = 'https://ows.emodnet-humanactivities.eu/wms?',
                                        layers = 'platforms',
                                        transparent = True, 
                                        control = True,
                                        fmt="image/png",
                                        name = 'EmodNet - Oil&Gas Offshore platforms',
                                        overlay = True,
                                        show = False,
                                        ).add_to(m)



        

        #folium.raster_layers.WmsTileLayer(url ='https://wmts.marine.copernicus.eu/teroWmts/MEDSEA_ANALYSISFORECAST_PHY_006_013/cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m_202411?request=GetCapabilities&service=WMS').add_to(m)
    else:
        pass

    # ADD TimeStampedGeojson from HECO to the map

    # prepare feature style

    datajson = json.loads(datajson)
    for feature in datajson['features']:
        feature['properties']['time'] = feature['id']
        feature['properties']['style'] = {'fillColor': 'black',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
            'radius': 5}


    TimestampedGeoJson(
        data=datajson,
        transition_time=50,
        period='PT1H',
        duration='PT1H',
        add_last_point=True,
        auto_play=True,
        loop=True,
        max_speed=10,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True
    ).add_to(m)

    # add SVG pin point on map

    sim_parameter_html_table = f'''<table><small>
                        <tr><td><b>Simulation settings</b></td></tr>
                        
                        <tr><td>Spill time: {inputdata['time0']}</td></tr>
                        <tr><td>Position ~ lat:{round(inputdata['lat0'],3)}</td><td>lon: {round(inputdata['lon0'],3)}</td></tr>
                        <tr><td>Dh constant: {inputdata['sim_diffusion_coeff']}</td></tr>
                        <tr><td>Spill duration: {inputdata['spill_release_duration_h']} [h]</td></tr>
                        <tr><td>Volume spilled: {inputdata['volume_spilled_m3']} [m^3]</td></tr>
                        </table></small>
                        </div>'''


    svg = '''<div>
            <svg viewBox="0 0 72 72" id="emoji" xmlns="http://www.w3.org/2000/svg">
            <g id="color">
                <path fill="#3f3f3f" d="M51.5,48.5c0,3,6,4,11,5s6,3,0,4-4.7378,3.5887-9.8689,3.2944S31.5,60.5,33.5,57.5s13-2,10-4-11.0977-2.8961-6.5489-4.948L45,46Z"/>
                <path fill="#d0cfce" d="M50.3434,20.303a8.4825,8.4825,0,0,0-5.303,1.9623,8.4825,8.4825,0,0,0-5.303-1.9623,8.27,8.27,0,0,0-4.5455,1.4181,7.9938,7.9938,0,0,0-9.0909,0,8.2692,8.2692,0,0,0-4.5454-1.4181C15.7259,20.303,11,26.7474,11,34.697s4.7259,14.3939,10.5556,14.3939a8.2691,8.2691,0,0,0,4.5454-1.418,7.9943,7.9943,0,0,0,9.0909,0,8.27,8.27,0,0,0,4.5455,1.418,8.4819,8.4819,0,0,0,5.303-1.9623,8.4819,8.4819,0,0,0,5.303,1.9623c5.83,0,10.5556-6.4444,10.5556-14.3939S56.1731,20.303,50.3434,20.303Z"/>
                <ellipse cx="50.3939" cy="34.697" rx="7.5758" ry="11.3636" fill="#9b9b9a"/>
                <ellipse cx="51.1515" cy="39.2424" rx="2.2727" ry="3.7879" fill="#3f3f3f"/>
            </g>
            <g id="line">
                <path fill="none" stroke="#000000" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M32.92,38.551a20.2817,20.2817,0,0,1-.708-5.3692,18.1953,18.1953,0,0,1,2.8846-10.178,11.7367,11.7367,0,0,1,1.7055-1.9947"/>
                <path fill="none" stroke="#000000" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M23.8292,38.551a20.2776,20.2776,0,0,1-.708-5.3692,18.1953,18.1953,0,0,1,2.8846-10.178,11.7367,11.7367,0,0,1,1.7055-1.9947"/>
                <path fill="none" stroke="#000000" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M50.3434,20.303a8.4825,8.4825,0,0,0-5.303,1.9623,8.4825,8.4825,0,0,0-5.303-1.9623,8.27,8.27,0,0,0-4.5455,1.4181,7.9938,7.9938,0,0,0-9.0909,0,8.2692,8.2692,0,0,0-4.5454-1.4181C15.7259,20.303,11,26.7474,11,34.697s4.7259,14.3939,10.5556,14.3939a8.2691,8.2691,0,0,0,4.5454-1.418,7.9943,7.9943,0,0,0,9.0909,0,8.27,8.27,0,0,0,4.5455,1.418,8.4819,8.4819,0,0,0,5.303-1.9623,8.4819,8.4819,0,0,0,5.303,1.9623c5.83,0,10.5556-6.4444,10.5556-14.3939S56.1731,20.303,50.3434,20.303Z"/>
                <ellipse cx="50.3939" cy="34.697" rx="7.5758" ry="11.3636" fill="none" stroke="#000000" stroke-miterlimit="10" stroke-width="2"/>
                <ellipse cx="51.1515" cy="39.2424" rx="2.2727" ry="3.7879" fill="none" stroke="#000000" stroke-linecap="round" stroke-miterlimit="10" stroke-width="2"/>
                <path fill="none" stroke="#000000" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M36.9511,48.552C32.4023,50.6039,40.5,51.5,43.5,53.5s-8,1-10,4,14,3,19.1311,3.2944S56.5,58.5,62.5,57.5s5-3,0-4-11-2-11-5V43"/>
            </g>
            </svg></div>'''

    origin_coord = gdf.iloc[0].geometry
    folium.Marker(
            location=[origin_coord.y, origin_coord.x],
            icon=folium.DivIcon(html=svg, 
                                icon_size=(50, 50),
                                icon_anchor=(50,50)),
            popup=sim_parameter_html_table,
            tooltip='Spill Origin'
           
            ).add_to(m)


    



    # prepare htmlbox for map
    urllegend1 = 'https://ows.emodnet-humanactivities.eu/wms?REQUEST=GetLegendGraphic&FORMAT=image/png&LAYER=vesseldensity_all'

    #


    box_html = f'''
            <div style="position: fixed;
                        bottom: 80px; left: 50px; width: 150px; height: 280px;
                        background-color: white; border: 1px solid black;
                        z-index:9999; font-size:10px;
                        "> 
                        <h3>HECO</h3> 
                        <p>Simulation of oil spill impacs using EmodNEt data, vessel and route density</p>
                        <img src="{urllegend1}" width="50">
                        <p>Event: {inputdata['time0']}</p>
                        </div>
             '''
    m.get_root().html.add_child(folium.Element(box_html))


    # add layer control
    folium.LayerControl().add_to(m)

    m.save(output_path)


    # save convex hull polygons to GeoJSON
    if savepolygons == True:
        convex_hull.to_file("heco-polygons.geojson", driver='GeoJSON')
    else:
        pass
    
    return

def create_points_animation(geojson_HECOpoints_path, output_gif_path):
    ''' Function that creates an animation of the points in the geojson file'
    @param geojson_HECOpoints_path: path to the geojson file with the points
    @param output_gif_path: path to the output gif file

    '''
    # check output gif path contains .gif
    if not output_gif_path.endswith('.gif'):
        output_gif_path = output_gif_path + '.gif'

    
    fig, ax = plt.subplots()

    gdf = gpd.read_file(geojson_HECOpoints_path)
    gdf = gdf.sort_values(by='time')
    uniquetimes = gdf['time'].unique()

    bounds = gdf.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    subset = gdf[gdf['time'] == gdf['time'].min()]
    scat = ax.scatter(subset.lon, subset.lat, c='black', s=5)

    def animate(i):
        ax.clear()
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(f"Time: {gdf['time'].min() + pd.Timedelta(i, unit='h')}")
        subset = gdf[gdf['time'] == gdf['time'].min() + pd.Timedelta(i, unit='h')]
        scat = ax.scatter(subset.lon, subset.lat, c='black', s=5)
        return (scat,)

    ani = FuncAnimation(fig, animate, frames=len(uniquetimes), interval=100, blit=True, repeat=True)

    writer = PillowWriter(fps=15,
                            metadata=dict(artist='Me'),
                            bitrate=1800)
    
    ani.save(output_gif_path, writer=writer)
    #ani.save('scatter.gif', writer=writer)

    plt.show()