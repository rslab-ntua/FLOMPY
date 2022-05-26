#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The routine application for floodwater detection using Sentinel-1 GRD products

Copyright (C) 2021 by K.Karamvasis
Email: karamvasis_k@hotmail.com
Last edit: 10.4.2022

This file is part of FLOMPY - FLOod Mapping PYthon toolbox.
    FLOMPY is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    FLOMPY is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with FLOMPY. If not, see <https://www.gnu.org/licenses/>.

"""

import os
import datetime
import time
import argparse

from utils.read_template_file import read_template
from utils.read_AOI import Coords_to_geojson, Input_vector_to_geojson
from Download.Sentinel_1_download import Download_S1_data
from Download.Download_orbits import download_orbits
from Download.Download_ERA5 import Get_ERA5_data_time_period
from Preprocessing_S1_data.Classify_S1_images import Get_images_for_baseline_stack
from Preprocessing_S1_data.Preprocessing_S1_data import Run_Preprocessing, get_flood_image
from Statistical_analysis.Generate_aux import get_S1_aux
from Statistical_analysis.calc_t_scores import Calc_t_scores
from Floodwater_classification.Classification import Get_flood_map
# from Validation.Validation import Accuracy_metrics_calc
# from Validation.EMS_preparation import rasterize

print('FLOod Mapping PYthon toolbox (FLOMPY) v.1.0')
print('Copyright (c) 2022 Kleanthis Karamvasis, karamvasis_k@hotmail.com')
print('Remote Sensing Laboratory of National Technical University of Athens')
print('-----------------------------------------------------------------')
print('License: GNU GPL v3+')
print('-----------------------------------------------------------------')

##########################################################################            
STEP_LIST = [
    'Download_Precipitation_data',
    'Download_S1_data',
    'Preprocessing_S1_data',
    'Statistical_analysis',
    'Floodwater_classification',]

##########################################################################
STEP_HELP = """Command line options for steps processing with \n names are chosen from the following list:

{}
{}
{}
{}
{}

In order to use either --start or --dostep, it is necessary that a
previous run was done using one of the steps options to process at least
through the step immediately preceding the starting step of the current run.
""".format(STEP_LIST[0], STEP_LIST[1], STEP_LIST[2], STEP_LIST[3], STEP_LIST[4])

##########################################################################
EXAMPLE = """example:
  FLOMPYapp.py FLOMPYapp.cfg            #run with FLOMPYapp.cfg template
  FLOMPYapp.py -h / --help             #help
  FLOMPYapp.py -H                      #print    default template options

  # Run with --start/stop/dostep options
  FLOMPYapp.py LPS2022.cfg --dostep Download_Precipitation_data  #run at step 'Download_Precipitation_data' only
  FLOMPYapp.py LPS2022.cfg --end download_S2_data    #end after step 'download_S2_data'
"""
##########################################################################
REFERENCE = """reference:
     Karamvasis K, Karathanassi V. FLOMPY: An Open-Source Toolbox for 
     Floodwater Mapping Using Sentinel-1 Intensity Time Series. 
     Water. 2021; 13(21):2943. https://doi.org/10.3390/w13212943 
"""

def create_parser():
    parser = argparse.ArgumentParser(description='FLOod Mapping PYthon toolbox (FLOMPY)',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=REFERENCE+'\n'+EXAMPLE)

    parser.add_argument('customTemplateFile', nargs='?',
                        help='custom template with option settings.')
    parser.add_argument('-H', dest='print_template', action='store_true',
                        help='print the default template file and exit.')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='do not plot results at the end of the processing.')
    step = parser.add_argument_group('steps processing (start/end/dostep)', STEP_HELP)
    
    step.add_argument('--start', dest='startStep', metavar='STEP', default=STEP_LIST[0],
                      help='start processing at the named step, default: {}'.format(STEP_LIST[0]))
    step.add_argument('--end','--stop', dest='endStep', metavar='STEP',  default=STEP_LIST[-1],
                      help='end processing at the named step, default: {}'.format(STEP_LIST[-1]))
    step.add_argument('--dostep', dest='doStep', metavar='STEP',
                      help='run processing at the named step only')
    
    return parser

def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    inps = parser.parse_args(args=iargs)


    # print default template
    if inps.print_template:
        default_template_file = os.path.join(os.path.dirname(__file__), 'FLOMPYapp.cfg')
        raise SystemExit(open(default_template_file, 'r').read())

    if (not inps.customTemplateFile):
        
        parser.print_usage()
        print(EXAMPLE)
        msg = "ERROR: no template file found! It requires:"
        msg += "\n  input a custom template file"
        print(msg)
        raise SystemExit()

    # invalid input of custom template
    if inps.customTemplateFile:
        inps.customTemplateFile = os.path.abspath(inps.customTemplateFile)
        if not os.path.isfile(inps.customTemplateFile):
            raise FileNotFoundError(inps.customTemplateFile)
            
    
    # check input --start/end/dostep
    for key in ['startStep', 'endStep', 'doStep']:
        value = vars(inps)[key]
        if value and value not in STEP_LIST:
            msg = 'Input step not found: {}'.format(value)
            msg += '\nAvailable steps: {}'.format(STEP_LIST)
            raise ValueError(msg)

    # ignore --start/end input if --dostep is specified
    if inps.doStep:
        inps.startStep = inps.doStep
        inps.endStep = inps.doStep

    # get list of steps to run
    idx0 = STEP_LIST.index(inps.startStep)
    idx1 = STEP_LIST.index(inps.endStep)
    if idx0 > idx1:
        msg = 'input start step "{}" is AFTER input end step "{}"'.format(inps.startStep, inps.endStep)
        raise ValueError(msg)
    inps.runSteps = STEP_LIST[idx0:idx1+1]

    # mssage - processing steps
    if len(inps.runSteps) > 0:
        print('--RUN-at-{}--'.format(datetime.datetime.now()))
        print('Run routine processing with {} on steps: {}'.format(os.path.basename(__file__), inps.runSteps))
        if inps.doStep:
            Remaining_steps = STEP_LIST[idx0+1:]
            print('Remaining steps:')
            [print(step) for step in Remaining_steps]
            print('--dostep option enabled, disable the plotting at the end of the processing.')
            inps.plot = False

    print('-'*50)
    return inps

class FloodwaterEstimation:
    """ Routine processing workflow for floodwater estimation from satellite
    remote sensing data.
    """

    def __init__(self, customTemplateFile=None, workDir=None):
        ''' customTemplateFile and scihub account is required. ''' 
        self.customTemplateFile = customTemplateFile
        self.cwd = os.path.abspath(os.getcwd())
        return

    def startup(self):
        """The starting point of the workflow. It runs everytime. 
        It 1) get and read template(s) options
           2) create geojson file
           3) creates directory structure
        
        """
        template_file = os.path.join(self.cwd, self.customTemplateFile)
        self.template_dict=read_template(template_file)
        [print(key,':',value) for key, value in self.template_dict.items()]
        
        self.credentials    = {self.template_dict['scihub_username']:self.template_dict['scihub_password']}
        self.projectname    = self.template_dict['Projectname']
        self.projectfolder  = self.template_dict['projectfolder']
        self.scriptsfolder  = self.template_dict['src_dir']
        self.flood_datetime = datetime.datetime.strptime(self.template_dict['Flood_datetime'],
                                                         '%Y%m%dT%H%M%S')
        self.baseline_days  = int(self.template_dict['before_flood_days'])
        self.after_flood_days = int(self.template_dict['after_flood_days'])
        self.relOrbit       = self.template_dict['relOrbit']
        self.S2tileid       = self.template_dict['S2_TILE']
        self.rain_thres     = float(self.template_dict['rain_thres'])
        self.min_map_area   = float(self.template_dict['minimum_mapping_unit_area_m2'])
        self.gptcommand     = self.template_dict['GPTBIN_PATH']
        self.snap_dir       = self.template_dict['snap_dir']
        self.CPU            = int(self.template_dict['CPU'])
        self.RAM            = self.template_dict['RAM']
        self.AOI_File       = self.template_dict['AOI_File']
        self.LATMIN         = float(self.template_dict['LATMIN'])
        self.LONMIN         = float(self.template_dict['LONMIN'])
        self.LATMAX         = float(self.template_dict['LATMAX'])
        self.LONMAX         = float(self.template_dict['LONMAX'])
        
        
        
        if self.AOI_File.upper() == "NONE":
            self.bbox           = [self.LONMIN,
                                   self.LATMIN,
                                   self.LONMAX,
                                   self.LATMAX,] 
    
            self.geojson_S1     = Coords_to_geojson(self.bbox,
                                                    self.projectfolder,
                                                    '{}_AOI.geojson'.format(self.projectname))
        else:
            self.bbox, self.geojson_S1 = Input_vector_to_geojson(self.AOI_File,
                                                                 self.projectfolder,
                                                                 '{}_AOI.geojson'.format(self.projectname))
        
        self.start_datetime = self.flood_datetime-datetime.timedelta(days=self.baseline_days)
        self.end_datetime = self.flood_datetime+datetime.timedelta(days=self.after_flood_days)                               
        self.Start_time=self.start_datetime.strftime("%Y%m%d")
        self.End_time=self.end_datetime.strftime("%Y%m%d")


        #--- Creating directory structure
        if not os.path.exists(self.projectfolder): os.mkdir(self.projectfolder)
        self.graph_dir = os.path.join(self.scriptsfolder,'C_Preprocessing/Graphs')
        assert os.path.exists(self.graph_dir)
        assert os.path.exists(self.gptcommand)
        
        self.S1_GRD_dir = os.path.join(self.projectfolder,'Sentinel_1_GRD_imagery')
        self.ERA5_dir = os.path.join(self.projectfolder,'ERA5')
        self.Preprocessing_dir = os.path.join(self.projectfolder, 'Preprocessed')
        self.Results_dir = os.path.join(self.projectfolder, 'Results')
        self.temp_export_dir = os.path.join(self.S1_GRD_dir,"S1_orbits")
        
        self.directories = [self.projectfolder,
                            self.ERA5_dir,
                            self.S1_GRD_dir,
                            self.Preprocessing_dir,
                            self.Results_dir,
                            self.temp_export_dir,]
        
        [os.mkdir(directory) for directory in self.directories if not os.path.exists(directory)]

        return 0
    

    def run_download_Precipitation_data(self, step_name):
        
        Get_ERA5_data_time_period(['total_precipitation',],
                                  self.Start_time,
                                  self.End_time,
                                  self.bbox,
                                  self.ERA5_dir)
        print("Precipitation data can be found at {}".format(self.ERA5_dir))
        return 0    
    
    def run_download_S1_data(self, step_name):
        
        Download_S1_data(scihub_accounts = self.credentials,
                          S1_GRD_dir = self.S1_GRD_dir,
                          geojson_S1 = self.geojson_S1,
                          Start_time = self.Start_time,
                          End_time = self.End_time,
                          relOrbit = self.relOrbit,
                          flood_datetime = self.flood_datetime,
                          time_sleep=100, # 1 minute
                          max_tries=50)
        
        download_orbits(snap_dir = self.snap_dir,
                temp_export_dir = self.temp_export_dir,
                S1_GRD_dir = self.S1_GRD_dir)
        
        print("Sentinel-1 data and orbit information have been successfully downloaded")
        
        return 0
    
    def run_preprocessing_S1_data(self, step_name):
        
        Get_images_for_baseline_stack(ERA5_dir = self.ERA5_dir,
                                      S1_GRD_dir = self.S1_GRD_dir,
                                      Start_time = self.Start_time,
                                      End_time = self.End_time,
                                      flood_datetime = self.flood_datetime,
                                      days_back=5,
                                      rain_thres=self.rain_thres)
        
        get_flood_image(self.S1_GRD_dir, 
                        self.flood_datetime)
        
        Run_Preprocessing(gpt_exe = self.gptcommand,
                  graph_dir = self.graph_dir,
                  S1_GRD_dir = self.S1_GRD_dir,
                  geojson_S1 = self.geojson_S1,
                  Preprocessing_dir = self.Preprocessing_dir) 
        
        return 0 

    
    def run_multitemporal_statistics(self, step_name):
        
        get_S1_aux (self.Preprocessing_dir)
        
        Calc_t_scores(projectfolder = self.projectfolder,
                      Results_dir = self.Results_dir,
                      S1_GRD_dir = self.S1_GRD_dir,
                      Preprocessing_dir = self.Preprocessing_dir)
        
        return 0  
    
    def run_get_flood_map(self, step_name):
        
        Get_flood_map(Preprocessing_dir = self.Preprocessing_dir,
                      Results_dir = self.Results_dir,
                      Projectname = self.projectname,
                      num_cores = self.CPU,
                      fast_flag = True,
                      minimum_mapping_unit_area_m2=self.min_map_area)
        return 0
    
    def plot_results(self, print_aux, plot):
        pass

    def run(self, steps=STEP_LIST, plot=True):
        # run the chosen steps
        for sname in steps:
            print('\n\n******************** step - {} ********************'.format(sname))
            
            if sname == 'Download_Precipitation_data':
                self.run_download_Precipitation_data(sname)

            elif sname == 'Download_S1_data':
                self.run_download_S1_data(sname)

            elif sname == 'Preprocessing_S1_data':
                self.run_preprocessing_S1_data(sname)
                
            elif sname == 'Statistical_analysis':
                self.run_multitemporal_statistics(sname)

            elif sname == 'Floodwater_classification':
                self.run_get_flood_map(sname)

        # plot result (show aux visualization message more multiple steps processing)
        print_aux = len(steps) > 1
        self.plot_results(print_aux=print_aux, plot=plot)

        # go back to original directory
        #print('Go back to directory:', self.cwd)
        os.chdir(self.cwd)

        # message
        msg = '\n################################################'
        msg += '\n   Normal end of FLOMPY processing!'
        msg += '\n################################################'
        print(msg)
        return


##########################################################################
def main(iargs=None):
    start_time = time.time()
    inps = cmd_line_parse(iargs)

    app = FloodwaterEstimation(inps.customTemplateFile)
    app.startup()
    if len(inps.runSteps) > 0:
        app.run(steps=inps.runSteps, plot=inps.plot)

    # Timing
    m, s = divmod(time.time()-start_time, 60)
    print('Time used: {:02.0f} mins {:02.1f} secs\n'.format(m, s))
    return

###########################################################################################
if __name__ == '__main__':
    main()
    