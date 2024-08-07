#%%
#  Import packages
# from pandas import date_range
import os, sys, glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
# import plotly.graph_objects as go

# date time modules
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import DateFormatter
# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pythonAssist as pa
from spectr import SpectralAnalysis
from joblib import Parallel, delayed
from dateutil import parser
import pythonAssist as pa

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#%%
class ProcessSRWS:
    # custom Inputs
    # user definitions
    inp = pa.struct()
    inp.tiny = 12
    inp.Small = 14
    inp.Medium = 16
    inp.Large = 18
    inp.Huge = 22
    inp.WriteData = 1
    plt.rc('font', size=inp.Small)          # controls default text sizes
    plt.rc('axes', titlesize=inp.Small)     # fontsize of the axes title
    plt.rc('axes', labelsize=inp.Medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=inp.Small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=inp.Small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=inp.Medium)    # legend fontsize
    plt.rc('figure', titlesize=inp.Huge)  # fontsize of the figure title
    # %matplotlib qt
    plt.close('all')

    def __init__(self, path, inp):
        self.path = path
        self.inp = inp

    @staticmethod
    def calc_angles(x, y, z, offset=None):
        #     Input:
        # x - x coordinates
        # y - y coordinates
        # z - z coordintates vertical direction
        # offset - offset in x, y, and z
        #     Output:
        # theta - azimuth angles in radians
        # phi - elevation angles in radians
        # L - distance vector sum

        # import numpy as np

        if offset == None:
            pass
        elif offset:
            x, y, z = x - offset[0], y - offset[1], z - offset[2]

        L = np.sqrt(x**2+y**2+z**2) 
        theta = np.arctan2(y, x)
        # phi = np.pi/2 -np.arccos((z)/L) # definition from DTU, phi from z-axis (vertical)
        phi = np.arctan2(z, np.sqrt(x**2+y**2))

        return theta, phi, L

    def MotorToCoordxyz(self, m1, m2, m3):
        # import numpy as np

        alpha = 0.1e-6  # Focus motor counts -> [m]
        beta = -1456.3555556  # Prism motor counts -> [degrees]
        d = 110e-3+43e-3  # Distance between prisms
        DL = 212e-3  # distance between lens and inner prism
        fl = 0.567  # Lens focal length
        m2 = m2  # inner prism
        m1 = m1 + (2**16*4)  # outer prism
        phi1 = (m2)/beta
        phi2 = (m1-m2)/beta  # Motor pos to prism angles
        phi1 = phi1*np.pi/180
        phi2 = phi2*np.pi/180  # Degrees to radians

        Df = fl+fl**2/(alpha*(-m3))  # Focus distance measured from lens
        L = Df-d-DL  # Focus distance measured from outer prism
        theta = 30.6*np.pi/180  # deflection anle
        xcalc = d*np.cos(phi1)*np.sin(theta) + L*(np.cos(phi1)*np.cos(theta)*np.sin(theta) -
                                                    np.sin(theta)*(np.sin(phi1)*np.sin(phi2) - np.cos(phi1)*np.cos(phi2)*np.cos(theta)))
        ycalc = d*np.sin(phi1)*np.sin(theta) + L*(np.cos(theta)*np.sin(phi1)*np.sin(theta) +
                                                    np.sin(theta)*(np.cos(phi1)*np.sin(phi2) + np.cos(phi2)*np.cos(theta)*np.sin(phi1)))
        zcalc = d*np.cos(theta) + L*(np.cos(theta)*np.cos(theta) -
                                        np.cos(phi2)*np.sin(theta)*np.sin(theta))

        # filtering out non-physical values
        xcalc[~xcalc.between(-500, 500)] = np.nan
        ycalc[~ycalc.between(-500, 500)] = np.nan
        zcalc[~zcalc.between(-500, 500)] = np.nan

        # replace nans with interpolated data
        xcalc = xcalc.interpolate()
        ycalc = ycalc.interpolate()
        zcalc = zcalc.interpolate()

        return xcalc, ycalc, zcalc

    def CoordxyzToMotor(self, x, y, x0, theta, d):
        # Script to convert the x, y, z coordinates from a scan pattern with defined theta to Phi1, Phi2 and L parameters. [Not working as expected]

        # Output:
        # xout - optimized solution for the Lidar equation to solve x, y z positions

        # Input:
        # x - scanned coordinates or scanning pattern coordinates
        # y - center position of the beams (i.e. origin [0,0,0] generally)
        # x0 - initial guesses for phi1, phi2 and L parameters refers to [phi1_0, phi2_0, L0]
        # theta - azimuth positions from measurements or simulations i.e. generally arctan2(y,x)
        # d - distance between prisms

        # modules required: numpy, matplotlib, scipy.optimize.fsolve
        # classes required:
        # Data-files required: none
        #
        # See also: MotorToCoordxyz.py,  OTHER_FUNCTION_NAME2

        # References:
        # Angelou N, Mann J, Sj?holm M and Courtney M 2012 Direct measurement of the spectral transfer function of a laser based anemometer Rev. Sci. Instrum. 83
        #
        # Author: Ashim Giyanani, Research Associate
        # Fraunhofer Institute of Wind Energy
        # Windpark planning and operation department
        # Am Seedeich 45, Bremerhaven
        # email: ashim.giyanani@iwes.fraunhofer.de
        # Git site: https://gitlab.cc-asp.fraunhofer.de/giyash/HighRe.git
        # Created: 28.12.2021 ; Last revision: 30.12.2021

        # import scipy as sp
        # from scipy import optimize

        def rotation_solve(x, *data):
            y, theta, d = data
            F = np.array([d*np.cos(x[0])*np.sin(theta) + x[2]*(np.cos(x[0])*np.cos(theta)*np.sin(theta)
                                                                - np.sin(theta)*(np.sin(x[0])*np.sin(x[1])-np.cos(x[0])*np.cos(x[1])*np.cos(theta))),
                            d*np.sin(x[0])*np.sin(theta) + x[2]*(np.sin(theta)*(np.cos(x[0])*np.sin(x[1])
                                                                                + np.cos(x[1])*np.sin(x[0])*np.cos(theta))+np.cos(theta)*np.sin(x[0])*np.sin(theta)),
                            d*np.cos(theta) + x[2]*(np.cos(theta)**2 - np.cos(x[1])*(np.sin(theta)**2))]) - np.array([y[0], y[1], y[2]])
            return F

        data = (y, theta, d)
        xout = sp.optimize.fsolve(rotation_solve, x0=x0, args=data)
        return xout

    # # Example:
    # import numpy as np
    # import sys
    # sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")
    # from FnLissajous3D import FnLissajous3D

    # #
    # d = 110e-3 + 43e-3 # distance between prisms
    # theta = np.deg2rad(30.6) # deflection angle
    # x0 = [0,0,125] # initial guess
    # xc, yc, zc = 125, 0, 0 # origin coordinates

    # # create Lissajous pattern
    # a, b = 3,2
    # px0, py0, pz0 = 125,0,0
    # Px = 0 # increases the x directions
    # Py = 100 # z and y are kept same here
    # Pz = 100
    # Np = 2**9 # no. of pts in Lissajous scan pattern
    # xp, yp, zp, tp = FnLissajous3D(Px, Py, Pz, a=a, b=b, px0=px0, Np=Np, plot_figure=1)
    # theta = np.arctan2(yp,zp)

    # # xin = [0, 0, 125] # scanning coordinates
    # # solve optimize
    # xout = np.empty((xp.shape[0], 3))
    # y = [xc,yc,zc]
    # for i in np.arange(len(xp)):
    #     x = [xp[i], yp[i], zp[i]]
    #     xout[i] = CoordxyzToMotor( x, y, x0, theta[i], d)

    # # plots
    # import matplotlib.pyplot as plt
    # # plt.plot(np.rad2deg(xout[:,1]/-1456.3555556))
    # plt.plot(np.rad2deg(xout[:,0]/-1456.3555556))
    # # plt.plot(xout[:,2])
    def CampaignConstants(self, coord, plot_figure):
        # load modules
        # import pandas as pd
        # import numpy as np
        import circle_fit
        # import matplotlib.pyplot as plt
        # ---Read local positions
        df = pd.read_csv(coord, header=None, delimiter=",")
        df = df.set_axis(['Name', 'Easting', 'Northing','Height'], axis=1)
        wt_E = 472078.92
        wt_N = 5928607.08

# WS1	=[472051.21,	5928311.68,	4.53]
# WS2	=[471954.26,	5928581.97,	4.81]
# WS3	=[472155.19,	5928626.49,	4.13]
# turbine_center	=[472078.89,	5928607.13,	118.46]
# mast_center	=[472016.75,	5928212.7,	10]
# ball	=[472047.03,	5928476.58,	11.65]
# vals = list(zip(WS1, WS2, WS3, turbine_center, mast_center, ball))
# named_tuple = ['WS1', 'WS2', 'WS3', 'turbine_center', 'mast_center', 'ball']
# df_corrected = pd.DataFrame(np.transpose(vals), index = named_tuple, columns = ['Easting', 'Northing', 'Height'])
# df_corrected.index.name='Name'

# # calculate the plane of measurement
# mast_center = df_corrected[df_corrected.index.str.match('mast_center')]
# TW_center = df_corrected[df_corrected.index.str.match('turbine_center')]

# # --coordinate system new axis
# L = 125
# D = np.array(TW_center)[0][0:2]-np.array(mast_center)[0][0:2]
# alpha = np.arctan2(D[1], D[0])
# Center_coord = [np.array(TW_center)[0][0] - L*np.cos(alpha), np.array(TW_center)[0][1] - L*np.sin(alpha)]
# x_t = [Center_coord[0] + L*np.cos(alpha), Center_coord[0] - L*np.cos(alpha)]
# y_t = [Center_coord[1] + L*np.sin(alpha+np.pi), Center_coord[1] - L*np.sin(alpha+np.pi)]


# # plots for corrected df
# fig, ax1 = plt.subplots()
# fig.set_size_inches(18.5, 10.5)
# df_corrected.loc[['mast_center', 'turbine_center']].plot(x="Easting", y="Northing", ax=ax1, color='black')
# ax1.scatter(Center_coord[0], Center_coord[1], color='red')
# df_corrected.plot(x="Easting",y="Northing", kind='scatter', ax=ax1)
# # ax1.plot(x_t, y_t, color='black')
# ax1.legend(['Coordinate system x-axis', 'Coordinate system y-axis',
#             'Center of coordinatesystem', 'Leica positions'])
# # ax1.axis('scaled')
# for idx, row in df_corrected.drop_duplicates().iterrows():
#     plt.text(row['Easting'], row['Northing'], idx)

        

        # ---choosing only the points on the tower flance to find the center of the turbine tower
        df_tower_flance = df[df['Name'].str.match('TOWER F')]
        # ---fitting circle to turbine tower points
        x = np.array(df_tower_flance['Easting'])
        y = np.array(df_tower_flance['Northing'])
        data = [np.array([x[i], y[i]]) for i in range(len(x))]
        xc, yc, r, _ = circle_fit.least_squares_circle(data)
        phi = np.linspace(0, np.pi*2, 100)
        x_fit, y_fit = xc+r*np.cos(phi), yc+r*np.sin(phi)
        df_temp = pd.DataFrame({'Name':'Tower_center', 'Easting':xc, 'Northing':yc, 'Height':0}, index=df.index)
        df = pd.concat([df, df_temp], ignore_index=True).drop_duplicates()

        # ---plotting
        if self.inp.plot_figure== True:
            ax = df_tower_flance.plot(x="Easting", y="Northing", kind="scatter")
            ax.plot(x_fit, y_fit, color='blue')
            ax.scatter(df['Easting'], df['Northing'])
            # df.plot(x="Easting", y="Northing", kind="scatter", ax=ax)
            for idx, row in df.drop_duplicates().iterrows():
                    plt.text(row['Easting'], row['Northing'], row['Name'])
            ax.axis('scaled')
            df.plot(x="Easting", y="Northing", kind="scatter", ax=ax)

        # ---Calculate center of scanpattern coordinate system
        mast_center = df[df['Name'].str.match('TOP CENTER')]
        TW_center = df[df['Name'].str.match('Tower_center')]

        # --coordinate system new axis
        L = 125
        D = np.array(TW_center)[0][1:3]-np.array(mast_center)[0][1:3]
        alpha = np.arctan2(D[1], D[0])
        Center_coord = [np.array(TW_center)[0][1] - L*np.cos(alpha), np.array(TW_center)[0][2] - L*np.sin(alpha)]
        x_t = [Center_coord[0] + L*np.cos(alpha), Center_coord[0] - L*np.cos(alpha)]
        y_t = [Center_coord[1] + L*np.sin(alpha+np.pi), Center_coord[1] - L*np.sin(alpha+np.pi)]

        # from gps_transformation import Affine
        # df_mask = df.loc[mask].drop_duplicates()
        # df_transformed = df_mask.apply(lambda x: Affine.transform_srws([list(x)])[1] , axis=1)
        # df_transformed = pd.DataFrame([*df_transformed], columns = df_mask.columns, index = df_mask.index)

        # ---Plotting
        df = df.set_index('Name')
        mask = ['TOP CENTER', 'WS1', 'WS2', 'WS3', 'Tower_center', 'BALL CENTER']
        if plot_figure==True:
            fig, ax1 = plt.subplots()
            fig.set_size_inches(18.5, 10.5)
            df.loc[['TOP CENTER', 'Tower_center']].plot(x="Easting", y="Northing", ax=ax1, color='black')
            ax1.scatter(Center_coord[0], Center_coord[1], color='red')
            df.loc[mask].plot(x="Easting",y="Northing", kind='scatter', ax=ax1)
            ax1.plot(x_t, y_t, color='black')
            ax1.legend(['Coordinate system x-axis', 'Coordinate system y-axis',
                        'Center of coordinatesystem', 'Leica positions'])
            ax1.axis('scaled')
            for idx, row in df.drop_duplicates().loc[mask].iterrows():
                plt.text(row['Easting'], row['Northing'], idx)


        # Local (Mikael) coordniate system
        df_rotate = df
        x = np.array(df.Easting)
        y = np.array(df.Northing)
        z = np.array(df.Height)

        x_new, y_new, z_new = ProcessSRWS.Rotate_and_translate(x,y,z,[Center_coord[0],Center_coord[1],0],0, -alpha+np.pi/2)
        df_rotate['Easting'] = x_new
        df_rotate['Northing'] = y_new
        df_rotate['Height'] = z_new

        y_t = [0, 0]
        x_t = [-L, L]

        if self.inp.plot_figure == True:
            fig, ax1 = plt.subplots()
            fig.set_size_inches(18.5, 10.5)
            ax1.scatter(0, 0, color='red')
            df_rotate.loc[mask].plot(x="Easting", y="Northing", kind='scatter', ax=ax1)
            for idx, row in df_rotate.drop_duplicates().loc[mask].iterrows():
                plt.text(row['Easting'], row['Northing'], idx, fontsize=14)
            ax1.set_xlabel('x-axis', fontsize=14)
            ax1.set_ylabel('y-axis', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ax1.grid()
            # ax1.legend(['CS x-axis','CS y-axis', 'Center of coordinatesystem','Leica positions'], fontsize=14)
            ax1.axis('scaled')

        # Get data into global coordinate system
        # elevation and azimuth angles
        elev = np.array([56.50, 51.71,56.37])*np.pi/180
        azi = np.array([186.21, 55.88, 335.13])*np.pi/180

        # correct for offsets during installation [to be confirmed]
        # offset_elev = np.array([0.0912, 0.3274, 0.1034]) * np.pi/180
        # offset_azi = np.array([-0.037, 0.0965, -0.4979]) * np.pi/180
        # for i in range(len(elev)):
        #     elev[i] = elev[i] + offset_elev[i]
        #     azi[i] = azi[i] + offset_azi[i]

        # absolute positions of the windscanners (x,y,z) with origin at (0,0,125)
        WS1 = [171.067, 18.6175, 11.19]
        WS2 = [-80.7441, -119.187, 11.43]
        WS3 = [-155.978, 72.3011, 10.72]
        return (WS1, WS2, WS3, elev, azi, df_rotate)

    from typing import Tuple
    @staticmethod
    def Read_SRWS_bin(file_name: str, mode: str, write_spec: bool=False, **kwargs) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Function that reads data from SRWS and converts it to a numpy array and pandas dataframes.

        Parameters:
        file_name (str): Path to the SRWS binary file.
        mode (str): Specifies the level of data extraction. Options are 'basic', 'standard', and 'all'.
        write_spec (bool, optional): If True, a separate pandas DataFrame containing the spectral data will be returned. Defaults to False.
        kwargs:
            cols: cols to be read for the data, Timestamps are added to the list automatically spectrum data is not tested (is meant for data)

        Returns:
        Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
            - np.ndarray: The raw data rea

        # Given configurations
        # lambda0 - laser wavelength = 1.55e-6
        # a0 - beam radius at waist = 32e-3
        # import numpy as np
        # import os
        # import pythonAssist as pa
        # import pandas as pd
        # from datetime import timedelta
        """
        cols = kwargs.setdefault('cols', ['Timestamp 1', 'Timestamp 2', 'Timestamp 3', 'Timestamp 4'])

        dt= np.dtype([('ID',np.int32),
            ('ComPosition_pm1 1',np.float64),  # commanded position'
            ('ComPosition_pm2 1',np.float64),
            ('ComPosition_fm3 1',np.float64),
            ('ComPosition_pm1 2',np.float64),
            ('ComPosition_pm2 2',np.float64),
            ('ComPosition_fm3 2',np.float64),
            ('ComPosition_pm1 3',np.float64),
            ('ComPosition_pm2 3',np.float64),
            ('ComPosition_fm3 3',np.float64),
            ('ActPosition_pm1 1',np.float64),
            ('ActPosition_pm2 1',np.float64),
            ('ActPosition_fm3 1',np.float64),
            ('ActPosition_pm1 2',np.float64),
            ('ActPosition_pm2 2',np.float64),
            ('ActPosition_fm3 2',np.float64),
            ('ActPosition_pm1 3',np.float64),
            ('ActPosition_pm2 3',np.float64),
            ('ActPosition_fm3 3',np.float64),
            ('ActPulse 1',np.int32), # used as a channel to send a trigger pulse when working in a windtunnel
            ('ActPulse 2',np.int32),
            ('ActPulse 3',np.int32),
            ('ModeVar',np.int32),
            ('Status 1',np.uint8),
            ('Timestamp 1',np.uint64),
            ('ID_nr 1',np.uint16), # for sync purposes
            ('AverageCount 1',np.uint16), # no. of spectra averaged in each sample, high when sampling frequency is low
            ('vlos 1',np.float32),
            ('MaximumValue 1',np.float32),
            ('TotalPower 1',np.int32),
            ('Qualitydata 1',np.float32),
            ('Beta 1',np.float32), # calculated from the spectrum, to do something with the noise
            ('LaserPowerEstimate 1',np.float32),
            ('Spectrum 1',(np.uint16,512)),
            ('Status 2',np.uint8),
            ('Timestamp 2',np.uint64),
            ('ID_nr 2',np.uint16), # for sync purposes
            ('AverageCount 2',np.uint16), # no. of spectra averaged in each sample, high when sampling frequency is low
            ('vlos 2',np.float32),
            ('MaximumValue 2',np.float32),
            ('TotalPower 2',np.int32),
            ('Qualitydata 2',np.float32),
            ('Beta 2',np.float32), # calculated from the spectrum, to do something with the noise
            ('LaserPowerEstimate 2',np.float32),
            ('Spectrum 2',(np.uint16,512)),
            ('Status 3',np.uint8),
            ('Timestamp 3',np.uint64),
            ('ID_nr 3',np.uint16), # for sync purposes
            ('AverageCount 3',np.uint16), # no. of spectra averaged in each sample, high when sampling frequency is low
            ('vlos 3',np.float32),
            ('MaximumValue 3',np.float32),
            ('TotalPower 3',np.int32),
            ('Qualitydata 3',np.float32),
            ('Beta 3',np.float32), # calculated from the spectrum, to do something with the noise
            ('LaserPowerEstimate 3',np.float32),
            ('Spectrum 3',(np.uint16,512)),
            ('Status 4',np.uint8),
            ('Timestamp 4',np.uint64),
            ('ID_nr 4',np.uint16), # for sync purposes
            ('AverageCount 4',np.uint16), # no. of spectra averaged in each sample, high when sampling frequency is low
            ('vlos 4',np.float32),
            ('MaximumValue 4',np.float32),
            ('TotalPower 4',np.int32),
            ('Qualitydata 4',np.float32),
            ('Beta 4',np.float32), # calculated from the spectrum, to do something with the noise
            ('LaserPowerEstimate 4',np.float32),
            ('Spectrum 4',(np.uint16,512))
            ])

        # number of data packages  (timestamps)
        numspec = int(os.path.getsize(file_name)/4408)
        data = np.fromfile(file_name, dt, numspec)

        # Extract information from the structured array
        keys = list(data.dtype.fields.keys())  # Parameter names
        dtype = [str(v[0]) for v in data.dtype.fields.values()]  # Dtypes
        bits = [v[1] for v in data.dtype.fields.values()]  # Bit positions

        # seperate the multidimensional spectral data to reduce load on the compiler
        if mode == 'basic':
            remove_cols = ['Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'ID_nr 1', 'ID_nr 2','ID_nr 3''ID_nr 4', 'Act Pulse 1','Act Pulse 2', 'Act Pulse 3', 'ModVar', 'AverageCount 1', 'AverageCount 2', 'AverageCount 3', 'AverageCount 4',\
                            'Beta 1', 'Beta 2', 'Beta 3', 'Beta 4', \
                                ]
        elif mode == 'standard':
            remove_cols = ['Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4', 'Beta 1', 'Beta 2', 'Beta 3', 'Beta 4','Act Pulse 1','Act Pulse 2', 'Act Pulse 3',]
        elif mode == 'all':
            remove_cols = ['Spectrum 1', 'Spectrum 2', 'Spectrum 3', 'Spectrum 4']
        elif mode == 'custom':
            cols = cols + ['Timestamp 1', 'Timestamp 2', 'Timestamp 3', 'Timestamp 4']
            remove_cols = [x for x in keys if x not in cols]
        else:
            print('Input the mode of data extraction: options (basic, standard, all)')

        keep_cols = [x for x in keys if x not in remove_cols]
        rest_data = pa.remove_fields(data, keep_cols)
        spec_data = pa.remove_fields(rest_data, remove_cols[4:])
        srws_data = pa.remove_fields(data, remove_cols)

        df = pd.DataFrame.from_records(srws_data)

        spec_df = pd.DataFrame()
        if write_spec==True:
            # create a pandas dataframe from the data
            for name in spec_data.dtype.names:
                df_temp = pd.DataFrame(spec_data[name])
                # # remove columns with only zero values
                df_temp = pa.remove_columns(df_temp, mode='0')
                # # replace rows with zeros with previous rows
                df_temp = pa.replace_rows(df_temp, replace='bfill')
                # concatenate spectrum if 
                # remove spectrum if all rows are zero or all (more than 50%) columns are zero 
                if ((df_temp**2).sum(axis=1) == 0).sum() == len(df_temp) or (((df_temp**2).sum()==0).sum() > df_temp.shape[1]/2):
                    max_name = name.replace('Spectrum',  'MaximumValue')
                    if (df[max_name].sum() == 0) or (df[max_name].mean() > 2**16):
                        df.drop(columns = [max_name], inplace=True)
                    pass
                else:
                    spec_df = pd.concat([spec_df, df_temp], axis=1)

            # creat a multiIndex from the number of spectra
            spec_names = spec_data.dtype.names[0:3]
            max_names = [s.replace('Spectrum', 'MaximumValue') for s in spec_names]
            max_new_names = dict(zip(df.filter(like='Maxi').columns, max_names))
            df.rename(columns=max_new_names, inplace=True)
            index = [spec_names, list(range(512))]
            index = pd.MultiIndex.from_product(index, names=['spec', 'Nspec'])
            spec_df.columns = index
            spec_df.set_index(df.index)

        # create extra datetime columns WS1, 2, 3
        time = df.filter(like='Time')
        zero_cols = time.columns[(time==0).all()]
        time = time.drop(zero_cols, axis=1) 
        T0 = parser.parse(file_name[-20:]) # get the start time from filename
        # replace zeros in timestamps
        time = time.replace(0, np.nan).ffill()
        # get timedelta, store timedeltas as individual timestamps
        dt = time.diff().fillna(0).astype(np.uint8)
        # preparing the timedelta as a dataframe
        df_dt = pd.DataFrame(dt)
        new_col_names = [f'dt {i}' for i in range(1,dt.shape[1]+1)]
        df_dt.columns = new_col_names

        # generate a new timestamp based on number of points in the raw data file
        try:
            if df.index.stop >= 19000: 
                T_gen = pd.date_range(T0, periods=df.index.stop,freq=str(int(1e9*60/df.index.stop))+'ns')
            else:
                T_gen = pd.date_range(T0, periods=df.index.stop, freq='3099814ns')
        except ZeroDivisionError:
            T_gen = []
            print(f"{[pa.now()]: This file seems to be empty}")

        # get the datetime dataframe integrated into the raw df
        df = pd.concat([df,df_dt], axis=1).set_index(T_gen)
        if write_spec:
            spec_df = spec_df.set_index(T_gen)

        # convert to UTC timestamps
        df.index = df.index.tz_convert('UTC')
        spec_df.index = spec_df.index.tz_convert('UTC')

        return data, df, spec_df

    def ReconstructUV(self, V_1, V_2, theta_1, theta_2, phi_1, phi_2):
        # Function for reconstructing the 2D wind velocity from two lidar radial
        # velocities thus ignoring the vertical wind component.
        # theta - azimuth angle in radians (positive direction is counter clockwise)
        # phi   - elevation angle in radians (measured from horizontal)

        # import numpy as np

        denom = np.cos(phi_1)*np.cos(phi_2)*(np.sin(theta_1) *
                                                np.cos(theta_2)-np.cos(theta_1)*np.sin(theta_2))

        u = (V_1*np.cos(theta_2)*np.cos(phi_2) -
                V_2*np.cos(theta_1)*np.cos(phi_1))/denom

        v = (V_2*np.sin(theta_1)*np.cos(phi_1) -
                V_1*np.sin(theta_2)*np.cos(phi_2))/denom
        return u, v

    def ReconstructUVW(self, V_1, V_2, V_3, theta_1, theta_2, theta_3, phi_1, phi_2, phi_3, mode="modified"):
        """
            Function for reconstructing the 3D wind velocity from three lidar radial velocities.
        
            Inputs:
                V_1, V_2, V_3 - line of sight velocities from three SRWS
                theta_# - azimuth angle in radians (positive direction is counter clockwise)
                phi_#   - elevation angle in radians (measured from horizontal)
       
            Outputs:
               u, v, w - wind components
               Note that the u, v are rotated compared to Tegtmeier2020 as the input x, y, z were aligned towards North, East, zenith. 
               The output provides therefore the same result as Tegtmeier2020 u pointing towards East, v towards North and w towards zenith 

            Example:
                import numpy as np
                V = -5.5, 3.2, 4.7
                phi = 0.57, 0.57, 0.57
                theta = -2.9, 0.9, -0.5
                u,v,w = ReconstructUVW(V[0],V[1],V[2],theta[0],theta[1],theta[2],phi[0],phi[1],phi[2])
                Vhorz = np.sqrt(u**2 + v**2 + w**2)
                zeta = np.rad2deg(np.arctan2(v,u)) + 180
                psi = np.rad2deg(np.arctan(w/Vhorz))
                print(u,v,w)
                print(Vhorz, zeta, psi)
        
            modules required: none
            classes required: none
            Data-files required: none
        
            References:
                Tegtmeier, A. (2020). 2018 V52-CCA Shortrange Windscanner. DTU Wind Energy E-Report-0198.
                Hansen, K. H., Pedersen, A. T., & Thorsen, G. R. (2018). WindScanner beam path geometry.
                Giyanani, A.et al. (2022). Wind speed reconstruction from three synchronized short-range WindScanner lidars 
                    in a large wind turbine inflow field campaign and the associated uncertainties.
        """

        import numpy as np

        alpha_1 = np.cos(theta_1)*np.cos(phi_1)
        alpha_2 = np.cos(theta_2)*np.cos(phi_2)
        alpha_3 = np.cos(theta_3)*np.cos(phi_3)
        beta_1 = np.sin(theta_1)*np.cos(phi_1)
        beta_2 = np.sin(theta_2)*np.cos(phi_2)
        beta_3 = np.sin(theta_3)*np.cos(phi_3)

        if mode == 'modified':
            a = alpha_3*beta_2-alpha_2*beta_3
            b = alpha_1*beta_3-alpha_3*beta_1
            c = alpha_2*beta_1-alpha_1*beta_2
        else:
            a = beta_3*alpha_2-beta_2*alpha_3
            b = beta_1*alpha_3-beta_3*alpha_1
            c = beta_2*alpha_1-beta_1*alpha_2

        denom = a*np.sin(phi_1)+b*np.sin(phi_2)+c*np.sin(phi_3)

        u = (V_1*(beta_2*np.sin(phi_3)-beta_3*np.sin(phi_2)) +
                V_2*(beta_3*np.sin(phi_1)-beta_1*np.sin(phi_3)) +
                V_3*(beta_1*np.sin(phi_2)-beta_2*np.sin(phi_1)))/denom

        v = (V_1*(alpha_3*np.sin(phi_2)-alpha_2*np.sin(phi_3)) +
                V_2*(alpha_1*np.sin(phi_3)-alpha_3*np.sin(phi_1)) +
                V_3*(alpha_2*np.sin(phi_1)-alpha_1*np.sin(phi_2)))/denom

        w = (a*V_1+b*V_2+c*V_3)/denom
        return u, v, w


    def ReconstructUVW_pedersen(self, V_1, V_2, V_3, theta_1, theta_2, theta_3, phi_1, phi_2, phi_3):
    # Function for reconstructing the 3D wind velocity from three lidar radial
    # velocities.
    # theta - azimuth angle in radians (positive direction is counter clockwise)
    # phi   - elevation angle in radians (measured from horizontal)
        # import numpy as np
        
        alpha_1 = np.cos(theta_1)*np.cos(phi_1)
        alpha_2 = np.cos(theta_2)*np.cos(phi_2)
        alpha_3 = np.cos(theta_3)*np.cos(phi_3)
        beta_1 = np.sin(theta_1)*np.cos(phi_1)
        beta_2 = np.sin(theta_2)*np.cos(phi_2)
        beta_3 = np.sin(theta_3)*np.cos(phi_3)

        a = alpha_3*beta_2-alpha_2*beta_3
        b = alpha_1*beta_3-alpha_3*beta_1
        c = alpha_2*beta_1-alpha_1*beta_2

        denom = a*np.sin(phi_1)+b*np.sin(phi_2)+c*np.sin(phi_3)

        u = (V_1*(alpha_2*np.sin(phi_3)-alpha_3*np.sin(phi_2))+
            V_2*(alpha_3*np.sin(phi_1)-alpha_1*np.sin(phi_3))+
            V_3*(alpha_1*np.sin(phi_2)-alpha_2*np.sin(phi_1)))/denom

        v = (V_1*(beta_3*np.sin(phi_2)-beta_2*np.sin(phi_3))+
            V_2*(beta_1*np.sin(phi_3)-beta_3*np.sin(phi_1))+
            V_3*(beta_2*np.sin(phi_1)-beta_1*np.sin(phi_2)))/denom

        w = (a*V_1+b*V_2+c*V_3)/denom
        return u,v,w

    @staticmethod
    def Rotate_and_translate(x, y, z, Cen, elev, azi):
        """
            Rotate and translate a point (x, y, z) around a specified center (Cen) in 3D space.

            Parameters:
            - x (float): x-coordinate of the point obtained from motor 2 xyz conversions.
            - y (float): y-coordinate of the point.
            - z (float): z-coordinate of the point.
            - Cen (list or tuple): Center point around which rotation and translation are performed.
            - elev (float): Elevation angle in radians.
            - azi (float): Azimuth angle in radians.

            Returns:
            - xrot (float): Translated and rotated x-coordinate.
            - yrot (float): Translated and rotated y-coordinate.
            - zrot (float): Translated and rotated z-coordinate.
        """
        ax_y = [0, 1, 0]
        ax_z = [0, 0, 1]
        Rz = ProcessSRWS.rotation_matrix(ax_z, azi)
        Ry = ProcessSRWS.rotation_matrix(ax_y, elev)
        G = np.array([x-Cen[0], y-Cen[1], z-Cen[2]])
        Grot = np.dot(Rz, np.dot(Ry, G))
        xrot = Grot[0, :]
        yrot = Grot[1, :]
        zrot = Grot[2, :]
        return xrot, yrot, zrot

    @staticmethod
    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        # import math
        # import numpy as np

        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    @staticmethod
    def scanner2globelCoord(x, y, z, WS, elev, azi):

        # import numpy as np
        # import pandas as pd

        ax_y = [0, 1, 0]
        ax_z = [0, 0, 1]
        Rz = ProcessSRWS.rotation_matrix(ax_z, azi)
        Ry = ProcessSRWS.rotation_matrix(ax_y, elev)
        G = np.array([x, y, z])
        Grot = np.dot(Rz, np.dot(Ry, G))
        xrot = pd.Series(Grot[0, :]+WS[0])
        yrot = pd.Series(Grot[1, :]+WS[1])
        zrot = pd.Series(Grot[2, :]+WS[2])

        # filtering out non-physical values
        xrot[~xrot.between(-500, 500)] = np.nan
        yrot[~yrot.between(-500, 500)] = np.nan
        zrot[~zrot.between(-500, 500)] = np.nan

        # replace nans with interpolated data
        xrot = xrot.interpolate()
        yrot = yrot.interpolate()
        zrot = zrot.interpolate()

        return xrot, yrot, zrot

    def uvw2los(self, u, v, w, phi, theta):
        """
        #uvw2los - script to covert the u,v and w components to the line of sight velocity (if not already available)
        #
        # Syntax:  Vlos = uvw2los(u,v,w,phi,theta,R)
        #
        # Inputs:
        #    u,v,w - wind components in longitudinal, lateral and vertical direction in cartesian coordinates with x [m/s]
        #    phi - elevation angle in the local Lidar coordinate system [deg]
        #    theta - azimuth angle in the local lidar coordinate system [deg]
        #
        # Outputs:
        #    Vlos - line of sight velocity in local Lidar coordinate system [m/s]
        #
        # Example:
        # import numpy as np
        # theta = 30 + np.random.randn(1000)
        # phi = 5 + np.random.randn(1000)
        # u = 8 + np.random.randn(1000)
        # v = 3 + np.random.randn(1000)
        # w = 1 + np.random.randn(1000)
        # Vlos = uvw2los(u,v,w,phi,theta)    #

        # modules required: numpy
        # classes required: none
        # Data-files required: none
        #
        # See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2

        # References:
        https://github.com/niva83/YADDUM/blob/master/yaddum/yaddum.py
        Vasiljevic N, Courtney M and Pedersen A T 2020 Uncertainty model for dual-Doppler retrievals of wind speed and wind direction Atmos. Meas. Tech. Discuss. 1�25
        #
        # Author: Ashim Giyanani, Research Associate
        # Fraunhofer Institute of Wind Energy
        # Windpark planning and operation department
        # Am Seedeich 45, Bremerhaven
        # email: ashim.giyanani@iwes.fraunhofer.de
        # Git site: https://gitlab.cc-asp.fraunhofer.de/giyash/HighRe.git
        # Created: 06-08-2020; Last revision: 12-May-200406-08-2020
        """
    # ------------- BEGIN CODE --------------
        # import numpy as np

        # convert angles from deg to rad
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

        # calculate the line of sight velocity
        Vlos = u*np.sin(theta)*np.cos(phi) + v*np.cos(theta) * \
            np.cos(phi) + w*np.sin(phi)

        return Vlos
    # ------------- END CODE --------------

    # ------------- Example --------------
    # import numpy as np
    # theta = 30 + np.random.randn(1000)
    # phi = 5 + np.random.randn(1000)
    # u = 8 + np.random.randn(1000)
    # v = 3 + np.random.randn(1000)
    # w = 1 + np.random.randn(1000)
    # Vlos = uvw2los(u,v,w,phi,theta)
    # @profile
    def srws2df(self, inp, file, campaign_constants, Nfiles):
        """generate a dataframe using the srws files"""
		
        import pythonAssist as pa
        # from enumerations import *
        # from metadata_helpers import (get_CampaignInfo, get_guiData)
        # from metadata_tables import (DatasetInfo, DataFile, UserInterface, Network, Site, WindFarm, InstrumentSpecs, LidarConfig)
        from datetime import datetime
        dateformat = "%Y-%m-%dT%H:%M:%SZ"

        # Step 1: get measurement campaign information
        WS1, WS2, WS3, elev, azi, df_rotate = campaign_constants

        path_Data="sample" 
        path_df_H="sample" 
        path_df_bowtie= "sample" 
        path_df_m="sample"
        Data_mean = None
        Data_min = None
        Data_std = None
        Data_max = None

        # Step 2a: Get all files in folder with data
        filename, _ = os.path.splitext(file)

        if inp.srws.use_corrected_vlos == True:
            inp.spec = pa.struct()
            inp.spec.test = False
            inp.spec.height = [1.25, 5]
            inp.spec.prominence = [0.25, 5]
            inp.spec.distance = 3
            inp.spec.width = 1
            inp.spec.threshold = 3500
            inp.spec.window = 128
            inp.spec.plot_figure = False
            inp.spec.verbose=False
            inp.spec.read_dtu = True
            inp.spec.window_size=10
            inp.spec.ax=1
            inp.spec.mode='median'
            inp.spec.moving_average = False
            inp.spec.interpolate = True
            inp.spec.n_peaks = 1
            inp.spec.wlen = 60

            # run the vlos estimation from spectra
            _, raw_df, spec_df = self.Read_SRWS_bin(filename, mode='all', write_spec=True)
            # reduce the cycle time during debugging. 
            # raw_df = raw_df.iloc[:2000,:]
            # spec_df = spec_df.iloc[:2000,:]
            
            vlos_names = [f'vlos_pp {i}' for i in range(1,4)]
            snr_names = [f'snr_pp {i}' for i in range(1,4)]
            objs = [SpectralAnalysis(spec_df['Spectrum {}'.format(i)], inp.spec, raw_df['MaximumValue {}'.format(i)]) for i in range(1,4)]
            with Parallel(n_jobs=3) as parallel:
                delayed_funcs = [delayed(lambda x:x.RunSpectralAnalysis())(obj) for obj in objs]
                run_df = parallel(delayed_funcs)
            
            vlos_df = pd.concat([pd.DataFrame(df[0]) for df in run_df], ignore_index=True, axis=1).set_axis(vlos_names, axis=1).reset_index(drop=True)
            snr_df = pd.concat([pd.DataFrame(df[1]) for df in run_df], ignore_index=True, axis=1).set_axis(snr_names, axis=1).reset_index(drop=True)
            
            Data_raw = pd.concat([raw_df.reset_index(drop=True), vlos_df, snr_df], axis=1).set_index(raw_df.index)
        else:
            # Step 2b: read the raw data file
            vlos_names = ['vlos {}'.format(i) for i in range(1,4)]
            _, Data_raw, spec_df = self.Read_SRWS_bin(filename, mode='all', write_spec=True)

        # get metadata for the file
        # meta_ds = self.add_metadata(file)

        # get the quality flags
        from DataQualityFlags import quality_control
        qc = quality_control(Data_raw)

        # drop columns with zeros in all rows and having 4 as the number at the end of the name. mark rows as missing and failing all qc checks
        Data_raw = qc.prepare_df(filter_out_regex="4$")

        # detect missing values in the dataframe
        ws_names = Data_raw.filter(regex='vlos').columns
        Data_raw = qc.detect_missing(channel_name=ws_names.to_list())

        # check time gaps in the dataframe
        inp.srws.dt = np.diff(Data_raw.index)[1]
        Data_raw = qc.detect_time_gaps(dt_in=inp.srws.dt, channel_name=None, correct_gaps=False)

        # detect outliers in ws range
        Data_raw = qc.detect_outliers_ws_range(channel_names=ws_names, ranges=[-35, 35])
        
        # assign valid qc flag to ones without any errors
        Data_raw = qc.assign_qc_valid(Data_raw)

        # extract time from the data
        Data_raw = Data_raw.reset_index()
        files_tstart = parser.parse(os.path.basename(file)[-20:])
        # files_tend = files_tstart + timedelta(minutes=1)
        # time1 = Data_raw['Timestamp 1']
        # time = Data_raw.filter(like='Time').drop('Timestamp 4', axis=1)

        # Step 2c: Gettting motor positions in local scanner coordinates
        x1, y1, z1 = self.MotorToCoordxyz(Data_raw['ActPosition_pm1 1'], Data_raw['ActPosition_pm2 1'], Data_raw['ActPosition_fm3 1'])
        x2, y2, z2 = self.MotorToCoordxyz(Data_raw['ActPosition_pm1 2'], Data_raw['ActPosition_pm2 2'], Data_raw['ActPosition_fm3 2'])
        x3, y3, z3 = self.MotorToCoordxyz(Data_raw['ActPosition_pm1 3'], Data_raw['ActPosition_pm2 3'], Data_raw['ActPosition_fm3 3'])

        # Step 2c: commanded positions for BowTie pattern
        xo1, yo1, zo1 = self.MotorToCoordxyz(Data_raw['ComPosition_pm1 1'], Data_raw['ComPosition_pm2 1'], Data_raw['ComPosition_fm3 1'])
        xo2, yo2, zo2 = self.MotorToCoordxyz(Data_raw['ComPosition_pm1 2'], Data_raw['ComPosition_pm2 2'], Data_raw['ComPosition_fm3 2'])
        xo3, yo3, zo3 = self.MotorToCoordxyz(Data_raw['ComPosition_pm1 3'], Data_raw['ComPosition_pm2 3'], Data_raw['ComPosition_fm3 3'])


        # Step 3a: Coordinate in Mikael-coordinates system
        # azi = (azi + np.pi) % (2*np.pi)
        xrot1, yrot1, zrot1 = ProcessSRWS.scanner2globelCoord(x1, y1, z1, WS1, elev[0], azi[0])
        xrot2, yrot2, zrot2 = ProcessSRWS.scanner2globelCoord(x2, y2, z2, WS2, elev[1], azi[1])
        xrot3, yrot3, zrot3 = ProcessSRWS.scanner2globelCoord(x3, y3, z3, WS3, elev[2], azi[2])

        # actual Theta, Phi and ranges
        theta1, phi1, R1 = ProcessSRWS.calc_angles(x1, y1, z1)
        theta2, phi2, R2 = ProcessSRWS.calc_angles(x2, y2, z2)
        theta3, phi3, R3 = ProcessSRWS.calc_angles(x3, y3, z3)

        # commanded Theta, Phi and ranges
        thetao1, phio1, Ro1 = ProcessSRWS.calc_angles(xo1, yo1, zo1)
        thetao2, phio2, Ro2 = ProcessSRWS.calc_angles(xo2, yo2, zo2)
        thetao3, phio3, Ro3 = ProcessSRWS.calc_angles(xo3, yo3, zo3)

        # Step 3b: rotated theta, phi and ranges
        theta1_rot, phi1_rot, _ = ProcessSRWS.calc_angles(WS1[0]-xrot1, WS1[1]-yrot1, WS1[2]-zrot1)
        theta2_rot, phi2_rot, _ = ProcessSRWS.calc_angles(WS2[0]-xrot2, WS2[1]-yrot2, WS2[2]-zrot2)
        theta3_rot, phi3_rot, _ = ProcessSRWS.calc_angles(WS3[0]-xrot3, WS3[1]-yrot3, WS3[2]-zrot3)
        theta_rot = (theta1_rot, theta2_rot, theta3_rot)
        phi_rot = (phi1_rot, phi2_rot, phi3_rot)

        # Step 4a: wind vector reconstruction based on dtu wind speed
        from scipy.spatial.transform import Rotation as R
        Rz = R.from_euler('z', 90, degrees=True).as_matrix()  # 26.01.2024 Paul Meyer found that the reconstructed wind components are 90° offseted i.e. vx is towards AD8 wt and vy is towards west
        # getting the radial wind speeds
        V1, V2, V3 = Data_raw.loc[:,ws_names[0]],Data_raw.loc[:,ws_names[1]], Data_raw.loc[:,ws_names[2]] 
        vx, vy, vz = self.ReconstructUVW(V1, V2, V3, theta_rot[0], theta_rot[1], theta_rot[2], phi_rot[0], phi_rot[1], phi_rot[2])
        [vx, vy, vz] = np.dot(Rz, [vx, vy, vz])
        # Step 4b:  wind vector from beam 1 and beam 2
        vx12, vy12 = self.ReconstructUV(V1, V2, theta_rot[0], theta_rot[1], phi_rot[0], phi_rot[1])
        [vx12, vy12, vz] = np.dot(Rz, [vx12, vy12, vz])
        # wind vector from beam 2 and beam 3
        vx23, vy23 = self.ReconstructUV(V2, V3, theta_rot[1], theta_rot[2], phi_rot[1], phi_rot[2])
        [vx23, vy23, vz] = np.dot(Rz, [vx23, vy23, vz])
        # wind vector from beam 3 and beam 1
        vx31, vy31 = self.ReconstructUV(V3, V1, theta_rot[2], theta_rot[0], phi_rot[2], phi_rot[0])
        [vx31, vy31, vz] = np.dot(Rz, [vx31, vy31, vz])

        # Step 4b: wind vector reconstruction based on IWES Vlos after performing spectral correction 
        # getting the radial wind speeds
        V1_pp, V2_pp, V3_pp = Data_raw.loc[:,vlos_names[0]],Data_raw.loc[:,vlos_names[1]], Data_raw.loc[:,vlos_names[2]] 
        vx_pp, vy_pp, vz_pp = self.ReconstructUVW(V1_pp, V2_pp, V3_pp, theta_rot[0], theta_rot[1], theta_rot[2], phi_rot[0], phi_rot[1], phi_rot[2])
        [vx_pp, vy_pp, vz_pp] = np.dot(Rz, [vx_pp, vy_pp, vz_pp])
        # Step 4b:  wind vector from beam 1 and beam 2
        vx12_pp, vy12_pp = self.ReconstructUV(V1_pp, V2_pp, theta_rot[0], theta_rot[1], phi_rot[0], phi_rot[1])
        [vx12_pp, vy12_pp, vz_pp] = np.dot(Rz, [vx12_pp, vy12_pp, vz_pp])
        # wind vector from beam 2 and beam 3
        vx23_pp, vy23_pp = self.ReconstructUV(V2_pp, V3_pp, theta_rot[1], theta_rot[2], phi_rot[1], phi_rot[2])
        [vx23_pp, vy23_pp, vz_pp] = np.dot(Rz, [vx23_pp, vy23_pp, vz_pp])
        # wind vector from beam 3 and beam 1
        vx31_pp, vy31_pp = self.ReconstructUV(V3_pp, V1_pp, theta_rot[2], theta_rot[0], phi_rot[2], phi_rot[0])
        [vx31_pp, vy31_pp, vz_pp] = np.dot(Rz, [vx31_pp, vy31_pp, vz_pp])

        # Step 4c: account for N=0 to metmast to wt line degree [to be confirmed]
        from scipy.spatial.transform import Rotation as R
        alpha = -9 # source geo Daten Total station survey, changed from +9 to -9� [29.06.2023]
        Rz = R.from_euler('z', alpha, degrees=True).as_matrix()

        # rotations on vx, vy, vz from Vlos (DTU)
        vz12, vz23, vz31 = vx12.copy()*0, vx23.copy()*0, vx31.copy()*0
        [u, v, w] = np.dot(Rz , [vx, vy, vz]) # TODO: check if this results in the error of u/v
        [u12, v12, w12] = np.dot(Rz , [vx12, vy12, vz12])
        [u23, v23, w23] = np.dot(Rz , [vx23, vy23, vz23])
        [u31, v31, w31] = np.dot(Rz , [vx31, vy31, vz23])
            
        # rotations on vx, vy, vz from Vlos (IWES)
        vz12_pp, vz23_pp, vz31_pp = vx12.copy()*0, vx23.copy()*0, vx31.copy()*0
        [u_pp, v_pp, w_pp] = np.dot(Rz , [vx_pp, vy_pp, vz_pp]) # TODO: check if this line results in the error of u/v
        [u12_pp, v12_pp, w12_pp] = np.dot(Rz , [vx12_pp, vy12_pp, vz12_pp])
        [u23_pp, v23_pp, w23_pp] = np.dot(Rz , [vx23_pp, vy23_pp, vz23_pp])
        [u31_pp, v31_pp, w31_pp] = np.dot(Rz , [vx31_pp, vy31_pp, vz31_pp])

        # Step 4d: Vector quantities using Vlos (DTU)
        Vvec = np.sqrt(vx**2 + vy**2 + vz**2)
        Vhorz = np.sqrt(vx**2 + vy**2)

        nan_mask = np.isnan(vx) | np.isnan(vy)
        gamma = np.nan * np.ones(vx.shape)
        gamma[~nan_mask] = (np.rad2deg(np.arctan2(vx[~nan_mask], vy[~nan_mask])) + 90) % 360
        psi = np.rad2deg(np.arctan2(vz,Vhorz))
        Vhorz12 = np.sqrt(vx12**2 + vy12**2)
        Vhorz23 = np.sqrt(vx23**2 + vy23**2)
        Vhorz31 = np.sqrt(vx31**2 + vy31**2)

        # for post-processed using Vlos(IWES)
        Vvec_pp = np.sqrt(vx_pp**2 + vy_pp**2 + vz_pp**2)
        Vhorz_pp = np.sqrt(vx_pp**2 + vy_pp**2)
        nan_mask = np.isnan(vx_pp) | np.isnan(vy_pp)
        gamma_pp = np.nan * np.ones(vx_pp.shape)
        gamma_pp[~nan_mask] = (np.rad2deg(np.arctan2(vx_pp[~nan_mask], vy_pp[~nan_mask])) + 90) % 360
        psi_pp = np.rad2deg(np.arctan2(vz_pp,Vhorz_pp))
        Vhorz12_pp = np.sqrt(vx12_pp**2 + vy12_pp**2)
        Vhorz23_pp = np.sqrt(vx23_pp**2 + vy23_pp**2)
        Vhorz31_pp = np.sqrt(vx31_pp**2 + vy31_pp**2)

        # Step 5: Projection of SRWS points on a grid
        from FnFindNearestGridPts import FnFindNearestGridPts
        xlg, ylg, zlg = xrot1+125, yrot1, zrot1
        Nx, Ny, Nz = 3, 11, 11
        # longitudinal dir, (row vector), [m]
        X = np.linspace(120, 130, Nx)
        # lateral direction, (row vector), [m]
        Y = np.linspace(-125, +125, Ny)
        # grid vertical direction (row vector), [m]
        Z = np.linspace(0, 250, Nz)
        _, _, _, _, _, k = FnFindNearestGridPts(X, Y, Z, xlg, ylg, zlg, grid=True)

        # errors in following the scan pattern
        # deviation in commanded and actual Theta, phi and ranges
        dT1, dT2, dT3 = thetao1 - theta1, thetao2 - theta2, thetao3 - theta3
        dP1, dP2, dP3 = phio1 - phi1, phio2 - phi2, phio3 - phi3
        dR1, dR2, dR3 = Ro1 - R1, Ro2 - R2, Ro3 - R3

        # mean following error in Range, theta and phi
        from scipy.stats import circmean
        fe_R = (dR1 + dR2 + dR3)/3
        fe_T = pd.DataFrame(np.transpose([dT1, dT2, dT3]), columns=['dT1', 'dT2', 'dT3']).apply(circmean, axis=1)
        fe_P = pd.DataFrame(np.transpose([dP1, dP2, dP3]), columns=['dT1', 'dT2', 'dT3']).apply(circmean, axis=1)

        # adding sum of separation
        # triple intersection of circles in space and sum of separation of beams
        p1 = np.array((xrot1, yrot1, zrot1))
        p2 = np.array((xrot2, yrot2, zrot2))
        p3 = np.array((xrot3, yrot3, zrot3))
        from FnEffProbeVolume import FnEffProbeVolume
        dZ_eff, dZ, dL, Leff = FnEffProbeVolume(p1, p2, p3)

        # append variables to existing dataframe for an elaborate dataframe
        listTuples = list(zip(vx, vy, vz, Vvec, Vhorz, gamma, psi, x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                xrot1, xrot2, xrot3, yrot1, yrot2, yrot3, zrot1, zrot2, zrot3,
                                theta1_rot, theta2_rot, theta3_rot, phi1_rot, phi2_rot, phi3_rot,
                                theta1, theta2, theta3, phi1, phi2, phi3, R1, R2, R3, 
                                vx12, vy12, Vhorz12, vx23, vy23, Vhorz23, vx31, vy31, Vhorz31,
                                vx_pp, vy_pp, vz_pp, vx12_pp, vy12_pp, vx23_pp, vy23_pp, vx31_pp, vy31_pp,
                                Vvec_pp, Vhorz_pp, gamma_pp, psi_pp, Vhorz12_pp, Vhorz23_pp, Vhorz31_pp,                                
                                fe_R, fe_T, fe_P, dZ_eff, dZ[0], dZ[1], dZ[2], dL, Leff,
                                u, v, w, u12, v12, w12, u23, v23, w23, u31, v31, w31,
                                u_pp, v_pp, w_pp, u12_pp, v12_pp, w12_pp, u23_pp, v23_pp, w23_pp, u31_pp, v31_pp, w31_pp,
                                ))
        nameTuples = ['vx', 'vy', 'vz', 'Vvec', 'Vhorz', 'gamma', 'psi', 'xs1', 'ys1', 'zs1', 'xs2', 'ys2', 'zs2', 'xs3', 'ys3', 'zs3',
                        'xr1', 'xr2', 'xr3', 'yr1', 'yr2', 'yr3', 'zr1', 'zr2', 'zr3',
                        'thetar1', 'thetar2', 'thetar3','phir1', 'phir2', 'phir3',
                        'theta1', 'theta2', 'theta3', 'phi1', 'phi2', 'phi3', 'R1', 'R2', 'R3',
                        'vx12', 'vy12', 'Vhorz12', 'vx23', 'vy23', 'Vhorz23', 'vx31', 'vy31', 'Vhorz31',
                        'vx_pp', 'vy_pp', 'vz_pp', 'vx12_pp', 'vy12_pp', 'vx23_pp', 'vy23_pp', 'vx31_pp', 'vy31_pp',                                
                        'Vvec_pp', 'Vhorz_pp', 'gamma_pp', 'psi_pp', 'Vhorz12_pp', 'Vhorz23_pp', 'Vhorz31_pp',                                
                        'fe_R', 'fe_T', 'fe_P',  'dZ_eff', 'dZ1', 'dZ2', 'dZ3', 'dL', 'Leff',
                        'u', 'v', 'w', 'u12', 'v12', 'w12', 'u23', 'v23', 'w23', 'u31', 'v31', 'w31',
                        'u_pp', 'v_pp', 'w_pp', 'u12_pp', 'v12_pp', 'w12_pp', 'u23_pp', 'v23_pp', 'w23_pp', 'u31_pp', 'v31_pp', 'w31_pp',
                        ]
        wfr_all = pd.DataFrame(listTuples, columns=nameTuples).reset_index(drop=True)
        Data = pd.concat([Data_raw, wfr_all], axis=1).set_index('index', drop=True)
        Data = Data.astype(dtype=Data.dtypes.replace('float64', 'float32'))
        # Data.index = Data.index.tz_convert('UTC')

        # Step 6: mean, std, min and max of the dataframe
        if inp.generate_stats:
            circ_cols = ['gamma', 'gamma_pp', 'psi', 'psi_pp', 'thetar1', 'thetar2', 'thetar3','phir1', 
                        'phir2', 'phir3', 'theta1', 'theta2', 'theta3', 'phi1', 'phi2', 'phi3','fe_T', 'fe_P']
            nondefault_cols = ['gamma', 'gamma_pp']
            Data.update(Data.filter(regex="gamma|psi").apply(np.deg2rad))
            Data_mean = pd.DataFrame(pa.multiple_stats_on_df(Data, circular_data_cols=circ_cols, circ_pi2pi=nondefault_cols)).T 
            Data_std = pd.DataFrame(pa.multiple_stats_on_df(Data, circular_data_cols=circ_cols, stats='std', circ_pi2pi=nondefault_cols)).T 
            Data_min =  pd.DataFrame(Data.apply(np.nanmin)).T
            Data_max =  pd.DataFrame(Data.apply(np.nanmax)).T

            # assigning the datetime index to the dataframes
            Data_mean['Tavg'] = pd.to_datetime(os.path.split(filename)[-1])
            Data_mean = Data_mean.reset_index(drop=True).set_index('Tavg', drop=True)
            Data_std.index = Data_mean.index
            Data_min.index = Data_mean.index
            Data_max.index = Data_mean.index

            self.write_srws2csv(Data_mean, csv_name='srws_data_mean.csv')
            self.write_srws2csv(Data_min,csv_name='srws_data_min.csv')
            self.write_srws2csv(Data_std, csv_name='srws_data_std.csv')
            self.write_srws2csv(Data_max, csv_name='srws_data_max.csv')


        # # write variables to a dataframe
        listTuples = list(zip(vx, vy, vz, Vvec,  Vhorz, gamma, psi, vx12, vy12,
                            Vhorz12, vx23, vy23, Vhorz23, vx31, vy31, Vhorz31,
                            vx_pp, vy_pp, vz_pp, vx12_pp, vy12_pp, vx23_pp, vy23_pp, vx31_pp, vy31_pp,
                            Vvec_pp, Vhorz_pp, gamma_pp, psi_pp, Vhorz12_pp, Vhorz23_pp, Vhorz31_pp,                                
                            fe_R, fe_T, fe_P, dZ_eff, dZ[0], dZ[1], dZ[2], dL, Leff,
                            u, v, w, u12, v12, w12, u23, v23, w23, u31, v31, w31,
                            u_pp, v_pp, w_pp, u12_pp, v12_pp, w12_pp, u23_pp, v23_pp, w23_pp, u31_pp, v31_pp, w31_pp,
                            Data_raw['dt 1'], Data_raw['dt 2'], Data_raw['dt 3'], Data_raw.qc1, Data_raw.qc2))
        nameTuples = ['vx', 'vy', 'vz','Vvec', 'Vhorz', 'gamma', 'psi', 'vx12', 'vy12', 'Vhorz12',
                        'vx23', 'vy23', 'Vhorz23', 'vx31', 'vy31', 'Vhorz31',
                        'vx_pp', 'vy_pp', 'vz_pp', 'vx12_pp', 'vy12_pp', 'vx23_pp', 'vy23_pp', 'vx31_pp', 'vy31_pp',                                
                        'Vvec_pp', 'Vhorz_pp', 'gamma_pp', 'psi_pp', 'Vhorz12_pp', 'Vhorz23_pp', 'Vhorz31_pp', 
                        'fe_R', 'fe_T', 'fe_P', 'dZ_eff', 'dZ1', 'dZ2', 'dZ3', 'dL', 'Leff',
                        'u', 'v', 'w', 'u12', 'v12', 'w12', 'u23', 'v23', 'w23', 'u31', 'v31', 'w31',
                        'u_pp', 'v_pp', 'w_pp', 'u12_pp', 'v12_pp', 'w12_pp', 'u23_pp', 'v23_pp', 'w23_pp', 'u31_pp', 'v31_pp', 'w31_pp',
                        'dt1', 'dt2', 'dt3', 'qc1', 'qc2']
        wfr = pd.DataFrame(listTuples, columns=nameTuples).reset_index(drop=True)
        wfr = wfr.set_index(Data.index)
        wfr = wfr.astype(dtype=wfr.dtypes.replace('float64', 'float32'))

        # # filter the data corresponding to the hub height
        hubcentre = [125, 0, 125]
        idx_hub = np.where(np.sum(k == hubcentre, axis=1) == 3)
        if not idx_hub[0].any():
            mean_pt = [pa.find_nearest(X,xlg.mean()), pa.find_nearest(Y,ylg.mean()), pa.find_nearest(Z,zlg.mean())]
            idx_hub = np.where(np.sum(k == mean_pt, axis=1) == 3)
        df_H = wfr.iloc[np.ravel(idx_hub), :].set_index(Data.index[np.ravel(idx_hub)])

        # filter the data corresponding to hubheight across the rotor
        # hubH = [125,:,125]
        # idx_hub = np.where(np.sum(k == hubH, axis=1)==2)

        # # write the mean for the whole pattern # TODO apply specific mean for mean wind speeds and wind direction (discrepency for values like 350° and 10° cases)
        circ_cols = wfr.filter(regex="gamma|psi|fe_(P|T)").columns
        nondefault_cols = ['gamma', 'gamma_pp']
        wfr.update(wfr.filter(regex="gamma|psi").apply(np.deg2rad))
        df_bowtie = pd.DataFrame(pa.multiple_stats_on_df(wfr, circular_data_cols=circ_cols, circ_pi2pi=nondefault_cols)).T 
        df_bowtie['Tavg'] = pd.to_datetime(os.path.split(filename)[-1])
        df_bowtie = df_bowtie.set_index(['Tavg'])

        # # write the mean of the parameters to a file
        df_H.update(df_H.filter(regex="gamma|psi").apply(np.deg2rad))
        df_m = pd.DataFrame(pa.multiple_stats_on_df(df_H, circular_data_cols=circ_cols, circ_pi2pi=nondefault_cols)).T
        df_m['Tavg'] = pd.to_datetime(os.path.split(filename)[-1])
        df_m = df_m.set_index(['Tavg'])

        # Step 7: write the dataframes to csv
        if inp.write_csv == True:
            self.write_srws2csv(df_m, csv_name='srws_hubH_mean_ID6362.csv')
            self.write_srws2csv(df_bowtie,csv_name='srws_avg_bowtie_ID6362.csv')
            self.write_srws2csv(df_H, csv_name='srws_data_hubH_ID6362.csv')

        return Data, df_H, df_bowtie, df_m, Data_mean, Data_std, Data_min, Data_max, spec_df

    @staticmethod
    def add_metadata(files):
        """
        add metadata generated for SRWS
        input: srws file for some details, and some details from the HighRe measurement campaign
        use the output ds to update another ds to add metadata
        """
        import pythonAssist as pa
        import json
        import xarray as xr
        from sqlalchemy import (Table, Column, Integer, String, ForeignKey, MetaData, insert, create_engine, Enum, \
            Float, select, Boolean, DateTime, Interval, ARRAY, )
        from sqlalchemy.orm import (relationship, declarative_base, registry, scoped_session, sessionmaker, Session, backref, mapped_column)
        from sqlalchemy.ext.serializer import loads, dumps
        # from enumerations import *
        from metadata_helpers import (get_CampaignInfo, get_guiData, as_dict, json_serialize, qc_flags_basic)
        from metadata_tables import (DatasetInfo, DataFile, UserInterface, Network, Site, WindFarm, InstrumentSpecs, LidarConfig, AggQualityControl, DataAccessLayer)
        from datetime import datetime
        dateformat = "%Y-%m-%dT%H:%M:%SZ"

        if np.size(files) > 1:
            file = files[0]
            meta_gui2 = get_guiData(files[-1])
        else:
            file = files[0]

        meta_gui,_ = get_guiData(file)
        meta_campaign = get_CampaignInfo() # duration and resolution not correct
        meta_dsi = DatasetInfo(
            url="zdrive_highre",
            webpage = "zdrive",
            name = "highre",
            cdm_data_type="trajectory",
            subject = "wind_3d",
            description = "Measurement of 3D wind fields in front of Adwen AD8-180 wind turbine at Testfeld BHV, measurement performed by Fraunhofer IWES in cooperation with DTU between October 2021 and June 2022",
            references = "Wind speed reconstruction from three synchronized short-range WindScanner lidars in a large wind turbine inflow field campaign and the associated uncertainties",
            doi = "10.1088/1742-6596/2265/2/022032",
            contributor_name = "ashim",
            contributor_role = "Data analysis",
            format="nc",
        )
        meta_ui = UserInterface(
            project_name = "highre",
            system_name = "zdrive",
            url = "zdrive_highre",
            access = "confidential",
            start_datetime = meta_gui.start_datetime[0].to_pydatetime(),
            end_datetime = meta_gui.end_datetime[0].to_pydatetime(),
            file_granularity = "minutes",
            sample_rate = "20 Hz",
            file_format_version = "0.1",
            samples_per_file = meta_gui.samples_per_file[0].to_pydatetime(),
            files_per_day = 144,
            project_id = 321012,
        )
        meta_net = Network(
            daylight_saving = True,
            data_trans_freq = "daily"
        )

        meta_dataf = DataFile(
            date_created = meta_gui.date_created[0],
            date_modified =meta_gui.date_modified[0],
            time_coverage_start = meta_campaign.time_coverage_start[0][0],
            time_coverage_end = meta_campaign.time_coverage_end[0][0],
            time_coverage_duration = meta_campaign.time_coverage_duration[0],
            time_coverage_resolution = meta_campaign.time_coverage_resolution[0],
            project = "highre",
            program = "Bowtie1",
            title = "radial velocities and spectrum data from three synchronized SRWS",
        )
        meta_site = Site(
            geospatial_bounds = "polygon((8.575881, 53.501206, (8.579721, 53.501168, (8.580494, 53.506630, (8.573821, 53.505265))",
            geospatial_lat_min = "53.501168",
            geospatial_lat_max = "53.506630",
            geospatial_lon_min = "8.573821",
            geospatial_lon_max = "8.580494",
            geospatial_vertical_min = 25,
            geospatial_vertical_max = 225,
            main_wind_direction_min = "240 degrees",
            main_wind_direction_max = "280 degrees",
            weibull_param_scale = 8.26,
            weibull_param_shape = 2.0,
            )
        meta_wf = WindFarm(
            id = "TBHV02",
            wt_rated_power = "8000 kW",
            wt_longitude = "8.579003",
            wt_latitude = "53.505657",
            wt_oem = "SGRE",
        )
        meta_srws = InstrumentSpecs(
            sensor_id = "srws_v2",
            manufacturer = "dtu",
            make_model = "R2D1(WS1), R2D2(WS2) and R2D3(WS3)",
            model = "2.0",
            size = [1.5,1.5,1.65],
            weight = 100,
            type = "lidar",
            operation = "srws",
            power = "230V AC, 16A 1000W",
            total_number = 3,
            short_name = "srws",
            long_name = "short-range continuous-wave WindScanners 6-inch",
            safety_check="safe for operation with special considerations (see report)",
            measurement_station = "onshore",
        )
        meta_lidcfg = LidarConfig(
            wavelength = "1565 nm",
            Pmax = "1000 mW",
            prism_angle = "30 deg",
            focus_min = "8 m",
            focus_max = "300 m",
            laser_safety_class = "lsk4",
            lidar_type = "cw",
            focal_length = "580 mm",
            aperture_radius_lens = "56 mm",
            beam_waist = "0.88 mm",
            spectra_N = 512,
            pulse_duration = None,
            range_max = None,
            aperture_diameter_telescope= "6 in",
        )
        meta_aqc = AggQualityControl(
            flag_values=qc_flags_basic(not_checked=1)[0],
            flag_masks=['good'],
            flag_meanings=qc_flags_basic(not_checked=1)[1]
        )
        
        meta_objs = [meta_dsi, meta_ui, meta_net, meta_dataf, meta_site, meta_wf, meta_srws, meta_lidcfg, meta_aqc]

        dal = DataAccessLayer()
        dal.connect()
        dal.session = dal.Session()
        # required if the database does not exist or not in the os.getcwd() folder
        dal.session.add_all(meta_objs)
        # dal.session.flush()
        dal.session.commit()
        query = select(DatasetInfo,UserInterface , Network,DataFile,Site,WindFarm, InstrumentSpecs,LidarConfig,AggQualityControl)
        ResultProxy = dal.session.execute(query)
        ResultSet = ResultProxy.fetchall()
        dal.session.close()

        ds = xr.Dataset()
        for r in ResultSet[0]:
            dct = as_dict(r)
            json_obj = json.dumps(dct, indent=4, default=lambda x: json_serialize(x))
            ds.attrs[r.__tablename__] = json_obj
        
        return ds

    def write_srws2csv(self, df, csv_name):
        """ 
        write srws data to csv within the working directory by creating a data/csv folder
        input: df - dataframe, csv_name - name of the csv file where the df should be appended
        output: appends df to the csv file located at workDir/data/csv/csv_name
        """
        path = os.path.join(self.inp.target_path,'data', 'Bowtie1', 'csv', csv_name)
        if os.path.exists(path):
            df.to_csv(path, mode='a',header=False, float_format="%.4f", index_label='datetime')
        else:
            df.to_csv(path, mode='w',header=True, float_format="%.4f", index_label='datetime')

    def FnConvertRawData(self, inp):
        r"""
        FUNCTION_NAME - function to import iSpin data from the provided path for raw/10 min data within the interval (tstart, tend)
        
        Syntax:  data_ispin = FnImportIspin(path,type, tstart, tend)
        
        Inputs:
        inp - input data structure with suboptions coord, path regStr, merge
        inp.path - path the folder where the files are stored
        inp.regStr - regexp for file selection in a folder
        inp.coord - coordinates file for CampaignConstants definition in ProcessSRWS.py
        
        Outputs:
        Data, Data_mean, Data_std, Data_min, Data_max, df_H, df_bowtie, df_m
            Data - All the data read directly
            Data_mean - Mean of the data read for each file read default: 1 min
            Data_std - Std dev of the data read for each file read default: 1 min
            Data_min - Minumum of the data read for each file read default: 1 min
            Data_max - maximum of the data read for each file read default: 1 min
            df_H - Pandas dataframe for hubheight time series
            df_bowtie - Pandas dataframe for average over the complete scan period
            df_m - average for hubheight time series i.e. average of df_H
    
        Example:
        # set path
            import sys
            sys.path.append("../../userModules")
            sys.path.append("../fun")
            import pythonAssist as pa
            inp = pa.struct()
            inp.srws = pa.struct()
            inp.srws.path.root = r"z:\Projekte\112933-HighRe\20_Durchfuehrung\OE410\SRWS\Data\Bowtie1_unzipped"
            inp.srws.coord = r"C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\HighRe\src\BREMERHAVEN TOT.txt"
        # get the size and other details of files
            inp.srws.regStr = "*2022-04-26T181[0|1]*[!.zip][!.txt]"
            Data, Data_mean, Data_std, Data_min, Data_max, df_H, df_bowtie, df_m = FnConvertRawData(inp, inp.srws.regStr, inp.srws.coord)
        
        modules required: see above in the preamble
        classes required: ProcessSRWS, FileOperations, 
        Data-files required: none

        See also: ProcessSRWS,  FnWsRange, FileOperations

        References:
            Author: Ashim Giyanani, Research Associate 
            Fraunhofer Institute of Wind Energy 
            Windpark planning and operation department
            Am Seedeich 45, Bremerhaven 
            email: ashim.giyanani@iwes.fraunhofer.de
            Git site: https://gitlab.cc-asp.fraunhofer.de/giyash/testfeld-bhv.git  
            Created: 06-08-2020; Last revision: 2022-08-31 17:14:48
        """

    #------------- BEGIN CODE --------------
        from dateutil import parser
        from joblib import Parallel, delayed
        import pythonAssist as pa

        # Step 1: Get measurement campaign information
        campaign_constants = self.CampaignConstants(inp.srws.coord, inp.plot_figure)

        # read the number of files to be read, unzip if necessary
        Nfiles = 0
        files = glob.glob(os.path.join(inp.srws.path.root, inp.srws.regStr), recursive=True)

        # Initialization
        DATA, Data_Mean, Data_Std, Data_Max = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        Data_Min, DF_H, DF_bowtie, DF_m, spec_DF = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(), pd.DataFrame()
        data_raw=pd.DataFrame()
        data_hub = pd.DataFrame()
        data_1min = pa.struct()
        base_path = os.path.join(self.inp.target_path, 'data', 'Bowtie1', 'parquet')

        for file in sorted(files[:]):
            filename, _ = os.path.splitext(file)
            files_tstart = parser.parse(os.path.basename(file)[-20:])
            try:
                Nfiles += 1
                Data, df_H, df_bowtie, df_m, Data_mean, Data_std, Data_min, Data_max, spec_df, meta_ds =  self.srws2df(inp, file, campaign_constants, Nfiles)
                print('[{0}]: Files {1}/{2} {3} completed'.format(pa.now(), Nfiles, len(files), filename[-20:]))
            except OverflowError:
                Nfiles += 1
                print('[{0}]: File {1}/{2} {3} skipped (OverflowError)'.format(pa.now(), Nfiles, len(files), filename[-20:]))
            # Concatenate the last 10 files
            if inp.srws.merge == True:  # dataframe concatenate yes=True, no=False
                DATA = pd.concat([DATA, Data])
                Data_Mean = pd.concat([Data_Mean, Data_mean])
                Data_Std = pd.concat([Data_Std, Data_std])
                Data_Min = pd.concat([Data_Min, Data_min])
                Data_Max = pd.concat([Data_Max, Data_max])
                DF_H = pd.concat([DF_H, df_H])
                DF_bowtie = pd.concat([DF_bowtie, df_bowtie])
                DF_m = pd.concat([DF_m, df_m])
                spec_DF = pd.concat([spec_DF, spec_df])
            elif (inp.write_parquet == True) & (inp.srws.merge==False):
                # base_path = os.path.join(self.inp.workDir, 'data', 'parquet')
                filepath = os.path.join(f'{base_path}', f'data_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.parq')
                # write to parquet files
                pa.write_parquet(filepath, Data)
                pa.write_fastparquet(os.path.join(f'{base_path}', f'df_H.parq'), df_H)
                pa.write_fastparquet(os.path.join(f'{base_path}', f'df_bowtie.parq'), df_bowtie)
                pa.write_fastparquet(os.path.join(f'{base_path}', f'df_m.parq'), df_m)
                # rename to standardize
                DATA = Data.copy()
                DF_H = df_H.copy()
                DF_bowtie = df_bowtie.copy()
                DF_m = df_m.copy()

            # write to ************************parquet files**************************************
            pa.write_parquet(filepath, DATA)
            pa.write_fastparquet(os.path.join(f'{base_path}', f'df_H_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.parq'), DF_H)
            pa.write_fastparquet(os.path.join(f'{base_path}', f'df_bowtie_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.parq'), DF_bowtie)
            pa.write_fastparquet(os.path.join(f'{base_path}', f'df_m_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.parq'), DF_m)

            # write to ************************netCDF files***************************************
            # convert to a mult-index dataframe
            from metadata_helpers import df2mi
            import xarray as xr
            # reg expression to match columns that ^(?!qc) do not match qc at the start, 
            # \s? has zero or one nr. of whitespaces, \D with any nonDigit, {} with a number 1,2,3,4 using a for loop, 
            # r?$ has zero or one nr. of 'r' at the end
            regStr = "^(?!qc).*\s?\D{}r?$"
            indexStr = 'srws_nr{}'
            nc_base_path = os.path.join(self.inp.target_path, 'data', 'Bowtie1', 'netcdf')

            _, _, ds_raw = df2mi(DATA, regStr, indexStr)
            _, _, ds_H = df2mi(DF_H, regStr, indexStr)
            _, _, ds_bowtie = df2mi(DF_bowtie, regStr, indexStr)
            _, _, ds_m = df2mi(DF_m, regStr, indexStr)

        # combine dataframes into structs
        # data_raw = DATA
        # data_hub = DF_H
        # data_1min = pa.struct()
        # data_1min.mean = Data_Mean
        # data_1min.std = Data_Std
        # data_1min.min = Data_Min
        # data_1min.max = Data_Max
        # data_1min.bowtie = DF_bowtie
        # data_1min.hub_mean = DF_m

        # with Parallel(n_jobs=6) as parallel:
        #     delayed_funcs = [delayed(self.srws2df)(inp, file, campaign_constants, Nfiles) for file in sorted(files[:])]
        #     run_df = parallel(delayed_funcs)
        #     DATA, DF_H, data_1min, data_raw, data_hub, DF_bowtie, DF_m = zip(*run_df)

        # print Statements    
        # print('[{1}]: Saved all data to {0}'.format(path_Data, pa.now()))
        # print('[{1}]: Saved data at hub height to {0}'.format(path_df_H, pa.now()))
        # print('[{1}]: Saved data averaged over the scan pattern to {0}'.format(path_df_bowtie, pa.now()))
        # print('[{1}]: Saved all data to {0}'.format(path_df_m, pa.now()))

        if (inp.write_parquet == True) & (inp.srws.merge==True):
            pa.write_parquet(os.path.join(self.inp.target_path,'data', 'Bowtie1','parquet',f'data_raw.parq'), data_raw)
            # pa.write_parquet(os.path.join(self.inp.target_path,'data','parquet',f'data_1min_mean.parq'), data_1min.mean)
            pa.write_parquet(os.path.join(self.inp.target_path,'data', 'Bowtie1','parquet',f'data_hub.parq'), data_hub)
            print('[{0}]: Saved srws to parquet files'. format(pa.now()))

            # adding metadata to the srws files
            if self.inp.add_metadata:
                ds_raw = ProcessSRWS.add_metadata_srws(ds_raw, nc_path=os.path.join(f"{nc_base_path}", f'ds_raw_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.nc'))
                ds_H = ProcessSRWS.add_metadata_srws(ds_H, nc_path=os.path.join(f"{nc_base_path}", f'ds_H_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.nc'))
                ds_bowtie = ProcessSRWS.add_metadata_srws(ds_bowtie, nc_path=os.path.join(f"{nc_base_path}", f'ds_bowtie_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.nc'))
                ds_m = ProcessSRWS.add_metadata_srws(ds_m, nc_path=os.path.join(f"{nc_base_path}", f'ds_m_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.nc'))
                meta_ds = self.add_metadata(files)

                ds_raw.attrs = meta_ds.attrs
                ds_H.attrs = meta_ds.attrs
                ds_bowtie.attrs = meta_ds.attrs
                ds_m.attrs = meta_ds.attrs
        #------------- END OF CODE ------------

        return DATA, DF_H, DF_bowtie, DF_m, ds_raw, ds_H, ds_bowtie, ds_m 

    def mp_FnConvertRawData(self, inp):
        """multiprocessing function for the same function above"""
        from dateutil import parser
        from joblib import Parallel, delayed
        import pythonAssist as pa
        from datetime import timezone
        import logging
        import traceback
        from FileOperations import FileOperations

        # Step 1: Get measurement campaign information
        campaign_constants = self.CampaignConstants(inp.srws.coord, inp.plot_figure)

        # Initialization
        DATA, DF_H, DF_bowtie, DF_m = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        data_raw=pd.DataFrame()
        data_hub = pd.DataFrame()
        data_1min = pa.struct()

        # read the number of files to be read, unzip if necessary
        print(f'[{pa.now()}]: Reading directories for files')
        Nchunks, Nfiles = 0, 0
        base_path = os.path.join(inp.target_path, 'data', 'Bowtie1', 'parquet')
        all_files = glob.glob(os.path.join(inp.srws.path.root, inp.srws.regStr), recursive=True)
        # remove completed files from the files to be converted
        completed_files_duobook = glob.glob(os.path.join(base_path, 'data*.parq'))
        completed_files_onedrive = glob.glob(os.path.join(inp.workDir, 'data', 'Bowtie1', 'parquet', 'data*.parq'))
        completed_files = completed_files_duobook + completed_files_onedrive
        completed_dates = [os.path.splitext(os.path.basename(cf).split('_')[1])[0] for cf in completed_files]
        with open(inp.srws.path.finished_files, 'r') as fp:
            finished_dates = fp.read().splitlines()
        finished_dates = [pd.to_datetime(fd, utc=True).strftime("%Y%m%dT%H%M%S%Z%z") for fd in finished_dates]
        completed_dates = sorted(set(completed_dates+finished_dates))
        # remove files from the list of srws files that had some error in the past, these files will be skipped
        with open(inp.srws.path.error_files, 'r') as fp:
            error_dates = fp.read().splitlines()[1:]
        error_dates = [pd.to_datetime(e, utc=True).strftime("%Y%m%dT%H%M%S%Z%z") for e in error_dates]
        all_files = sorted(all_files[:])
        print(f'[{pa.now()}]: Reading directories for files completed')

        # # special lines
        # for f in all_files[:]:
        #     if (pd.to_datetime(os.path.basename(f), utc=True) <= pd.to_datetime("2021-10-25T000000+02", utc=True)):
        #         all_files.remove(f)
        #         print(f"files before HighRe campaign: {f}")
        #     elif pd.to_datetime(os.path.basename(f), utc=True).strftime("%Y%m%dT%H%M%S%Z%z") in completed_dates:
        #         all_files.remove(f)
        #         print(f"removing completed files: {f}")
        #     elif pd.to_datetime(os.path.basename(f), utc=True).strftime("%Y%m%dT%H%M%S%Z%z") in error_dates:
        #         all_files.remove(f)
        #         print(f"removing error files: {f}")

        roll_chunks = 1 # user input
        chunks = [all_files[i:i+roll_chunks] for i in range(0,len(all_files),roll_chunks)]
        if len(all_files) <= 20:
            Njobs = 1
        else:
            Njobs = 6

        for files in chunks:
            Nchunks += 1
            try:
                with Parallel(n_jobs=Njobs, timeout=1800) as parallel:
                    delayed_funcs = [delayed(self.srws2df)(inp, file, campaign_constants, Nfiles) for file in sorted(files[:])]
                    run_df = parallel(delayed_funcs)
                    # Data_chunk, df_H_chunk, df_bowtie_chunk, df_m_chunk, Data_mean_chunk, Data_std_chunk, Data_min_chunk, Data_max_chunk = zip(*run_df)
                    print(f'[{pa.now()}]: Chunk {Nchunks}/{len(chunks)} {files[-1][-20:]} completed')

                # TODO convert this into a generator output, will need a generator function
                for i, df_i in enumerate(run_df):
                    files_tstart = parser.parse(os.path.basename(files[i])[-20:]).astimezone(timezone.utc)
                    filepath = os.path.join(f'{base_path}', f'data_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.parq')
                    data_raw, df_H, df_bowtie, df_m, _, _, _, _, _ = zip(df_i)

                    if inp.srws.merge == True:
                        DATA = pd.concat([DATA, data_raw[0]])
                        DF_H = pd.concat([DF_H, df_H[0]])
                        DF_bowtie = pd.concat([DF_bowtie, df_bowtie[0]])
                        DF_m = pd.concat([DF_m, df_m[0]])
                    else:
                        DATA = data_raw[0]
                        DF_H = df_H[0]
                        DF_bowtie = df_bowtie[0]
                        DF_m = df_m[0]

                    if self.inp.write_parquet:
                        # write to ************************parquet files**************************************
                        pa.write_parquet(filepath, DATA)
                        pa.write_fastparquet(os.path.join(f'{base_path}', f'df_H_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.parq'), DF_H)
                        pa.write_fastparquet(os.path.join(f'{base_path}', f'df_bowtie_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.parq'), DF_bowtie)
                        pa.write_fastparquet(os.path.join(f'{base_path}', f'df_m_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.parq'), DF_m)

                    # write to ************************netCDF files***************************************
                    # convert to a mult-index dataframe
                    # from metadata_helpers import df2mi
                    # import xarray as xr
                    # reg expression to match columns that ^(?!qc) do not match qc at the start, 
                    # \s? has zero or one nr. of whitespaces, \D with any nonDigit, {} with a number 1,2,3,4 using a for loop, 
                    # r?$ has zero or one nr. of 'r' at the end
                    # regStr = "^(?!qc).*\s?\D{}r?$"
                    # indexStr = 'srws_nr{}'
                    # nc_base_path = os.path.join(self.inp.target_path, 'data', 'Bowtie1', 'netcdf')

                    # _, _, ds_raw = df2mi(DATA, regStr, indexStr)
                    # _, _, ds_H = df2mi(DF_H, regStr, indexStr)
                    # _, _, ds_bowtie = df2mi(DF_bowtie, regStr, indexStr)
                    # _, _, ds_m = df2mi(DF_m, regStr, indexStr)

                    # adding metadata to the srws files
                    if self.inp.add_metadata:
                        nc_base_path = os.path.join(self.inp.target_path, 'data', 'Bowtie1', 'netcdf')
                        ds_raw = ProcessSRWS.add_metadata_srws(ds_raw, nc_path=os.path.join(f"{nc_base_path}", f'ds_raw_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.nc'))
                        ds_H = ProcessSRWS.add_metadata_srws(ds_H, nc_path=os.path.join(f"{nc_base_path}", f'ds_H_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.nc'))
                        ds_bowtie = ProcessSRWS.add_metadata_srws(ds_bowtie, nc_path=os.path.join(f"{nc_base_path}", f'ds_bowtie_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.nc'))
                        ds_m = ProcessSRWS.add_metadata_srws(ds_m, nc_path=os.path.join(f"{nc_base_path}", f'ds_m_{files_tstart.strftime("%Y%m%dT%H%M%S%Z%z")}.nc'))
                        meta_ds = self.add_metadata(files)

                        ds_raw.attrs = meta_ds.attrs
                        ds_H.attrs = meta_ds.attrs
                        ds_bowtie.attrs = meta_ds.attrs
                        ds_m.attrs = meta_ds.attrs
                    else:
                        ds_raw = None
                        ds_H = None
                        ds_bowtie = None
                        ds_m = None

                # add reconstructed files to finished_files.txt
                FileOperations.write_paths_to_file(inp.srws.path.finished_files, [os.path.basename(f) for f in files])
                print(f"[{pa.now()}]: Adding {os.path.basename(files[0])}-{os.path.basename(files[-1])} to finished_files.txt")
            
                # del Data, df_H, df_bowtie, df_m 
            except Exception as e:
                FileOperations.write_paths_to_file(inp.srws.path.error_files, [os.path.basename(f) for f in files])
                print(f"[{pa.now()}]: Adding {os.path.basename(files[0])}-{os.path.basename(files[-1])} to error_files.txt")
                logging.error(traceback.format_exc())
                DATA, DF_H, DF_bowtie, DF_m, ds_raw, ds_H, ds_bowtie, ds_m = None, None, None, None, None, None, None, None

        return DATA, DF_H, DF_bowtie, DF_m, ds_raw, ds_H, ds_bowtie, ds_m

    @staticmethod
    def create_nc_groups(ds, groups, group_vars, nc_path):

        nc_groups = dict(zip(groups, group_vars))
        ds.drop_vars(ds.data_vars).to_netcdf(nc_path, "w", format="NETCDF4", engine="netcdf4")
        for g in groups:
            temp_ds = ds[nc_groups[g]]
            temp_ds = temp_ds.fillna(np.nan).astype('float32')
            temp_ds.to_netcdf(nc_path, "a", group=g, format="NETCDF4", engine="netcdf4")

        return None

    @staticmethod
    def add_metadata_srws(ds, nc_path, grouping=True):
        """
        Add metadata to the xarray dataset
        """

        import pandas as pd
        from DataQualityFlags import quality_control as qc
        import ast
        from ProcessSRWS import ProcessSRWS



        # read in the metadata from the excel file
        metadata_xl_path = r"../srws_trial_run/Metadata.xlsx"
        sheet="Variables" 
        srws_variables = pd.read_excel(metadata_xl_path, sheet_name=sheet)

        # the ranges in excel are read as string, string_eval converts them from string to list type
        srws_variables['valid_range'] = srws_variables.valid_range.apply(qc.string_eval)
        # using set theory to remove some useless variables
        valid_variables = (ds.variables.mapping.keys()-set({ky for ky in ds.dims.keys()}))
        for v in sorted(valid_variables):
            idx = srws_variables[srws_variables.name==v].index
            flg = qc.qc_range_testing(ds[v].median().values, srws_variables.loc[idx,'valid_range'].values[0])
            srws_variables.loc[idx, 'flag_values']=flg
            srws_variables.loc[idx, 'flag_meanings']=flg.name

        # srws_variables['flag_values'] = srws_variables.flag_values.apply(lambda x: format(x,'08'))
        srws_variables['data_type'] = srws_variables.data_type.apply(lambda x: 'np.{}'.format(x))
        for s in srws_variables.name:
            try:
                ds[s].attrs = srws_variables[srws_variables.name == s].iloc[0,:].to_dict()
            except KeyError:
                continue

        if grouping == True:
            # create groups based on input regarding the group names (groups) and variables assigned to the group (group_vars)
            # define the groups needed for dataset to be divided in
            N_ds_vars = len(ds.variables.mapping.keys())
            groups = ['dtu_raw', 'dtu_postprocessed', 'iwes_postprocessed']
            # get the group variables, if not defined already into sets
            dtu_raw_vars = [        
                'ComPosition_pm1',
                'ComPosition_pm2',
                'ComPosition_fm3',
                'ActPosition_pm1',
                'ActPosition_pm2',
                'ActPosition_fm3',
                'ActPulse',
                'Status',
                'Timestamp',
                'ID_nr',
                'AverageCount',
                'vlos',
                'MaximumValue',
                'TotalPower',
                'Qualitydata',
                'Beta',
                'LaserPowerEstimate',
                'ModeVar',
                ]
            dtu_raw_vars = set([v for v in dtu_raw_vars if v in ds.variables.mapping.keys()])             
            iwes_pp_vars = set({ky: v for ky, v in ds.variables.items() if ky.endswith('_pp')})
            dtu_pp_vars = ds.variables.mapping.keys() - iwes_pp_vars - dtu_raw_vars - set({ky for ky in ds.dims.keys()})

            group_vars = [dtu_raw_vars, dtu_pp_vars, iwes_pp_vars]
            ProcessSRWS.create_nc_groups(ds, groups, group_vars, nc_path)

        return ds 

if __name__ == "__main__":
    # when running under terminal exec(open(r'C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules\ProcessSRWS.py').read())
    import sys
    import os
    from datetime import datetime
    import matlab2py as m2p
    import time
    # time.sleep(40*60)

    from ProcessSRWS_spark import ProcessSRWS
    from FileOperations import FileOperations
    from PostProcess_sonics import PostProcess_sonics
    import pythonAssist as pa

    # input parameters
    inp = pa.struct()
    inp.srws = pa.struct()
    inp.srws_path = pa.struct()
    # inp.srws_path_root = os.path.join(workDir, 'data', 'nawea', 'srws')
    # User Input: Add here the root, the files will be searched based on the regular expression
    inp.srws_path_root= r"C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\srws_trial_run\data"
    # inp.srws_path_root= r"z:\Projekte\112933-HighRe\20_Durchfuehrung\OE410\SRWS\Data\Bowtie1\2021\11\02"
    inp.srws_coord = os.path.join(r'C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\HighRe','data','BREMERHAVEN TOT.txt')
    inp.srws_regStr = "**/*2022-04-26T1858*[!.zip][!.txt]" # 
    # inp.srws_regStr = "**/*2021-11-02T1[2-9]*[!.zip][!.txt]" # 
    inp.srws_merge = False  # dataframe concatenate yes=1, no=0
    inp.srws_relative_align = True # align the WS setup with the North (9° at Testfeld BHV)
    inp.srws_use_corrected_vlos = True
    inp.pickle = False
    inp.write_csv=False
    inp.write_parquet = True
    inp.generate_stats = False
    inp.filter_data = False
    inp.plot_figure = False
    inp.write_spec = True
    inp.workDir = os.path.dirname(sys.path[0])
    # inp.workDir = r'C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\HighRe'
    inp.tstart = datetime.strptime('2021-11-01_12-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
    inp.tend = datetime.strptime('2021-11-01_14-59-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
    inp.add_metadata = True
    inp.target_path = r"C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\srws_trial_run"
    # inp.srws_path_error_files = os.path.join(inp.workDir, r"data/Bowtie1/error_files.txt")
    # inp.srws_path_finished_files = os.path.join(inp.workDir, r"data/Bowtie1/finished_files.txt")
    # inp.srws_path_root= r"z:\Projekte\112933-HighRe\20_Durchfuehrung\OE410\SRWS\Data\Bowtie1\2021\11"
    # inp.srws_regStr = "**/*2021-11-*T*[!.zip][!.txt]" # 
    inp.srws_path_error_files = None
    inp.srws_path_finished_files = None

    # load the clases
    srws = ProcessSRWS(inp.srws_path_root, inp)
    # fo = FileOperations(inp.srws_path_root)

    # get the size and other details of files
    # file_prop = fo.FnGetFileSize(inp.srws_path_root, inp.srws_regStr)

    if inp.pickle == True:
        ppsData = pd.read_pickle(os.path.join(inp.workDir, 'Bowtie1', 'ppsData.pickle'))
        cups_1min = pd.read_pickle(os.path.join(inp.workDir, 'Bowtie1', 'cups_1min.pickle'))
        sonics_1min = pd.read_pickle(os.path.join(inp.workDir, 'Bowtie1', 'sonics_1min.pickle'))
        data_1min = pd.DataFrame(pd.read_pickle(os.path.join(inp.workDir, 'Bowtie1', 'data_1min_mean.pickle')))
    else:
        # convert raw data to 1-min averages,
        Data, df_H, df_bowtie, df_m, ds, ds_H, ds_bowtie, ds_m = srws.mp_FnConvertRawData(inp)

    sys.exit('manual')

    # convert to a mult-index dataframe
    from metadata_helpers import df2mi
    import xarray as xr
    # reg expression to match columns that ^(?!qc) do not match qc at the start, 
    # \s? has zero or one nr. of whitespaces, \D with any nonDigit, {} with a number 1,2,3,4 using a for loop, 
    # r?$ has zero or one nr. of 'r' at the end
    regStr = "^(?!qc).*\s?\D{}r?$"
    indexStr = 'srws_nr{}'
    df_mi, dropped_cols, ds = df2mi(Data[0], regStr, indexStr)
    


    # get post-processed sonics data
    inp.pps = pa.struct()
    inp.ppsPath = os.path.join(self.inp.workDir,'data', '1min_SRWS', '110m')
    inp.pps.regStr = '*.txt'
    inp.pps.searchStr = '(\d{12})'
    inp.pps.dateformat='%Y%m%d%H%M%S'
    inp.pps.cols = ['Time', 'v1 [m/s]', 'v2 [m/s]', 'v3 [m/s]']
    inp.tstart = datetime.strptime('2022-04-26_16-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
    inp.tend = datetime.strptime('2022-04-26_17-00-00', '%Y-%m-%d_%H-%M-%S') # Select start date in the form yyyy-mm-dd_HH-MM-SS
    pps = PostProcess_sonics(inp)
    ppsData = pps.read_sonics(inp.pps.regStr, inp.pps.searchStr, inp.pps.dateformat, inp.pps.cols)
    # post-processed data at the start of the period, moving to the end of period
    ppsData.index = ppsData.index + timedelta(minutes=1)

    # compare srws with metmast data
    inp.sonics = pa.struct()
    inp.sonics.paths = [
            '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/gill_115_u/20 Hz',  # thies_25_Vx
            '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/gill_115_v/20 Hz',  # gill_115_u
            '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/gill_115_w/20 Hz',  # 
            '/AIRPORT/AD8_PROTOTYPE/METMAST_EXTENSION/gill_115_SonicTempC/20 Hz'  # degrees
    ]
    inp.sonics.names = [
        'X',
        'Y',
        'Z',
        'T', # degrees
    ]
    inp.sonics.sampleRate=20
    inp.folder = r""
    # inp.ppsPath = r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\HighRe\data\nawea\postprocessed_sonics"
    # inp.ppsPath = r"z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\01_Wartung_Messmast_GE-NET_DWG_20190226\Data\post_processed\1min_SRWS\110m"
    inp.sonics.offset = 121.31 # offset for gill at 110m=121.31, at 55m=121.84, thies at 25m=122.04 deg 
    inp.sonics.sensor = 'gill' #  sensor brand
    inp.sonics.z = 115 # height of the sensor
    inp.status = 'offline'

    # cups configuration
    inp.cups = pa.struct()
    inp.cups.paths = [
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0000_V1/25 Hz',
        # '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0010_V2/25 Hz',
        # '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0020_V3/25 Hz',
        # '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0030_V4/25 Hz',
        # '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0040_V5/25 Hz',
        # '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0050_V6/25 Hz',
        '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0070_D1/25 Hz',
        # '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0100_D4/25 Hz',
        # '/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ/M0110_D5/25 Hz',
    ]
    inp.cups.names = [
        'v1',
        # 'v2',
        # 'v3',
        # 'v4',
        # 'v5',
        # 'v6',
        'd1',
        # 'd4',
        # 'd5',
    ]
    inp.cups.sampleRate = 25

    # cups configuration
    inp.gpo = pa.struct()
    inp.gpo.paths = [
        '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_118m_HWS_hub/600 s',
        '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_393m_HWS_hub/600 s',
        '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_118m_DIRECTION_hub/600 s',
        '/AIRPORT/AD8_PROTOTYPE/LIDAR_2020/GreenPO_393m_DIRECTION_hub/600 s',
    ]
    inp.gpo.names = [
        'v118m',
        'v393m',
        'wd118m',
        'wd393m',
    ]
    inp.gpo.sampleRate = 1/600
    inp.gpo.folder=""

    # def FnProcess_nac_lidar_onedas(self):
    # # script to read Nacelle Lidar data from OneDAS
    import sys
    sys.path.append(r"../../userModules")
    from FnImportOneDas import FnImportOneDas
    _, gpo,_ = FnImportOneDas(inp.tstart, inp.tend, inp.gpo.paths, inp.gpo.names, inp.gpo.sampleRate, inp.gpo.folder)
    path = r"z:\Projekte\109797-TestfeldBHV\30_Technical_execution_Confidential\TP3\AP2_Aufbau_Infrastruktur\Infrastruktur_Windmessung\02_Equipment\04_Nacelle-Lidars-Inflow_CONFIDENTIAL\30_Data\GreenPO\2022-04\WIPO0100301_average_data_2022-04-27_00-00-00.csv"
    gpo = pd.read_csv(path,delimiter=";", parse_dates=True,  index_col='Date and Time')
    gpo1 = gpo.loc[gpo['Distance']==118]
    gpo1_10min = gpo1.loc['2022-04-26T18:00': '2022-04-26T19:00']
    gpo2 = gpo.loc[gpo['Distance']==393]
    gpo2_10min = gpo2.loc['2022-04-26T18:00': '2022-04-26T19:00']

    # # apply range filters
    # gpo = gpo.drop(columns=['t'])
    # for param in cups.columns:
    #     if param != 'd1':
    #         cups[param][(cups[param] < -100) | (cups[param] > 100)] = np.nan
    #         cups[param] = cups[param].interpolate(method='linear')
    #     else:
    #         continue


    # load the class
    pps = PostProcess_sonics(inp)
    WS1, WS2, WS3, elev, azi, df_rotate = srws.CampaignConstants(inp.srws_coord)

    # get line of sight velocities
    sonics, sonics_1min, sonics_10min = pps.FnPostProcess()

    # get the cup anemometer readings
    cups, cups_1min, cups_10min = pps.FnProcess_cups()

    # ## plot Scan pattern and Windscanner in Mikael coordinate system
    import plotly.graph_objects as go
    fig = go.Figure()
    id = range(0, 740)
    bowtie1 = go.Scatter3d(x=data_raw.xr1.iloc[id], y=data_raw.yr1.iloc[id], z=data_raw.zr1.iloc[id],mode='lines',name = 'bowtie1')
    bowtie2 = go.Scatter3d(x=data_raw.xr2.iloc[id], y=data_raw.yr2.iloc[id], z=data_raw.zr2.iloc[id],mode='lines',name = 'bowtie2')
    bowtie3 = go.Scatter3d(x=data_raw.xr3.iloc[id], y=data_raw.yr3.iloc[id], z=data_raw.zr3.iloc[id],mode='lines',name = 'bowtie3')
    WS1_pl  = go.Scatter3d(x=[WS1[0],WS2[0],WS3[0]], y=[WS1[1],WS2[1],WS3[1]], z=[WS1[2],WS2[2],WS3[2]],mode='markers', name  ='Windscanners')

    fig.add_trace(bowtie1)
    fig.add_trace(bowtie2)
    fig.add_trace(bowtie3)
    fig.add_trace(WS1_pl)                                     

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        )
    fig.show()

    # import plotly.express as px
    # fig=px.scatter(x=data_raw.index, y = data_raw.V1)
    # fig.update_layout(autosize=False,width=800,height=800)
    # fig.show()
    # fig.data=[]
    # fig.layout = {}


    #% plot Pedro's components
    # plt.figure()
    # plt.plot(ppsData.index, ppsData['v1 [m/s]'], 'k.', label='pedro v1')
    # plt.plot(ppsData.index, ppsData['v2 [m/s]'], 'b.', label='pedro v2')
    # plt.plot(ppsData.index, ppsData['v3 [m/s]'], 'g.', label='pedro v3')
    # plt.legend()
    # plt.xlabel('Datetime')
    # plt.ylabel('Vlos')

    # plot self projected sonics
    # plt.figure()
    # plt.plot(sonics_1min.index, sonics_1min.Vrot1, 'ko', alpha = 0.4, label='self v1')
    # plt.plot(sonics_1min.index, sonics_1min.Vrot2, 'bo', alpha = 0.4, label='self v2')
    # plt.plot(sonics_1min.index, sonics_1min.Vrot3, 'go', alpha = 0.4, label='self v3')
    # plt.legend()
    # plt.xlabel('Datetime')
    # plt.ylabel('Vlos')

    # compare Vlos from cups, sonics and SRWS for 1 min 
    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(12,12))
    axs[0].plot(sonics_1min.index, sonics_1min.Vrot1, 'ko', alpha = 0.4, label='self v1')
    axs[0].plot(ppsData.index, ppsData['v1 [m/s]'], 'k.', label='pedro v1')
    axs[0].plot(cups_1min.index, cups_1min.Vrot1, 'kd', alpha = 0.5, label='cups v1')
    axs[0].plot(data_1min.mean.index+ timedelta(minutes=1), data_1min.mean.V1, 'r+', alpha = 0.8, label='srws v1')
    axs[0].legend()

    axs[1].plot(sonics_1min.index, sonics_1min.Vrot2, 'bo', alpha = 0.4, label='self v2')
    axs[1].plot(ppsData.index, ppsData['v2 [m/s]'], 'b.', label='pedro v2')
    axs[1].plot(cups_1min.index, cups_1min.Vrot2, 'kd', alpha = 0.5, label='cups v2')
    axs[1].plot(data_1min.mean.index+ timedelta(minutes=1), data_1min.mean.V2, 'r+', alpha = 0.8, label='srws v2')
    axs[1].legend()

    axs[2].plot(sonics_1min.index, sonics_1min.Vrot3, 'go', alpha = 0.4, label='self v3')
    axs[2].plot(ppsData.index, ppsData['v3 [m/s]'], 'g.', label='pedro v3')
    axs[2].plot(cups_1min.index, cups_1min.Vrot3, 'kd', alpha = 0.5, label='cups v3')
    axs[2].plot(data_1min.mean.index+ timedelta(minutes=1), data_1min.mean.V3, 'r+', alpha = 0.8, label='srws v3')
    axs[2].legend()
    date_fmt = DateFormatter("%H:%M")
    axs[2].xaxis.set_major_formatter(date_fmt)
    axs[2].set_xlabel('Time [hh:mm] ({0}-{1})'.format(inp.tstart.strftime('%d.%m.%y'), inp.tend.strftime('%d.%m.%y')), fontsize=18)
    # plt.xlabel('Datetime')
    plt.ylim([-7,7])
    plt.ylabel('Vlos')

    # compare Vhub from cups, sonics and SRWS
    # fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(12,12))
    fig = plt.figure(figsize=(13,4))
    ax = plt.gca()
    plt.plot(sonics_1min.index, sonics_1min.Vhorz, 'bo-', alpha = 0.4, label='sonics')
    plt.plot(data_1min.mean.index+ timedelta(minutes=2), data_1min.hub_mean.Vhorz, 'r.-', label='srws')
    # plt.plot(pd.to_datetime(data_1min.mean.index), data_1min.hub_mean.Vhorz12, 'm.-', label='srws_12')
    # plt.plot(pd.to_datetime(data_1min.mean.index), data_1min.hub_mean.Vhorz23, 'ro-', label='srws_23')
    # plt.plot(pd.to_datetime(data_1min.mean.index), data_1min.hub_mean.Vhorz31, 'r+-', label='srws_31')
    plt.plot(cups_1min.index, cups_1min.v1, 'kd-', alpha = 0.5, label='cups v1')
    plt.plot(gpo1_10min.index, gpo1_10min['HWS hub'], 'bx-', label='nacelle_lidar_V118m')
    # plt.plot(gpo2_10min.index, gpo2_10min['HWS hub'], 'gx-', label='nacelle_lidar_V393m')
    date_fmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_fmt)
    plt.xlabel('Time [hh:mm] ({0}-{1})'.format(inp.tstart.strftime('%d.%m.%y'), inp.tend.strftime('%d.%m.%y')), fontsize=18)
    plt.ylabel('Horizontal wind speed [m/s]')
    plt.legend()

    fig = plt.figure(figsize=(13,4))
    ax = plt.gca()
    sonics_1min['wdir'] = (np.rad2deg(np.arctan2(sonics_1min.U, sonics_1min.V)) + 180) % 360
    plt.plot(sonics_1min.index, sonics_1min.wdir, 'bo-', alpha = 0.4, label='sonics')
    plt.plot(data_1min.mean.index+ timedelta(minutes=1), (data_1min.hub_mean.gamma)+9, 'r.-', label='srws')
    plt.plot(cups_1min.index, cups_1min.d1, 'kd-', alpha = 0.5, label='cups v1')
    date_fmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_fmt)
    plt.xlabel('Time [hh:mm] ({0}-{1})'.format(inp.tstart.strftime('%d.%m.%y'), inp.tend.strftime('%d.%m.%y')), fontsize=18)
    plt.ylabel('Wind direction [°]')
    plt.legend()

    fig = plt.figure(figsize=(13,4))
    ax = plt.gca()
    plt.plot(sonics_1min.index, sonics_1min.tilt, 'bo-', alpha = 0.4, label='sonics')
    plt.plot(data_1min.mean.index+ timedelta(minutes=1), (data_1min.hub_mean.psi), 'r.-', label='srws')
    plt.plot(cups_1min.index, cups_1min.d1, 'kd-', alpha = 0.5, label='cups v1')
    date_fmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_fmt)
    plt.xlabel('Time [hh:mm] ({0}-{1})'.format(inp.tstart.strftime('%d.%m.%y'), inp.tend.strftime('%d.%m.%y')), fontsize=18)
    plt.ylabel('Wind direction [°]')
    plt.legend()

    # 3D plot
    my_cmap = plt.get_cmap('jet')
    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.add_subplot(projection='3d')
    p1 = ax.scatter(df_id.xr1, df_id.yr1, df_id.zr1, alpha=0.8, c=df_id.Vhorz,cmap=my_cmap,marker='.', s=5, vmin=0, vmax=30)
    plt.title("Scanning pattern plot")
    ax.set_xlim3d(120,130)
    ax.set_ylim3d(-120,120)
    ax.set_zlim3d(0,250)
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    fig.colorbar(p1, ax = ax, shrink = 0.5, aspect = 5)
    plt.show()

#%% 
# plt.plot(ds.vlos.isel(srws_nr=0)[range(700)], label='dtu')
# plt.plot(ds.vlos_pp.isel(srws_nr=0)[range(700)], label='ash')
# plt.legend()
# plt.show()

# %% Timestamps with problems and not recondstruncted
# 2021-11-02T1255 - > reason specrtra has variation at the tail and the cerntre is shifted.
# 2021-11-02T154600+01
# 2021-11-02T154900+01
# 2021-11-02T190400+01


