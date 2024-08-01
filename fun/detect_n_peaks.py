# load modules
import pandas as pd
import os, sys, glob
import numpy as np
import xarray as xr

sys.path.append(r"../fun")
import fun.pythonAssist as pa
from fun.spectr import SpectralAnalysis
from fun.ProcessSRWS import ProcessSRWS
from fun.FileOperations import FileOperations

def detect_n_peaks(inp, f):
    """generate parquet files and netcdf files with multiple peaks from srws raw spectral data"""
    
    srws = ProcessSRWS(f, inp)

    ## Get all files in folder with data
    filename, _ = os.path.splitext(f)
    
    _, df_raw, spec_df = srws.Read_SRWS_bin(filename, mode='basic', write_spec=True)

    try:
        sa = [SpectralAnalysis(spec_df[f"Spectrum {i+1}"], inp, df_raw[f"MaximumValue {i+1}"]) for i in range(3)]
        run_dict = {i: sa[i].RunSpectralAnalysis(peak_mode='prom') for i in range(3)}
        run_df = list(run_dict.values())        

        # prepare names
        vlos_names = [f'vlos_pp s{i}p{j}' for i in range(1,4) for j in range(1,inp.n_peaks+1)]
        snr_names = [f'snr_pp s{i}p{j}' for i in range(1,4) for j in range(1,inp.n_peaks+1)]
        props = ['peaks','peak_start','peak_end','peak_heights','prominences','h_center', 'v_center']
        prop_names = [f'{c} s{i}p{j}' for i in range(1,4) for j in range(1,inp.n_peaks+1) for c in props]
        p_names = [f'peak_power s{i}p{j}' for i in range(1,4) for j in range(1,inp.n_peaks+1)]


        # concat the dataframes
        vlos_df = pd.concat([pd.DataFrame(df[0]) for df in run_df], ignore_index=True, axis=1).set_axis(vlos_names, axis=1)

        snr_df = pd.concat([pd.DataFrame(df[1]) for df in run_df], ignore_index=True, axis=1).set_axis(snr_names, axis=1)

        df_peaks_df = pd.concat([pd.concat(df[2], axis=1) for df in run_df], axis=1).set_axis(prop_names, axis=1)

        peak_power_df = pd.concat([pd.DataFrame(df[3]) for df in run_df], ignore_index=True, axis=1).set_axis(p_names, axis=1)

        dfp = pd.concat([df_peaks_df, snr_df, vlos_df, peak_power_df], axis=1).set_index(df_raw.index)
            
        # save to parquet and netcdf
        pa.write_fastparquet(os.path.join(base_path, rf"data_peaks_{os.path.basename(f)}_V3.parq"), dfp)

        print(f'[{pa.now()}]: {f[-20:]} completed')
    except Exception as e:
        err_file = r"error_files.txt"
        fo = FileOperations(os.path.dirname(f))
        error_files = fo.write_paths_to_file(err_file, [f])
        print(f'[{pa.now()}]: Problems with {f[-20:]} due to error {e}')
        
    return None

if __name__ == "__main__":

    # initialize parameters
    inp = pa.struct()
    inp.test = False
    inp.height = [1.25, 5]
    inp.prominence = [0.25, 5]
    inp.distance = 3
    inp.width = 1
    inp.threshold = 3500
    inp.window = 128
    inp.plot_figure = False
    inp.verbose=False
    inp.read_dtu = True
    inp.window_size=10
    inp.ax=1
    inp.mode='median'
    inp.moving_average = False
    inp.interpolate = True
    inp.write_spec=True
    inp.generate_stats = False
    inp.n_peaks = 3
    inp.wlen = 60
        
    f = r"C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\srws_trial_run\data\srws_data\2022-04-26T185900+00"
    detect_n_peaks(inp, f)