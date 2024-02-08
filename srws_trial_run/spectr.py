# load modules
import pandas as pd
import os
import numpy as np
import sys
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

workDir = os.path.dirname(sys.path[0])
sys.path.append(os.path.join(workDir,'fun'))
sys.path.append(os.path.join(workDir,'src'))
sys.path.append(os.path.join(workDir,'data', 'nawea', 'srws'))

sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")

import pythonAssist as pa
from FnPeaks import FnPeaks

# import user functions
# from MotorToCoordxyz import *
# from rotation_matrix import rotation_matrix
# from scanner2globelCoord import *
# from Rotate_and_translate import Rotate_and_tranlate
# from ReconstructUVW import *
# from ReconstructUV import *
# from calc_angles import *
from scipy.signal import find_peaks

# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
class SpectralAnalysis:
    # scale the spectrum to its original value and find peaks using the maximum value:


    def __init__(self, spec, inp, max_value):
        import numpy as np
        self.spec = spec
        self.inp = inp
        self.max_value = max_value
        self.inp.lambda0 = 1565e-9 # R2D1 = 1564.93e-9
        self.inp.Fs = 120e6 # Hz
        self.inp.Nfft = np.shape(spec)[1]
        self.inp.Nspec = np.shape(spec)[1]
        self.inp.resolution = 0.1

    def FnScaleSpectrum(self):

        def scale_func(spec, max_val):
            """
            Function to scale the spectrum back to its original value
            Input: 
                spec  - instantaneous spectrum response from a CW lidar. usually shape: (1, 512)
                max_value - scaling value stored in the data, with which the spectrum was scaled
                factor = 2**16-1 as 16 bytes are stored in the data (corresponds to 512 points) 
            Output:
                scaled spectrum (no name)
            """
            return spec * (max_val/(2**16-1))
        import numpy as np
        import pandas as pd
        vfunc = np.vectorize(scale_func)
        scaled_arr = vfunc(self.spec.transpose(),self.max_value)
        scaled_spec = pd.DataFrame(scaled_arr.T)
        return scaled_spec

    def FnSpectrumPeaks_max(self):
        """
        Function to find the spectral peak using maximum value
        Input: 
            spec  - instantaneous spectrum response from a CW lidar. usually shape: (1, 512)
            inp - stuct from pa.struct() providing the following attributes
                threshold = threshold above which the peaks should be founds
                window - rolling window within which the peak should be found
                plot_figure - plot figure True / False
                verbose - text output on screen True / False 
        Output:
            peaks - dataframe with min, max, min_position, max_position, lags, nr. of peaks
        Remarks:
            iterrows() slows the peak finding process and therefore this function should be avoided for large spectrum data
        """
        
        peaks = pd.DataFrame(columns = ['max', 'min','max_pos', 'min_pos', 'lags', 'Npeaks'])
        
        for idx, row in self.spec.iterrows():
            # from scipy.signal import find_peaks
            # peaks, _ = find_peaks(row, height=300, distance=64, prominence=1000, width=3)
            peaks_idx, Npeaks, peaks_val = FnPeaks(row, self.inp.threshold, self.inp.window, self.inp.plot_figure, verbose=self.inp.verbose)
            try:
                maxima = peaks_val[peaks_val == peaks_val.max()]
                minima = peaks_val[peaks_val == peaks_val.min()]
            except:
                print("probably len(Npeaks) = 0")
            
            peaks_temp = pd.DataFrame({
                                'max': maxima.values[0],
                                'min': minima.values[0],
                                'max_pos': maxima.index[0],
                                'min_pos': minima.index[0],
                                'lags': maxima.index[0] - minima.index[0],
                                'Npeaks' : Npeaks,
                                },
                                index=[0]
                            )
            peaks = pd.concat([peaks, peaks_temp], ignore_index=True)

        return peaks

    #%% method 2: using vectorization
    # Vectorization function
    def FnFlattenSpectrum(self, correct=False):
        """
        Function to flatten the spectrum
        Input: 
            spec  - instantaneous spectrum response from a CW lidar. usually shape: (N, 512)
            max_value - max_value used for scaling the spectrum (see SRWS Data for parameter max_value)
        Output:
            spec_flat  - normalized and flattened spectrum. shape: (N, 512)
            spec_real - spectrum scaled to real output spectrum using the max value. shape: (N, 512)
            spec_median - median spectrum by find the median for the bins for a 1-min file.  shape: (N, 512)
        Remarks:
            The function is designed only for 512 bins, update if the bin sizes change
        """
        import numpy as np
        import pandas as pd
        from scipy.signal import find_peaks, peak_widths

        Nrows, Ncols = self.spec.shape
        
        if correct == True:
            if ~np.isfinite(self.max_value.mean()) | (self.max_value.mean() > 2**16):
                self.max_value = pd.Series([900]*Nrows, index=self.max_value.index)
                
            if (self.spec.median()==0).sum() > 1:
                # correct for zeros at tails
                self.spec = self.detect_spectral_zerotail(self.spec, duplicated_val=0, correct=True)
            else:
                # correct for large deviations at tails
                self.spec = self.detect_spectral_noisytail(self.spec)
        else:
            if (self.spec.median()==0).sum() >= 32:
                print("ValueError: many zeros in this spectra")
            elif self.max_value.mean() > 2**16:
                print("ValueError: maximum value too high with high noise at tail")


        # scale the spectrum from normalized to true state
        spec_real = ((self.spec.transpose()) * (self.max_value.to_numpy()) / (2**16-1)).transpose() # size: N X 512
        # get the median of the whole 1-min spectrum
        spec_median = spec_real.median() # size: 512 x 1
        spec_mean = spec_real.mean()
        
        # find the 1-min median peak from the spectrum
        heights = [200, 120, 50, None]
        distances = [8, 4, 2, 1]
        proms = [10, 10, 10, 4]
        specs = [spec_median, spec_median, spec_median, spec_mean]
        loops = 0
        peaks = []
        while len(peaks)==0:
            peaks, _ = find_peaks(specs[loops], height=heights[loops], distance=distances[loops], prominence=proms[loops], width=1)
            loops += 1
            if loops == 4:
                print(f"[{pa.now()}]: maximum nr of loops passed, using default argmax value")
                peaks = spec_median.argmax()
                break

        idx = abs(peaks-Ncols/2).argmax()


        # # find the location and side of the peak
        # try:
        #     idx = abs(peaks - Ncols/2).argmax()
        # except ValueError:
        #     print(f"[{pa.now()}]: ValueError due to no peaks found, using mean instead of median")
        #     if len(peaks)==0:
        #         spec_median = spec_real.mean()
        #         peaks,_ = find_peaks(spec_median.values, height=120, distance=8, prominence=10, width=1)
        #     elif len(peaks)==0:
        #         peaks, _ = find_peaks(spec_median.values,height=50, distance=8, prominence=10, width=1)
        #     elif len(peaks)==0:
        #         print("No peaks were found, lowering the bar")
        #         peaks,_ = find_peaks(spec_median.values, distance=1, prominence=4)
        # finally:
        #     try:
        #         idx = abs(peaks - Ncols/2).argmax()
        #     except ValueError:
        #         print(f"[{pa.now()}]: ValueError due to no peaks found, assigning argmax value")
        #         peaks = spec_median.argmax()
        #         idx = abs(peaks - Ncols/2).argmax()

        sign = np.sign((peaks[idx] - Ncols/2))
        spec_base = spec_median.copy()
        if sign > 0:
            spec_base[(int(Ncols/2)+1):] = spec_median[list(reversed(range(int(Ncols/2)-1)))]
        elif ((peaks <= int(Ncols/2)-1).all() or (sign < 0)):
            spec_base[range(int(Ncols/2)-1)] = spec_median.iloc[list(reversed(range(int(Ncols/2)+1,Ncols)))].values

        # Flattening the spectrum
        spec_flat = spec_real / spec_base
        # since sometimes this results in a nan or 0 (interpolate in future)
        spec_flat = spec_flat.interpolate(axis=1, limit=32).copy()

        return spec_flat, spec_real, spec_median, spec_base

    #%% find centroid peaks for time x fft freq matrix
    # @profile
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks, peak_widths
    import pythonAssist as pa

    def FnDopplerPeaks(self, spec_flat, peak_mode):
        """
        Function to find the doppler peak from a flattened spectrum
        Input: 
            spec_flat  - normalized and flattened spectrum as output from FnFlattenSpectrum. shape: (N, 512)
        Output:
            df_peaks - dataframe consisting of doppler peak position and peak start and end indices for post-processing
            df_prop - dataframe consisting of peak properties as given in scipy.signal.find_peaks() function
        Remarks:
            The function is designed only for 512 bins, update if the bin sizes change
        """
        # Steps involved in this function:
        # find peaks and peak properties for a flattened spectrum
        # If there are no peaks found, use relaxed find_peaks configuration to find them
        # find the peak with maximum lags, where the doppler peak centroid is defined by h_center


        # correction for blade detection in spectra using a moving average
        # @cache
        # def gauss(x ,h, a, x0, sigma):
        #     """Generate a gaussian function"""
        #     y = h +a * np.exp(-(x-x0)**2/(2*sigma**2))
        #     return y 

        # spec_flat, idx = self.blade_correction(spec_flat, gauss)
        # print(f"{pa.now()}: blade passing correction performed on {len(idx)} samples")
        
        # if  (spec_flat.max(axis=1)>10).sum() > 50:
        #     print(f"{pa.now()}: Performing blade correction on spectrum")
        #     idx = spec_flat[spec_flat.max(axis=1)>4].index
        #     win_length = 11
        #     lag = int((win_length-1)/2)
        #     w_mean = spec_flat.rolling(window=win_length, closed='both').mean().shift(periods=-lag).ffill()
        #     spec_temp = spec_flat.iloc[idx,:].divide(w_mean.iloc[idx,:])
        #     spec_flat.update(spec_temp)

        # drop all nan rows
        nan_rows = spec_flat[spec_flat.isna().sum(axis=1) == spec_flat.shape[1]].index

        spec_flat = spec_flat.dropna(axis=0, how='all')
        # find the peaks and properties
        S_peak = spec_flat.apply(find_peaks,height=self.inp.height, distance=self.inp.distance, prominence=self.inp.prominence, width=self.inp.width, axis=1, result_type='expand')
        
        df_peaks = pd.DataFrame(S_peak[0].to_list()).add_prefix('peaks_')
        
        df_prop = pd.DataFrame(S_peak[1].to_list())

        # find the peaks where no peaks were detected with the default setting
        no_peaks = df_peaks[df_peaks.isna().sum(axis=1) == df_peaks.shape[1]]
        default_reducer = 0.025
        while (not no_peaks.empty):
            self.inp.height[0] = self.inp.height[0] - default_reducer
            S_no_peaks = spec_flat.iloc[no_peaks.index,:].apply(find_peaks, height=self.inp.height, prominence=self.inp.prominence, width=self.inp.width, axis=1, result_type='expand')
            df_no_peaks = pd.DataFrame(S_no_peaks[0].to_list(), index=no_peaks.index).add_prefix('peaks_')
            df_no_prop = pd.DataFrame(S_no_peaks[1].to_list(), index=no_peaks.index)
            df_peaks = df_no_peaks.combine_first(df_peaks)
            df_prop = df_no_prop.combine_first(df_prop)
            # no_peaks = df_peaks[df_peaks.isna().sum(axis=1) == (df_peaks.shape[1]-1)]
            if  (np.round(self.inp.height[0], decimals =4) == 1.0000):
                print(f"[{pa.now()}]: Prominence height requirement relaxed by 0.2") 
                default_reducer = 0.2
            elif (np.round(self.inp.height[0], decimals =4) == 0.0000):
                break
        
        if df_peaks.empty:
            return pd.DataFrame(np.tile(np.nan, spec_flat.shape[0]))
        
        # find peak based on maximum positional value
        @staticmethod
        def find_max_index(row):
            Ncols = len(row)
            return np.nanargmax((row - (Ncols-(Ncols%2))/2).abs())

        # sort the columns in the dataframe
        df_peaks = df_peaks.reindex(sorted(df_peaks.columns, key=lambda x: float(x.split('_')[1])), axis=1)

        # find peak based on maximum positional value

        if peak_mode == 'max':
            df_peaks[peak_mode + '_idx'] = df_peaks.filter(like='peaks').apply(find_max_index, axis=1).astype('int16').copy()
        # find peak based on highest prominence
        elif peak_mode == 'prom':
            df_peaks[peak_mode + '_idx'] = df_prop.prominences.apply(pa._nanargmax, ax=0).astype('int16',errors='ignore').copy()
        # find peaks based on maximum widths
        elif peak_mode == 'width':
            df_peaks[peak_mode + '_idx'] = df_prop.widths.apply(np.nanargmax).astype('int16').copy()
        # assign the peak index, here we assign prominence index as peak index
        df_peaks['doppler_idx'] = df_peaks[peak_mode+'_idx'].astype('int16', errors='ignore').copy()
        df_peaks['doppler_peak'] = df_peaks.apply(lambda x: x[x.at[peak_mode+'_idx'].astype('int16')], axis=1).copy()

        # assign information which procedure has been used for peak selection
        df_peaks['qc_flag'] = (np.zeros(len(df_peaks))).astype('int16').copy()

        # 2nd iteration using moving window averaged outlier estimates.
        outliers,df_peaks['ma_mean'], df_peaks['ma_std'] = pa.signal_outliers(df_peaks.doppler_peak, win_length=11, sigma=1, detrend=True)

        def closest_peak(df_peaks_row):
            """function to find the closest peak to the moving average"""
            import numpy as np
            peaks = df_peaks_row.filter(like='peaks_')
            mean_peak = df_peaks_row.ma_mean
            try:
                iv = np.nanargmin(np.abs(peaks  - mean_peak))
            except ValueError:
                iv = 0 
            return peaks[iv].astype('int'), np.uint16(iv)

        applied_df = pd.DataFrame(1, index=outliers.index, columns=['qc_flag'])
        # replace rows which have NaN in all peaks, erroneous rows (probably waiting rows)
        idx_nan = df_peaks[df_peaks.filter(like='peaks_').isna().all(axis=1)].index
        df_peaks.iloc[idx_nan] = df_peaks.iloc[idx_nan-1].copy()
        df_prop.iloc[idx_nan] = df_prop.iloc[idx_nan-1].copy()
        
        applied_df[['doppler_peak', 'doppler_idx']] = df_peaks.iloc[outliers.index].apply(closest_peak,axis=1, result_type='expand').copy()
        df_peaks.update(applied_df)
        # some points result in error if a peak is closest to the ma_mean and is defined as a peak

        # finding peak start and end , h_center and v_center for comparison and further evaluation
        df_prop = pd.concat([df_prop, df_peaks.doppler_idx], axis=1)

        df_prop.doppler_idx = df_prop.doppler_idx.astype('float').astype('Int32')
        peak_start = df_prop.apply(lambda r: np.floor(r['left_ips'][np.uint16(r.doppler_idx)]), axis=1).copy() 
        peak_end = df_prop.apply(lambda r: np.ceil(r['right_ips'][np.uint16(r.doppler_idx)]), axis=1).copy()
        peak_heights = df_prop.apply(lambda r: r['peak_heights'][np.uint16(r.doppler_idx)], axis=1).copy()
        prominences = df_prop.apply(lambda r: r['prominences'][np.uint16(r.doppler_idx)], axis=1).copy()
        h_center = (peak_start + peak_end)/2
        v_center = (peak_heights + prominences)/2
        df_temp = pd.DataFrame(data = zip(peak_start, peak_end, peak_heights, prominences, h_center, v_center), 
                                columns=['peak_start', 'peak_end', 'peak_heights', 'prominences', 'h_center', 'v_center'])
        df_peaks = pd.concat([df_peaks, df_temp], axis=1)

        # reshapeing the df_peaks to original spec dimensions along axis=0 (rows)
        df_nans = pd.DataFrame(index=range(self.spec.shape[0]), columns=df_peaks.columns)
        df_peaks = df_nans.combine_first(df_peaks)
        # df_peaks.iloc[nan_rows,:] = np.nan

        return df_peaks

    def blade_correction(self, spec_flat, gauss):
        idx = spec_flat[spec_flat.max(axis=1)>4].index
        spec_blade = spec_flat.iloc[idx].median()
        y_max_idx = np.argmax(spec_blade)
        y_max = spec_blade[y_max_idx]
        y_std = np.std(spec_blade)
        y_base = np.round(np.mean(spec_blade))
        gauss_vfunc = np.vectorize(gauss)
        spec_gauss = gauss_vfunc(range(512), y_base, y_max, y_max_idx, y_std)
        spec_temp = spec_flat.iloc[idx,:].divide(spec_gauss)
        spec_flat.update(spec_temp)
        return spec_flat, idx

    @staticmethod
    def zero_spectra_correction(spec):
        # correct spectra with zeros in all fft bins (e.g. waiting spectras for other sensors)
        spectrum = spec.copy()
        idx = spectrum[spectrum.sum(axis=1)==0].index
        spectrum.loc[idx,:] = spectrum.loc[idx-1,:].copy(deep=True)
        # print(f"{pa.now()}: spectrum corrected for zero values in all fft bins")
        return spectrum

    # @profile
    def FnFindPeaks_exact(self, spec_1d, df_peaks):
        """
        FnFindPeaks_exact - function to find the peak positions along a 1D array / list / dataframe
        Syntax:  peaks = FnFindPeaks_exact(spec, df_peaks, inp)
        Inputs:
            spec - input flattened spectrum from CW Lidar SRWS [1D array or a row from spec_flat]
            df_peaks  - peaks found using FnFindPeaks function (required doppler_peak, peak_start, peak_end)
            inp - an input  struct with the following attributes
                mode - select the method of finding the peak -> options: "median" / "centroid" / "fwhm" / "maximum"
                interpolation - perform interpolation or not -> options: True / False
                moving_average - perform moving average or not -> options True / False
                resolution - resolution of the bins to be interpolated -> works for 0.1
                height - height input for scipy.signal.find_peaks
                prominence - prominence input for scipy.signal.find_peaks
                width - width input for scipy.signal.find_peaks -> width = 1 helps

        Outputs:
            peak - peak
        
        # Steps involved in this function:
        #  - define a threshold based on standard deviation 
        #  - assign zero to the points below this threshold (make sure that no large peaks are occurring at the tail of spectra)
        #  - interpolate the spectrum to a new resolution
        #  - apply moving average based on window_size
        #  - perform peak location finding based on the method selected
        # - get relative peak position as an input to spectrum2vlos function
        """

        # import modules
        import numpy as np
        import pandas as pd

        # slice the spectrum for between the peak start and peak end
        spec = self.FnSliceSpectrum(spec_1d, df_peaks)

        # perform interpolation in between bins
        if self.inp.interpolate == True:
            x_new = np.linspace(df_peaks.peak_start, df_peaks.peak_end, np.uint16(len(spec)/self.inp.resolution)+1)
            spec = self.spectral_interp1d(spec, x_new)

        # apply moving average to the spectrum
        if self.inp.moving_average == True:
            # print('using default window size of 10 points')
            spec = self.FnSpectral_moveavg(spec)

        # find the peak locations according to median, centroid, FWHM and maximum methods
        if self.inp.mode== 'median':
            # print("median method uses cumsum(). use a different integration scheme if problems occur")
            peak_idx = self.find_peak_median(spec)
        elif self.inp.mode == "centroid":
            peak_idx = self.find_centroid(spec)
        elif self.inp.mode == "fwhm":
            peak_idx = self.find_peak_fwhm(spec)
        elif self.inp.mode == "midbase":
            peak_idx = self.find_peak_midbase(spec)
        else:
            print("add the peak finding mode")
        
        # get relative lags in the peak from 256
        try:
            peak = peak_idx - len(spec_1d)/2
            peak_val = spec[peak_idx]
        except KeyError:
            if spec_1d.isna().sum() == len(spec_1d):
                peak, peak_val = np.nan, np.nan
            else:
                peak_idx = df_peaks.doppler_peak,  
                peak_val = spec[peak_idx]
                peak = peak_idx - len(spec_1d)/2

        return peak, peak_val

    # @jit(nopython=True, nogil=True)
    # @profile
    def find_peak_median(self, spec):
        spec_cdf = spec.cumsum().astype(float)
        peak_idx = (spec_cdf - spec_cdf.max()/2).abs().idxmin()
        return peak_idx

    def find_peak_midbase(self, spec):
        """ find the peak based on mean of the left and right ips of a curve, see find_peaks"""
        from scipy.signal import find_peaks
        _, prop=find_peaks(np.ravel(spec.values), height=self.inp.height,prominence=self.inp.prominence, width= inp.width)
        peak_idx = spec.index[round(np.mean([prop['left_ips'][1], prop['right_ips'][1]]))]
        return peak_idx

    def find_peak_fwhm(self, spec):
        """ find the peak based on center at FWHM of a curve, see find_peaks"""
        from scipy.signal import find_peaks
        _, prop=find_peaks(np.ravel(spec.values), height=self.inp.height,prominence=self.inp.prominence, width= inp.width)
        peak_idx = (df_prop.iloc[0,:].left_ips[0] + df_prop.iloc[0,:].right_ips[0]) / 2
        return peak_idx

    def find_centroid(self, spec):
        """ find the centroid of given curve, see definition of centroid"""
        peak_idx = np.dot(spec.index, spec.values)/np.sum(spec.values)
        return peak_idx

    # @profile
    def FnSliceSpectrum(self, spec_1d, df_peaks):

        # load modules
        import numpy as np
        import pandas as pd

        mean_spec = np.nanmean(spec_1d.astype('float32'))

        if isinstance(spec_1d, np.ndarray):
            spec_1d = pd.Series(spec_1d,index=range(512))

        # slice the spectrum between left and right base and remove the values below the threshold        
        try:
            spec = spec_1d.iloc[np.uint16(df_peaks.peak_start):(np.uint16(df_peaks.peak_end)+1)]
        except (ValueError, AttributeError) as e:
            print(f"[{pa.now()}]: {spec_1d.name} needs loosened peak conditions")
            _, pk_prop = find_peaks(spec_1d, distance=100, width=1)
            peak_idx = np.argmax(pk_prop['prominences'])
            spec = spec_1d.iloc[np.uint16(pk_prop['left_ips'][peak_idx]):(np.uint16(pk_prop['right_ips'][peak_idx])+1)]

        spec_std = np.nanstd(spec.astype('float32'))
        # threshold of 1-sigma
        threshold = mean_spec + spec_std * 1 
        # any outlier will be null
        spec[(spec < threshold)] = 0

        return spec

    #%% Find centroid peaks for 1d array
    def FnFindPeaksCentroid1D(self, spec_1d):
        df_peaks, df_prop = find_peaks(np.ravel(spec_1d), height=self.inp.height, distance=self.inp.distance, prominence=self.inp.prominence, width=self.inp.width)

        if (np.sum(np.isnan(df_peaks))!=0) or (len(df_peaks)==0):
            df_peaks, df_prop = find_peaks(np.ravel(spec_1d), height=self.inp.relaxed_height, distance=self.inp.distance, prominence=self.inp.prominence, width=self.inp.width)

        peak_idx = np.argmax(np.abs(df_peaks - (len(spec_1d)-(len(spec_1d)%2))/2))
        df_prop['doppler_peak']= df_peaks[peak_idx]
        df_prop['h_center']= np.mean([df_prop['left_ips'][peak_idx],df_prop['right_ips'][peak_idx]])
        df_prop['v_center']= np.mean([df_prop['peak_heights'][peak_idx],df_prop['prominences'][peak_idx]])
        return df_peaks, df_prop

    # Simple construct:
    def spectral_interp1d(self, spec_1d, *x_new):
        # interpolate a 1d spectral array
        from scipy import interpolate
        import numpy as np
        import pandas as pd

        # adjust Nspec for 1D arrays instead of 2D dataframes, remove this if else
        if len(np.shape(spec_1d)) == 1:
            self.inp.Nspec = len(spec_1d)
        else:
            self.inp.Nspec = np.shape(spec_1d)[1]
        Nbins = np.uint16(self.inp.Nspec/self.inp.resolution + 1)

        # temporary: use kwargs/args in the future
        try:
            x_new
        except NameError:
            x_new = np.linspace(0, self.inp.Nspec, Nbins)
        finally:
            if len(x_new) == 0:
                x_new = np.linspace(0, self.inp.Nspec, Nbins)
                

        f = interpolate.interp1d(spec_1d.index, spec_1d.values, fill_value='extrapolate')
        spectrum_interp = f(x_new)
        return pd.Series(np.ravel(spectrum_interp), index=np.ravel(x_new))

    def df_spec_interp(self, spec_flat):
        # interpolate a complete nd array spectrum in the form of a dataframe
        # Input:
        #       spec_flat: pandas dataframe containing an nd-array
        # Output: 
        # spec_flat_interp: interpolated spectrum
        spec_flat_interp = spec_flat.apply(lambda s: self.spectral_interp1d(s), axis=1, result_type='expand')
        return spec_flat_interp

    def FnSpectral_moveavg(self, spec_flat_interp):
        # from scipy import interpolate
        if len(np.shape(spec_flat_interp)) == 1:
            self.inp.ax = 0

        spec_flat_interp = pd.DataFrame(spec_flat_interp)
        S_ma = spec_flat_interp.interpolate(axis=self.inp.ax).rolling(window=self.inp.window_size, min_periods=1,axis=self.inp.ax, center=True).mean().shift(periods=np.uint16((self.inp.window_size-1)/2))
        S_ma.ffill().bfill()
        return S_ma

    def spectrum2vlos(self, lags):
        # resolution of the measurement
        vlos = lags*(self.inp.lambda0*self.inp.Fs/(2*self.inp.Nfft))
        return vlos
    
    # from numba import jit
    @staticmethod
    # @jit(nopython=True)
    def calc_snr(spec, peaks, method='peak'):
        """
        Calculate the SNR (Signal-to-Noise Ratio) from spectral data.

        Parameters:
            spec (numpy.ndarray): Array containing spectral data. Use flattened spectra from spectr.py
            **kwargs: Keyword arguments containing peak values.
            method - method to find the Psignal
            Psignal = P_peak (peak method)
            Psignal = area under the peak hill

        Returns:
            float: The calculated Signal-to-Noise Ratio (SNR).
        Remarks:
            # problems: perform the snr based on the peak value from FnFindPeaks_exact
        """
        import warnings
        warnings.filterwarnings("error", category=RuntimeWarning)

        spec = spec.astype('float64')
        if (peaks is None) or np.isnan(peaks):
            peak_id = np.argmax(spec)
        else:
            peak_id = np.round(peaks).astype('int')
            peak_id_none = np.argmax(spec)
            if (abs(peak_id - peak_id_none) > 50) & abs(peak_id-256)> 150:
                peak_id = peak_id_none            

        # getting the power in the signal
        peak_start = peak_id - 25
        peak_end = min(511, peak_id + 25)
        if method == 'peak':
            P_signal = spec[peak_id]
        elif method=="area":
            spec_power = spec[range(peak_start,peak_end)]
            spec_power[spec_power==0] = np.nan
            try:
                P_signal = np.nanmean(spec[peak_start:peak_end])
            except RuntimeWarning:
                P_signal = np.nan

        # getting the power in the noise
        bins_wo_peak = [x for x in range(0, 512) if x < peak_start or x > peak_end]
        spec_noise = spec[bins_wo_peak]
        P_noise = np.nanmean(spec_noise)
        if (P_noise == 0) | ~np.isfinite(P_noise) | (P_noise > 1000):
            P_noise = 400
        elif round(P_noise) != 1:
            peak_start = peak_start - 25
            peak_end = peak_end - 25
            bins_wo_peak = [x for x in range(0, 512) if x < peak_start or x > peak_end]
            spec_noise = spec[bins_wo_peak]
            P_noise = np.nanmean(spec_noise)
        # else:
        #     P_noise = 1

        # getting the variance in signal (power) and noise
        var_s = P_signal
        var_n = P_noise
        snr = 10*np.log10(np.abs(P_signal - P_noise) / P_noise)

        return snr, var_s, var_n

    # @profile
    def RunSpectralAnalysis(self):
        # flatten the spectrum
        # load modules
        import numpy as np
        import pandas as pd


        self.spec = self.zero_spectra_correction(self.spec)
        self.max_value = self.series_zero_correction(self.max_value)

        # flatten the spectrum
        try:
            spec_flat, spec_real, _, _ = self.FnFlattenSpectrum()
        except:
            spec_flat, spec_real, _, _ = self.FnFlattenSpectrum(correct=True)
            
        # find doppler peak and left and right bases
        df_peaks = self.FnDopplerPeaks(spec_flat, peak_mode='prom')

        # drop rows with NaNs in spec and df_peaks
        spec_idx_finite = spec_real.notna().all(axis=1)
        df_peaks_idx_finite = df_peaks.notna().all(axis=1)
        spec_real = spec_real[(spec_idx_finite | df_peaks_idx_finite)]
        df_peaks = df_peaks[(spec_idx_finite | df_peaks_idx_finite)]

        # finding peaks based on FnFindPeaks_exact method
        spec_all = pd.concat([spec_real, df_peaks], axis=1)
        peak_df = spec_all.apply(lambda s: self.FnFindPeaks_exact(s.filter(regex="^\d{1,3}"),s.filter(like='peak')), axis=1, result_type='expand')
        peak, peak_val = peak_df.iloc[:,0], peak_df.iloc[:,1]
        peak.name = "peak_exact"
        # calculate the snr based on exact peak
        spec_all = pd.concat([spec_all, np.round(peak)+256], axis=1)
        snr = spec_all.apply(lambda s: self.calc_snr(s.filter(regex="^\d{1,3}").values, peaks=s.peak_exact, method='area'), axis=1, result_type='expand')
        
        # convert lags into vlos
        vlos = self.spectrum2vlos(peak)

        # convert the snr df into different elements
        snr, var_s, var_n = snr.iloc[:,0], snr.iloc[:,1], snr.iloc[:,2]

        # del peak, df_peaks, spec_flat, spec_real

        return vlos, snr
    
    @staticmethod
    def series_zero_correction(df_series):
        """correct a series [df_series] for zeros"""
        dfs = df_series.copy()
        dfs[dfs==0] = np.nan
        dfs = dfs.fillna(method='ffill')
        return dfs
    
    @staticmethod
    def detect_spectral_zerotail(spec_df, duplicated_val=0, correct=True):
        """
        detect zeros or other constant values at the tail of a spectrum, resulting in erroneous vlos as the center might be shifted
        diff_val=0 and constant_val = 0 as default, can be changed to desired values if needed
        Syntax:  spec_df = detect_spectral_zerotail(spec_df, duplicated_val=0, correct=True)
        """
        temp_df = spec_df.median().copy()
        # find duplicated values or values not changing with frequency bins (along axis=1)
        diff_zero = temp_df.duplicated(keep=False) # keep=False lets include the first duplicate element
        # find values that are equal to duplicated value (default=0)
        is_zero = (temp_df == duplicated_val)
        # index of the duplicated values and equal to zero
        idx_cond = temp_df.loc[(diff_zero & is_zero)].index
        idx_diff = list(np.diff(idx_cond)==1)
        idx_diff = idx_diff + idx_diff[-1:]
        
        new_df = spec_df.copy()
        new_df.loc[:, idx_cond[idx_diff]] = np.nan

        
        if correct == True:
            # new_df.drop(idx_cond[idx_diff],axis=1, inplace=True)
            lags_0, sign = SpectralAnalysis.spectral_center_position(new_df)
            new_df = SpectralAnalysis.mirror_spectral_tail(new_df, limits=[idx_cond[idx_diff][0], idx_cond[idx_diff][-1]])
          
        
        return new_df

    @staticmethod
    def detect_spectral_noisytail(spec_df):
        """
        detect noise in the tail regions. This can be due to aliasing or spectral leakage.
        threshold = 30  (optimum) for Nfft = 512 bins,
        window_length = 64 (optimum) for Nfft=512 bins
        limits = [250,350] for scaled spectrum with mean around 300 dB
        """
        import scipy as sp
        temp_df = spec_df.median().copy()
        Nbins = len(temp_df)

        # find the threshold for deviations in spectrum
        threshold = sp.stats.median_abs_deviation(temp_df)
        # find the rolling standard deviation considering spectral values between limits
        rolling_std = temp_df.rolling(window=8, min_periods=1).std().bfill()
        # find outliers
        large_dev = rolling_std > threshold
        # find if the large deviations lie at the tails
        is_tail = (temp_df.index <  int(Nbins/4)) | (temp_df.index >  int(Nbins*3/4)) 
        # is_tail = ((temp_df.loc[large_dev].index <= round(Nbins/4)) | (temp_df.loc[large_dev].index > round(Nbins*3/4))
        outlier_first = temp_df[large_dev & is_tail].index[0]
        outlier_last = temp_df[large_dev & is_tail].index[-1]
        outlier_limits = [outlier_first, outlier_last]
        
        # is there a zero shift necessary
        # lags_0, sign = SpectralAnalysis.spectral_center_position(temp_df)
        
        new_df = SpectralAnalysis.mirror_spectral_tail(spec_df, limits=outlier_limits)
        
        return new_df

    @staticmethod
    def mirror_spectral_tail(spec_df, limits):
        """
        Mirrors a part of spectra (especially tail) depending on the sign and part on the other side
        spec_df - 1D/2D spectra with Nrow and Ncols
        limits -  limits are the limits where large deviations/errors start and end limits = [start, end] 
        
        """
        # get the lags with which zero is shifted, get the sign of deviation from center
        
        Nrows, Ncols = np.shape(spec_df)
        lags_0, sign = SpectralAnalysis.spectral_center_position(spec_df)

        part_df = spec_df.copy()    
        if sign < 0:
            part_df.drop(range(Ncols - abs(lags_0), Ncols),axis=1, inplace=True)
            new_df = pd.concat([spec_df.iloc[:,list(reversed(range(limits[0]-abs(lags_0),limits[0])))], part_df],axis=1, ignore_index=True)
        elif sign > 0:
            part_df.drop(range(abs(lags_0)),axis=1, inplace=True)
            new_df = pd.concat([part_df, spec_df[:,list(reversed(range([limits[1]]-abs(lags_0), limits[1])))]], ignore_index=True)

        return new_df

    @staticmethod
    def spectral_center_position(spec_df):
        """
        get the position of the spectral center in the spectra. erronous spectra leads to shift of central spectral zero
        """
        try:
            Nrows, Ncols = np.shape(spec_df)
            mean_df = spec_df.median().copy()
        except ValueError:
            Ncols = len(spec_df)
            mean_df = spec_df.copy()
            
        idx_0 = mean_df[mean_df == 0].index
        # are there more than one zero values
        idx_0 = idx_0[np.argmin(abs(int(Ncols/2) - idx_0))]
        lags_0 = int(idx_0 - len(mean_df)/2)
        sign = np.sign(lags_0)
        return lags_0, sign

    @staticmethod
    def generate_gold_spectrum(Nfft, peak_loc, peak_val, **kwargs):
        import scipy as sp
        
        # frequency range
        f = range(Nfft)

        # defintion of the specrtum based on inverted mexican hat (i.e. ricker wavelet)
        mexhat = 400*sp.signal.ricker(len(freq), 4.0)
        # invert the ricker wavelet
        inv_mexhat = -(mexhat) + 300
        inv_mexhat[np.argmin(inv_mexhat)]=0
        spec = inv_mexhat  + 30*np.random.randn(512)
        
        # adding a doppler peak and some real spectral features
        spec[peak_loc] = peak_val
        
        if 'shifted' in kwargs:
            spec = np.roll(spec, kwargs['shifted'])
            print(f"Shifted spectrum with {kwargs['shifted']} positions")        
            
        if 'zero_tail' in kwargs:
            spec[400:] = 0
            print(f"Added zeros at the tail of the spectrum")
            
        if 'noisy_tail' in kwargs:
            spec[0:128] += 100* np.random.randn(len(spec[0:128]))
            print(f"Added high noise at the beginning of the spectrum")

        return spec
    
    @staticmethod
    def fn_acf(x, L, plotopt=1):
        """
        Computes the one-sided Autocorrelation function and optionally plots the ACF.

        Args:
        x (numpy.ndarray): Sequence whose ACF needs to be calculated.
        L (int): Number of lags up to which the ACF needs to be calculated.
        plotopt (int, optional): Option for plotting. 1 (Yes, Default), 0 (No plot).

        Returns:
        acfx (numpy.ndarray): Values of the ACF (first value at lag zero is unity).
        lags (numpy.ndarray): Lag values.

        """
        import numpy as np
        import matplotlib.pyplot as plt
        if plotopt != 1 and plotopt != 0:
            raise ValueError("plotopt must be 0 or 1")

        x = x.reshape(-1)  # Ensure x is a 1D array
        c = np.correlate(x, x, mode='full')  # Compute the cross-correlation
        lags = np.arange(-L, L + 1)
        acfx = c[L:(2 * L + 1)] / c[L]  # Normalize to have the first value at lag zero as unity

        if plotopt == 1:
            plt.figure()
            plt.stem(lags, acfx, linefmt='-b', markerfmt='or', basefmt=' ')
            errlim = 2.58 / np.sqrt(len(x))
            plt.plot(lags, np.ones(len(lags)) * errlim, 'r--')
            plt.plot(lags, -np.ones(len(lags)) * errlim, 'g--')
            plt.box(False)
            plt.ylabel('ACF', fontsize=12, fontweight='bold')
            plt.xlabel('Lags', fontsize=12, fontweight='bold')
            plt.xticks(fontsize=12, fontweight='bold')
            plt.yticks(fontsize=12, fontweight='bold')
            plt.gca().set_facecolor((1, 1, 1))
            plt.show()

        return acfx, lags





# %%
if __name__ == "__main__":
    # load modules
    import pandas as pd
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import sys

    import pythonAssist as pa
    from FnWsRange import FnWsRange
    from FnUncertainty_Vlos import FnUncertainty_Vlos
    from FnPeaks import FnPeaks

    workDir = os.path.dirname(r"C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\HighRe\fun")
    sys.path.append(os.path.join(workDir,'fun'))
    sys.path.append(os.path.join(workDir,'src'))
    sys.path.append(os.path.join(workDir,'data', 'nawea', 'srws'))
    sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")

    from spectr import SpectralAnalysis
    import pythonAssist as pa
    import matlab2py as m2p
    from ProcessSRWS import ProcessSRWS

    file = os.path.join(workDir, "data", "nacelle_lidar", "2021-11-10T140500+00")
    file = r"z:\\Projekte\\112933-HighRe\\20_Durchfuehrung\\OE410\\SRWS\\Data\\Bowtie1\\2021\\11\\08\2021-11-08T135033+00"
    ## Get all files in folder with data
    filename, file_extension = os.path.splitext(file)

    m2p.tic()
    # initialize parameters
    inp = pa.struct()
    inp.test = False
    inp.height = [1.25, 15]
    inp.relaxed_height = [1.1, 5]
    inp.prominence = [0, 15]
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
    srws = ProcessSRWS(file, inp)
    data, df, spec_df = srws.Read_SRWS_bin(filename, mode='basic')
    spectrum = spec_df['Spectrum 1']
    max_value = df['MaximumValue 1']
    
    # load the class
    # correct spectra for zero values
    from spectr import SpectralAnalysis
    spectrum = SpectralAnalysis.zero_spectra_correction(spectrum)
    max_value = SpectralAnalysis.series_zero_correction(max_value)

    sa = SpectralAnalysis(spectrum, inp, max_value)
    vlos, snr = sa.RunSpectralAnalysis()

    sys.exit('manual')
    # scale the spectrum 
    scaled_spectrum = sa.FnScaleSpectrum()
    
    # check for zeros at tail
    # corr_spec = sa.detect_spectral_zerotail(spec_df['Spectrum 2'], duplicated_val=0, correct=True)
    # spec_wo_noise = sa.detect_spectral_noisytail(spec_df['Spectrum 1'])
    

    # # flatten the spectrum
    try:
        spec_flat,  spec_real, spec_median, spec_base = sa.FnFlattenSpectrum()
    except ValueError:
        spec_flat,  spec_real, spec_median, spec_base = sa.FnFlattenSpectrum(correct=True)
        
    # # find doppler peak and left and right bases350
    df_peaks = sa.FnDopplerPeaks(spec_flat, peak_mode='prom')
    # # # find the peaks by interpolation and moving average over the cropped peak area
    spec_all = pd.concat([spec_real, df_peaks], axis=1)
    snr = spec_all.apply(lambda s: sa.calc_snr(s.filter(regex="^\d{1,3}").values, peaks=s.doppler_peak, method='area'), axis=1, result_type='expand')
    peak = spec_all.apply(lambda s: sa.FnFindPeaks_exact(s.filter(regex="^\d{1,3}"),s.filter(like='peak')), axis=1, result_type='reduce')
    # # derive the vlos from peak positions
    vlos = sa.spectrum2vlos(peak)
    # # split the snr df into elements
    snr, var_s, var_n = snr.iloc[:,0], snr.iloc[:,1], snr.iloc[:,2]


    m2p.toc()

    sys.exit('manual stop')

    # plot iwes corrected data vs measured vlos data
    id = range(2000)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[id], y=data['Windspeed 3'][id], mode='markers', name="data"))
    fig.add_trace(go.Scatter(x=df.index[id], y=vlos[id], mode='markers', name="iwes"))
    fig.show()

    v3_mean = data['Windspeed 3'].mean()
    corr_v3_mean = vlos.mean()

    # snr = scaled_spectrum.apply(lambda x: sa.calc_snr(x), axis=1, raw=True)

    sys.exit('manual stop')

    # interpolate the whole spectrum (takes too long)
    spec_flat_interp = sa.df_spec_interp(spec_flat)
    # apply moving average to the interpolated spectrum
    S_ma = sa.FnSpectral_moveavg(spec_flat_interp)

    df_peaks, df_prop = sa.FnFindPeaksCentroid1D(S_ma)

    centroid_vlos = sa.spectrum2vlos(df_prop['h_center']/10 - 256)
    maximum_vlos = sa.spectrum2vlos(df_prop['doppler_peak']/10 - 256)
    median_vlos = sa.spectrum2vlos(df_prop['median_peak']/10 - 256)


    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spec_flat.columns, y=spec_flat.iloc[4,275:290].values, mode='markers', name="flat"))
    fig.add_trace(go.Scatter(x=S_ma.columns/10, y=S_ma.iloc[4,2750:2900].values, mode='markers', name="linear"))
    fig.show()


    # plot the snr plots against vlos to see what happens at outliers
    id = range(2000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=snr.index[id], y=snr[id].values, mode='markers', name="snr"))
    fig.add_trace(go.Scatter(x=vlos.index[id], y=vlos[id].values, mode='markers', name="vlos"))
    fig.show()

    # %% plotly plot
    import plotly.graph_objects as go
    fig = go.Figure()
    for i in range(5):
        fig.add_trace(go.Scatter(x=spec_flat.columns, y=spec_flat.iloc[i,:], mode='markers', name="flat{}".format(str(i))))
    fig.show()

    fig = go.Figure()
    rng = [13880, 13977, 15721, 16836]
    for i in rng:
        fig.add_trace(go.Scatter(x=spec_flat.columns, y=spec_flat.iloc[i,:], mode='markers', name="flat{}".format(str(i))))
        print(find_peaks(spec_flat.iloc[i,:], height=[1.1,5], distance=64, prominence=[0,10], width=1)[0])
        # fig.add_trace(go.Scatter(x=spec_flat.columns, y=spec_real.iloc[i,:], mode='markers', name="flat{}".format(str(i))))
        fig.show()

# %% Test Mikael spectrum 

    if inp.test == True:
        spectrum1= pd.Series([0, 0, 0, 1, 2, 3, 3, 4, 5, 3, 2, 1, 0, 0, 0, 0], name='Spectrum1')
        spectrum2 = pd.Series([0, 0, 1, 2, 3, 3, 4, 5, 4, 3, 3, 2, 1, 0, 0, 0], name='Spectrum2')
        freq = pd.Series(np.arange(1,17), name='Frequency')
        mikael_test_spectrum = pd.concat([spectrum1, spectrum2, freq], axis=1)

        # freq_new = pd.Series(np.linspace(0.1,16,160 ), name='Frequency')
        # spec_interp = spectral_interp1d(mikael_test_spectrum['Spectrum2'], freq_new)
        freq_new = pd.Series(np.linspace(0,512,5120 ), name='Frequency')
        spec_interp = spectral_interp1d(spec_flat.iloc[0,:], freq_new)
        spec_ma = FnSpectral_moveavg(spec_interp, window_size=5, ax=1)
        # inp.height = [1,10]
        # inp.relaxed_height = [1,10]
        # inp.prominence = [0,10]
        # inp.distance = 10
        # inp.width = 1

        df_peaks, df_prop = FnFindPeaksCentroid1D(spec_interp, inp)
    

# %%
    if inp.read_dtu == True:
        df_dtu = pd.read_csv("../data/comparison/BenchmarkLOSuvw_update_2022-04-26T180000+00.csv")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.head().index, y=df['Windspeed 1'].head(), mode='markers', name="data"))
        fig.add_trace(go.Scatter(x=df_dtu.head().index, y=df_dtu['v LOS R2D1'].head(), mode='markers', name="mikael"))
        fig.add_trace(go.Scatter(x=centroid_vlos.head().index, y=centroid_vlos.head().values, mode='markers', name="centroid"))
        fig.add_trace(go.Scatter(x=median_vlos.head().index, y=median_vlos.head().values, mode='markers', name="median"))
        fig.add_trace(go.Scatter(x=maximum_vlos.head().index, y=maximum_vlos.head().values, mode='markers', name="max"))
        fig.show()


# %% Extras

    # df_prop['doppler_peak'] = df_peaks['doppler_peak']
    # df_prop['median_peak'] = [np.median([df_prop['left_ips'][i][peak_idx[i]],df_prop['right_ips'][i][peak_idx[i]]]) for i in range(len(peak_idx))]
    # df_prop['h_center']= [np.mean([df_prop['left_ips'][i][peak_idx[i]],df_prop['right_ips'][i][peak_idx[i]]]) for i in range(len(peak_idx))]
    # df_prop['v_center']= [np.mean([df_prop['peak_heights'][i][peak_idx[i]],df_prop['prominences'][i][peak_idx[i]]]) for i in range(len(peak_idx))]

# check the following points 09.01.23:
# Index   vlos
# 10958	9.004863
# 16160	9.114902

# Remarks:
# inp.prominence = [0,10] - > results in many Vlos below 0, although a peak exists for about 4 m/s
# inp.prominence = [0, 3] - > improved version to reduce Vlos below 0, some bad peaks occurring at below 2 prominence
# inp.width = 1 - > optimum value width of 2 creates overestimation of vlos
# peak_start and peak_end using the width parameter  from find_peaks allows for the spikes far away to contribute to the median/centroid
