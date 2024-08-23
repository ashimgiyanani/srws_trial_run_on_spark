import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pythonAssist as pa
from enum import IntFlag, auto

class qc1_flags(IntFlag):
    valid = auto()
    invalid = auto()
    doubtful = auto()
    missing = auto()

class qc2_flags(IntFlag):
    valid = auto()
    sensor_failure = auto()
    all_checks_failed = auto()
    range_failed = auto()
    bad_quality = auto()

class quality_control:
    """
    quality_control - class to perform quality control on data

    Syntax:  

    Inputs:
        df  - dataframe with flat data with time as index and other general variables for wind energy
        path - path to the raw data or 10 min avg data
        fmt - format of iSpin data 10 min avg. data (1/600 Hz) or raw data (10 Hz)
        tstart - start of datetime interval (datetime format)
        tend - end of datetime (datetime format)

    Outputs:
        data_ispin - pandas dataframe output

    Example:

    Raises:
    adding qc_flag in decimal will create errors if the same value is added
    http://karthur.org/2021/fast-bitflag-unpacking-python.html

    modules required: none
    classes required: none
    Data-files required: none

    See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2

    References:
    Author name, year, Title, Link
    Website, Title, link,

    Author: Ashim Giyanani, Research Associate
    Fraunhofer Institute of Wind Energy
    Windpark planning and operation department
    Am Seedeich 45, Bremerhaven
    email: ashim.giyanani@iwes.fraunhofer.de
    Git site: https://gitlab.cc-asp.fraunhofer.de/giyash
    Created: 06-08-2020; Last revision: 12-May-200406-08-2020
    """

    inp = pa.struct()
    inp.tiny = 12
    inp.Small = 14
    inp.Medium = 16
    inp.Large = 18
    inp.Huge = 22
    inp.WriteData = 1
    plt.rc('font', size=inp.Small)  # controls default text sizes
    plt.rc('axes', titlesize=inp.Small)  # fontsize of the axes title
    plt.rc('axes', labelsize=inp.Medium)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=inp.Small)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=inp.Small)  # fontsize of the tick labels
    plt.rc('legend', fontsize=inp.Medium)  # legend fontsize
    plt.rc('figure', titlesize=inp.Huge)  # fontsize of the figure title
    # %matplotlib qt
    plt.close('all')

    def __init__(self, df):
        df['qc1'] = 0
        df['qc2'] = 0
        self.df = df
        # self.inp = inp

    def assign_qc_valid(self, df):

        qc1_valid = (df['qc1'] == 0)
        qc2_valid = (df['qc2'] == 0)
        df.loc[qc1_valid, 'qc1'] = qc1_flags.valid
        df.loc[qc2_valid, 'qc2'] = qc2_flags.valid

        self.df = df.copy()

        return self.df

    def detect_missing(self, channel_name=None):
        # detect nans or inf in time series data.
        # if no channel_name is given, rows of df checked for nan/inf and marked as flag if nans exceed 50% of columns size.
        # allows multiple channel_name as a list (note: checks if all are nans/inf)

        if channel_name is None:
            finite_idx = ((~self.df.isna()) | np.isfinite(self.df)).sum(
                axis=1) > (len(self.df.columns) // 2)
        else:
            finite_idx = (np.isfinite(self.df[channel_name])
                          | self.df[channel_name].notnull()).all(axis=1)

        self.df.loc[~finite_idx,'qc1'] = self.df.loc[~finite_idx,'qc1'] | qc1_flags.missing
        self.df.loc[~finite_idx,'qc2'] = self.df.loc[~finite_idx,'qc2'] | qc2_flags.all_checks_failed

        return self.df

    def detect_time_gaps(self, **kwargs):
        # detect gaps in datetime index or series

        # dt_in  - input as datetime.timedelta()
        # channel_name - column name in the df to be used for detection
        # correct_gaps - true/false for correcting gaps using linear range creation
        #
        import pandas as pd
        from datetime import timedelta

        channel_name = kwargs.setdefault('channel_name', None)
        correct_gaps = kwargs.setdefault('correct_gaps', False)
        variable_sample_freq = kwargs.setdefault('variable_sampl_freq', False)
        freq_range = kwargs.setdefault('freq_range', [0, 600])

        try:
            dt_in = kwargs.setdefault('dt_in',
                                    (self.df.index[1] - self.df.index[0]))

            dt_in = pd.to_timedelta(dt_in, 'ns')
            if channel_name is None:
                delta_T = self.df.sort_index().index.to_series().diff().bfill()
            else:
                delta_T = self.df[channel_name].sort_values(by=channel_name).apply(
                    lambda x: pd.to_timedelta(x, 'ns')).diff().bfill()

            if not variable_sample_freq:
                gaps_idx = delta_T != dt_in
                gaps = delta_T[delta_T != dt_in]
            else:
                gaps_idx = delta_T.between(
                    timedelta(seconds=freq_range[0]),
                    timedelta(seconds=freq_range[1])).apply(lambda x: not (x))
        except IndexError:
            gaps_idx = pd.Series([True]*len(self.df))

        # add into the qc flag columns
        self.df.loc[gaps_idx.values,
                    'qc1'] = self.df.loc[gaps_idx.values,
                                         'qc1'] | qc1_flags.invalid
        self.df.loc[gaps_idx.values,
                    'qc2'] = self.df.loc[gaps_idx.values,
                                         'qc2'] | qc2_flags.all_checks_failed

        if correct_gaps:
            self.df = self.df.asfreq(dt_in.to_pytimedelta())

        return self.df

    def detect_outliers_ws_range(self, channel_names, ranges):

        # correction for less ranges(lazy) or no ranges given by user
        if len(ranges) != len(channel_names):
            ranges = np.tile(ranges, (len(channel_names), 1))
        elif ranges == None:
            ranges = np.tile([-30, 30], (len(channel_names), 1))
        else:
            pass

        for channel_name, range in zip(channel_names, ranges):
            idx = self.df[channel_name].astype(np.float64).between(
                range[0], range[1], inclusive='both')
        self.df.loc[~idx,
                    'qc1'] = self.df.loc[~idx, 'qc1'] | qc1_flags.invalid
        self.df.loc[~idx,
                    'qc2'] = self.df.loc[~idx, 'qc2'] | qc2_flags.range_failed

        return self.df

    def detect_outliers_range(self, channel_names):
        from outliers import detect_outliers

        for channel_name in channel_names:
            idx = self.df[channel_name].astype(np.float64).apply(
                detect_outliers, alpha=0.05)
            self.df.loc[~idx,
                        'qc1'] = self.df.loc[~idx,
                                             'qc1'] | qc1_flags.invalid
            self.df.loc[~idx,
                        'qc2'] = self.df.loc[~idx, 'qc2'] | qc2_flags.range_failed

        return self.df

    def detect_spikes(self, channel_names, **kwargs):

        from FnPeaks import FnPeaks
        import pythonAssist as pa

        verbose = kwargs.setdefault('verbose', False)
        window_length = kwargs.setdefault('window_length', 100)
        plot_figure = kwargs.setdefault('plot_figure', False)
        circular_data_cols = kwargs.setdefault('circular_data_cols', ['None'])
        circ_pi2pi = kwargs.setdefault('circ_pi2pi', ['None'])
        
        df_sd = pa.multiple_stats_on_df(self.df.filter(channel_names), circular_data_cols=circular_data_cols, stats='std', degrees=True, circ_pi2pi=circ_pi2pi)
        thresh = (df_sd * 6).to_list()

        thresholds = kwargs.setdefault('thresholds', thresh)

        # correction for less thresholds(lazy) or no threshold given by user
        if len(thresholds) != len(channel_names):
            thresholds = np.tile(thresholds, len(channel_names))

        for channel_name, threshold in zip(channel_names, thresholds):
            peaks_idx, Npeaks, peaks_val = FnPeaks(self.df[channel_name],
                                                   threshold, window_length,
                                                   plot_figure, verbose)
            idx = self.df.iloc[peaks_idx].index
            self.df.loc[idx,
                        'qc1'] = self.df.loc[idx,
                                             'qc1'] | qc1_flags.invalid
            self.df.loc[idx,
                        'qc2'] = self.df.loc[idx,
                                             'qc2'] | qc2_flags.bad_quality

        return self.df

    def detect_sudden_changes(self, channel_names, **kwargs):
        # https://www.iese.fraunhofer.de/blog/change-point-detection/


        from changepy import pelt
        from changepy.costs import normal_mean

        plot_figure = kwargs.setdefault('plot_figure', False)
        thresh_mean = kwargs.setdefault('thresh_mean', 1.5) 
        thresh_std = kwargs.setdefault('thresh_std', 3) 

        for channel_name in channel_names:
            self.df[channel_name] = self.df[channel_name].ffill().bfill()
            std = np.nanstd(self.df[channel_name].values)
            mean = np.nanmean(self.df[channel_name].values)
            result = pelt(normal_mean(self.df[channel_name].values, std**2),
                          len(self.df[channel_name].values))

            for cp1, cp2 in zip(result, result[1:]):

                # print(cp1, cp2)
                result_mean = np.nanmean(self.df[channel_name].iloc[cp1:cp2].values)
                result_std = np.nanstd(self.df[channel_name].iloc[cp1:cp2].values)

                if (abs(result_mean - mean) > thresh_mean) | (abs(result_std - std) > thresh_std):
                    idx = self.df.index[cp1:cp2]
                    self.df.loc[
                        idx,
                        'qc1'] = self.df.loc[idx, 'qc1'] | qc1_flags.doubtful
                    self.df.loc[idx,
                                'qc2'] = self.df.loc[idx,
                                                     'qc2'] | qc2_flags.sensor_failure

            if plot_figure == True:
                # Plot the time series
                plt.plot(self.df[channel_name].values)
                # Plot change points
                for cp in result:
                    plt.axvline(cp, color='red', linestyle='--')
                plt.title('Sudden Changes in Time Series')
                plt.xlabel('Data Count')
                plt.ylabel('Value')
                plt.show()

        return self.df

    def detect_stuck_sensor(self, channel_names, thresholds=[6]):
        # flag stuck values in a timeseries

        # correction for less thresholds(lazy) or no threshold given by user
        if len(thresholds) != len(channel_names):
            thresholds = np.tile(thresholds[0], len(channel_names))

        for channel_name, threshold in zip(channel_names, thresholds):
            N_values = self.df[channel_name].value_counts(dropna=True)
            stuck_values = N_values[N_values > threshold].index
            stuck_values_idx = self.df[self.df[channel_name].isin(
                stuck_values)].index

            self.df.loc[stuck_values_idx,
                        'qc1'] = self.df.loc[stuck_values_idx,
                                             'qc1'] | qc1_flags.doubtful
            self.df.loc[stuck_values_idx,
                        'qc2'] = self.df.loc[stuck_values_idx,
                                             'qc2'] | qc2_flags.sensor_failure

        return self.df

    def detect_time_errors(self, channel_names, threshold):
        # flag stuck time (index) in a timeseries

        # correction for less thresholds(lazy) or no threshold given by user
        if len(thresholds) != len(channel_names):
            thresholds = np.tile(thresholds, len(channel_names))
        else:
            thresholds = np.tile(2, len(channel_names))

        return self.df

    def detect_bad_wdir(self, channel_names, **kwargs):
        # flag bad wind direction signal, similar to detect_outlier_ws_range
        
        verbose = kwargs.setdefault('verbose', False)

        # drop the nan values
        df = self.df.copy()

        # range error
        for channel_name in channel_names:
            df.dropna(subset=[channel_name], inplace=True)
            flag_df = df[~df[channel_name].between(0, 360, inclusive='both')]
        if ~flag_df.empty:
            self.df.loc[flag_df.index,
                        'qc1'] = self.df.loc[flag_df.index,
                                             'qc1'] | qc1_flags.doubtful
            self.df.loc[flag_df.index,
                        'qc2'] = self.df.loc[flag_df.index,
                                             'qc2'] | qc2_flags.range_failed
            if verbose:
                print(f"[{pa.now()}]: Corrected {len(flag_df)}/{len(df)} data points for range errors ")
        else:
            if verbose:
                print(f"[{pa.now()}]: No range errors found in wind direction")

        return self.df

    def detect_wake_effects(self, channel_name, wdir_range):
        # flag bad wind direction signal and wind speed signal due to wake effects
        # wdir_range - wind direction ranges to be filtered out in the form of [a1 a2; b1 b2; c1 c2;  d1 d2]

        # prepare the dataframe
        df = self.df.copy()
        df.dropna(subset=channel_name, inplace=True)

        #
        import numpy as np

        m, n = np.shape(wdir_range)
        nSec = len(wdir_range)
        print(f'[{pa.now()}]: number of sectors to be filtered: {nSec}')

        for i in range(m):
            flag_df = df[df[channel_name].between(wdir_range[i][0],
                                                  wdir_range[i][1])]
            if ~flag_df.empty:
                self.df.loc[flag_df.index,
                            'qc1'] = self.df.loc[flag_df.index,
                                                 'qc1'] | qc1_flags.doubtful
                self.df.loc[flag_df.index,
                            'qc2'] = self.df.loc[flag_df.index,
                                                 'qc2'] | qc2_flags.bad_quality
            else:
                continue

        return self.df

    def detect_bad_cnr(self, channel_names, cnr_ranges):
        # cnr range between -30 to 30 dB, valid WS range and low availability at the same time
        df = self.df.copy()
        # keep rows with two non-NA values
        df.dropna(thresh=2, inplace=True)

        for channel_name, cnr_range in zip(channel_names, cnr_ranges):
            flag_df = df[~df[channel_name].between(cnr_range[0], cnr_range[1])]
            if ~flag_df.empty:
                self.df.loc[flag_df.index,
                            'qc1'] = self.df.loc[flag_df.index,
                                                 'qc1'] | qc1_flags.doubtful
                self.df.loc[flag_df.index,
                            'qc2'] = self.df.loc[flag_df.index,
                                                 'qc2'] | qc2_flags.bad_quality

        return self.df

    def detect_bad_availability(self, channel_names, avail_ranges):
        # bad availability < 75 %

        df = self.df.copy()
        df.dropna(threshold=2, inplace=True)

        for channel_name, avail_ranges in zip(channel_names, avail_ranges):
            flag_df = df[~df[channel_name].
                         between(avail_ranges[0], avail_ranges[1])]
            if ~flag_df.empty:
                self.df.loc[flag_df.index,
                            'qc1'] = self.df.loc[flag_df.index,
                                                 'qc1'] | qc1_flags.doubtful
                self.df.loc[flag_df.index,
                            'qc2'] = self.df.loc[flag_df.index,
                                                 'qc2'] | qc2_flags.bad_quality

        return self.df

    def detect_bad_row(self, **kwargs):
        # detect bad rows i.e. rows with all zeros or NaNs, correct using linear interpolation
        return self.df

    def detect_bad_column(self, **kwargs):
        # detect bad columns i.e. columns with all zeros or NaNs,
        return self.df

    def detect_primary_sensor_errors(self):
        # errors due to another variable measured by self
        return self.df

    def detect_secondary_sensor_errors(self):
        # errors due to comparison with another sensor but within the same conditions
        return self.df

    def detect_model_errors(self):
        # errors due to comparison with models or another information
        return self.df

    def detect_high_stddev(self, channel_names, stddev_limits):
        df = self.df.copy()
        df.dropna(threshold=2, inplace=True)

        for channel_name, stddev_limit in zip(channel_names, stddev_limits):
            flag_df = df[~df[channel_name] > stddev_limit *
                         np.nanstd(df[channel_name])]
            if ~flag_df.empty:
                self.df.loc[flag_df.index,
                            'qc1'] = self.df.loc[flag_df.index,
                                                 'qc1'] | qc1_flags.doubtful
                self.df.loc[flag_df.index, 'qc2'] = self.df.loc[
                    flag_df.index, 'qc2'] | qc2_flags.bad_quality

        return df

    @staticmethod
    def generate_gold_dataset(mean_wind, mean_wdir):

        import numpy as np
        import pandas as pd
        import pythonAssist as pa

        np.random.seed(1384)

        # generate a random wind speed data
        up = np.random.randn(8760, 3)
        U_hub = mean_wind
        U = U_hub + up
        df_gold = pd.DataFrame(U, columns=['U1', 'U2', 'U3'])

        # define the bins and weights assigned to respective bins
        wdir_bins = np.linspace(0, 324, 11)
        weights = 0.05 * np.ones(np.shape(wdir_bins))
        # find the index of the bin closest to the mean wind direction
        _, closest_wdir_idx, _ = pa.FnClosestMember([mean_wdir], wdir_bins)
        weights[closest_wdir_idx[0] - 2:closest_wdir_idx[0] + 1] = 0.2
        # Generate random wind direction values using weighted probabilities
        df_gold['wdir'] = np.random.choice(wdir_bins, size=len(U), p=weights)
        # Create a time range with 1-hour intervals
        df_gold['T'] = pd.date_range(start="2021-01-01",
                                     periods=8760,
                                     freq="1H")
        df_gold = df_gold.set_index('T')

        # add nans at regular intervals
        nan_index = np.linspace(0, 8760, 25)
        nan_index[1:] = nan_index[1:] - 1
        df_gold.iloc[nan_index.astype('uint16'), :] = np.nan

        # add a stuck sensor result into the timeseries
        df_gold.loc[:, 'U1'][100:110] = 10.0

        # adding a sudden change in the gold dataset
        df_gold.loc[df_gold.index[7000:8000],
                    df_gold.filter(regex='U').columns] += 10

        # add spikes in data
        df_gold.loc[:, 'U1'][299] = 35
        df_gold.loc[:, 'U1'][150] = -35

        # add bad wdir in data
        df_gold.loc[:, 'wdir'][365] = 365
        df_gold.loc[:, 'wdir'][770] = -5

        # add time not sequential
        new_index = df_gold.index.values
        new_index[200:205] = df_gold.index[100:105].values
        # add time duplication
        new_index[100:105] = df_gold.index[99]
        df_gold.index = pd.Index(new_index)

        # add columns with zeros in all rows
        df_gold['U3'] = 0

        # add rows with zeros, nans and infinite numbers or even numbers like -999
        df_gold.iloc[8700, :] = 0
        df_gold.iloc[8701] = np.nan
        df_gold.iloc[8702, :] = np.inf
        df_gold.iloc[8703, :] = -9999.99
        df_gold.iloc[8704, :] = 999

        # # add empty columns qc1 and qc2
        # df_gold = df_gold.reindex(columns=['U', 'qc1' ,'qc2'])

        return df_gold

    def prepare_df(self, filter_out_regex, replace_nan=[-9999.99, -999]):
        """
        prepares a df where columns with all zeros and satisfying the condition of reg. expres. using filter_out_regex are removed from the df.

        Also assigns empty rows as missing
        """
        # find rows with special missing values (e.g. -999, 999)
        for rep in replace_nan:
            self.df[self.df == rep] = np.nan
        self.df[~np.isfinite(self.df)] = np.nan

        # find columns with zeros in all the rows/instances
        idx_zero_cols = (self.df**2).sum() == 0
        # find columns ending with 4, tending to be zero
        idx_four = self.df.filter(regex=filter_out_regex).columns
        # combined
        idx_combined = (idx_zero_cols.values & self.df.columns.isin(idx_four))
        # print(f"[{pa.now()}]: removed following columns {self.df.columns[idx_combined].to_list()}")
        self.df = self.df.loc[:, ~idx_combined]

        # finding rows with all zeros or nans
        idx_zero = (self.df.sum(axis=1) == 0)
        idx_nan = self.df.isna().all(axis=1)
        idx_combined = (idx_zero & idx_nan)
        # apply quality flags
        self.df.loc[idx_combined,
                    'qc1'] = self.df.loc[idx_combined,
                                         'qc1'] | qc1_flags.missing
        self.df.loc[idx_combined,
                    'qc2'] = self.df.loc[idx_combined,
                                         'qc2'] | qc2_flags.all_checks_failed

        return self.df

    @staticmethod
    def update_bitflags(df, existing, new):
        return df

    @staticmethod
    def qc_range_testing(val, range):
        from metadata_helpers import qc1_flags
        # checks if a value lies within a range or not and returns a basic quality flag
        if val >= range[0] and val <= range[1]:
            qc_flag = qc1_flags.valid
        else:
            qc_flag = qc1_flags.doubtful
        return qc_flag

    @staticmethod
    def string_eval(inp_string):
        # takes in an input string with a list and returns a list,
        # useful for converting lists "[0, 10]" to a list [0, 10]
        import ast
        try:
            lst = ast.literal_eval(inp_string)
        except:
            lst = eval(inp_string)
        return lst

    @staticmethod
    def get_qc_str(IntFlag, qc_flags):
        """get the string values assigned using auto in the class qc1_flags & qc2_flags"""

        flags = qc_flags(IntFlag)
        flag_names = [flag.name for flag in qc_flags if flag & flags]

        return flag_names



if __name__ == "__main__":

    import pythonAssist as pa
    from DataQualityFlags import quality_control, qc1_flags, qc2_flags
    
    # apply quality control
    df_gold = quality_control.generate_gold_dataset(mean_wind=10,
                                                    mean_wdir=270)
    qc = quality_control(df_gold)

    # detect time gaps
    df_gold = qc.detect_time_gaps()

    # detect missing samples
    df_gold = qc.detect_missing(channel_name=['U1', 'U2', 'U3'])  # works

    # detect outliers in ws ranges
    df_gold = qc.detect_outliers_ws_range(channel_names=['U1', 'U2'],
                                          ranges=[[-30, 30]])  # works

    # detect stuck sensors
    df_gold = qc.detect_stuck_sensor(channel_names=['U1', 'U2'],
                                     thresholds=[6, 6])  # works

    df_gold = qc.detect_bad_wdir(channel_names=['wdir'])  # works

    # df_gold = qc.detect_sudden_changes(channel_names = ['U1', 'U2', 'U3'],plot_figure=True) # does not work

    df_gold = qc.detect_wake_effects(channel_name='wdir',
                                     wdir_range=[[10, 30], [150, 210]])

    # df_gold = qc.detect_spikes(channel_name='U1', threshold=5, window_length=100, plot_figure=True, verbose=True)
    # drop a row of dates
    # df_gold.drop(index='2021-12-31 21:00:00')

    # filter out missing values using qualty flags qc1 and qc2
    # df_gold[df_gold.qc1==qc1_flags.missing.value]

    # Check and validate different results
    df_gold.iloc[365, :]['qc1']

    # prepare df
    df_gold = qc.prepare_df(filter_out_regex='3$', replace_nan=[-9999.99, 999])

    df_gold = qc.assign_qc_valid(df_gold)

