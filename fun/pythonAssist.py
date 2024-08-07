from typing import List, Any, Optional, Union, Tuple
from matplotlib.pyplot import Axes
import datetime
import warnings

class struct():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


# function to check if a variable exists in local or global workspace
def exists(var):
    var_exists = var in locals() or var in globals()
    return var_exists
    # Syntax:
    #    exists("variable_name")
    # Output:
    #    True or False

def myround(x, base=5):
    import numpy as np
    # function to round the x value to the nearest muliple of base
    # returns 9 for myround(10.25,3) -> 3*3 = 9
    return base * np.round(x/base)

def now():
    # returns the current datetime
    import datetime
    return datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')

def today():
    from datetime import datetime
    today =  datetime.today().strftime("%d.%m.%Y")
    return today

def find_nearest(array, value):
    """
    find the nearest value in the array corresponding to the value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def FnClosestMember(values, array):
# function to find the closest value for every 'value' in 'array'', size(values) > size(array)
    import numpy as np
    import scipy

# 	values = -5 + 35*np.random.random((5,1));
# 	array = np.linspace(0,25,6);
    #make sure that array is a numpy array
    array = np.array(array)
    values = np.array(values)

    # if soreted arrays are necessary
    # idx_sorted = np.argsort(array)
    # sorted_array = np.array(array[idx_sorted])

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_less] -= 1
    close_dist = (values - array[idxs])
    close_idx = idxs
    close_values = array[idxs]
    return close_dist, close_idx, close_values

def remove_exponent(x):
# removes the exponent and proves a decimal representation of a number i.e. removes trailing zeros
    from decimal import Decimal
    d = Decimal(x)
    return d.quantize(Decimal(1)) if d == d.to_integral() else d.normalize()

# Example:
# values = -5 + 35*rand(1,100);
# array = 0:5:25;
# close_dist, close_idx, close_value = FnClosestMember(values, array)

def save_workspace(filename, names_of_spaces_to_save, dict_of_values_to_save):
    import shelve
# '''
#     filename = location to save workspace.
#     names_of_spaces_to_save = use dir() from parent to save all variables in previous scope.
#         -dir() = return the list of names in the current local scope
#     dict_of_values_to_save = use globals() or locals() to save all variables.
#         -globals() = Return a dictionary representing the current global symbol table.
#         This is always the dictionary of the current module (inside a function or method,
#         this is the module where it is defined, not the module from which it is called).
#         -locals() = Update and return a dictionary representing the current local symbol table.
#         Free variables are returned by locals() when it is called in function blocks, but not in class blocks.
#
#     Example of globals and dir():
#         >>> x = 3 #note variable value and name bellow
#         >>> globals()
#         {'__builtins__': <module '__builtin__' (built-in)>, '__name__': '__main__', 'x': 3, '__doc__': None, '__package__': None}
#         >>> dir()
#         ['__builtins__', '__doc__', '__name__', '__package__', 'x']
# '''
    print('save_workspace')
    print('C_hat_bests') in names_of_spaces_to_save
    print(dict_of_values_to_save)
    my_shelf = shelve.open(filename,'n') # 'n' for new
    for key in names_of_spaces_to_save:
        try:
            my_shelf[key] = dict_of_values_to_save[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            #print('ERROR shelving: {0}'.format(key))
            pass
    my_shelf.close()

def load_workspace(filename, parent_globals):
    import shelve
#     '''
#         filename = location to load workspace.
#         parent_globals use globals() to load the workspace saved in filename to current scope.
#     '''
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        parent_globals[key]=my_shelf[key]
    my_shelf.close()

# an example script of using this:
# x = 3
# save_workspace('a', dir(), globals())

def FindLastRows(ws):
# script with iteration to find the last row with values in it
    nrows = ws.max_row
    lastrow = 0
    if nrows > 1000:
        nrows = 1000
    while True:
        if ws.cell(nrows, 3).value != None:
            lastrow = nrows
            break
        else:
            nrows -= 1
    return nrows

def create_pdf():
    pdf_file = 'multipage.pdf'

    can = canvas.Canvas(pdf_file)

    can.drawString(20, 800, "First Page")
    can.showPage()

    can.drawString(20, 800, "Second Page")
    can.showPage()

    can.drawString(20, 700, "Third Page")
    can.showPage()

    can.save()

def add_image():
# https://code-maven.com/add-image-to-existing-pdf-file-in-python
    from PyPDF2 import PdfFileWriter, PdfFileReader
    import io

    in_pdf_file = 'multipage.pdf'
    out_pdf_file = 'with_image.pdf'
    img_file = '../../static/img/code_maven_440x440.png'

    packet = io.BytesIO()
    can = canvas.Canvas(packet)
    #can.drawString(10, 100, "Hello world")
    x_start = 0
    y_start = 0
    can.drawImage(img_file, x_start, y_start, width=120, preserveAspectRatio=True, mask='auto')
    can.showPage()
    can.showPage()
    can.showPage()
    can.save()

    #move to the beginning of the StringIO buffer
    packet.seek(0)

    new_pdf = PdfFileReader(packet)

    # read the existing PDF
    existing_pdf = PdfFileReader(open(in_pdf_file, "rb"))
    output = PdfFileWriter()

    for i in range(len(existing_pdf.pages)):
        page = existing_pdf.getPage(i)
        page.mergePage(new_pdf.getPage(i))
        output.addPage(page)

    outputStream = open(out_pdf_file, "wb")
    output.write(outputStream)
    outputStream.close()

def ReplaceZeros(y):
# replace zeros using linear interpolation
    from scipy.interpolate import interp1d
    y = y.to_numpy()
    idx = np.where(y!=0)
    x = np.arange(len(y))
    f = interp1d(x[idx], y[idx])
    ynew = f(x)
    return ynew

def ReplaceNaNs(y):
# replace NaNs using linear interpolation
    from scipy.interpolate import interp1d
    y = y.to_numpy()
    idx = np.where(y!=np.nan)
    x = np.arange(len(y))
    f = interp1d(x[idx], y[idx])
    ynew = f(x)
    return ynew

def LHospitalsDivision(a, b):
    # performs standard division, but 0/0 results in a 1
    return (b and a / b or 1)  # a / b

def date_range_fs(duration, fs, start=0):
    """ Create a DatetimeIndex based on sampling frequency and duration
    Args:
        duration: number of seconds contained in the DatetimeIndex
        fs: sampling frequency
        start: Timestamp at which de DatetimeIndex starts (defaults to POSIX
               epoch)
    Returns: the corresponding DatetimeIndex
    """
    return pd.to_datetime(
        np.linspace(0, 1e9*duration, num=fs*duration, endpoint=False),
        unit='ns',
        origin=start)

def notify_win(e):
    import datetime
    from plyer import notification
    notification.notify(
        #title of the notification,
        title = "Python Script needs attention".format(datetime.date.today()),
        #the body of the notification
        message = "Error: {0}\n".format(e),
        #creating icon for the notification
        #we need to download a icon of ico file format
        # app_icon = "Paomedia-Small-N-Flat-Bell.ico",
        # the notification stays for 50sec
        timeout  = 50
    )
    #sleep for 4 hrs => 60*60*4 sec
    #notification repeats after every 4hrs
    # time.sleep(60*60*4)

def clean_nested(l):
# script to remove empty sublists from a nested list
# https://stackoverflow.com/questions/60891417/removing-empty-sublists-from-a-nested-list
    cleaned = []
    for v in l:
        if isinstance(v, list):
            v = clean_nested(v)
            if not v:
                continue
        cleaned.append(v)
    return cleaned

# Reference: https://stackoverflow.com/questions/37079175/how-to-remove-a-column-from-a-structured-numpy-array-without-copying-it
def view_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to keep.

    Returns a view of the array `a` (not a copy).
    """
    import numpy as np
    dt = a.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                formats=formats,
                offsets=offsets,
                itemsize=itemsize))
    b = a.view(newdt)
    return b

def remove_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to remove.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    keep_names = [name for name in dt.names if name not in names]
    return view_fields(a, keep_names)

def add_top_column(df, top_col, inplace=False):
    # Create multi-index top column and assign the dataframe as a subcolumn
    # REf: https://stackoverflow.com/questions/51021468/can-sub-columns-be-created-in-a-pandas-data-frame
    import pandas as pd

    if not inplace:
        df = df.copy()

    df.columns = pd.MultiIndex.from_product([[top_col], df.columns])
    return df

def remove_columns(df, mode='nan+0'):
    """
         remove columns with zeros or NaNs from a dataframe
         mode = 'nan+0' removes columns with nans and zeros, mode=0 considers only zeros.
    """
    # find if there are 0s in all the columns in a row, 
    # note that for using power of 2, dtype should be valid, therefore uint64

    import pandas as pd
    temp_df = df.astype('float64').copy() # applicable to spectrum only

    # FIXME: gettting error for astype
    # idx_0 =  temp_df.loc[:,(temp_df**2).sum()==0].columns
    df_shape_0 = temp_df.loc[:,(temp_df**2).sum()!=0].shape[1]

    if (mode == 'nan+0') & (df_shape_0 == 0):
        df = temp_df.loc[:, (df**2).sum() != 0]
    elif (mode == '0') & (df_shape_0 == 0):
        df = temp_df.loc[:, (df!=0).any(axis=0)]
    elif (df_shape_0 == temp_df.shape[1]-1):
        df = temp_df.copy()
        # print('dataframe returned without removing columns')
    elif  (df_shape_0 > 256) & (df_shape_0 < 511):
        zero_streaks = (temp_df.median() == 0).astype(int).diff().eq(-1).cumsum() * (temp_df.median() == 0)
        zero_sequence_idx = zero_streaks[zero_streaks==1].index
        temp_df = temp_df.drop(zero_sequence_idx, axis=1)
        
        # find the symmetrical shape of noise base spectrum
        if zero_sequence_idx[0] > 256:
            valid_idx = zero_sequence_idx - len(zero_sequence_idx)
        elif zero_sequence_idx[0] < 256:
            valid_idx = zero_sequence_idx - len(zero_sequence_idx)
        
        # flip and append spectrum
        padding_spec = temp_df.iloc[:,valid_idx].iloc[:,::-1]
        padding_spec.columns = np.flip(511-zero_sequence_idx)
        if padding_spec.columns[0] == 0:
           temp_df = pd.concat([padding_spec, temp_df], axis=1, ignore_index=True)
        elif padding_spec.columns[-1]==511:
            temp_df = pd.concat([temp_df, padding_spec], axis=1, ignore_index=True)
    else:
        warnings.warn('Something went wrong, test the function')

    if (df.dtypes == 'uint16').all():
        temp_df.astype('uint16')

    return df

def replace_rows(df, replace):
    """
    replace rows in a dataframe containing all zeros/nans with ffill / bfill / linear interpolation
    """
    import pandas as pd

    if not df.empty:
        mask = df.loc[(df**2).sum(axis=1) == 0]
        dfc = df.copy()
        dfc.loc[mask.index,:] = np.nan

        if (len(mask)/len(df) > 0.01):
            print('Too many zeros/nans in the dataframe')
            df = dfc.copy()
            pass
        elif (replace == 'bfill') | (replace == 'ffill'):
            df = dfc.ffill().bfill()
        elif replace == 'interpolate':
            df = dfc.interpolate(method='linear', axis=0)
    else:
        df = pd.DataFrame()

    return df

def remove_rows(df, mode):
    """
        remove rows with zeros or NaNs from a dataframe, mode options 'nan+0' / 0
    """
    if (mode == 'nan+0'):
        df = df.loc[(df**2).sum(axis=1) != 0]
    elif mode == '0':
        df = df.loc[(df!=0).any(axis=1)]
    else:
        df = df.copy()
        print('dataframe returned without removing rows')
    return df

def signal_outliers(df, win_length=60, sigma=3, detrend=False):
    # find the outliers in the signal following a moving average of window length win
    from scipy import signal
    import pandas as pd
    import numpy as np

    df.interpolate(inplace=True)
    lag = int((win_length-1)/2)

    if detrend:
        window_val = []
        df_temp=df.rolling(window=win_length, step=int((win_length-1)/2)).apply(lambda x: window_val.append(signal.detrend(x)) or 0, raw=True)
        df_temp = df_temp.reset_index(drop=True)
        ma_std = pd.Series(np.nan, range(len(df)))
        ma_std.iloc[df_temp.dropna().index] = np.std(window_val, axis=1)
        # ma_std[range(win_length)] = ma_std[-1]
    else:
        ma_std = df.rolling(window=win_length, step=int((win_length-1)/2)).std().shift(periods=-lag)

    ma_mean = df.rolling(window=win_length, step=int((win_length-1)/2)).mean().shift(periods=-lag)
    
    # filling in NaN since the values are mean and std for the curves over a limited window centers
    ma_std = ma_std.reindex(df.index).interpolate()
    ma_mean = ma_mean.reindex(df.index).interpolate()

    # Forward and backward filling using neighbouring values
    ma_mean = ma_mean.ffill().bfill()
    ma_std = ma_std.ffill().bfill()

    outliers = df[abs(df - ma_mean) > sigma*ma_std]

    return outliers, ma_mean, ma_std

import numpy as np
from scipy.signal import detrend

def window_op(x, method='std'):
    """perform operations on a rolling window (currently only nanstd/nanmean/nanmedian)"""
    if method=='std':
        x_new =  np.nanstd(detrend(x), ddof=1)
    if method=='mean':
        x_new = np.nanmean(x,ddof=1)
    if method == 'median':
        x_new = np.nanmedian(x, ddof=1)
    return x_new

def signal_outliers_vectorized(df, win_length=60, sigma=3):
    # find the outliers in the signal following a moving average of window length win
    from scipy import signal
    import pandas as pd
    import numpy as np
    from pythonAssist import window_op

    df.interpolate(inplace=True)
    lag = int((win_length-1)/2)

    ma_std = df.rolling(window=win_length, step=int((win_length-1)/2)).apply(lambda x: window_op(x), raw=True).shift(periods=-lag)
    ma_std = ma_std.reindex(df.index).interpolate()
    ma_std = ma_std.ffill().bfill()

    ma_mean = df.rolling(window=win_length, step=int((win_length-1)/2)).mean().shift(periods=-lag)
    ma_mean = ma_mean.reindex(df.index).interpolate()
    ma_mean = ma_mean.ffill().bfill()

    outliers = df[abs(df - ma_mean) > sigma*ma_std]

    return outliers, ma_mean, ma_std

def import_modules(import_file):
    with open(import_file) as f:
        code = f.read()

    return exec(code)

def import_toml(toml_file):
    import tomli
    with open(toml_file, mode="rb") as f:
        config = tomli.load(f)

    return config

def dict2struct(dictionary):
    # function to convert a dict type object to a struct like object(accessible via dot convention as in Matlab)
    # Remarks: input should be a pure dict at all levels
    from collections import namedtuple
    from recordclass import (recordclass, make_dataclass)

    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = dict2struct(value)

    # convert the dictionary to a namedTuple
    Struct = make_dataclass('Struct', dictionary.keys(), iterable=True)
    # Struct = namedtuple('Struct', dictionary.keys())
    struct = Struct(**dictionary)
    return struct

def cleanup_dataframe(df, sort_by_column):
    """ cleans up the input dataframe """
    # add all the steps here from RunCompare_prep_srwws

    # clear rows with only NaNs
    df = out.srws.df_1min.sort_values(by ='datetime')

def represent(obj):
    from IPython.lib.pretty import pretty
    return pretty(obj)

def mutate_obj(input_obj):
    from collections import namedtuple
    obj = namedtuple('input_obj', input_obj.__fields__)
    output_obj = obj(*input_obj)
    return output_obj

def write_parquet(filepath, data2pq):
    """ script to write pandas dataframe to parquet files """
    import pyarrow as pw
    import pyarrow.parquet as pq

    try:
        data2pq.index = data2pq.index.tz_convert('UTC')
    except AttributeError:
        data2pq.index = pd.to_datetime(data2pq.index, utc=True)
    Data_table = pw.Table.from_pandas(data2pq)
    pq.write_table(Data_table, filepath, allow_truncated_timestamps=True, coerce_timestamps='us', compression='GZIP', compression_level=9)

def write_fastparquet(filepath, data2pq):
    """script to write pandas dataframe to parquet files using fastparquet with append=True function """
    from fastparquet import write
    import pytz
    # from fastparquet import ParquetFile

    data2pq.index = data2pq.index.tz_convert(pytz.utc)
    # data2pq.index = pd.DatetimeIndex(data2pq.index)
    # data2pq.index = data2pq.index.strftime('%Y-%m-%d %H:%M:%S.%f %Z%z')
    try:
        write(filepath, data2pq, object_encoding='infer', has_nulls='infer',compression='GZIP')
    except ValueError:
        data2pq['ID'] = data2pq['ID'].astype(np.int32)
        write(filepath, data2pq, object_encoding='infer',has_nulls='infer', compression='GZIP')

def date_converter(date: Union[str, datetime.date, datetime.datetime]) -> datetime.date:
    """Convert input date to datetime.date format.

    Args:
        date: Date to be converted.
    Returns:
        Converted date in datetime.date format.

    Raises:
        TypeError: If input date is not type string, datetime.date, or
            datetime.datetime.
    Ref: https://github.com/e-hulten/july/blob/master/src/july/utils.py
    """
    if isinstance(date, str):
        return dt.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, datetime.date):
        # Check this last, as isinstance(<datetime object>, datetime.date)
        # is True as well.
        return date
    else:
        raise TypeError(
            "Expected 'date' to be type: [str, datetime.date, datetime.datetime]. "
            f"Got: {type(date)}."
        )


def date_range(
        start_date: Union[str, datetime.date, datetime.datetime],
        end_date: Union[str, datetime.date, datetime.datetime],
    ) -> List[datetime.date]:
    """Create rate of datetime.dates from start_date and end_date.

    Args:
        start_date: First date of date range.
        end_date: Last date of date range (inclusive).
    Returns:
        List of all dates in range [start_date, end_date].
    REf: https://github.com/e-hulten/july/blob/master/src/july/utils.py
    """
    start_date = date_converter(start_date)
    end_date = date_converter(end_date)
    rng_diff = end_date - start_date
    return [start_date + timedelta(days=x) for x in range(0, rng_diff.days + 1)]

def _nanargmax(arr, ax=0):
    """
    script to find the nanargmax of without the error all-nan slice, returns NaN if all NaN slice
    """
    if (np.isnan(arr).all() or not arr.any()):
        val = np.nan
    else:
        val = np.nanargmax(arr, axis=ax)
    
    return val

def rotate_list(x,n=1,direction='right'):
    """
    rotate a list to the right [1,2,3,4]-> [4,1,2,3] or vice-versa with n number of elements
    """
    if direction == 'right':
        y = x[-n:] + x[:-n]
    else:
        y = x[n:] + x[:n]

    return y
    
def convert_by_name(type_name, value):
    """
    dynamically convert the python datatype, type_name is the type name (e.g. str, float, int)
    value - value of the object to covert
    Ref:https://stackoverflow.com/questions/70958215/dynamically-convert-python-datatype
    Remarks: only works in the current compiler, cannot be implemented as a function method
    """
    trusted_types = ["int", "float", "complex", "bool", "str"] ## others as needed
    if type_name in trusted_types:
        return getattr(__builtins__, type_name)(value)
    else:
        return globals().get(type_name)

    return value
# print(convert_by_name("bool", 1))

def map_index_level(index, mapper, level=0):
    """
    Returns a new Index or MultiIndex, with the level values being mapped.
    df = pd.DataFrame([[1,2],[3,4]])
    df.index = pd.MultiIndex.from_product([["a"],["i","ii"]])
    df.columns = ["x","y"]

    df.index = map_index_level(index=df.index, mapper=str.upper, level=1)
    df.columns = map_index_level(index=df.columns, mapper={"x":"foo", "y":"bar"})

    # Result:
    #       foo  bar
    # a I     1    2
    #   II    3    4    
    
    """
    assert(isinstance(index, pd.Index))
    if isinstance(index, pd.MultiIndex):
        new_level = index.levels[level].map(mapper)
        new_index = index.set_levels(new_level, level=level)
    else:
        # Single level index.
        assert(level==0)
        new_index = index.map(mapper)
    return new_index

def resample_mean(x, threshold=2, window=10, bounds = [2, 25]):
    """
    # perform nanmean on resampled data
    """
    # filter peaks using moving avergage window
    from FnPeaks import FnPeaks
    peaks_idx, Npeaks, peaks_val = FnPeaks(x, threshold=threshold, window=window)
    x[peaks_idx]=np.nan

    # filter values outside the bounds
    x[(x < bounds[0]) | (x > bounds[1])] = np.nan

    return np.nanmean(x)

def resample_median(x, threshold=2, window=10, bounds = [2, 25]):
    """
    # perform nanmedian on resampled data
    """

    # filter peaks using moving avergage window
    from FnPeaks import FnPeaks
    peaks_idx, Npeaks, peaks_val = FnPeaks(x, threshold=threshold, window=window)
    x[peaks_idx]=np.nan

    # filter values outside the bounds
    x[(x < bounds[0]) | (x > bounds[1])] = np.nan

    return np.nanmedian(x)

def resample_std(x, threshold=2, window=10, bounds = [2, 25]):
    """
    # perform nanstd on resampled data
    """

    # filter peaks using moving avergage window
    from FnPeaks import FnPeaks
    peaks_idx, Npeaks, peaks_val = FnPeaks(x, threshold=threshold, window=window)
    x[peaks_idx]=np.nan

    # filter values outside the bounds
    x[(x < bounds[0]) | (x > bounds[1])] = np.nan

    return np.nanstd(x)

def multiple_stats_on_df(df, circular_data_cols, stats='mean', degrees=False, circ_pi2pi=None):
    """
    function to perform stats on a df with two different data types one linear and one circular,
    e.g. in case of wind speed and wind direction within the data
    Input:
        df - non homogeneous dataframe
        circular_data_cols  - list of columns containing circular data (note: input/output in radians)
        stats - mean / std dev / variance -> 'mean', 'std', 'var' or 'median'
        degrees: True/False if input is in degrees -> means the output is also in degrees
        kwargs['circ_pi2pi'] - list containing True/False for using the default bounds (False) of [0, 2*pi], state True for [-pi, pi] bounds

    Output:
        df  - mean values along the index (rows) axis returning a df with one value / column

    reference: numpy.nanmean, np.nanstd, np.nanvar, scipy.stats.circmean, sp.circstd, sp.circvar

    Note: the high/low limits do not apply for median estimates, need to improve the algorithm
    """
    import numpy as np
    import scipy.stats as sp

    def circmedian(angs):
        """
        still to be implemented in scipy.stats
        https://github.com/scipy/scipy/issues/6644
        """
        pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
        pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
        pdists = np.abs(pdists).sum(1)
        return angs[np.argmin(pdists)]
    
    # clean the df for columns that will create an error

    if stats == 'mean':
        func = np.nanmean
        func_circ = sp.circmean
    elif stats == 'std':
        func = np.nanstd
        func_circ = sp.circstd
    elif stats == 'median':
        func = np.nanmedian
        func_circ = circmedian
    elif stats == 'var':
        func = np.nanvar
        func_circ = sp.circvar

    # take the mean of columns containing only nans in all rows
    mean_df_nans = df.loc[:,df.isna().all(axis=0)].mean()
    # remove the nan columns
    df = df.loc[:,~df.isna().all(axis=0)]
    # remove exptremely high values
    df[df.abs() > 1e7] = np.nan

    mean_df = df.apply(func, axis=0)
    if degrees is False:
        circ_mean_df = df.filter(circular_data_cols).apply(func_circ, axis=0)
        circ_mean_df_nondef = df.filter(circ_pi2pi).apply(func_circ, axis=0, high=np.pi, low=-np.pi)
    else: # FIXME for degrees =True, the output shall be in degrees
        circ_mean_df = df.filter(circular_data_cols).apply(np.deg2rad).apply(func_circ, axis=0, nan_policy='omit').apply(np.rad2deg)
        circ_mean_df_nondef = df.filter(circ_pi2pi).apply(np.deg2rad).apply(func_circ, axis=0, high=np.pi, low=-np.pi, nan_policy='omit').apply(np.rad2deg)

    circ_mean_df.update(circ_mean_df_nondef)

    # updates  the values of circ_mean_df in mean df
    cols = list(df.columns)
    new_df = circ_mean_df.combine_first(mean_df)
    new_df = new_df.combine_first(mean_df_nans)
    new_df = new_df[cols]

    return new_df

def dict_to_xarray_metadata(xr_obj, metadata_dict, prefix=""):
    for key, value in metadata_dict.items():
        # Create a unique key for the metadata attribute
        attr_key = prefix + key if prefix else key

        if isinstance(value, dict):
            # If the value is a dictionary, recursively call the function
            dict_to_xarray_metadata(xr_obj, value)
        elif isinstance(value, (DataResponse, CatalogItem, ResourceCatalog, Resource, Representation)):
            dict_to_xarray_metadata(xr_obj, value.__dict__)
        else:
            # If the value is not a dictionary, add it as a metadata attribute
            xr_obj.attrs[attr_key] = value

def json_to_xarray_metadata(xr_obj, metadata_dict):
    for f in metadata_dict:
        if f['name'].split(' (')[0].split('_40_ms')[0] == xr_obj.name:
            dict_to_xarray_metadata(xr_obj, f)
        elif f['name'].split(' (')[0].split('_40_ms')[0] == "ISO 8601 time":
            dict_to_xarray_metadata(xr_obj, f)
        else:
            print(f"{f['name']} not found")

    
def hdf5_to_ds(inp, hdf5_file):
    """ convert hdf5 files to xarray dataset"""
    import xarray as xr
    
    ds = xr.Dataset()
    for dataset_name in hdf5_file:
        dataset = hdf5_file[dataset_name]
        metadata = dict(dataset.attrs.items())['properties']
        dict_to_xarray_metadata(ds, json.loads(metadata))
        for k, v in dataset.items():
            metadata = dict(dataset[k].attrs.items())['properties']
            for k1, v1 in dataset[k].items():
                    # print(dict(dataset[k][k1].attrs))
                timestamps = [inp.tstart + inp.sample_period * x for x in range(0, len(np.array(dataset[k][k1][:])))]
                ds[k] = xr.DataArray(dataset[k][k1][:], coords={'timestamp':timestamps}, dims='timestamp')
            dict_to_xarray_metadata(ds[k], json.loads(metadata))
    return ds

def replace_nans_in_dict(d:dict, value_to_replace='') -> dict:
    """"replace nan or none in a dict to a desired value, defaults to no string"""
    for key, value in d.items():
        if isinstance(value, dict):
            replace_nans_in_dict(value)
        elif value is None:
            d[key] = value_to_replace
        elif value == 'nan':
            d[key] = value_to_replace
        elif np.isnan(value):
            d[key] = value_to_replace
    
    return d

def convert_to_string(data):
    if isinstance(data, dict):
        return {k: convert_to_string(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_string(item) for item in data]
    else:
        return str(data)


if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    import pythonAssist as pa
    import matlab2py as m2p

    # sigma = 3
    # mu = 10
    # U = sigma * np.random.randn(100,1) + mu 
    # U[50] = 30
    # df = pd.DataFrame(U, dtype=np.float64)
    # win_length = 11

    # # test the method applied
    # m2p.tic()
    # outliers, U_mean, U_std = pa.signal_outliers(df, win_length=win_length, sigma=sigma, detrend=True)
    # m2p.toc()

    # # test the method applied with vectorization
    # m2p.tic()
    # outliers2, U_mean2, U_std2 = pa.signal_outliers_vectorized(df, win_length=win_length, sigma=sigma)
    # m2p.toc()

    # #plot the difference between signal outliers and signal_outliers_vectorized
    # import matplotlib.pyplot as plt
    # plt.plot(U_std, U_std2, 'k.')

    # # import the toml file and convert to struct
    # inp = import_toml(toml_file=r"C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\HighRe\src\highre_input.toml")
    # inp = dict2struct(inp)

    # x = [1,2,3]
    # y =  rotate_list(x, n=1, direction='right')

    # test code for multiple stats on df # TODO build a pytest
    import numpy as np
    import pandas as pd
    import scipy.stats as sp

    ws = np.random.randn(1000) + 10
    wd = np.deg2rad(((np.random.randn(1000)*10 + 350) % 360))
    theta = np.deg2rad((np.random.randn(1000) + 30) % 360)
    idx = pd.period_range(start='2021-10-25', periods=1000, freq='1T')
    df = pd.DataFrame(np.transpose([ws, wd, theta]), columns=['ws', 'wd', 'theta'], index=idx)
    df.index.name='Timestamp'

    # display the error in using mean
    print(f"Numpy mean wdir (mean=350): {np.rad2deg(np.nanmean(wd))}")    

    # the correct mean using ircular data
    print(f"Circular mean wdir (mean=350): {np.rad2deg(sp.circmean(wd))}")

    # apply different functions on a df with non-homogeneous data i.e. having linear and circular data
    df_mean = multiple_stats_on_df(df, circular_data_cols=['wd', 'theta'], stats='mean', nondefault_bound_cols=['theta'])
    df_std = multiple_stats_on_df(df, circular_data_cols=['wd', 'theta'], stats='std', nondefault_bound_cols=['theta'])
