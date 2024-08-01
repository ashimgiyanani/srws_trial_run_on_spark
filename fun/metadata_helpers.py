import enum
import uuid
import sys, os
import numpy as np
sys.path.append(r"../fun")
import pythonAssist as pa
import pandas as pd
import time
from datetime import *
sys.path.append(r"../../userModules")
from FileOperations import FileOperations
from ProcessSRWS import ProcessSRWS
import matlab2py as m2p

def generate_uuid():
    # generates a unique id
    return str(uuid.uuid4())

def generate_metadata(FilePath):
    return sys.path(FilePath)

def qc_flags_basic(**kwargs):
    # assigns a 10 bit flag using the inputs
    defaultKwargs = {
        'valid': 0,
        'modified': 0,
        'estimated': 0,
        'resampled': 0,
        'not_checked': 0,
        'inconsistent': 0,
        'doubtful': 0,
        'missing': 0,
    }
    kwargs = defaultKwargs | kwargs

    flag_meanings, flag_values = '', ''
    for k,v in kwargs.items():
        flag_meanings = flag_meanings + ', ' + str(k)
        flag_values = flag_values + str(v)
    flag_meanings = flag_meanings[2:]
    flag_values_int = bin2dec(flag_values)

    return flag_values_int, flag_values, flag_meanings

from enum import IntFlag, auto
class qc1_flags(IntFlag):
    valid = auto()
    modified = auto()
    estimated = auto()
    resampled = auto()
    not_checked = auto()
    inconsistent = auto()
    doubtful = auto()
    missing = auto()

class qc2_flags(IntFlag):
    valid = auto()
    bias = auto()
    custom_method_applied = auto()
    sensor_failure = auto()
    qc_all_checks = auto()
    qc_range = auto()
    qc_windspeed = auto()
    qc_winddir = auto()
    qc_cnr = auto()
    qc_stddev_high = auto()


def qc_flags_advanced(**kwargs):
    # assigns a 10 bit flag using the inputs
    defaultKwargs = {
        'bias': 0,
        'custom_method_applied': 0,
        'sensor_failure': 0,
        'qc_all_checks':0,
        'qc_range':0,
        'qc_windspeed':0,
        'qc_winddir':0,
        'qc_cnr': 0,
        'qc_stddev_high':0,
        }
    kwargs = defaultKwargs | kwargs

    flag_meanings, flag_values = '', ''
    for k,v in kwargs.items():
        flag_meanings = flag_meanings + ', ' + str(k)
        flag_values = flag_values + str(v)
    flag_meanings = flag_meanings[2:]
    flag_values_int = bin2dec(flag_values)

    return flag_values_int, flag_values, flag_meanings

def generate_flag_combinations(qc1=True):
    """generate flag combinations relating to different qc flags"""
    
    from metadata_helpers import qc1_flags, qc2_flags
    if qc1==True:
        func= qc1_flags
        combinations = [qc1_flags.valid, qc1_flags.invalid, qc1_flags.doubtful, qc1_flags.missing]
    elif qc1==False:
        func = qc2_flags
        combinations = [qc2_flags.valid, qc2_flags.sensor_failure, qc2_flags.all_checks_failed, qc2_flags.range_failed, qc2_flags.bad_quality]
    else:
        print("Add other qc_flag class variables in this function")

    names, values = [], []
    for i in range(1, 2**len(combinations)):
        flags = func(0)
        for j in range(len(combinations)):
            if i & (1 << j):
                flags |= combinations[j]
        names.append(flags)
        values.append(flags.value)
    
    return names, values

def as_dict(x):
    # defines a dict from tables
    return {c.name: getattr(x, c.name) for c in x.__table__.columns if c is not None}

def get_enum(x):
    names = []
    values = []
    for i in (x):
        print("{0}: {1}".format(i.name,i.value))
        names.append(i.name)
        values.append(i.value)

    return names, values

def read_cf_variables(path, ds, **kwargs):
    """reads cf variables matching the variables in the dataframe or xarray dataset and assign basic metadata"""
    import pandas as pd
    cf_flags = kwargs.setdefault('cf_flags', False)
    ignore_vars = kwargs.setdefault('ignore_vars', [])

    vars = pd.read_excel(path, sheet_name="Variables")
    # vars = vars.drop(columns=['units'])
    for v in ignore_vars:
        vars = vars.drop(vars[vars.name==v].index)  
        
    vars['data_type'] = vars['data_type'].apply(lambda x: '{}'.format(x))

    if cf_flags == True:
        from metadata_helpers import qc1_flags, get_qc_str
        vars['flag_values'] = ds.qc1.to_series().mode()[0]
        vars['flag_meanings'] = [get_qc_str(vf, qc1_flags) for vf in vars['flag_values']]
    
    for s in vars.name:
        if s in ds.variables.mapping.keys():
            if s == 'Timestamp':
                ds[s].attrs = vars[vars.name==s].iloc[0,:].drop(columns=['units', 'data_type']).to_dict()
            else:
                ds[s].attrs = vars[vars.name==s].iloc[0,:].to_dict()
            # if s == 'Timestamp':
            #     continue
            #     ds[s].attrs['units'] = 'nanoseconds since 1970-01-01 00:00:00'
            #     ds[s].encoding['units'] = 'nanoseconds since 1970-01-01 00:00:00'

    return ds

def get_cf_std(path):
    # gets the cf standard table as a pandas dataframe
    import pandas as pd
    cf_stdnames = pd.read_xml(path, xpath="//entry")
    cf_stdnames.rename(columns={"id": "standard_name"}, errors="raise", inplace=True)

    # adding alias defined for some variables only, see standard tables at bottom
    cf_stdnames['alias'] = None
    alias = pd.read_xml(path, xpath = "//alias")
    alias.rename(columns={"id": "alias", "entry_id": "standard_name"}, errors="raise", inplace=True)
    alias = alias[alias.standard_name.isin(cf_stdnames.standard_name)]
    for i, a in alias.iterrows():
        cf_stdnames.loc[cf_stdnames.standard_name.str.contains(a.standard_name), 'alias'] = a.alias

    return cf_stdnames

def identical(column_name):
    def mydefault(context):
        return context.current_parameters.get(column_name)
    return mydefault

def getColumnDtypes(dataTypes):
    """ 
    pandas df dataypes and SQL datatypes differ, so to align we perform dtype replacement 
    Syntax: columnDataType = getColumnDtypes(mi_metadata.dtypes)
    """
    dataList = []
    for x in dataTypes:
        if x == 'int64':
            dataList.append('int')
        elif (x=='float64'):
            dataList.append('float')
        else:
            dataList.append('varchar')
    return dataList

def get_CampaignInfo(**kwargs):
    """
        get_CampaignInfo - function to import campaign info from a files from data path
    
        Syntax:  campaign_df = get_CampaignInfo()
    
        Inputs:
            last_measurement_file - path to last measurement file
            first_measurement_file - path to first measurement file
            searchStr - datetime searchStr similar to in the file paths
            dateformat - dateformat to be converted into
                     
        Outputs:
                campaign_df - dataframe containing all the variables
                 - time_coverage_start, _end, _duration, resolution
    
        Example:
            path = inp.srws.path.root
            searchStr = '(\d{4}-\d{2}-\d{2}T\d{6}\+\d{2})'
            dateformat = '%Y-%m-%dT%H%M%S%z'
            last_measurement_file = r"z:\Projekte\112933-HighRe\20_Durchfuehrung\OE410\SRWS\Data\Rosette\2022\06\21\2022-06-21T060000+00"
            first_measurement_file = r"z:\Projekte\112933-HighRe\20_Durchfuehrung\OE410\SRWS\Data\Bowtie1\2021\10\25\2021-10-25T090000+02"
            campaign_df = get_CampaignInfo(...) # here less generic still
    
        Raises:
    
        modules required: sys, numpy
        classes required: FileOperations
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

    import sys
    import isodate

    sys.path.append(r"C:\Users\giyash\OneDrive - Fraunhofer\Python\userModules")
    from FileOperations import FileOperations
    # default values for kwargs
    default_kwargs = {
        'searchStr': '(\d{4}-\d{2}-\d{2}T\d{6}\+\d{2})',
        'dateformat':'%Y-%m-%dT%H%M%S%z',
        'last_measurement_file': r"../data/2021-08-26T163600+02",
        'first_measurement_file': r"../data/2021-08-26T163600+02",
        'path': sys.path[0],
        'regStr': "*T*[!.zip][!.csv][!.txt][!.x*]"
                    }
    kwargs = default_kwargs | kwargs

    time_coverage_start =  pd.to_datetime(os.path.basename(kwargs['first_measurement_file']), utc=True, infer_datetime_format=True),
    time_coverage_end = pd.to_datetime(os.path.basename(kwargs['last_measurement_file']), utc=True, infer_datetime_format=True),


    # searchStr = args.__dict__.get('searchStr', '(\d{4}-\d{2}-\d{2}T\d{6}\+\d{2})')
    # dateformat = args.__dict__.get('dateformat','%Y-%m-%dT%H%M%S%z')
    # last_measurement_file = args.__dict__.get('last_measurement_file', r"../data/2021-08-26T163600+02")
    # first_measurement_file = args.__dict__.get('first_measurement_file', r"../data/2021-08-26T163600+02")
    # time_coverage_start = args.__dict__.get('time_coverage_start', pd.to_datetime(os.path.basename(first_measurement_file), utc=True, infer_datetime_format=True))
    # time_coverage_end = args.__dict__.get('time_coverage_end', pd.to_datetime(os.path.basename(last_measurement_file), utc=True, infer_datetime_format=True))

    # calculated values
    time_coverage_duration = isodate.duration_isoformat((time_coverage_end[0] - time_coverage_start[0]))
    fo = FileOperations(os.path.dirname(kwargs['last_measurement_file']))
    file_prop = fo.FnGetFileSize(kwargs['path'], regStr = kwargs['regStr'])
    DT = fo.FnGetDateTime(file_prop.filenames, searchStr = kwargs['searchStr'], dateformat=kwargs['dateformat'])
    try:
        time_coverage_resolution = isodate.duration_isoformat((np.diff(DT)[0]))
    except (TypeError, IndexError):
        time_coverage_resolution = isodate.duration_isoformat(timedelta(minutes=1))

    campaign_df = pd.DataFrame(
        data = [[time_coverage_start, time_coverage_end, time_coverage_duration, time_coverage_resolution]], \
        columns = ['time_coverage_start', 'time_coverage_end', 'time_coverage_duration', 'time_coverage_resolution']
        )

    return campaign_df


def get_guiData(FilePath):
    """
        get_guiData - import the information regarding the default GUI / Data location interface
    
        Syntax: df = get_guiData(FilePath)
    
        Inputs:
           path - path to the raw data (srws data) 
    
        Outputs:
           df - pandas dataframe with subvariables
            - url, filename, extension, 
            - start_datetime, end_datetime, file_granularity, samples_per_file, 
            - date_created, date_modified, time_coverage_start, time_coverage_end
    
        Example:
            FilePath = r"C:\\Users\\giyash\\OneDrive - Fraunhofer\\Python\\Scripts\\Metadata\\data\\2021-08-26T163600+02"
            df = get_guiData(FilePath)

        Raises:
    
        modules required: datetime, sys, os
        classes required: Read_SRWS_bin
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
    import os, sys
    sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")
    from ProcessSRWS import ProcessSRWS
    inp = pa.struct()
    inp.srws = pa.struct()
    inp.srws.path = FilePath

    dateformat = "%Y-%m-%d %H:%M:%S.%f%Z"

    filename = os.path.basename(FilePath)
    url = os.path.dirname(FilePath)
    extension = os.path.splitext(os.path.basename(FilePath))[1]
    start_datetime = pd.to_datetime(filename, utc=True, infer_datetime_format=True)
    data, df, _ = ProcessSRWS.Read_SRWS_bin(FilePath, mode='basic')
    var_names = df.columns
    end_datetime = pd.to_datetime(df.index[-1], utc=True, infer_datetime_format=True)
    file_granularity = end_datetime - start_datetime
    samples_per_file = df.index[-1]
    date_created = pd.to_datetime(datetime.fromtimestamp(os.path.getctime(FilePath)), utc=True)
    date_modified = pd.to_datetime(datetime.fromtimestamp(os.path.getmtime(FilePath)),utc= True)
    time_coverage_start = pd.to_datetime(df.index[0], utc=True, infer_datetime_format=True)
    time_coverage_end = pd.to_datetime(df.index[-1],  utc=True, infer_datetime_format=True)

    df = pd.DataFrame(
        data = [[url, filename, extension, start_datetime, end_datetime, file_granularity,\
            samples_per_file, date_created, date_modified, time_coverage_start, time_coverage_end]], \
        columns = ["url", "filename", "extension", "start_datetime", "end_datetime", "file_granularity", \
            "samples_per_file", "date_created", "date_modified", 'time_coverage_start', 'time_coverage_end']
        )
    return df, var_names

def get_guiData_ascii(FilePath, dateformat, filename_format):
    """
        get_guiData - import the information regarding the default GUI / Data location interface
    
        Syntax: df = get_guiData(FilePath)
    
        Inputs:
           path - path to the raw data (srws data) 
           dateformat - format of the date within the filename
           filename_format to look for 
    
        Outputs:
           df - pandas dataframe with subvariables
            - url, filename, extension, 
            - start_datetime, end_datetime, file_granularity, samples_per_file, 
            - date_created, date_modified, time_coverage_start, time_coverage_end
    
        Example:
            FilePath = r"C:\\Users\\giyash\\OneDrive - Fraunhofer\\Python\\Scripts\\Metadata\\data\\2021-08-26T163600+02"
            df = get_guiData(FilePath)

        Raises:
    
        modules required: datetime, sys, os
        classes required: Read_SRWS_bin
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

    dateformat = "%Y-%m-%dT%H%M%S%Z"
    datepattern = r"\d{4}-\d{2}-\d{2}T\d{6}+\d{2}"
    import re
    match = re.search(datepattern, FilePath)
    url = os.path.dirname(FilePath)
    filename, extension = os.path.splitext(os.path.basename(FilePath))
    start_datetime = pd.to_datetime(filename, utc=True, format=dateformat, infer_datetime_format=True)
    data, df, _, df_all = Read_SRWS_bin(FilePath)
    var_names = df.columns
    end_datetime = pd.to_datetime(df.index[-1], utc=True, infer_datetime_format=True)
    file_granularity = end_datetime - start_datetime
    samples_per_file = df.index[-1]
    date_created = pd.to_datetime(datetime.fromtimestamp(os.path.getctime(FilePath)), utc=True)
    date_modified = pd.to_datetime(datetime.fromtimestamp(os.path.getmtime(FilePath)),utc= True)
    time_coverage_start = pd.to_datetime(df.index[0], utc=True, infer_datetime_format=True)
    time_coverage_end = pd.to_datetime(df.index[-1],  utc=True, infer_datetime_format=True)

    df = pd.DataFrame(
        data = [[url, filename, extension, start_datetime, end_datetime, file_granularity,\
            samples_per_file, date_created, date_modified, time_coverage_start, time_coverage_end]], \
        columns = ["url", "filename", "extension", "start_datetime", "end_datetime", "file_granularity", \
            "samples_per_file", "date_created", "date_modified", 'time_coverage_start', 'time_coverage_end']
        )
    return df, var_names

# Reference: https://github.com/Austyns/sqlite-to-json-python
import sqlite3
import json

def dict_factory(cursor, row):
    # get dict from sqlite query
    # ref: https://stackoverflow.com/questions/3300464/how-can-i-get-dict-from-sqlite-query
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# connect to the SQlite databases
def openConnection(pathToSqliteDb):
    connection = sqlite3.connect(pathToSqliteDb)
    connection.row_factory = dict_factory
    cursor = connection.cursor()
    return connection, cursor


def getAllRecordsInTable(table_name, pathToSqliteDb):
    conn, curs = openConnection(pathToSqliteDb)
    conn.row_factory = dict_factory
    curs.execute("SELECT * FROM '{}' ".format(table_name))
    # fetchall as result
    results = curs.fetchall()
    # close connection
    conn.close()
    return json.dumps(results)


def sqliteToJson(pathToSqliteDb):
    connection, cursor = openConnection(pathToSqliteDb)
    # select all the tables from the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    # for each of the tables , select all the records from the table
    for table_name in tables:
        # Get the records in table
        results = getAllRecordsInTable(table_name['name'], pathToSqliteDb)

        # generate and save JSON files with the table name for each of the database tables and save in results folder
        with open('./results/'+table_name['name']+'.json', 'w') as the_file:
            the_file.write(results)
    # close connection
    connection.close()


def df2mi(df_all, regStr, indexStr, **kwargs):
    # function to convert a dataframe with mutliple columns using regex into a multiindex dataframe
    # problems: all the values are converted to float values

    import re
    import xarray as xr

    N_index = kwargs.setdefault('N_index', 5)
    xarray_ds = kwargs.setdefault('xarray_ds', True)

    df_mi = pd.DataFrame()
    inp_cols = df_all.columns
    for i in range(1, N_index):
        # filter out single group
        temp_df = df_all.filter(regex=regStr.format(i))
        columns = [
            re.split(f"\s?{i}$", c)[0].replace(" ", "")
            for c in temp_df.columns
        ]
        inp_cols = inp_cols.drop(temp_df.columns)
        # cols = [c[:-2] for c in temp_df.columns]
        index = [
            i * np.ones(len(temp_df.index), dtype=np.int32), temp_df.index
        ]
        df_multi = pd.DataFrame(temp_df.values, columns=columns, index=index)
        df_multi.index.set_names([indexStr.split('{')[0], 'idx'], inplace=True)
        df_mi = pd.concat([df_mi, df_multi], axis=0)

    dropped_cols = inp_cols

    if xarray_ds:
        # convert the multi-index dataframe with common columns into an xarray dataset with non-homogeneous dataset
        ds = xr.Dataset.from_dataframe(df_mi)
        for col in dropped_cols:
            ds = ds.assign(**{col: ('idx', df_all[col])})
    else:
        ds = None

    return df_mi, dropped_cols, ds

def dict2yaml(filepath, dct):
    # write a dict to yaml file
    import yaml
    with open(filepath, 'w') as file:
        document = yaml.dump(dct, file)
    return None

def dict2json(filepath: str, dct: dict) -> None:
    """write a dict to json file"""
    import json
    from collections import OrderedDict

    with open(filepath, 'w', encoding='utf-8') as file:
        document = json.dump(OrderedDict(dct), file, indent=4, ensure_ascii=False)

    return None


def yaml2dict(filepath: str) -> dict:
    """read a yaml to dict"""

    import yaml
    with open(filepath, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    return data


def json2dict(filepath: str) -> None:
    """read a json to dict"""

    import json
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def bin2dec(f_bin):
    """ script to convert from binary number to decimal format"""
    f_dec = int(f_bin, 2)
    return f_dec

def dec2bin(f_dec):
    """ script to convert from binary number to decimal format"""
    f_bin = bin(f_dec).replace('0b', '').zfill(8)
    return f_bin


def json_serialize(x):
    # seriealize a json item
    # Application:
    # for r in dict_item:
    #     dct = as_dict(r)
    #     json_obj = json.dumps(dct, indent=4, default=lambda x: json_serialize(x))
    #     print(json_obj)

    try:
        xval = x.value
    except AttributeError:
        xval = str(x)

    return xval


def to_dict(row):
    if row is None:
        return None

    rtn_dict = dict()
    keys = row.__table__.column.keys()
    for key in keys:
        rtn_dict[key] = getattr(row, key)

    return rtn_dict


def exportexcel(Data):

    data = Data.query.all()
    data_list = [to_dict[item] for item in data]
    df = pd.DataFrame(data_list)
    filename = app.config['UPLOAD_FOLDER'] + "/autos.xlsx"
    print(f"FILENAME: {filename}")

    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, sheet_name='Reg')
    writer.save()

    return None


if __name__ == "__main__":
    from metadata_helpers import get_cf_std, get_guiData, qc_flags_basic, qc_flags_advanced
    path = r"../data/cf-standard-name-table.xml"
    cf_std = get_cf_std(path)

    FilePath = r"../data/2021-08-26T163600+02"
    df, var_names = get_guiData(FilePath)

    qc1_flag_values, qc1_flag_meanings = qc_flags_basic(not_checked=1)
    qc2_flag_values, qc2_flag_meanings = qc_flags_advanced()






# implement this for __str__ or __repr__ implementation of the sql tables
# https://stackoverflow.com/questions/54026174/proper-autogenerate-of-str-implementation-also-for-sqlalchemy-classes/54034230#54034230
