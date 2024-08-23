import os, sys, glob
sys.path.append("../fun/")
sys.path.append("../../userModules")
import pythonAssist as pa

# import modules
import numpy as np
import shutil
import chardet
import pandas as pd
import hashlib
# from numba import njit, jit
from typing import Union, Dict
import xarray as xr

class FileOperations:


    # class attributes:
    def __init__(self, path):
        self.path = path

    def printPath(self):
        # print the path where the operations are performed
        print(self.path)

    def FnGetFiles(self):
        # function to get all the files in input path
        # Input:
        #     path -  path to the directory (from class) 
        # Output:
        #     files - a list of files in path 
        # Example:
        #     path = r"Z:\Projekte\109797-TestfeldBHV\Data"
        #     files = FnGetFiles(path)

        import os
        import itertools

        files = []
        for _, _, f in os.walk(self.path):
            files.append(f)
        files = list(itertools.chain(*files))
        return files

    # get full paths for the files
    @staticmethod
    def FnGetFileSize(path, regStr):
    # function to get the file size and other details
    # Input:
        # path -  path to the directory (from class)
        # regStr - regex parameter for narrowing the search criteria 
    # Output:
        # size - size of the files
        # fullpath - full path of the files
        # names - names of the files including the extension
        # filenames - name of the file without path and without extension
        # extension - extension of the file
    # Example:
        # regStr = "*real*.csv"
        # path = r"Z:\Projekte\109797-TestfeldBHV\Data"

        # import os
        # import glob
        # import pandas as pd

        fullpaths = glob.glob(os.path.join(path, "**", regStr), recursive=True)
        # fullpaths = glob.glob(os.path.join(path , "**" , regStr), recursive=True)

        # remove folders from the dataframe, as algorith doesn't distinguish files w/o extensions and folders
        fullpaths = [f for f in fullpaths if os.path.isfile(f)]

        # @njit
        def process_times(fullpaths):
            sizes, times, names, filenames, extensions = [], [], [], [], []
            for f in fullpaths:
                names.append(os.path.basename(f))
                filenames.append(os.path.splitext(os.path.basename(f))[0])
                extensions.append(os.path.splitext(os.path.basename(f))[1])
                sizes.append(os.path.getsize(f))
                times.append(os.path.getmtime(f))

            # combine to a pandas dataframe
            listTuples = list(zip(sizes, times, fullpaths, names, filenames, extensions))
            nameTuples = ['sizes', 'times', 'fullpaths', 'names', 'filenames', 'extension']
            file_prop = pd.DataFrame(listTuples, columns = nameTuples)
            return file_prop

        # assign names
        file_prop = process_times(fullpaths)

        return file_prop


    @staticmethod
    def match_files(files):
    # function to match the files in a list (to be verified)
    # Input:
    #     files  - files is a list of filenames without path
    # Output:
    #     result - dict of files not matching to the list
    # Example:
    # filename1 = 'Data_A_2015-07-29_16-25-55-313.txt'
    # filename2 = 'Data_B_2015-07-29_16-25-55-313.txt'
    # file_list = [filename1, filename2]
    # match_files(file_list)
        result = {}
        for filename in files:
            data, letter, date, time_txt = filename.split('_')
            time, ext = time_txt.split('.')
            hour, min, sec, ns = time.split('-')

            key = date + '_' + hour + '-' + min

            # Initialize dictionary if it doesn't already exist.
            if not result.has_key(key):
                result[key] = {}

            result[key][letter] = filename
        return result

    # find the gaps in files based on differencing
    @staticmethod
    def FnFindFileGaps(searchStr, filenames, dateformat, period):
    
    # Input:
    #     path - () path to the Directory
    #     searchStr - regex String search for datetime in the filenames
    #     filenames - list of filenames incl. path, like the output of FnGetFileSize()
    #     dateformat - datetimeformat for datetime.strptime() function
    #     period - period of the timedelta in the timeseries according to datetime module (e.g. 1 min for srws data)

    # Output:
    #     minor_gaps - small gaps in the time Series
    #     major gaps - major gaps in the time Series

    # Example:
    #     filenames = fullpath1
    #     searchStr = '(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
    #     dateformat = '%Y-%m-%d_%H-%M-%S'
    #     period = '1M'  
    #     minor_gaps, major_gaps = FnFindFileGaps(searchStr, filenames, dateformat, period)

        import sys
        sys.path.append(r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules")
        import matlab2py as m2p
        import re
        from datetime import datetime, timedelta
        import numpy as np

        match = []
#         searchStr = '(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
        for n in filenames:
            match.append(re.search(searchStr, n).group(0))

        dt1 = [datetime.strptime((x), dateformat) for x in match]
        ddt1 = np.diff(dt1, prepend=dt1[0]).astype('timedelta64[D]')

        # find the index of gaps
        if period == '1M':
            idx_major = m2p.findi(ddt1, lambda ddt1: (ddt1 > timedelta(minutes=1)))
            idx_minor = np.flatnonzero((ddt1 < timedelta(minutes=1)))
        elif period == '10M':
            idx_major = m2p.findi(ddt1, lambda ddt1: (ddt1 > timedelta(minutes=10)))
            idx_minor = np.flatnonzero((ddt1 < timedelta(minutes=10)))
        elif period == 'D':
            idx_major = m2p.findi(ddt1, lambda ddt1: (ddt1 > timedelta(days=1)))
            idx_minor = np.flatnonzero((ddt1 < timedelta(days=1)))
        else:
            print('Choosing timedelta as  1 hr')
            idx_major = m2p.findi(ddt1, lambda ddt1: (ddt1 > timedelta(hours=1)))
            idx_minor = np.flatnonzero((ddt1 < timedelta(hours=1)))

        # assign dates to the gap index
        minor_gaps = [dt1[x] for x in idx_minor]
        major_gaps = [dt1[x] for x in idx_major]

        import matplotlib.pyplot as plt
        plt.plot(dt1, ddt1, 'k.')
        plt.xticks(rotation=30)

        return minor_gaps, major_gaps

    @staticmethod
    def FnGetDateTime(filenames,  searchStr, dateformat):
    # Input:
    #     path - (self) path to the Directory
    #     searchStr - regex String search for datetime in the filenames
    #     filenames - list of filenames incl. path, like the output of FnGetFileSize()
    #     dateformat - datetimeformat for datetime.strptime() function
    # Output:
    #     DT - datetime object from the filenames
	# Example:
	#       searchStr = '(\d{4}-\d{2}-\d{2}T\d{6}\+\d{2})'
	#	    dateformat = '%Y-%m-%dT%H%M%S%z'
	#       DT = fo.FnGetDateTime(prop.filenames, searchStr = searchStr, dateformat=dateformat)

        import re
        from datetime import datetime
        DT = []
        for i in range(len(filenames)):
            try:
                dt=re.search(searchStr, filenames[i]).group(1)
                DT.append(datetime.strptime(dt, dateformat))
            except ValueError:
                dt=re.search(searchStr, filenames[i]).group(1)
                DT.append(datetime.strptime(dt+'00', dateformat))
            except AttributeError:
                DT.append(np.nan)
                # print('File {0} skipped'. format(filenames[i]))
                continue
        return DT

    
    def getSftp(self):
        ## load modules
        import os
        import fnmatch
        import pysftp
        # import types
        import datetime
        import numpy as np
        # import pandas as pd

        def now():
            return datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')

        ## sftp server configuration for WindCube Data
        sftp_url = "ftp.fraunhofer.de" # change this for a different ftp address
        user = "testfeld01" # change username here
        pswd = "s1fOEXaaeT1C8D" # change password here
        # indicate no hostkeys to avoid the error related to ssh hostkeys or provide a ssh hostkey
        cnopts = pysftp.CnOpts() 
        cnopts.hostkeys = None # not recommended due to lack of secured connection

        ## definitions to append the filenames to the respective structure
        files = [] # files within the ftp server
        dirs = [] # directories in the ftp server, 1st level
        un_files = [] # extra files

        def store_files(fname):
                files.append(fname)

        def store_dirs(dirname):
                dirs.append(dirname)

        def store_un_files(name):
                un_files.append(name)

        ## connect to the sftp server
        sftp =  pysftp.Connection(host=sftp_url,username=user, password=pswd,cnopts=cnopts)
        print('[{}]: Connection succesfully established'.format(now()))
        # get the directory and file listing
        sftp.walktree('.',store_files,store_dirs, store_un_files, recurse=True)

        ## create a dict with directory data in each folders
        dirData = {}
        for i in np.arange(len(dirs)):
                files = []
                sftp.walktree(dirs[i], store_files,store_dirs, store_un_files, recurse=True)
                dirData[os.path.split(dirs[i])[1]] = files # subdirectory structure

        # get the directory and file listing
        sftp.walktree('.',store_files,store_dirs, store_un_files, recurse=True)

    def FnLogging(self, log_file, src_folder, dest_folder, regStr, move):
    # function to start the logging environment for a func
        import logging
        from FileOperations import FileOperations
        # load the File Operations class
        fo = FileOperations(src_folder)
        
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
        logging.info('Started')
        fo.copy_files(src_folder, dest_folder, move, regStr)
        logging.info('Finished')    

    def copy_files(self, src_folder, dest_folder, move, regStr):
        import logging
        import os, pathlib, itertools
        import re
        from collections import defaultdict
        from datetime import datetime, tzinfo
        import pandas as pd
        import sys
        sys.path.append(r'C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\DataQualityControl\src')
        sys.path.append(r'C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\userModules')
        from FileOperations import FileOperations
        import shutil
        from shutil import make_archive
        import pythonAssist as pa
        import csv
        
        logging.info('Copying files')
        # load the File Operations class
        fo = FileOperations(src_folder)
        
        # get the files in the given path
        # files = fo.FnGetFiles()
        
        # get the file sizes, paths and names
        # regStr = '*T*.zip'
        prop = fo.FnGetFileSize(src_folder, regStr)
        
        # get the dates of the files in datetime format
        dates = pd.to_datetime(prop.filenames)
        
        # change to the directory where folders need to be created
        os.chdir(dest_folder)
        
        # copy zip files from source folder to destination folder, if the file not exists
        N_empty, N_exists = 0, 0
        files_copied, files_notcopied = [], []
        for i in range(len(prop.filenames)):
            d = pd.to_datetime(prop.filenames[i])
            tempPath = os.path.join(dest_folder, str(d.year), str(d.month).zfill(2), str(d.day).zfill(2))
            os.makedirs(tempPath, exist_ok= True)

            try:
                dest_file = os.path.join(tempPath, prop.names[i])
                if not os.path.isfile(dest_file):
                    if move == False:
                        shutil.copy2(prop.fullpaths[i], tempPath)
                    elif move == True:
                        shutil.move(prop.fullpaths[i], tempPath)
                    else:
                        print("specify move = True or False")

                    print("[{0}]: Completed {3}/{4} -- File {1} copied to {2}".format(pa.now(), prop.filenames[i], tempPath[-18:], i, len(prop.filenames)))
                    N_empty += 1
                    files_copied.append(prop.fullpaths[i])
                    with open('CopiedFiles.txt', 'a') as file:
                        w = csv.writer(file)
                        w.writerow(prop.fullpaths[i]+'\n')
            except OSError:
                print("[{0}]: File {1} exists or has some error".format(pa.now(), prop.filenames[i]))
                N_exists += 1
                files_notcopied.append(prop.fullpaths[i])
                with open('NotCopiedFiles.txt', 'a') as file:
                    w = csv.writer(file)
                    w.writerow(str(prop.fullpaths[i])+'\n')

    # # https://github.com/menzi11/EZFile/blob/master/ezfile/__init__.py
    def get_relative_path(self, start_path):
        return os.path.relpath(self.path, start_path)

    def get_curr_dir(self):
        return os.path.abspath(sys.path[0])

    def exists(self):
        """tell if a path exists"""
        return os.path.exists(self.path)

    def exists_as_dir(self, path):
        """ check if a path is a dir and it is exists"""
        return self.exists() and os.path.isdir(path)

    def exists_as_file(self, path):
        ''' check if a path is a file and it is exists '''
        return self.exists() and os.path.isfile(path)

    def get_full_path_with_ext(self, path):
        """ return full path of a file(abspath)"""
        return os.path.abspath(path)

    def get_full_path_without_ext(path):
        """ return full path of a file without ext """
        return get_sibling_file( path , get_short_name_without_ext(path) )

    def get_ext(path):
        """ get file ext """
        return os.path.splitext(get_short_name_with_ext(path))[1]

    def get_short_name_without_ext(path):
        """ get file short name without ext, for example: "c:/1.txt" will return "1" """
        return os.path.splitext(get_short_name_with_ext(path))[0]

    def get_short_name_with_ext(path):
        """ get file short name without ext, for example: "c:/1.txt" will return "1.txt" """
        return os.path.basename(path)

    def get_child_file(path, child_name ):
        """ get child file of a path( no matter if the child file exists )
            for example, "get_child_file('c:/','1.txt')" will return "c:/1.txt" """
        return os.path.join(path, child_name )

    def get_sibling_file(path, siblingFileName):
        """ get sibling file of a path. for example, get_sibling_file('c:/1.txt','2.txt') will return 'c:/2.txt' """
        return get_parent_dir(path).get_child_file(siblingFileName)

    def get_parent_dir(self, path):
        """ get parent dir, get_parant_dir('c:/1.txt') will return 'c:/' """
        return os.path.abspath(os.path.join(path, '..'))

    def create_dir(self, path):
        """ create a dir. if the dir exists, than do nothing. """
        if not self.exists_as_dir(path):
            os.makedirs(path)

    def with_new_ext(path, newExt):
        """ change files ext, if path is a dir, than do nothing. """
        if get_ext(path) == '':
            return
        if '.' not in newExt[0]:
            newExt = '.' + newExt
        path = get_full_path_without_ext(path) + newExt
        return path

    def move_to(path, target):
        """将文件夹或文件移动到新的位置,如果新的位置已经存在,则返回False"""
        if exists_as_file(path) and not exists_as_file(target):
            create_dir( get_parent_dir(target) )
            shutil.move( get_full_path_with_ext(path), get_full_path_with_ext(target) )
        elif exists_as_dir(path) and not exists_as_file(target):
            shutil.move( get_full_path_with_ext(path), get_full_path_with_ext(target) )
        return True

    def remove(self, path):
        """删除文件或文件夹,不经过回收站"""
        if exists_as_dir(path):
            shutil.rmtree(get_full_path_with_ext(path))
        elif exists_as_file(path):
            os.remove(get_full_path_with_ext(path))

    def copy_to(self, path, target, replaceIfTargetExist=False ):
        """将文件拷贝到target中,target可为一个ezfile或者str. 若target已经存在,则根据replaceIfTargetExist选项来决定是否覆盖新文件. 返回是否复制成功."""
        if self.exists_as_file(target) and not replaceIfTargetExist:
            return False, print(f"skipped {os.path.basename(target)}")
        if self.exists_as_file(target):
            self.remove(target)
        if self.exists_as_file(path) and not self.exists_as_file(target):
            self.create_dir( self.get_parent_dir(target) )
            shutil.copy2(self.get_full_path_with_ext(path), self.get_full_path_with_ext(target))
        elif self.exists_as_dir(path) and not self.exists_as_file(target):
            shutil.copytree( self.get_full_path_with_ext(path), self.get_full_path_with_ext(target) )
        return True, print(f"copied {os.path.basename(target)}")

    def rename(path, newname, use_relax_filename=True, include_ext=False):
        """ rename a file. if 'use_relax_filename' enabled, than unsupported char will remove auto. """
        t = ['?', '*', '/', '\\', '<', '>', ':', '\"', '|']
        for r in t:
            if not use_relax_filename and r in newname:
                return False
            newname = newname.replace(r, '')
        X = os.path.join(get_parent_dir(path), newname)
        if exists(path):
            if include_ext:
                X = X + get_ext(path)
            shutil.move(get_full_path_with_ext(path), X)
        path = X
        return True

    def create_file(path):
        if exists(path):
            return
        open(path, 'a').close()

    def empty_file(path):
        if exists_as_dir(path):
            return
        remove(path)
        create_file(path)

    def replace_text_to_file(path,text, target_code = 'UTF-8-SIG' ):
        if exists_as_dir(path):
            return
        if exists_as_file(path):
            remove(path)
        create_file(path)
        fo = open( get_full_path_with_ext(path), "w", encoding=target_code )
        fo.write( text )
        fo.close()

    def read_text_from_file( path , code = '' ):
        if code == '':
            code = detect_text_coding(path)
        fo = open( get_full_path_with_ext(path), "r", encoding=code )
        x = fo.read()
        fo.close()
        return x

    def detect_text_coding(path):
        """以文本形式打开当前文件并猜测其字符集编码"""
        f = open(get_full_path_with_ext(path), 'rb')
        tt = f.read(200)
        f.close()
        result = chardet.detect(tt)
        return result['encoding']

    def change_encode_of_text_file(path, target_code , src_encoding = '' ):
        text = read_text_from_file(path,src_encoding)
        replace_text_to_file( path, text, target_code )

    def find_child_files(path, searchRecursively=False, wildCardPattern="."):
        """在当前目录中查找文件,若选择searchRecursively则代表着搜索包含子目录, wildCardPattern意思是只搜索扩展名为".xxx"的文件,也可留空代表搜索全部文件. """
        all_search_list = ['.','.*','*','']
        tmp = list()
        if not exists_as_dir(path):
            return tmp
        for fpath, _, fnames in os.walk(get_full_path_with_ext(path)):
            if fpath is not get_full_path_with_ext(path) and not searchRecursively:
                break
            for filename in fnames:
                if wildCardPattern in all_search_list:
                    pass
                else:
                    if wildCardPattern[0] != '.':
                        wildCardPattern = '.' + wildCardPattern
                    if not filename.endswith(wildCardPattern) and wildCardPattern != '.':
                        continue
                tmp.append( os.path.join(fpath,filename) )
        return tmp

    def FnDuplicates(self):
    # function to remove duplicates from a folder
    # https://stackoverflow.com/questions/13130048/find-duplicate-filenames-and-only-keep-newest-file-using-python
        import os
        from collections import namedtuple

        directory = self.path
        os.chdir(directory)

        newest_files = {}
        Entry = namedtuple('Entry',['date','file_name'])

        for file_name in os.listdir(directory):
            name, ext = os.path.splitext(file_name)
            cashed_file = newest_files.get(name)
            this_file_date = os.path.getmtime(file_name)
            if cashed_file is None:
                newest_files[name] = Entry(this_file_date,file_name)
            else:
                if this_file_date > cashed_file.date: #replace with the newer one
                    newest_files[name] = Entry(this_file_date,file_name)
                    # remove the old file 
                    os.remove(cashed_file.file_name) #this line added

    def FnZipFiles(self, extn):
        # function to zip files individually from a folder A with subfolders to a folder B with subfolders recursively
        # skips zipped files
        # zips files with extn
        # Input: 
        #     path - path to the Directory of interest
        #     extn - file extension to be zipped e.g., extn='.rtd', '.csv','',  acc. to os.path.splitext format
        #     regStr - reg expression to include for files for glob.glob implementation
        # Output:
        #     none - file operations
        import os, sys
        sys.path.append(r"../fun")
        import pythonAssist as pa
        import zipfile
        import glob

        regStr = '/**/*T*[!.zip]'
        files = glob.glob(''.join([self.path,regStr]),recursive=True)
    
        for f in files:
            if (os.path.splitext(f)[1]==extn):
                zp = zipfile.ZipFile(''.join([f,'.zip']), 'w')
                zp.write(f, compress_type=zipfile.ZIP_LZMA)
                zp.close()
                print('[{0}]: Compressed {1}'.format(pa.now(), os.path.basename(f)))
            else:
                print('[{0}]: Skipped {1}'.format(pa.now(), os.path.basename(f)))
    
    def FnFastZipFiles(self, extn, **kwargs):
        # function to perform quick zip files individually from a folder A with subfolders to a folder B with subfolders recursively
        # skips zipped files
        # skips files with extn
        # performs quickly with subprocess module (unstable)
        # Input: 
        #     path - path to the Directory of intersect
        #     extn - file extension to be zipped e.g., extn='', 'csv' acc. to os.path.splitext format
        #     outfmt - zip format as output, e.g. '.7z', '.zip', output by 7z.exe
        # Output:
        #     none - file operations
        import os, sys
        sys.path.append(r"../fun")
        import subprocess
        import pythonAssist as pa
        import zipfile
        import glob

        # get all the non-zip files in the directory
        regStr = kwargs['regStr']                                # '/**/*[!.(zip|7z|rar)]'
        target_folder = kwargs['target_folder']                  # r"zips/"
        outfmt = kwargs['outfmt']

        files = glob.glob(os.path.join(self.path, regStr),recursive=True)
        path_7zip = r"C:\Program Files\7-Zip\7z.exe"
    
        for f in files:
            n, e = os.path.splitext(f)

            # use target-folder or current directory
            if not target_folder:
                outfile_name = "".join([n,outfmt])
            else:
                outfile_name = os.path.join(target_folder, "".join([os.path.basename(n), outfmt]))

            if ((e == extn) & ~(os.path.isfile(outfile_name))):
                # os.chdir(os.path.dirname(f))
                ret = subprocess.check_output([path_7zip, "a", "-tzip", "-m5=lzma", outfile_name, f])
                print('[{0}]: Compressed {1}'.format(pa.now(), os.path.basename(n)))
            else:
                print('[{0}]: Skipped {1}'.format(pa.now(), os.path.basename(n)))

    def FnFastUnzipFiles(self):
        # function to perform quick unzip of files from a folder and subfolder

        import os, sys
        sys.path.append(r"../fun")
        import subprocess
        import pythonAssist as pa
        import zipfile
        import glob

        # get all the non-zip files in the directory
        files = glob.glob(''.join([self.path,'/**/*T*.zip']),recursive=True)
        path_7zip = r"C:\Program Files\7-Zip\7z.exe"
    
        for f in files:
            n, e = os.path.splitext(f)
            if (e == extn):
                outfile_name = "".join([n,'.zip'])
                # os.chdir(os.path.dirname(f))
                ret = subprocess.check_output([path_7zip, "x", "-tzip", "-m5=lzma", outfile_name, f])
                print('[{0}]: Compressed {1}'.format(pa.now(), os.path.basename(n)))
            else:
                print('[{0}]: Skipped {1}'.format(pa.now(), os.path.basename(n)))

    import os
    @staticmethod
    def FnUnzipFile(filepath, **kwargs):
        """
        function to unzip a single file

        input:
            filepath:  a windows path or similar
            target_path - a path where the data should be unzipped
        output:
            output_path - path where the contents of the folder/file were saved
        """
        import os
        import subprocess
        
        target_path = kwargs.setdefault('target_path', os.path.dirname(filepath))

        path_7zip =  r"C:\Program Files\7-Zip\7z.exe" 
        ret = subprocess.check_output([path_7zip, "x", "-tzip", "-m5=lzma", target_path, filepath])
        print('[{0}]: Extracted {1}'.format(pa.now(), os.path.basename(filepath)))
        output_path = os.path.join(target_path, os.path.splitext(os.path.basename(filepath))[0])
        return output_path

    def FnGetLatestFile(self, regStr, min_size):
        # Input:
            # regStr - filter files to read based on regex rules
            # min_size - minimum size of the files
        # Ouptut: 
            # latest_file - latest modified file in the directory

        # script to get the newest file in the folder
        import os
        from datetime import datetime
        from FileOperations import FileOperations
        from itertools import compress
        
        fo = FileOperations(self.path)
        # regStr = '*T*[!.zip][!.txt]'
        size, fullpath, _, _, _ = fo.FnGetFileSize(regStr)
        # filter out very low sized files
        list_of_files = list(compress(fullpath, [(s>min_size) for s in size]))
        # find the latest file
        latest_file = sorted(list_of_files, key=os.path.getctime)[-2]
        print('[{0}]: latest file = {1}'.format(datetime.now(),os.path.basename(latest_file)))

        return latest_file


    # def main():
    #         searchStr = '(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
    #         dateformat = '%Y-%m-%d_%H-%M-%S'
    #         minor_gaps1, major_gaps1 = FnFindFileGaps(searchStr, fullpath1, dateformat, period='H')
    #         minor_gaps2, major_gaps2 = FnFindFileGaps(searchStr, fullpath2, dateformat, period='H')

    def FnDateMask(self, regStr,  tstart, tend, searchStr, dateformat):
    # script to filter files based on dates using start date and end date
        import numpy as np
        from FileOperations import FileOperations
        fo = FileOperations(self.path)
        file_prop = fo.FnGetFileSize(self.path, regStr)
        dates = fo.FnGetDateTime(file_prop.fullpaths, searchStr, dateformat)

		# adjust if the files are of hourly resolution, while the tstart and tend are of higher resolution
        if np.diff(dates[0:2])[0] > (tend-tstart):
            tstart = tstart.replace(minute=0, hour=tstart.hour)
            tend = tend.replace(minute=0, hour=tend.hour+1)

        dates_mask = [(d >= tstart) & (d <= tend) for d in dates]
        idx_mask = list(np.ravel(np.where(dates_mask)))
		
        # sizes_mask = [file_prop.sizes[i] for i in idx_mask]
        # paths_mask = [file_prop.fullpaths[i] for i in idx_mask]
        # filenames_mask = [file_prop.filenames[i] for i in idx_mask]
        # names_mask = [file_prop.names[i] for i in idx_mask]
        # ext_mask = [file_prop.extension[i] for i in idx_mask]
        # return sizes_mask, paths_mask, fullnames_mask, names_mask, ext_mask
        return file_prop.iloc[idx_mask,:]

    @staticmethod
    def FnCheckSynced(files1, files2, size1):
    #  script to check the elements synced from list1 to list2, list1 is the elaborate list
    # Input:
    #     files1 - python list with filenames, should be bigger list than list2 i.e. the reference list
    #     files2 - python list with filenames
    #     sizes1 - size of files1
    # Output:
    #     files_synced  - files in list2 are available in list1
    #     files_notsynced - files absent in list2, but available in list1
    #     sizes_synced - sizes of files_synced
    #     sizes_notsynced - sizes of files_notsynced
        import numpy as np

        files_synced, files_notsynced = [],[]           
        sizes_synced, sizes_notsynced = [],[]           
        for filename, size in zip(files1, size1):
            if filename in files2:
                files_synced.append(filename)
                sizes_synced.append(size)
            else:
                files_notsynced.append(filename)
                sizes_notsynced.append(size)
        return files_synced, files_notsynced, sizes_synced, sizes_notsynced

    def FnRenameFiles(self, regStr, based_on):
        # script to Rename the files in binary folder to include the UTC time from the file headers
        import pandas as pd
        import os

        sensor = self.FnGetFileSize(regStr="*.dat")
        count = 0

        for f in sensor.fullpaths:
            df = pd.read_csv(f, usecols=[0] , header=1, parse_dates=True, skiprows=range(2,4))
            tstart = pd.to_datetime(df.TIMESTAMP.iloc[0]).strftime("%Y%m%d_%H")
            tend = pd.to_datetime(df.TIMESTAMP.iloc[-1]).strftime("%Y%m%d_%H")

            if based_on == "start":
                os.rename(f,"".join([os.path.dirname(f), tstart,'_',os.path.basename(f)]))
            elif based_on == "end":
                os.rename(f, "".join([os.path.dirname(f), tend, '_', os.path.basename(f)]))
            else:
                print('provide based_on = str(start/end) for renaming files')

            count = count + 1
            print('{0}: {1}/{2} File processed ({3})'.format(pa.now(), count ,len(sensor.fullpaths), os.path.basename(f)))

    def delete_files_below_threshold(self, prop, threshold):
        # function to delete files below a size threshold
        # input:
        #   prop - output from FileOperations.FnGetFileSize()
        #   threshold - input size of file in bytes as a threshold
        # output:
        #   filtered_files = files filtered and deleted
        #   ***filtered_files are deleted

        filtered_files = prop[prop.sizes <= threshold]
        print('[{0}]: Following files will be deleted {1} \n'.format(pa.now(), filtered_files.names))

        DeleteFiles = input('Enter yes or no!: ')
        if  DeleteFiles == ("yes"):
            N, Ntotal = 0, len(filtered_files.fullpaths) 
            for f in filtered_files.fullpaths:
                try:       
                    os.remove(f)
                    N += 1
                    print('[{0}]: Deleted file {1}/{2}'.format(pa.now(), N, Ntotal))
                except OSError as e:  ## if failed, report it back to the user ##
                    print ("Error: %s - %s." % (e.filename, e.strerror))
        else:
            print('[{0}]: Files not deleted!'.format(pa.now()))

        return filtered_files

    @staticmethod
    def write_paths_to_file(dest_path, list):
        # writes a list to a file, the list maybe be paths or string

        if not(os.path.exists(dest_path)):
            mode = 'w'
        else:
            mode = 'a'

        with open(dest_path, mode) as fp:
            for item in list:
                # write each item on a new line
                fp.write("%s\n" % item) 

    import pandas as pd
    import os
    import glob

    @staticmethod
    def remove_duplicates_and_save(src_path, dest_path, regStr):
        # List all files in the input folder
        files = glob.glob(os.path.join(src_path, regStr), recursive=False)

        # Initialize an empty DataFrame to store the data
        df_combined = pd.DataFrame()

        # Iterate through each file in the folder
        # update for speed using dask dataframes
        for file in files:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file, header=0, lineterminator='\n',encoding='UTF-8', encoding_errors='ignore', on_bad_lines='warn')

            # Append the unique data to the combined DataFrame
            df_combined = pd.concat([df_combined , df], ignore_index=True, axis=0)

            # Remove duplicates based on all columns
            df_combined = df_combined.drop_duplicates()

            print(f"{file} completed")

        # df_combined = df_combined.fillna(method='ffill')

        # TODO: remove files that have the newline error while writing
        df_combined = df_combined.set_index('datetime')

        # Save the combined DataFrame as a single CSV file
        df_combined.to_csv(dest_path, index=False)

        return df_combined
    
    def main_combined_srws_csv(self, regStr, **kwargs):
        """Example to above definition"""

        input_folder = kwargs.setdefault('input_folder', self.path)
        output_folder = kwargs.setdefault('output_folder', self.path)
        filename = kwargs.setdefault('filename', regStr.replace('*', ''))
        output_file = kwargs.setdefault('output_file', os.path.join(self.path, filename))

        regStr = regStr # 'finished*.txt'
        df_combined = self.remove_duplicates_and_save(input_folder, output_file, regStr)
        return df_combined

    def convert_file_granulity(data_in: Union[xr.Dataset, pd.DataFrame], data_out: Union[xr.Dataset, pd.DataFrame], time_groupby_str: str) -> Dict[str, xr.Dataset]:
        """convert the file granulity from a higher to lower granulity e.g. from daily to hour granulity"""
        
        if isinstance(data_in, xr.Dataset):
            groups = data_in.groupby(time_groupby_str)

        data_out = {}
        for res, group in groups:
            data_out[res] = group

        return data_out

    @staticmethod    
    def read_nc(nc_path: str) -> Union[dict, dict]:
        """ read nc files with groups"""

        import netCDF4 as nc
        import pandas as pd
        import xarray as xr

        # initialize the xarray dataset and pd. dataframe
        ds_combined = {}
        df_combined = {}
        
        ds = nc.Dataset(nc_path)
        if not ds.groups:
            ds_combined = xr.load_dataset(nc_path, engine='netcdf4',decode_times=True, decode_cf=True)
            # attrs_timestamp = ds_combined['Timestamp'].attrs
            # ds_combined['Timestamp'] = pd.to_datetime(ds_combined['Timestamp'].values, utc=True)
            # ds_combined['Timestamp'].attrs = attrs_timestamp
            # create a group pandas dataframe in a dict
            df_combined = ds_combined.to_dataframe()
        for g in ds.groups.keys():
            # initialize for group 
            ds_combined[f'{g}'] = {}
            df_combined[f'{g}'] = {}
            if not ds.groups[g].groups:
                ds_combined[f'{g}'] = xr.load_dataset(nc_path, group=f'{g}', engine='netcdf4', decode_times=False, decode_cf=False)
                df_combined[f'{g}'] = ds_combined[f'{g}'].to_dataframe()
            else:                                
                for sg in ds.groups[g].groups.keys():
                    # create an group xarray and store in a dict    
                    ds_combined[f'{g}'][f'{sg}'] = xr.load_dataset(nc_path, group=f'{g}/{sg}', engine='netcdf4', decode_times=False, decode_cf=False)
                    attrs_timestamp = ds_combined[f'{g}'][f'{sg}']['Timestamp'].attrs
                    ds_combined[f'{g}'][f'{sg}']['Timestamp'] = pd.to_datetime(ds_combined[f'{g}'][f'{sg}']['Timestamp'].values, utc=True)
                    ds_combined[f'{g}'][f'{sg}']['Timestamp'].attrs = attrs_timestamp
                    # create a group pandas dataframe in a dict
                    df_combined[f'{g}'][f'{sg}'] = ds_combined[f'{g}'][f'{sg}'].to_dataframe()

        return ds_combined, df_combined    


# Example:
if __name__ == "__main__":

    # from FileOperations import FnFindFileGaps
    import pandas as pd
    from FileOperations import FileOperations 
    import matlab2py as m2p
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from joblib import Parallel, delayed
    
    # nc_path = r"C:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\Flow\data\flow\cups\bhv_flow_cups_10min_L3_20211101T000000Z.nc"
    # dsc, dfc = FileOperations.read_nc(nc_path)
 
    # # find all the files in the root folder
    # dest_folder = r"d:\srws\Data\parquet"
    # log_file = r"d:\srws\Data\parquet\copy.log"
    # src_folder = r"z:\OE\OE400_STUD\RaoSuk\parquet"
    # prop_src = pd.DataFrame()
    # for subdir, dirs, files in os.walk(src_folder):
    #         fo = FileOperations(subdir)
    #         file_prop = fo.FnGetFileSize(subdir, regStr='data_*.parq')
    #         file_prop = pd.DataFrame(file_prop)
    #         prop_src = pd.concat([prop_src, file_prop], axis=0, ignore_index=True)
    # prop_src = prop_src.drop_duplicates()

    # # Set path
    # src_path = r"z:\OE\OE400_STUD\RaoSuk\parquet\2022"
    # dest_path = r"d:\srws\Data\parquet\2022"
    # fo = FileOperations(src_path)

    # with Parallel(n_jobs=10, prefer="threads") as parallel:
    #     delayed_funcs = [delayed(fo.copy_to)(f, os.path.join(dest_folder,str(pd.to_datetime(f[-28:-5]).year), os.path.basename(f))) for f in sorted(prop_src.fullpaths)]
    #     _ = parallel(delayed_funcs)
        
    # # get the files
    # m2p.tic()
    # files = fo.FnGetFileSize(path, regStr='*T*[!.zip][!.txt][!.parq]')
    # m2p.toc()

    # # get the size and other details of files
    # # sizes, fullpaths, names, filenames, extensions = fo.FnGetFileSize('*[^\.]')
    # sizes, fullpaths, names, filenames, extensions = fo.FnGetFileSize('*.*')

    # # zip files in the given path
    # # fo.FnFastZipFiles(extn='')

    # min_size= 1
    # regStr = '*T*[!.zip][!.txt]' # for srws 
    # regStr = '*'
    # latest = fo.FnGetLatestFile(regStr, min_size)

    # fo.FnDuplicates()

    # # fo.getSftp()
    # searchStr = '(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
    # dateformat = '%Y-%m-%d_%H-%M-%S'
    # minor_gaps1, major_gaps1 = fo.FnFindFileGaps(searchStr, path1, dateformat, period='H')
    # minor_gaps2, major_gaps2 = fo.FnFindFileGaps(searchStr, path2, dateformat, period='H')

    # # get the files within a folder structure
    # files1 = fo.FnGetFiles(path1)
    # files2 = fo.FnGetFiles(path2)

    # # get the file sizes and file paths
    # regStr = "*real_time*.csv"
    # size1, fullpath1, names1 = fo.FnGetFileSize(path1, regStr)
    # size2, fullpath2, names2 = fo.FnGetFileSize(path2, regStr)

    # df = pd.DataFrame(
    #     {
    #     'size':size,
    #     'fullpath':fullpath,
    #     'names':names,
    #     }
    # )

    # # df.to_pickle('../results/FileOperations_GreenPO.pkl')

    # Example usage for combining files due to error in synchronisation from cloud:
    src_path = r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\HighRe\data\Bowtie1\csv"
    # dest_path = r"c:\Users\giyash\OneDrive - Fraunhofer\Python\Scripts\HighRe\data\Bowtie1\finished_files_ID5899.txt"
    # regStr = 'finished*.txt'
    # df_combined=FileOperations.remove_duplicates_and_save(src_path, dest_path, regStr)
    fo = FileOperations(src_path)
    df_combined = fo.main_combined_srws_csv(regStr='*srws_data_hubH*.csv')

    sys.exit('manual stop')    
