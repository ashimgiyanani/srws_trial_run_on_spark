import os
import sys
import argparse
import pythonAssist as pa
from ProcessSRWS_spark import ProcessSRWS
import glob
from dateutil import parser
from spectr import SpectralAnalysis
from datetime import datetime

def main(args: list[str]):

    # define and set the attributes of struct as arguments
    parser = argparse.ArgumentParser(description="Read SRWS binary file and convert to numpy array")
    parser.add_argument("srws_path_root", help="path to all the files")
    parser.add_argument("srws_coord", help="Path to coordinate file")
    parser.add_argument("mode", choices=["basic", "standard", "all"], default="all", help="Mode of data extraction (basic, standard, all)")    
    parser.add_argument("write_spec",type=bool, choices=[True, False], default="False", help="Write the spectra from SRWS data to output df)")
    parser.add_argument("srws_regStr", default="*.*", help="provide a regular expression for filtering files")    
    parser.add_argument("srws_merge", type=bool, default=False, help="Flag to indicate whether to concatenate dataframes")
    parser.add_argument("srws_relative_align", type=bool, default=True, help="Flag to indicate whether to align with North")
    parser.add_argument("srws_use_corrected_vlos", default=True, type=bool, help="Flag to indicate whether to use corrected vlos")
    parser.add_argument("pickle", default=False, type=bool, help="Flag to indicate whether to pickle the data")
    parser.add_argument("write_csv", default=False, type=bool, help="Flag to indicate whether to write data to CSV")
    parser.add_argument("write_parquet", default=True, type=bool, help="Flag to indicate whether to write data to Parquet")
    parser.add_argument("generate_stats", default=False, type=bool, help="Flag to indicate whether to generate statistics")
    parser.add_argument("filter_data", default=False, type=bool, help="Flag to indicate whether to filter data")
    parser.add_argument("plot_figure", default=False, type=bool, help="Flag to indicate whether to plot figures")
    parser.add_argument("workDir", default="None", help="Working directory")
    parser.add_argument("tstart", type=lambda d: datetime.strptime(d, '%Y-%m-%d_%H-%M-%S'), help="Start date")
    parser.add_argument("tend", type=lambda d: datetime.strptime(d, '%Y-%m-%d_%H-%M-%S'), help="End date")
    parser.add_argument("add_metadata", default=True, type=bool, help="Flag to indicate whether to add metadata")
    parser.add_argument("target_path", default="None", help="Path to target directory")
    parser.add_argument("srws_path_error_files", default="None", help="Path to error files")
    parser.add_argument("srws_path_finished_files", default="None", help="Path to finished files")

    args = parser.parse_args(args_list)
    inp = pa.struct(**vars(args))

#     # read in files
    srws = ProcessSRWS(inp.srws_path_root, inp)
#     Data, df_H, df_bowtie, df_m, ds, ds_H, ds_bowtie, ds_m = srws.FnConvertRawData(inp)

    files = glob.glob(os.path.join(inp.srws_path_root, inp.srws_regStr), recursive=True)
    for f in files:
        print(f"[{pa.now()}]: processing file {f}")
        _, df, _ = srws.Read_SRWS_bin(f, inp.mode)

        # write to csv file
        df.to_csv(f + ".csv")

    return None

if __name__ == "__main__":

    base_dir = sys.argv[1]
    # base_dir = os.path.dirname(os.getcwd())    
    print(f"Using dir: {base_dir}")
    
    # for dubugging purposes, clear the contents of error_files and finished _files
    open(rf"{base_dir}\srws_data\error_files.txt", "w").close()
    open(rf"{base_dir}\srws_data\finished_files.txt", "w").close()

    # remove parquet output files saved [for debugging purposes]
    from FileOperations import FileOperations
    fo = FileOperations(base_dir)
    files = fo.FnGetFileSize(base_dir, "202*T*")

    # Define the arguments as a list
    args_list = [
        base_dir,
        f"{base_dir}/srws_data/BREMERHAVEN TOT.txt",
        "all",
        "1",
        "**/*T*[!.zip][!.txt][!.csv][!.parq]",
        "False",
        "True",
        "True",
        "False",
        "True",
        "True",
        "False",
        "False",
        False,
        os.path.dirname(sys.path[0]),
        "2021-11-01_12-00-00",
        "2021-11-01_14-59-00",
        "False",
        base_dir,
        f"{base_dir}/srws_data/error_files.txt",
        f"{base_dir}/srws_data/finished_files.txt"
    ]
    
    # Call the main function with the arguments list
    main(args_list)
    #main(sys.argv[1:])
