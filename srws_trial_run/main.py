import os
import sys
import argparse
import pythonAssist as pa
from ProcessSRWS import ProcessSRWS
import glob
from dateutil import parser
from spectr import SpectralAnalysis

def main(args: list[str]):


    # define and set the attributes of struct as arguments
    parser = argparse.ArgumentParser(description="Read SRWS binary file and convert to numpy array")
    parser.add_argument("root", help="path to all the files")
    parser.add_argument("mode", choices=["basic", "standard", "all"], default="all", help="Mode of data extraction (basic, standard, all)")    
    parser.add_argument("write_spec", choices=["True", "False"], default="False", help="Write the spectra from SRWS data to output df)")
    parser.add_argument("regStr", default="*.*", help="provide a regular expression for filtering files")    

    args = parser.parse_args()

    # read in files
    srws = ProcessSRWS(args.root, args)

    files = glob.glob(os.path.join(args.root, args.regStr), recursive=True)
    for f in files:
        print(f"[{pa.now()}]: processing file {f}")
        _, df, _ = srws.Read_SRWS_bin(f, args.mode)

        # write to csv file
        df.to_csv(f + ".csv")


if __name__ == "__main__":

    main(sys.argv[1:])
