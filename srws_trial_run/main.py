import os
import sys
import argparse
import pythonAssist as pa
from ProcessSRWS_ID6931 import ProcessSRWS
import glob

def main(args: list[str]):


    inp = pa.struct()
    inp.write_spec = None
    inp.filename = None
    inp.mode = None

    # define and set the attributes of struct as arguments
    parser = argparse.ArgumentParser(description="Read SRWS binary file and convert to numpy array")
    parser.add_argument("--root", help="path to all the files")
    parser.add_argument("--mode", choices=["basic", "standard", "all"], default="all", help="Mode of data extraction (basic, standard, all)")    
    parser.add_argument("--write_spec", choices=["True", "False"], default="False", help="Write the spectra from SRWS data to output df)")
    parser.add_argument("--regStr", default="*.*", help="provide a regular expression for filtering files")    

    args = parser.parse_args()

    # read in files
    srws = ProcessSRWS(inp.root, inp)

    files = glob.glob(os.path.join(args.root, args.regStr), recursive=True)
    for f in files:
        _, df, _ = srws.Read_SRWS_bin(f, args.mode)

        # write to csv file
        df.to_csv(os.path.join(f,".csv"))


if __name__ == "__main__":

    main(sys.argv[1:])
