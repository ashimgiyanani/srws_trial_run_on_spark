# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:01:38 2020

@author: giyash
"""

def FnWsRange(WsIn, WsMin, WsMax, flag, verbose):
# Function to filter out the wind speed for values lower than the minimum selected and maximum selected
# Syntax: WsOut = FnWsRange(WsIn, WsMin, WsMax, flag, verbose)

# Input:
    # WsIn - Input Wind speed matrix
    # WsMin - minimum wind speed filter eg: WsMin = 0.5 m/s
    # WsMax - maximum wind speed filter eg: WsMax = 40 m/s
    # flag - value to be assigned for an erroneous value, e.g. nan
    # verbose - '1' to print the no.of input and output data  on screen, '0' to supress

# Output:
    # WsOut - Output wind speed matrix
    
# Example:
    # import numpy as np
    # WsIn = 1*np.random.randn(10)
    # WsMin = 0
    # WsMax = 40
    # flag = np.nan
    # verbose=1
    # WsOut = FnWsRange(WsIn, WsMin, WsMax, flag, verbose)

# References:none
#
# Author: Ashim Giyanani, Research Associate 
# Fraunhofer Institute of Wind Energy 
# Windpark planning and operation department
# Am Seedeich 45, Bremerhaven 
# email: ashim.giyanani@iwes.fraunhofer.de
# Git site: https://gitlab.cc-asp.fraunhofer.de/giyash/testfeld-bhv.git  
# Created: 2020-10-02 10:31; 
# Last revision: 2020-10-02 10:31


    import numpy as np

    WsOut = WsIn.copy()                                      # output wind speed
    
    # perform range test on wind speed
    WsOut[WsIn.astype('float') <= WsMin] = flag                        # filtering below the range
    WsOut[WsIn.astype('float') > WsMax] = flag                         # filtering above the range
                                                       #
    nIn = np.sum(np.isnan(WsIn.astype('float')))                # number of input data
    nOut = np.sum(np.isnan(WsOut.astype('float')))              # no. of output data  after filtering
                                                       #
    if (verbose == 1): # verbose condition
      print(' No. of bad data in Input: {}'.format(nOut-nIn));         # display input nos.
    
    return WsOut

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    WsIn = 20*np.random.randn(13)	#

    WsIn = pd.DataFrame(WsIn, index = pd.date_range('01-01-2020', '01-01.2021', freq='MS'))
    WsMin = 0
    WsMax = 40
    flag = np.nan
    verbose=1
    WsOut = FnWsRange(WsIn, WsMin, WsMax, flag, verbose)
