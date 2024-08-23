def FnPeaks(x, threshold=3, window=60, plotFig=True, verbose=True):
	""" 
	FnPeaks - Function definition to find peaks in the input x

	Syntax:
		x = pd.DataFrame(np.random.randint(0,1000), columns='x')
		threshold, window, plotFig, verbose = 3, 60, True, True
		peaks_idx, Npeaks, peaks_val = FnPeaks(x, threshold, window, plotFig, verbose)


	Inputs:
	x - input, preferably pandas dataframe, will be eventually converted to df
	threshold - threshold between the value and the rolling mean of the dataset x
	window - length of the rolling window used for the median
	plotFig - boolean stating if plots required or not [True / False]
	verbose - boolean stating if the print output is wished or not [True / False]

	Outputs:
		peaks_idx - index of the peaks  
		peaks.sum() - number of peaks found
		peaks_val - value of the peaks

	Example:
	modules required: pandas, randn, numpy, matplotlib
	classes required: none
	Data-files required: none

	See also: find_peaks

	References:
	Author: Ashim Giyanani, Research Associate 
	Fraunhofer Institute of Wind Energy 
	Windpark planning and operation department
	Am Seedeich 45, Bremerhaven 
	email: ashim.giyanani@iwes.fraunhofer.de
	Git site: https://gitlab.cc-asp.fraunhofer.de/giyash/testfeld-bhv.git  
	Created: 06-08-2020; Last revision: 12-May-200406-08-2020

	"""

	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt

	# convert to pandas dataframe
	df=pd.DataFrame()
	df['x'] = pd.DataFrame(x)

	# find peaks
	df['x_corr'] = df['x'].rolling(window).median().fillna(method='bfill').fillna(method='ffill')
	difference = np.abs(df['x'] - df['x_corr'])
	peaks = difference > threshold

	if (plotFig == True) & (peaks.sum()>0):
		figsize = (7, 2.75)
		kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)
		fig, ax = plt.subplots(figsize=figsize)
		df['x'].plot()
		df['x'][peaks].plot(**kw)

	peaks_idx = np.where(peaks)
	peaks_val = df['x'].iloc[peaks_idx]

	if verbose == True:
		print('No. of peaks found: {}'.format(len(peaks_idx[0])))

	return peaks_idx, peaks.sum(), peaks_val

# def FnPeaks_xr(x, threshold, window, plotFig):
# 	import xarray as xr
# 	import pandas as pd
# 	import matplotlib.pyplot as plt
#
# 	threshold = 20000
# 	xra = xr.DataArray()
#     xra = spec1_xr.rolling(Nspec=4).median().bfill(dim='Nspec').ffill(dim='Nspec').isel(index=2)
# 	difference = np.abs(spec1_xr.isel(index=2) - xra)
# 	peaks = difference > threshold
# 	peaks_idx = np.where(peaks)
# 	peaks_val = df['x'].iloc[peaks_idx]
#
# 	return peaks_idx, peaks.sum(), peaks_val

if __name__ == '__main__':
	import numpy as np
	import pandas as pd

	x = pd.DataFrame(np.random.randn(1000,1))
	x.iloc[500] = x.iloc[500] + 25
	x.iloc[750] = x.iloc[750] - 25
	threshold, window, plotFig, verbose = 4, 60, True, True
	peaks_idx, Npeaks, peaks_val = FnPeaks(x, threshold, window, plotFig, verbose)
