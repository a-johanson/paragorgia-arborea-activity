# Copyright (c) 2016 Arne Johanson

import numpy as np
import pandas as pd
import json
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


### Load data

# Bins 1, 2, 3 of Up are to be removed later on
dirmagUpA = np.genfromtxt("input/MLM_adcpU_dirmag.csv", skip_header=3, delimiter=",", comments="#", dtype=float, invalid_raise=True)

# Bin 1 of Down is to be removed later on
dirmagDownA = np.genfromtxt("input/MLM_adcpD_dirmag.csv", skip_header=3, delimiter=",", comments="#", dtype=float, invalid_raise=True)

openessA = np.genfromtxt("input/coral_frames2.csv", skip_header=2, delimiter=",", comments="#", dtype=float, invalid_raise=True)

groupLabels = np.genfromtxt("input/frames_group_labels.csv", skip_header=1, usecols = (3,4,5,6), delimiter=",", comments="#", dtype=float, invalid_raise=True)

assert groupLabels.shape[0] == openessA.shape[0]


with open("input/scalar_POS434-156_conservativeTemperature_215_original.json") as fp:
	ctA = np.asarray(json.load(fp)["data"])

with open("input/scalar_POS434-156_absoluteSalinity_215_original.json") as fp:
	saA = np.asarray(json.load(fp)["data"])
	
with open("input/scalar_POS434-156_potentialDensityAnomaly_215_original.json") as fp:
	sigma0A = np.asarray(json.load(fp)["data"])

#with open("input/scalar_POS434-157_fluorescence_211_original.json") as fp:
#	fluorescenceA = np.asarray(json.load(fp)["data"])
#
#with open("input/scalar_POS434-157_pH_211_original.json") as fp:
#	pHA = np.asarray(json.load(fp)["data"])
#	
#with open("input/scalar_POS434-157_turbidity_211_original.json") as fp:
#	turbidityA = np.asarray(json.load(fp)["data"])




### Create time series date indices

dateOffset = np.datetime64("2012-06-01T00:00:01Z")

openessIndex = dateOffset + openessA[:,0].astype("timedelta64[s]")
ctIndex = dateOffset + ctA[:,0].astype("timedelta64[s]")
saIndex = dateOffset + saA[:,0].astype("timedelta64[s]")
sigma0Index = dateOffset + sigma0A[:,0].astype("timedelta64[s]")
#fluorescenceIndex = dateOffset + fluorescenceA[:,0].astype("timedelta64[s]")
#pHIndex = dateOffset + pHA[:,0].astype("timedelta64[s]")
#turbidityIndex = dateOffset + turbidityA[:,0].astype("timedelta64[s]")

dirmagUpIndex = dateOffset + dirmagUpA[:,0].astype("timedelta64[s]")
dirmagDownIndex = dateOffset + dirmagDownA[:,0].astype("timedelta64[s]")


### Create original time series / data frames

openessOrig = pd.Series(openessA[:,1], index=openessIndex)
ctOrig = pd.Series(ctA[:,1], index=ctIndex)
saOrig = pd.Series(saA[:,1], index=saIndex)
sigma0Orig = pd.Series(sigma0A[:,1], index=sigma0Index)
#fluorescenceOrig = pd.Series(fluorescenceA[:,1], index=fluorescenceIndex)
#pHOrig = pd.Series(pHA[:,1], index=pHIndex)
#turbidityOrig = pd.Series(turbidityA[:,1], index=turbidityIndex)

nBinsUnfilteredUp = round((dirmagUpA.shape[1]-1)/2)
dirUpOrig = pd.DataFrame(data=dirmagUpA[:,1:(1+nBinsUnfilteredUp)], index=dirmagUpIndex)
magUpOrig = pd.DataFrame(data=dirmagUpA[:,(1+nBinsUnfilteredUp):], index=dirmagUpIndex)

nBinsUnfilteredDown = round((dirmagDownA.shape[1]-1)/2)
dirDownOrig = pd.DataFrame(data=dirmagDownA[:,1:(1+nBinsUnfilteredDown)], index=dirmagDownIndex)
magDownOrig = pd.DataFrame(data=dirmagDownA[:,(1+nBinsUnfilteredDown):], index=dirmagDownIndex)


### Interpolate univariate time series

def interpolateUnivariateTSLinear(targetTSIndex, sourceTS, offset=np.timedelta64(0, "s"), derivative=False):
	target_index = targetTSIndex + offset
	indexAfter = np.searchsorted(sourceTS.index.values, target_index, side="right")
	value_a = sourceTS.values[indexAfter-1]
	value_b = sourceTS.values[indexAfter]
	time_a = sourceTS.index.values[indexAfter-1]
	time_b = sourceTS.index.values[indexAfter]
	t_span = time_b - time_a
	target_values = np.empty(target_index.shape[0])
	if not derivative:
		weight_b = (target_index - time_a) / t_span
		target_values[:] = (1-weight_b) * value_a + weight_b * value_b
	else:
		t_span_in_minutes = t_span / np.timedelta64(1, "m")
		target_values[:] = (value_b - value_a) / t_span_in_minutes
	return target_values


def createUnivariateTSProducts(targetName, targetTSIndex, sourceTS, offsetsInMin=[-2*60, -3*60, -4*60]):#offsetsInMin=[-60, -2*60, -3*60, -4*60, -5*60, -6*60, -7*60, -8*60, -9*60, -10*60, -11*60]): #offsetsInMin=[-3*60, -3*60-10, -3*60-20, -3*60-30]
	tsValuesDict = {}
	# interpolation
	name = targetName
	tsValuesDict[name] = {"name": name, "data": interpolateUnivariateTSLinear(targetTSIndex, sourceTS)}
	## derivative
	#name = "{}_dt".format(targetName)
	#tsValuesDict[name] = {"name": name, "data": interpolateUnivariateTSLinear(targetTSIndex, sourceTS, derivative=True)}
	# lags + derivative at lag
	for lag in offsetsInMin:
		name = "{}_lag_{}min".format(targetName, abs(lag))
		tsValuesDict[name] = {"name": name, "data": interpolateUnivariateTSLinear(targetTSIndex, sourceTS, offset=np.timedelta64(lag, "m"))}
		#name = "{}_dt_lag_{}min".format(targetName, abs(lag))
		#tsValuesDict[name] = {"name": name, "data": interpolateUnivariateTSLinear(targetTSIndex, sourceTS, offset=np.timedelta64(lag, "m"), derivative=True)}
	return tsValuesDict
	
openess = openessOrig
ignoreBecauseOfLags = 7
openess = openess[ignoreBecauseOfLags:]
groupLabels = groupLabels[ignoreBecauseOfLags:,:]
assert openess.values.shape[0] == groupLabels.shape[0]
print("k =", openess.shape[0])

groupLinComb = 0.7*groupLabels[:,0] + 0.2*groupLabels[:,1] + 0.07*groupLabels[:,2] + 0.03*groupLabels[:,3]




ct_products = createUnivariateTSProducts("ct", openess.index.values, ctOrig)
sa_products = createUnivariateTSProducts("sa", openess.index.values, saOrig)
sigma0_products = createUnivariateTSProducts("sigma0", openess.index.values, sigma0Orig)

univariate_products = {}
univariate_products.update(ct_products)
univariate_products.update(sa_products)
univariate_products.update(sigma0_products)


### Plot observations

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

axStart = np.datetime64("2012-06-04T23:59:59Z")
oneAxDuration = np.timedelta64(6*24, "h")
for i,ax in enumerate([ax1,ax2]):
	ax.plot(openess.index, openess.values, label = r"Labels $y^{(i)}$", color="b", marker="o", markeredgewidth=1.25, linewidth=1.25)
	ax.plot(openess.index, groupLinComb, label = r"Linear Combinations $\eta_i$", color="r", marker="o", markeredgewidth=1.25, linewidth=1.25)
	ax.plot(openess.index, np.full(openess.index.shape[0], 0.5), label = "Decision Boundary", color="k", linewidth=1.25)
	ax.set_xlim([axStart+i*oneAxDuration,axStart+(i+1)*oneAxDuration])
	ax.set_ylim([-0.1,1.1])
	ax.set_ylabel("Deg. of Extension")
	ax.xaxis.set_major_locator(mdates.DayLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
	ax.xaxis.set_minor_locator(mdates.HourLocator())

ax1.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, ncol=5, mode="expand", borderaxespad=0.0)
plt.xlabel("Date")
fig.savefig("figures/observations.pdf", bbox_inches="tight")
plt.close(fig)


### Interpolate multivariate time series

def interpolateMultivariateTSLinear(targetTSIndex, sourceDF, colPrefix, colOffset=0, offset=np.timedelta64(0, "s"), derivative=False, angles=False):
	target_index = targetTSIndex + offset
	indexAfter = np.searchsorted(sourceDF.index.values, target_index, side="right")
	time_a = sourceDF.index.values[indexAfter-1]
	time_b = sourceDF.index.values[indexAfter]
	t_span = time_b - time_a
	colNames = []
	target_values = np.empty((target_index.shape[0], (2 if angles and not derivative else 1)*(sourceDF.values.shape[1]-colOffset)))
	for j in range(colOffset, sourceDF.values.shape[1]):
		if angles and not derivative:
			colNames.append("{}_X{}".format(colPrefix, j+1))
			colNames.append("{}_Y{}".format(colPrefix, j+1))
		else:
			colNames.append("{}_{}".format(colPrefix, j+1))
		value_a = sourceDF.values[indexAfter-1, j]
		value_b = sourceDF.values[indexAfter  , j]
		if not derivative:
			weight_b = (target_index - time_a) / t_span
			weight_a = 1 - weight_b
			if not angles:
				target_values[:,j-colOffset] = weight_a * value_a + weight_b * value_b
			else:
				#target.values[:,j-colOffset] = np.arctan2(weight_a*np.sin(value_a) + weight_b*np.sin(value_b), \
				#								weight_a*np.cos(value_a) + weight_b*np.cos(value_b))
				interpolated_x = weight_a*np.cos(value_a) + weight_b*np.cos(value_b)
				interpolated_y = weight_a*np.sin(value_a) + weight_b*np.sin(value_b)
				interpolated_norm = np.sqrt(np.square(interpolated_x) + np.square(interpolated_y))
				target_values[:,(2*(j-colOffset))  ] = interpolated_x / interpolated_norm
				target_values[:,(2*(j-colOffset)+1)] = interpolated_y / interpolated_norm
		else:
			t_span_in_minutes = t_span / np.timedelta64(1, "m")
			if not angles:
				target_values[:,j-colOffset] = (value_b - value_a) / t_span_in_minutes
			else:
				#atan2(sin(x-y), cos(x-y))
				angle_diff = value_b - value_a
				target_values[:,j-colOffset] = np.arctan2(np.sin(angle_diff), np.cos(angle_diff)) / t_span_in_minutes
	return (colNames, target_values)


def createMultivariateTSProducts(targetName, targetTSIndex, sourceDF, colOffset=0, angles=False, offsetsInMin=[-2*60, -3*60, -4*60]): #offsetsInMin=[-60, -2*60, -3*60, -4*60, -5*60, -6*60, -7*60, -8*60, -9*60, -10*60, -11*60]): #offsetsInMin=[-3*60, -3*60-10, -3*60-20, -3*60-30]
	tsValuesDict = {}
	# interpolation
	name = targetName
	colNames, data = interpolateMultivariateTSLinear(targetTSIndex, sourceDF, name, colOffset=colOffset, angles=angles)
	tsValuesDict[name] = {"name": name, "colNames": colNames, "data": data}
	## derivative
	#name = "{}_dt".format(targetName)
	#colNames, data = interpolateMultivariateTSLinear(targetTSIndex, sourceDF, name, colOffset=colOffset, angles=angles, derivative=True)
	#tsValuesDict[name] = {"name": name, "colNames": colNames, "data": data}
	# lags + derivative at lag
	for lag in offsetsInMin:
		name = "{}_lag_{}min".format(targetName, abs(lag))
		colNames, data = interpolateMultivariateTSLinear(targetTSIndex, sourceDF, name, colOffset=colOffset, angles=angles, offset=np.timedelta64(lag, "m"))
		tsValuesDict[name] = {"name": name, "colNames": colNames, "data": data}

		#name = "{}_dt_lag_{}min".format(targetName, abs(lag))
		#colNames, data = interpolateMultivariateTSLinear(targetTSIndex, sourceDF, name, colOffset=colOffset, angles=angles, derivative=True, offset=np.timedelta64(lag, "m"))
		#tsValuesDict[name] = {"name": name, "colNames": colNames, "data": data}
	return tsValuesDict


# filter out Bins 1-3 for upward-facing ADCP
nBinsSkipUp = 3

dirUp_products = createMultivariateTSProducts("dirUp", openess.index.values, dirUpOrig, colOffset=nBinsSkipUp, angles=True)
magUp_products = createMultivariateTSProducts("magUp", openess.index.values, magUpOrig, colOffset=nBinsSkipUp)


# filter out Bin 1 for downward-facing ADCP
nBinsSkipDown = 1

dirDown_products = createMultivariateTSProducts("dirDown", openess.index.values, dirDownOrig, colOffset=nBinsSkipDown, angles=True)
magDown_products = createMultivariateTSProducts("magDown", openess.index.values, magDownOrig, colOffset=nBinsSkipDown)

multivariate_products = {}
multivariate_products.update(dirUp_products)
multivariate_products.update(magUp_products)
multivariate_products.update(dirDown_products)
multivariate_products.update(magDown_products)


### Create several data products with high resolution to generate quasi-continuous predictions with our LR models

origIndexStart = 336185 # in [s]
origIndexEnd = 9332570 # in [s] --- shorter: 1342685
origIndexStep = 600 # in [s]
commonOrigIndex = dateOffset + np.arange(origIndexStart, origIndexEnd, origIndexStep).astype("timedelta64[s]")

dirUpXYOrig_lag_3h = pd.DataFrame(data=np.empty((commonOrigIndex.size, 2*(dirUpOrig.values.shape[1]-nBinsSkipUp))), index=commonOrigIndex)
dirUpXYOrig_lag_3h.columns, dirUpXYOrig_lag_3h.values[:,:] = interpolateMultivariateTSLinear(dirUpXYOrig_lag_3h.index.values, dirUpOrig, "direction", colOffset=nBinsSkipUp, offset=np.timedelta64(-3, "h"), derivative=False, angles=True)

dirUpXYOrig_lag_4h = pd.DataFrame(data=np.empty((commonOrigIndex.size, 2*(dirUpOrig.values.shape[1]-nBinsSkipUp))), index=commonOrigIndex)
dirUpXYOrig_lag_4h.columns, dirUpXYOrig_lag_4h.values[:,:] = interpolateMultivariateTSLinear(dirUpXYOrig_lag_4h.index.values, dirUpOrig, "direction", colOffset=nBinsSkipUp, offset=np.timedelta64(-4, "h"), derivative=False, angles=True)

magUpOrig_lag_2h = pd.DataFrame(data=np.empty((commonOrigIndex.size, magUpOrig.values.shape[1]-nBinsSkipUp)), index=commonOrigIndex)
magUpOrig_lag_2h.columns, magUpOrig_lag_2h.values[:,:] = interpolateMultivariateTSLinear(magUpOrig_lag_2h.index.values, magUpOrig, "magnitude", colOffset=nBinsSkipUp, offset=np.timedelta64(-2, "h"), derivative=False, angles=False)

magDownOrig_lag_2h = pd.DataFrame(data=np.empty((commonOrigIndex.size, magDownOrig.values.shape[1]-nBinsSkipDown)), index=commonOrigIndex)
magDownOrig_lag_2h.columns, magDownOrig_lag_2h.values[:,:] = interpolateMultivariateTSLinear(magDownOrig_lag_2h.index.values, magDownOrig, "magnitude", colOffset=nBinsSkipDown, offset=np.timedelta64(-2, "h"), derivative=False, angles=False)

magDownOrig_lag_3h = pd.DataFrame(data=np.empty((commonOrigIndex.size, magDownOrig.values.shape[1]-nBinsSkipDown)), index=commonOrigIndex)
magDownOrig_lag_3h.columns, magDownOrig_lag_3h.values[:,:] = interpolateMultivariateTSLinear(magDownOrig_lag_3h.index.values, magDownOrig, "magnitude", colOffset=nBinsSkipDown, offset=np.timedelta64(-3, "h"), derivative=False, angles=False)


### Output data products

pathPrefix = "./data_products/open_closed/"

openess.to_pickle(pathPrefix+"ts_openess.pkl")

with open(pathPrefix+"univariate_products.pkl", "wb") as fp:
	pickle.dump(univariate_products, fp)

with open(pathPrefix+"multivariate_products.pkl", "wb") as fp:
	pickle.dump(multivariate_products, fp)

dirUpXYOrig_lag_3h.to_pickle(pathPrefix+"df_dirXYUpOriginal_lag_3h.pkl")
dirUpXYOrig_lag_4h.to_pickle(pathPrefix+"df_dirXYUpOriginal_lag_4h.pkl")
magUpOrig_lag_2h.to_pickle(pathPrefix+"df_magUpOriginal_lag_2h.pkl")
magDownOrig_lag_2h.to_pickle(pathPrefix+"df_magDownOriginal_lag_2h.pkl")
magDownOrig_lag_3h.to_pickle(pathPrefix+"df_magDownOriginal_lag_3h.pkl")

## Additional JSON output
#jsonProduct = {
#	"timestamp": ((openess.index-dateOffset)/np.timedelta64(1, "s")).tolist(),
#	"label": openess.values.tolist(),
#	"ct": univariate_products["ct"]["data"].tolist(),
#	"sa": univariate_products["sa"]["data"].tolist(),
#	"sigma0": univariate_products["sigma0"]["data"].tolist()
#}
#with open(pathPrefix+"extract.json", "w") as fp:
#    json.dump(jsonProduct, fp)

print("Output files written")
