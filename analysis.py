# Copyright (c) 2016 Arne Johanson 

import numpy as np
import pandas as pd
from sklearn import decomposition, preprocessing, cross_validation, linear_model, feature_selection, metrics
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import itertools


### Load data

pathPrefix = "./data_products/open_closed/"

openess = pd.read_pickle(pathPrefix+"ts_openess.pkl")
dirXYUpFullSeries_lag_3h = pd.read_pickle(pathPrefix+"df_dirXYUpOriginal_lag_3h.pkl")
dirXYUpFullSeries_lag_4h = pd.read_pickle(pathPrefix+"df_dirXYUpOriginal_lag_4h.pkl")
magUpFullSeries_lag_2h   = pd.read_pickle(pathPrefix+"df_magUpOriginal_lag_2h.pkl")
magDownFullSeries_lag_2h = pd.read_pickle(pathPrefix+"df_magDownOriginal_lag_2h.pkl")
magDownFullSeries_lag_3h = pd.read_pickle(pathPrefix+"df_magDownOriginal_lag_3h.pkl")

with open(pathPrefix+"univariate_products.pkl", "rb") as fp:
	univariate_products = pickle.load(fp)

with open(pathPrefix+"multivariate_products.pkl", "rb") as fp:
	multivariate_products = pickle.load(fp)

#for p in univariate_products:
#	print(p["name"])
#
#for p in multivariate_products:
#	print(p["name"])


### Apply PCA to multivariate series

def writeCSV(fileName, data, timeIndex, colNames, dateOffset=np.datetime64("2012-06-01T00:00:01Z")):
	timestamps = (timeIndex - dateOffset) / np.timedelta64(1, "s")
	timestamps.reshape((timestamps.shape[0], 1))
	with open(fileName, "wb") as f:
		f.write(b"timestamp,")
		f.write((",".join(colNames)).encode("utf-8"))
		f.write(b"\n")
		np.savetxt(f, np.column_stack((timestamps, data)), delimiter=",")

def doPCA(x, nComponents, printStats=False, printPrefix=""):
	pca = decomposition.PCA(n_components=nComponents, whiten=False)
	xTrans = pca.fit_transform(x)
	if printStats:
		print("PCA({},{}) variance explained by components:".format(printPrefix, nComponents), pca.explained_variance_ratio_)
		print("PCA({},{}) total variance explained:".format(printPrefix, nComponents), sum(pca.explained_variance_ratio_))
		#print(pca.components_)
	return xTrans, pca

def doPCATargetVariance(x, minVariance, maxFeatures):
	for i in range(x.shape[1]):
		pca = decomposition.PCA(n_components=i+1, whiten=False)
		pca.fit(x)
		if sum(pca.explained_variance_ratio_) >= minVariance or i+1 >= maxFeatures:
			xTrans = pca.transform(x)
			return xTrans, pca
	return x, None

def doPCAOnDict(inDict, minVar, maxFeatures):
	outDict = {}
	for key in inDict:
		p = inDict[key]
		pcaData, pcaObj = doPCATargetVariance(p["data"], minVar, maxFeatures)
		colNames = []
		for i in range(pcaData.shape[1]):
			colNames.append("{}_pca{}".format(p["name"], i+1))
		newKey = "{}_pca".format(p["name"])
		outDict[newKey] = {"name": newKey, "pcaObj": pcaObj, "pcaData": pcaData, "colNames": colNames, "variance": sum(pcaObj.explained_variance_ratio_)}
	return outDict

print("PCA Stats: Feature | # Components | Varaince Retained")
multivariate_pca = doPCAOnDict(multivariate_products, 0.75, 3)
for key in multivariate_pca:
	p = multivariate_pca[key]
	print(p["name"], len(p["colNames"]), p["variance"])

n_univariate = len(univariate_products)
n_multivariatePCA = 0
for key in multivariate_pca:
	p = multivariate_pca[key]
	n_multivariatePCA += p["pcaData"].shape[1]


### Extract labels (y) and features (X)

y = openess.values[:].astype(int)
X = np.empty((openess.index.shape[0], n_univariate+n_multivariatePCA))

featureNames = []
colID = 0
for key in sorted(univariate_products):
	p = univariate_products[key]
	featureNames.append(key)
	X[:,colID] = p["data"]
	colID += 1

for key in sorted(multivariate_pca):
	p = multivariate_pca[key]
	featureNames.extend(p["colNames"])
	X[:,colID:colID+p["pcaData"].shape[1]] = p["pcaData"]
	colID += p["pcaData"].shape[1]

featureNames = np.asarray(featureNames)
print("Features names:", featureNames)
print("X.shape:", X.shape)

assert featureNames.shape[0] == X.shape[1]
print("sum(y==1):", sum(y==1))
print("sum(y==0):", sum(y==0))
print("y.size:", y.size)
assert sum(y==1) + sum(y==0) == y.size


### Functions for rose diagrams
# cf. https://stackoverflow.com/questions/16264837/how-would-one-add-a-colorbar-to-this-example

def colored_bar(left, height, z=None, width=0.8, bottom=0, countLimit=1, ax=None, **kwargs):
	if ax is None:
		ax = plt.gca()
	width = itertools.cycle(np.atleast_1d(width))
	bottom = itertools.cycle(np.atleast_1d(bottom))
	rects = []
	for x, y, h, w in zip(left, bottom, height, width):
		rects.append(Rectangle((x,y), w, h))
	coll = PatchCollection(rects, array=z, **kwargs)
	coll.set_clim(0, countLimit)
	ax.add_collection(coll)
	#ax.autoscale()
	return coll

def addRosePlot(fig, title, bins, avgVelocity, count, velocityLimit, countLimit=1, axisID=111, addColorScale=False):
	ax = fig.add_subplot(axisID, projection="polar")
	plt.title(title, y=1.1)
	ax.set_theta_direction(-1)
	ax.set_theta_offset(0.5*math.pi)
	ax.set_ylim([0, velocityLimit])
	#ax.set_rlabel_position(0)
	cmap = plt.get_cmap("viridis")
	coll = colored_bar(bins, avgVelocity, count, ax=ax, width=np.radians(360/bins.size-1), countLimit=countLimit, cmap=cmap)
	#fig.colorbar(coll)
	if addColorScale:
		cbaxes = fig.add_axes([0.1, 0.1, 0.8, 0.03]) 
		fig.colorbar(coll, cax = cbaxes, orientation="horizontal")  
	ax.set_yticks(range(100, velocityLimit, 100))

def createBinnedData(dirs, mags, nBins=36, degreesBegin=-180):
	howToSort = dirs.argsort()
	dirsSorted = dirs[howToSort]
	magsSorted = mags[howToSort]

	plotBins = np.radians(np.arange(degreesBegin, degreesBegin+360, math.floor(360/nBins)))
	assert plotBins.size == nBins
	plotAvgMags = np.zeros(nBins)
	plotCounts = np.zeros(nBins)

	iBin = 0
	boundaryIncr = 2.0*math.pi/nBins
	rightBoundary = degreesBegin/180 * math.pi + boundaryIncr
	for i, d in enumerate(dirsSorted):
		while d > rightBoundary:
			iBin += 1
			rightBoundary += boundaryIncr
		plotCounts[iBin] += 1
		plotAvgMags[iBin] += magsSorted[i]
	plotAvgMags = np.nan_to_num(plotAvgMags/plotCounts)
	return (plotBins, plotAvgMags, plotCounts)


### Function for training a model

def trainModel(X, y, featureNames, nFeatures=None, features2Select=None, seedXVal=None, seedLR=None):
	# Output:
	# 1. Test metrics
	# 2a. Classifier intercept
	# 2b. Classifier coefficients
	# 3. Feature names
	# 4a. Scaler mean
	# 4b. Scaler scale
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=seedXVal)

	scaler = preprocessing.StandardScaler()
	scaler.fit(X_train)
	X_train_std = scaler.transform(X_train)
	X_test_std = scaler.transform(X_test)

	classifier = linear_model.LogisticRegression(solver="liblinear", random_state=seedLR, penalty="l2", C=1.0)
	#classifier.fit(X_train_std, y_train)

	if nFeatures is not None:
		selector = feature_selection.RFE(classifier, n_features_to_select=nFeatures, verbose=False)
		selector = selector.fit(X_train_std, y_train)
		selected = selector.support_
		#print(selector.ranking_)
	else:
		selected = (featureNames == "")
		for f in features2Select:
			selected[np.where(featureNames == f)] = True

	selectedFeatureNames = featureNames[selected]
	#print("Selected features:", selectedFeatureNames)
	
	classifier.fit(X_train_std[:,selected], y_train)
	intercept = classifier.intercept_[0]
	coefficients = classifier.coef_[0]

	predictions = classifier.predict(X_test_std[:,selected])

	likelihood = 1.0
	for x_i, y_i in zip(X_test_std[:,selected], predictions):
		logfkt = 1.0/(1.0+math.exp(-1.0*(intercept + np.dot(coefficients, x_i))))
		likelihood *= (logfkt if y_i == 1 else (1.0-logfkt))
	#print(classifier.predict_proba(X_test_std[:,selector.support_]))

	n0 = sum(y_test==0)
	n1 = sum(y_test==1)
	n_total = len(y_test)
	#assert n0+n1 == n_total

	# Nice visualization: https://en.wikipedia.org/wiki/Precision_and_recall
	# cf.: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
	metricResults = { \
						"accuracy": metrics.accuracy_score(y_test, predictions), \
						"overallAccuracy": metrics.accuracy_score(y, classifier.predict(scaler.transform(X)[:,selected])), \
						"likelihood": likelihood, \
						"loglikelihood" : math.log(likelihood), \
						"precision": metrics.precision_score(y_test, predictions, average=None), \
						"recall": metrics.recall_score(y_test, predictions, average=None), \
						"support": np.array([n0, n1, n_total]), \
						"label": ["closed", "open", "avg/total"] \
					}
	metricResults["precision"] = np.append(metricResults["precision"], [(n0*metricResults["precision"][0] + n1*metricResults["precision"][1]) / n_total])
	metricResults["recall"] = np.append(metricResults["recall"], [(n0*metricResults["recall"][0] + n1*metricResults["recall"][1]) / n_total])

	#print(metrics.classification_report(y_test, predictions, digits=5))
	#confidence = classifier.decision_function(X_test_std[:,selector.support_]) # i.e., distance from decision boundary

	return (metricResults, intercept, coefficients, selectedFeatureNames, scaler.mean_[selected], scaler.scale_[selected])


### Parameters for model generation

seedOffset = 437
nIterations = 1000

### Generate 1/2/6-feature models

metr1F, interc1F, coef1F, selFeatures1F, mean1F, scale1F = trainModel(X, y, featureNames, \
		features2Select=["dirUp_lag_180min_pca1"], \
		seedXVal=seedOffset, seedLR=seedOffset+nIterations)
print("1F-Model:", metr1F["accuracy"], interc1F, coef1F, selFeatures1F, mean1F, scale1F)
print(metr1F)

metr2F, interc2F, coef2F, selFeatures2F, mean2F, scale2F = trainModel(X, y, featureNames, \
		features2Select=["dirUp_lag_240min_pca1","magUp_lag_120min_pca1"], \
		seedXVal=seedOffset, seedLR=seedOffset+nIterations)
print("2F-Model:", metr2F["accuracy"], interc2F, coef2F, selFeatures2F, mean2F, scale2F)
print(metr2F)

metr6F, interc6F, coef6F, selFeatures6F, mean6F, scale6F = trainModel(X, y, featureNames, \
		features2Select=["dirUp_lag_180min_pca1","dirUp_lag_240min_pca1","magUp_lag_120min_pca1","magDown_lag_180min_pca2","magDown_lag_180min_pca3","magDown_lag_120min_pca2"], \
		seedXVal=seedOffset, seedLR=seedOffset+nIterations)
print("6F-Model:", metr6F["accuracy"], interc6F, coef6F, selFeatures6F, mean6F, scale6F)
print(metr6F)



### Find average flow direction for polyps to be more likely to be extended (according to dirUp_lag_180min_pca1)

pca_dirUP_lag3h = multivariate_pca["dirUp_lag_180min_pca"]["pcaObj"]
dirXYUpFullSeriesPCA_lag_3h = pca_dirUP_lag3h.transform(dirXYUpFullSeries_lag_3h)
pca_dirUP_lag4h = multivariate_pca["dirUp_lag_240min_pca"]["pcaObj"]
dirXYUpFullSeriesPCA_lag_4h = pca_dirUP_lag4h.transform(dirXYUpFullSeries_lag_4h)
pca_magUP_lag2h = multivariate_pca["magUp_lag_120min_pca"]["pcaObj"]
magUpFullSeriesPCA_lag_2h = pca_magUP_lag2h.transform(magUpFullSeries_lag_2h)
pca_magDOWN_lag2h = multivariate_pca["magDown_lag_120min_pca"]["pcaObj"]
magDownFullSeriesPCA_lag_2h = pca_magDOWN_lag2h.transform(magDownFullSeries_lag_2h)
pca_magDOWN_lag3h = multivariate_pca["magDown_lag_180min_pca"]["pcaObj"]
magDownFullSeriesPCA_lag_3h = pca_magDOWN_lag3h.transform(magDownFullSeries_lag_3h)

assert dirXYUpFullSeriesPCA_lag_3h.shape[0] == magDownFullSeriesPCA_lag_2h.shape[0]

#print(pca_dirUP_lag3h.components_[0])
#print(pca_dirUP_lag3h.components_[0,1::2])
#print(pca_dirUP_lag3h.components_[0,::2])
dirUP_lag3_h_pca1_xy_norm = np.sqrt(pca_dirUP_lag3h.components_[0,::2]**2 + pca_dirUP_lag3h.components_[0,1::2]**2)
print("Vector means of pca_dirUP_lag3h pc1:", dirUP_lag3_h_pca1_xy_norm/dirUP_lag3_h_pca1_xy_norm[0])

dirUP_lag3_h_pca1_x = coef1F[0] * np.mean(pca_dirUP_lag3h.components_[0,::2])
dirUP_lag3_h_pca1_y = coef1F[0] * np.mean(pca_dirUP_lag3h.components_[0,1::2])
dirUP_max_open = math.atan2(dirUP_lag3_h_pca1_y, dirUP_lag3_h_pca1_x)
dirUP_max_closed = math.atan2(-1*dirUP_lag3_h_pca1_y, -1*dirUP_lag3_h_pca1_x)
print("dirUP_max_open (lag=3h) in degrees, North=0, positive angles turn cw:", 180 * dirUP_max_open/math.pi)
print("dirUP_max_closed (lag=3h) in degrees, North=0, positive angles turn cw:", 180 * dirUP_max_closed/math.pi)

#pca_dirDown_lag3h = multivariate_pca["dirDown_lag_180min_pca"]["pcaObj"]
#print("pca_dirDown_lag3h.components_[1,:]")
#print(pca_dirDown_lag3h.components_[1,:])

print("pca_dirUP_lag3h.PC1", pca_dirUP_lag3h.components_[0,:])
print("pca_magUP_lag2h.PC1", pca_magUP_lag2h.components_[0,:])

print("pca_magDOWN_lag2h.PC1", pca_magDOWN_lag2h.components_[0,:])
print("pca_magDOWN_lag2h.PC2", pca_magDOWN_lag2h.components_[1,:])
print("pca_magDOWN_lag3h.PC1", pca_magDOWN_lag3h.components_[0,:])
print("pca_magDOWN_lag3h.PC2", pca_magDOWN_lag3h.components_[1,:])
print("pca_magDOWN_lag3h.PC3", pca_magDOWN_lag3h.components_[2,:])


### Create rose diagrams of dir/mag up lag 180min

plotDirs = np.arctan2( \
						np.mean(multivariate_products["dirUp_lag_180min"]["data"][:,1::2], axis=1), \
						np.mean(multivariate_products["dirUp_lag_180min"]["data"][:,::2], axis=1))
plotDirs_open = plotDirs[y==1]
plotDirs_closed = plotDirs[y==0]

plotMags = np.mean(multivariate_products["magUp_lag_180min"]["data"], axis=1)
plotMags_open = plotMags[y==1]
plotMags_closed = plotMags[y==0]

fig = plt.figure(figsize=(11,6.4))

nHistBins = 20
maxCount = 65
plotBins, plotAvgMags, plotCounts = createBinnedData(plotDirs_open, plotMags_open, nBins=nHistBins)
#print(np.max(plotCounts)) -> set maxCount
addRosePlot(fig, "Extended", plotBins, plotAvgMags, plotCounts/maxCount, velocityLimit=500, axisID=121) #211
plotBins, plotAvgMags, plotCounts = createBinnedData(plotDirs_closed, plotMags_closed, nBins=nHistBins)
#print(np.max(plotCounts)) -> set maxCount
addRosePlot(fig, "Retracted", plotBins, plotAvgMags, plotCounts/maxCount, velocityLimit=500, axisID=122, addColorScale=True)

fig.savefig("figures/currents_up_lag_180min_rose.pdf", bbox_inches="tight")
plt.close(fig)


### Generate 1/2/6-feature model predictions

p_openessFull_1F = pd.Series(np.empty(dirXYUpFullSeries_lag_3h.values.shape[0]), index=dirXYUpFullSeries_lag_3h.index.values)
p_openessFull_1F.values[:] = 1.0/(1.0+np.exp(-1.0*interc1F - coef1F[0]*(dirXYUpFullSeriesPCA_lag_3h[:,0]-mean1F[0])/scale1F[0]))
writeCSV(pathPrefix+"p_openessFull_1F.csv", p_openessFull_1F.values, p_openessFull_1F.index.values, ["p_openessFull_1F"])

p_openessFull_2F = pd.Series(np.empty(dirXYUpFullSeries_lag_3h.values.shape[0]), index=dirXYUpFullSeries_lag_3h.index.values)
p_openessFull_2F.values[:] = 1.0/(1.0+np.exp(-1.0*interc2F - coef2F[0]*(dirXYUpFullSeriesPCA_lag_4h[:,0]-mean2F[0])/scale2F[0] \
															- coef2F[1]*(magUpFullSeriesPCA_lag_2h[:,0]-mean2F[1])/scale2F[1]))
writeCSV(pathPrefix+"p_openessFull_2F.csv", p_openessFull_2F.values, p_openessFull_2F.index.values, ["p_openessFull_2F"])

p_openessFull_6F = pd.Series(np.empty(dirXYUpFullSeries_lag_3h.values.shape[0]), index=dirXYUpFullSeries_lag_3h.index.values)
p_openessFull_6F.values[:] = 1.0/(1.0+np.exp(-1.0*interc6F - coef6F[0]*(dirXYUpFullSeriesPCA_lag_3h[:,0]-mean6F[0])/scale6F[0] \
															- coef6F[1]*(dirXYUpFullSeriesPCA_lag_4h[:,0]-mean6F[1])/scale6F[1] \
															- coef6F[2]*(magDownFullSeriesPCA_lag_2h[:,1]-mean6F[2])/scale6F[2] \
															- coef6F[3]*(magDownFullSeriesPCA_lag_3h[:,1]-mean6F[3])/scale6F[3] \
															- coef6F[4]*(magDownFullSeriesPCA_lag_3h[:,2]-mean6F[4])/scale6F[4] \
															- coef6F[5]*(magUpFullSeriesPCA_lag_2h[:,0]-mean6F[5])/scale6F[5]))
writeCSV(pathPrefix+"p_openessFull_6F.csv", p_openessFull_6F.values, p_openessFull_6F.index.values, ["p_openessFull_6F"])


### Plot observations vs. 1/2/6-feature model predictions

fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

axStart = np.datetime64("2012-06-04T23:59:59Z")
oneAxDuration = np.timedelta64(3*24, "h")
for i,ax in enumerate([ax1,ax2,ax3,ax4]):
	ax.plot(openess.index, openess.values, label = "Observations", color="b", marker="o", markeredgewidth=1.25, linewidth=1.25)
	ax.plot(p_openessFull_1F.index, p_openessFull_1F.values, label = "1 Feature", color="r", linewidth=1.25)
	ax.plot(p_openessFull_2F.index, p_openessFull_2F.values, label = "2 Features", color="c", linewidth=1.25)
	ax.plot(p_openessFull_6F.index, p_openessFull_6F.values, label = "6 Features", color="g", linewidth=1.25)
	ax.plot(openess.index, np.full(openess.index.shape[0], 0.5), label = "Decision Boundary", color="k", marker="o", fillstyle="none", markeredgewidth=1.25, linewidth=1.25)
	ax.set_xlim([axStart+i*oneAxDuration,axStart+(i+1)*oneAxDuration])
	ax.set_ylim([-0.1,1.1])
	ax.set_ylabel(r"$p$ Extension")
	ax.xaxis.set_major_locator(mdates.DayLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
	ax.xaxis.set_minor_locator(mdates.HourLocator())

ax1.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, ncol=5, mode="expand", borderaxespad=0.0)
plt.xlabel("Date")
fig.savefig("figures/predictions.pdf", bbox_inches="tight")
plt.close(fig)


### Create models with 1,...,n features and average their properties (Monte Carlo cross-validation)

accuracies = []
likelihoods = []
#nFeaturesToTry = [X.shape[1]+1]
#nFeaturesToTry = [1, X.shape[1]+1]
nFeaturesToTry = range(1, X.shape[1]+1)  
for nFeatures in nFeaturesToTry:
	avgAccuracy = 0.0
	avgLikelihood = 0.0
	featureTable = np.zeros((featureNames.shape[0], 4))
	for i in range(nIterations):
		metr, interc, coef, selFeatures, ignoreMean, ignoreScale = trainModel(X, y, featureNames, nFeatures=nFeatures, seedXVal=seedOffset+i, seedLR=seedOffset+nIterations+i)
		acc = metr["accuracy"]
		likel = metr["likelihood"]
		avgAccuracy += acc/nIterations
		avgLikelihood += likel/nIterations
		for j, feature in enumerate(selFeatures):
			fIndex = np.where(featureNames == feature)[0][0]
			featureTable[fIndex,0] += 1
			featureTable[fIndex,1] += interc
			featureTable[fIndex,2] += coef[j]
			featureTable[fIndex,3] += math.fabs(coef[j])
	accuracies.append(avgAccuracy)
	likelihoods.append(avgLikelihood)
	featureTable[:,1] /= featureTable[:,0]
	featureTable[:,2] /= featureTable[:,0]
	featureTable[:,3] /= featureTable[:,0]
	print("Feature selection table for {} features and {} iterations:".format(nFeatures, nIterations))
	print("Feature | % Chosen | Avg. Intercept | Avg. Coef. | Avg. Abs. Coef. | Odd Increase p. Unit Feature De-/Increase")
	for i, feature in enumerate(featureNames):
		print("{:<32}{:>8.1%}{:>10.5F}{:>10.5F}{:>10.5F}{:>+8.1F}%".format(feature, featureTable[i][0]/nIterations, featureTable[i][1], featureTable[i][2], featureTable[i][3], 100*math.exp(math.fabs(featureTable[i][2]))-100))
print("nFeatures | Avg. Accuracy | Avg. Likelihood | Log Likelihood | Likelihood Ratio")
for index, (acc, likel) in enumerate(zip(accuracies, likelihoods)):
	prevLikel = likelihoods[index-1] if index>0 else likel
	print("{:>4}{:>12.3%}{:>14.5e}{:>14.6F}{:>14.6F}".format(nFeaturesToTry[index], acc, likel, math.log(likel), 2.0*(math.log(likel)-math.log(prevLikel))))


### Average properties of 1-feature models for every feature

print("All possible 1D Model with {} iterations each:".format(nIterations))
print("Feature | Avg. Accuracy | Avg. Intercept | Avg. Coef. | Avg. Abs. Coef. | Avg. Odd Increase")
for f in featureNames:
	avgAccuracy = 0.0
	avgIntercept = 0.0
	avgCoef = 0.0
	avgAbsCoef = 0.0
	for i in range(nIterations):
		metr, interc, coef, selFeatures, ignoreMean, ignoreScale = trainModel(X, y, featureNames, features2Select=[f], seedXVal=seedOffset+i, seedLR=seedOffset+nIterations+i)
		acc = metr["accuracy"]
		avgAccuracy += acc/nIterations
		avgIntercept += interc/nIterations
		avgCoef += coef[0]/nIterations
		avgAbsCoef += math.fabs(coef[0])/nIterations
	print("{:<32}{:>10.3%}{:>10.5F}{:>10.5F}{:>10.5F}{:>+8.1F}%".format(f, avgAccuracy, avgIntercept, avgCoef, avgAbsCoef, 100*math.exp(math.fabs(avgCoef))-100))

