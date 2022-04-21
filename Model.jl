using DataFrames
using Statistics
using ScikitLearn
using CSV
using Plots
using LinearAlgebra
using PyCall

import StatsPlots as sp

using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
@sk_import ensemble: RandomForestRegressor
#################################
# Notes
# Function with leverage
#  80-20 training test split , 5-fold CrossValidation
# Minimum number of samples per leaf and number of trees => Optimization
# Max_features_parameter = Sqrt(#descriptors)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=400, test_size=200, random_state=4)
#
#
#################################
## Importing the data

# Change data from here (gre or am)
am = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norm_Amide.csv", DataFrame)
# GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norm_GR.csv", DataFrame)
data = am
data_name = "Amide"

RTI = data[:,2]
desc = Matrix(data[:,8:end])

#################################
sp.scatter(RTI, desc, normalized=true, label=false)
xlims!(-1000,1000)
ylims!(-5000,5000)


#################################
## Regression

MaxFeat = Int(round(sqrt(size(desc,2))))
reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=7,verbose =1, oob_score =true, max_features= MaxFeat)
## QUESTION: WHAT IS VERBOSITY?


X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
fit!(reg, X_train, y_train)
accuracy = score(reg, X_train, y_train)
n_folds=5
cross_val_score(reg, X_train, y_train; cv=n_folds)
