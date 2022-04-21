using DataFrames
using Statistics
using ScikitLearn
using CSV
using Plots
# Notes
# Function with leverage
#  80-20 training test 5-fold 
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=400, test_size=200, random_state=4)
# CrossValidation
########################
# Importing the data
gr = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norm_GR.csv",DataFrame)
am = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norm_Amide.csv", DataFrame)


############
# Regression

RTI_col =
First_desc_col =
