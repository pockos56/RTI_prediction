using DataFrames
using Statistics
using ScikitLearn
using CSV
using Plots
using LinearAlgebra
using PyCall

import StatsBase as BS
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
# Empirical good default values are max_features=n_features/3 for regression
# problems, and max_features=sqrt(n_features) for classification tasks
#
#################################
## Importing the data

# Change data from here (gre or am)
AM = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Amide.csv", DataFrame)
GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Greek.csv", DataFrame)
norm_GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norman_(Greek model).csv", DataFrame)
norm_AM = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norman_(Amide model).csv", DataFrame)

data = GR
data_name = "Greek"

#= For the Greek dataset
retention = data[!,[:2]]
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_$(data_name).csv", retention)
retention_cor = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_$(data_name).csv", DataFrame, decimal = ',')

RTI = retention_cor[:,1]
desc = Matrix(data[:,4:end])
=#
# For the Amide dataset
        RTI = data[:,2]
        desc = Matrix(data[:,6:end])
#
#################################
# Experimental RTI Plotting
RTI1 = sort(RTI)
sp.histogram(RTI, bins=60, label=false, xaxis = "Experimental RTI", yaxis = "Frequency", title = "RTI distribution for the $(data_name) dataset")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RTI_distribution_$data_name.png")
#=     ###              PROBLEM -> NOT ENOUGH descriptors that contain MW and logP       ###
logp = data.MLogP[:]
mass = data.MW[:]
sp.scatter(mass,logp,legend=false,title="LogP vs Mass for the $(data_name) dataset",xaxis="MW",yaxis="LogP")
#sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\LogP-Mass_$data_name.png")
##     ###                                                                              ### =#
#################################
## Regression
MaxFeat = Int(round((size(desc,2))/3))
reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=6, oob_score =true, max_features= MaxFeat)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
fit!(reg, X_train, y_train)

#=# Parameter Optimization

# Optimization of samples per leaf

cross_val_accuracies = zeros(16,5)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
for sample_per_leaf = 2:2:16
        reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=sample_per_leaf, oob_score =true, max_features= MaxFeat, n_jobs=-1)
        fit!(reg, X_train, y_train)
        cross_val_accuracies[sample_per_leaf,:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(sample_per_leaf)
end

cross_val_accuracies_mean = mean(cross_val_accuracies, dims=2)

sp.scatter(cross_val_accuracies_mean, legend=false, xaxis = "Sample per leaf", yaxis = "R2", title = "Cross validation score for different sample per leaf")
sp.ylims!(0.77,0.83)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Sample_per_leaf_$data_name.png")

# Optimization of number of trees

cross_val_accuracies2 = zeros(1200,5)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
for number_of_trees = vcat(50, collect(100:100:1200))
        reg = RandomForestRegressor(n_estimators=number_of_trees, min_samples_leaf=4, oob_score =true, max_features= MaxFeat, n_jobs=-1)
        fit!(reg, X_train, y_train)
        cross_val_accuracies2[number_of_trees,:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(number_of_trees)
end

cross_val_accuracies_mean2 = mean(cross_val_accuracies2, dims=2)

sp.scatter(cross_val_accuracies_mean2, legend=false, xaxis="number of trees",yaxis="R2",title="Cross validation score for different number of trees")
sp.ylims!(0.8,0.82)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Number_of_trees_$data_name.png")

# Optimization of max number of features

cross_val_accuracies3 = zeros(1000,5)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21)
for MaxFeat = [15,46,90,200,390,600,1000]
        reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, oob_score =true, max_features= MaxFeat, n_jobs=-1)
        fit!(reg, X_train, y_train)
        cross_val_accuracies3[MaxFeat,:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(MaxFeat)
end
cross_val_accuracies_mean3 = mean(cross_val_accuracies3, dims=2)

sp.scatter(cross_val_accuracies_mean3, legend=false,xaxis="Max Features",yaxis="R2",title="Cross validation score for different max features")
sp.ylims!(0.67,0.825)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Max_features_$data_name.png")
=#
## Regression with optimal parameters
###

MaxFeat = Int(round((size(desc,2))/3))
n_estimators = 400
min_samples_leaf = 4
reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features= MaxFeat, n_jobs=-1, oob_score =true)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
fit!(reg, X_train, y_train)

## Calculation of R2
accuracy_train = score(reg, X_train, y_train)

accuracy_test = score(reg, X_test, y_test)

## Prediction of RTIs using the model
y_hat_test = predict(reg,X_test)
y_hat_train = predict(reg,X_train)

## Cross Validation
CV = cross_val_score(reg, X_train, y_train, cv=5)
CV_mean = mean(CV)

## Finding the most important descriptors
importance = 100 .* sort(reg.feature_importances_, rev=true)
importance_index = sortperm(reg.feature_importances_, rev=true)
significant_columns = importance_index[importance .>=0.1]
important_desc = names(data[:,6:end])[significant_columns]

# To be updated...
# 1. Crippen's LogP
# 2. Mannhold LogP
# 3. XLogP
# 4. Number of carbon atoms
# 5. Molar refractivity
# 6. Average Broto-Moreau autocorrelation - lag 1 / weighted by first ionization potential
# 7. Smallest absolute eigenvalue of Burden modified matrix - n 2 / weighted by relative Sanderson electronegativities
# 8. Crippen's molar refractivity

sp.scatter(y_train,y_hat_train,label="Training set", legend=:topleft, color = :magenta)
sp.scatter!(y_test,y_hat_test,label="Test set",legend=:topleft, color=:orange)
sp.plot!([0,1000],[0,1000],label="1:1 line",linecolor ="black",width=2)
sp.xlabel!("Experimental RTI")
sp.ylabel!("Predicted RTI")
sp.title!("RTI regression all descriptors")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Regression_$data_name.png")


## Finding the most significant columns
CV_sign = zeros(22,2)
k = 2

reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=390, n_jobs=-1, oob_score =true)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
fit!(reg, X_train, y_train)

importance = 100 .* sort(reg.feature_importances_, rev=true)
importance_index = sortperm(reg.feature_importances_, rev=true)
important_desc = names(data[:,6:end])[importance_index]

for i = 10:15            #vcat(collect(2:15), collect(20:5:50))
    CV_sign[k,1] = i

    selection = important_desc[1:i]
    desc_temp = Matrix(select(data, selection))
    MaxFeat = Int64(ceil(i/3))
    reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=MaxFeat, n_jobs=-1, oob_score =true)
    X_train, X_test, y_train, y_test = train_test_split(desc_temp, RTI, test_size=0.20, random_state=21)
    fit!(reg, X_train, y_train)

    CV = cross_val_score(reg, X_train, y_train, cv=5)
    CV_sign[k,2] = mean(CV)

    k = k + 1
    println(i)
end

sp.scatter(CV_sign[:,1],CV_sign[:,2], legend = false, ylims = (0.6,0.76))
sp.xlabel!("Number of descriptors used")
sp.ylabel!("CV score")
sp.title!("CV Score vs. no. of descriptors")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Partial_Model_CV_Score_$data_name.png")

# For the greek dataset we need at least 3 descriptors - let's save them
# For the amide dataset we need at least 13 descriptors - let's save them

selection = important_desc[1:13]
using BSON
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Descriptor_names_partial_model_$data_name", selection)


## Model - Selected descriptors
desc_temp = Matrix(select(data, important_desc[1:13]))
MaxFeat = Int64(ceil(size(desc_temp,2)/3))
reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=MaxFeat, n_jobs=-1, oob_score =true)
X_train, X_test, y_train, y_test = train_test_split(desc_temp, RTI, test_size=0.20, random_state=21)
fit!(reg, X_train, y_train)

## Calculation of R2 - Selected descriptors
accuracy_train = score(reg, X_train, y_train)
accuracy_test = score(reg, X_test, y_test)

## Prediction of RTIs using the model - Selected descriptors
y_hat_test = predict(reg,X_test)
y_hat_train = predict(reg,X_train)

## Cross Validation - Selected descriptors
CV = cross_val_score(reg, X_train, y_train, cv=5)
CV_mean = mean(CV)

## Plots
sp.scatter(y_train,y_hat_train,label="Training set", legend=:topleft, color = :magenta)
sp.scatter!(y_test,y_hat_test,label="Test set",legend=:topleft, color=:orange)
sp.plot!([0,1000],[0,1000],label="1:1 line",linecolor ="black",width=2)
sp.xlabel!("Experimental RTI")
sp.ylabel!("Predicted RTI")
sp.title!("RTI regression $(size(desc_temp,2)) descriptors")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Regression_Partial_Model_$data_name.png")


##Remarks:
# Increasing max_features shows a better regression
# min_samples_leaf should stay as low as 2
# number of trees has no significant benefit after a certain number (eg.100)

## Leverage (from ToxPredict)
#= Why is it like this?

function leverage_dist(X_train,itr)

    lev = zeros(itr)

    for i =1:itr
        ind = BS.sample(1:size(X_train,1))
        x = X_train[ind,:]
        lev[i] = transpose(x) * pinv(transpose(X_train) * X_train) * x
        println(i)
    end

    return lev

end

itr = 2
lev = leverage_dist(X_train,itr)
=#

## My Leverage function

function leverage_dist(X_train)
    lev = zeros(size(X_train,1))
    for ind =1:size(X_train,1)
        x = X_train[ind,:]
        lev[ind] = transpose(x) * pinv(transpose(X_train) * X_train) * x
        println(ind)
    end
    return lev
end

lev = leverage_dist(X_train)
df = DataFrame(lev=lev)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_$(data_name).csv",df)

df = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_$(data_name).csv",DataFrame)
lev = Matrix(df)

histogram(lev, bins=35, label = false, title="Applicability Domain for the Amide dataset", xaxis="Leverage")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_histogram_$data_name.png")
