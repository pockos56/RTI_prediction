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
am = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norm_Amide.csv", DataFrame)
GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norm_GR.csv", DataFrame)
data = GR
data_name = "Greek"
show(data)
# For the Greek dataset
 s = data[!,[:2]]
 CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_$(data_name).csv", s)
 s_cor = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_$(data_name).csv", DataFrame, decimal = ',')
 RTI = s_cor[:,1]
 desc = Matrix(data[:,5:end])

# For the Amide dataset
# RTI = data[:,2]
# desc = Matrix(data[:,8:end])

#################################
# Experimental RTI Plotting
RTI1 = sort(RTI)
sp.histogram(RTI, bins=60, label=false, xaxis = "Experimental RTI", yaxis = "Frequency", title = "RTI distribution for the $(data_name) dataset")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RTI_distribution_$data_name.png")

logp = data.XLogP[:]
mass = data.MW[:]
sp.scatter(mass,logp,legend=false,title="LogP vs Mass for the $(data_name) dataset",xaxis="Mass",yaxis="LogP")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\LogP-Mass_$data_name.png")

#################################
## Regression
MaxFeat = Int(round((size(desc,2))/3))
reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=6, oob_score =true, max_features= MaxFeat)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
fit!(reg, X_train, y_train)

## Parameter Optimization

# Optimization of samples per leaf

cross_val_accuracies = zeros(16,5)
for sample_per_leaf = 2:2:16
        reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=sample_per_leaf, oob_score =true, max_features= MaxFeat)
        X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
        fit!(reg, X_train, y_train)
        cross_val_accuracies[sample_per_leaf,:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(sample_per_leaf)
end

cross_val_accuracies_mean = mean(cross_val_accuracies, dims=2)

sp.scatter(cross_val_accuracies_mean, legend=false, xaxis = "Sample per leaf", yaxis = "R2", title = "Cross validation score for different sample per leaf")
sp.ylims!(0.74,0.78)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Sample_per_leaf_$data_name.png")

# Optimization of number of trees

cross_val_accuracies2 = zeros(1200,5)
for number_of_trees = 100:100:1200
        reg = RandomForestRegressor(n_estimators=number_of_trees, min_samples_leaf=6, oob_score =true, max_features= MaxFeat, n_jobs=-1)
        X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
        fit!(reg, X_train, y_train)
        cross_val_accuracies2[number_of_trees,:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(number_of_trees)
end

cross_val_accuracies_mean2 = mean(cross_val_accuracies2, dims=2)

sp.scatter(cross_val_accuracies_mean2, legend=false, xaxis="number of trees",yaxis="R2",title="Cross validation score for different number of trees")
sp.ylims!(0.75,0.77)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Number_of_trees_$data_name.png")

# Optimization of max number of features

cross_val_accuracies3 = zeros(1000,5)
for MaxFeat = [15,46,90,200,500,699,1000]
        reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=6, oob_score =true, max_features= MaxFeat, n_jobs=-1)
        X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21)
        fit!(reg, X_train, y_train)
        cross_val_accuracies3[MaxFeat,:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(MaxFeat)
end
cross_val_accuracies_mean3 = mean(cross_val_accuracies3, dims=2)

sp.scatter(cross_val_accuracies_mean3, legend=false,xaxis="Max Features",yaxis="R2",title="Cross validation score for different max features")
sp.ylims!(0.7,0.78)

sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Max_features_$data_name.png")

## Regression with optimal parameters

MaxFeat = Int(round((size(desc,2))/3))
n_estimators = 300
min_samples_leaf = 4
reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features= MaxFeat, n_jobs=-1, oob_score =true)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
fit!(reg, X_train, y_train)

## Calculation of R2
accuracy = score(reg, X_train, y_train)

## Prediction of RTIs using the model
y_hat_test = predict(reg,X_test)
y_hat_train = predict(reg,X_train)

## Cross Validation
CV = cross_val_score(reg, X_train, y_train, cv=5)
CV_mean = mean(CV)

## Finding the most important descriptors
importance = 100 .* sort(reg.feature_importances_, rev=true)
importance_index = sortperm(reg.feature_importances_, rev=true)
significant_columns = importance_index[importance .>=1.5]
important_desc = names(data[:,8:end])[significant_columns]

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

##Remarks:
# Increasing max_features shows a better regression
# min_samples_leaf should stay as low as 2
# number of trees has no significant benefit after a certain number (eg.100)

## Leverage (from ToxPredict)
# Why is it like this?

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
