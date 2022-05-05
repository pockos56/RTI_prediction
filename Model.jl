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
# Experimental RTI Plotting
RTI1 = sort(RTI)
sp.histogram(RTI,bins=70, label=false)

#################################
## Regression
MaxFeat = Int(round(sqrt(size(desc,2))))
reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=6, oob_score =true, max_features= MaxFeat)
X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
fit!(reg, X_train, y_train)

## Parameter Optimization
accuracies = zeros(8,12,10)
for sample_per_leaf = 4:4:16
    for no_trees = 100:100:1200 #:200:900
        reg = RandomForestRegressor(n_estimators=no_trees, min_samples_leaf=sample_per_leaf, oob_score =true, max_features= MaxFeat)
        X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
        fit!(reg, X_train, y_train)
        accuracy = score(reg, X_train, y_train)
        accuracies[Int((sample_per_leaf)/4),Int(no_trees/100)] = accuracy
        println("The obtained accuracy [$(no_trees) trees, $(sample_per_leaf) samples/leaf and $(MaxFeat) max features] is $accuracy")
    end
end
heatmap(accuracies)
findmax(accuracies)
surface(accuracies)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Accuracies.png")

# Optimization of samples per leaf
cross_val_accuracies = zeros(8,5)
for sample_per_leaf = 2:2:16
        reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=sample_per_leaf, oob_score =true, max_features= 46)
        X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
        fit!(reg, X_train, y_train)
        cross_val_accuracies[Int((sample_per_leaf)/2),:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(sample_per_leaf)
end

sp.scatter(cross_val_accuracies, label = ["2" "4" "6" "8" "10" "12" "14" "16"], legend=false)
sp.xaxis!("(Sample per leaf)/2")
sp.yaxis!("R2")
sp.title!("Cross validation score for different sample per leaf")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Sample_per_leaf_$data_name.png")

# Optimization of number of trees
cross_val_accuracies2 = zeros(12,5)
for number_of_trees = 100:100:1200
        reg = RandomForestRegressor(n_estimators=number_of_trees, min_samples_leaf=6, oob_score =true, max_features= 46)
        X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
        fit!(reg, X_train, y_train)
        cross_val_accuracies2[Int((number_of_trees)/100),:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(number_of_trees)
end
cross_val_accuracies_mean = mean(cross_val_accuracies2, dims=2)
sp.scatter(cross_val_accuracies_mean, legend=false)
sp.xaxis!("(number of trees)/100")
sp.yaxis!("R2")
sp.title!("Cross validation score for different number of trees")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Number_of_trees_$data_name.png")

# Optimization of max number of features
cross_val_accuracies3 = zeros(56,5)
for MaxFeat = 36:2:56
        reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=6, oob_score =true, max_features= MaxFeat)
        X_train, X_test, y_train, y_test = train_test_split(desc, RTI, test_size=0.20, random_state=21);
        fit!(reg, X_train, y_train)
        cross_val_accuracies3[MaxFeat,:] = cross_val_score(reg, X_train, y_train, cv=5)
        println(MaxFeat)
end
cross_val_accuracies_mean = mean(cross_val_accuracies3, dims=2)
sp.scatter(cross_val_accuracies_mean, legend=false)
sp.ylims!(0.7,0.715)
sp.xlims!(35,57)
sp.xaxis!("Max Features")
sp.yaxis!("R2")
sp.title!("Cross validation score for different max features")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Max_features_$data_name.png")

## Calculation of R2
accuracy = score(reg, X_train, y_train)

## Prediction of RTIs using the model
y_hat_test = predict(reg,X_test)
y_hat_train = predict(reg,X_train)

## Cross Validation
cross_val_score(reg, X_train, y_train, cv=5)

## Finding the most important descriptors
importance = 100 .* sort(reg.feature_importances_, rev=true)
importance_index = sortperm(reg.feature_importances_, rev=true)
significant_columns = importance_index[importance .>=1.5]
important_desc = names(data[:,8:end])[significant_columns]

# 1. Crippen's LogP
# 2. Mannhold LogP
# 3. XLogP
# 4. Number of carbon atoms
# 5. Molar refractivity
# 6. Average Broto-Moreau autocorrelation - lag 1 / weighted by first ionization potential
# 7. Smallest absolute eigenvalue of Burden modified matrix - n 2 / weighted by relative Sanderson electronegativities
# 8. Crippen's molar refractivity

sp.scatter(y_train,y_hat_train,label="Training set",bins = 50,legend=:topleft)
sp.scatter!(y_test,y_hat_test,bins = 50,label="Test set",legend=:topleft)
sp.plot!([0,1000],[0,1000],label="1:1 line",linecolor ="black")
sp.xlabel!("Measured RTI")
sp.ylabel!("Predicted RTI")
sp.title!("RTI regression all descriptors")

##Remarks:
# Increasing max_features shows a better regression
# min_samples_leaf should stay as low as 2
# number of trees has no significant benefit after a certain number (eg.100)

## Leverage (from ToxPredict)

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
histogram(lev, bins=30, label = false)
boxplot(lev, label = false)
