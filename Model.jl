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

# Change data from here (GR or AM)
AM = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Amide.csv", DataFrame)
GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Greek.csv", DataFrame)
norm_GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norman_(Greek model).csv", DataFrame)
norm_AM = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norman_(Amide model).csv", DataFrame)

data = AM
data_name = "Amide"

#= For the Greek dataset
retention = data[!,[:2]]
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_$(data_name).csv", retention)
retention_cor = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_$(data_name).csv", DataFrame, decimal = ',')

RTI = retention_cor[:,1]
desc = Matrix(data[:,4:end])
=#
# For the Amide dataset
        RTI = data[:,2]
        desc = Matrix(data[:,6:end])           # Careful! Matrix should have 1170 descriptors
#
#################################
# Experimental RTI Plotting
RTI1 = sort(RTI)
sp.histogram(RTI, bins=60, label=false, xaxis = "Experimental RTI", yaxis = "Frequency", title = "RTI distribution for the $(data_name) dataset")
#sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RTI_distribution_$data_name.png")
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
n_estimators = 500          #500 for the Greek, 400 for the Amide
min_samples_leaf = 4
reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features= MaxFeat, n_jobs=-1, oob_score =true, random_state=21)
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
important_desc = names(data[:,6:end])[significant_columns]     # For Amide
#important_desc = names(data[:,4:end])[significant_columns]     # For Greek

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

# For the greek dataset we need at least 3 descriptors - let's save the first 5
# For the amide dataset we need at least 13 descriptors - let's save them

selection = important_desc[1:5]
using BSON
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Descriptor_names_partial_model_$data_name", selection)


## Model - Selected descriptors
using BSON
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Descriptor_names_partial_model_$data_name", selection)

desc_temp = Matrix(select(data, selection))
MaxFeat = Int64(ceil(size(desc_temp,2)/3))
reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=21)
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

## Distribution (1st try)
# Lowest point
sort(y_hat_test)[1]
sort(y_hat_train)[1]

lowest = sortperm(y_hat_test)[1]
y_hat_test[lowest]

X_low = zeros(5000,length(X_test[lowest,:]))
for i = 1:size(X_low,1)
    change = BS.sample(1:length(X_test[lowest,:]))
    for j = 1:length(X_test[lowest,:])
        if j == change
            small_change = BS.sample(-0.3:0.001:0.3)
            X_low[i,j] = X_test[lowest,j] + small_change
        else
            X_low[i,j] = X_test[lowest,j]
        end
    end
end

y_hat_low = predict(reg,X_low)
y_hat_lowest = sort(y_hat_low)[1]
histogram(y_hat_low, label=false, yaxis = "Frequency",xaxis = "Predicted RTI",title = "Lowest point - Distribution")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Lowest_point_distribution_$data_name.png")

#= Distribution (2nd try) -Very inconsistent results
    # Lowest point
sort(y_hat_test)[1]
sort(y_hat_train)[1]

lowest = sortperm(y_hat_train)[1]
y_hat_train[lowest]

X_low = zeros(50000,length(X_train[lowest,:]))
X_low[1,:] = X_train[lowest,:]
for i = 2:size(X_low,1)
    change = BS.sample(1:length(X_train[lowest,:]))
    for j = 1:length(X_train[lowest,:])
        if j == change
            percentage = BS.sample(-0.01:0.0001:0.01)
            X_low[i,j] = X_low[i-1,j] + percentage
        else
            X_low[i,j] = X_low[i-1,j]
        end
    end
end

y_hat_low = predict(reg,X_low)
histogram(y_hat_low, label=false, yaxis = "Frequency",xaxis = "Predicted RTI",title = "Lowest point - Distribution")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Lowest_point_distribution_$data_name.png")
=#

# Highest point
sort(y_hat_test)[end]
sort(y_hat_train)[end]

highest = sortperm(y_hat_train)[end]
y_hat_train[highest]

X_high = zeros(5000,length(X_train[highest,:]))
for i = 1:size(X_high,1)
    change = BS.sample(1:length(X_train[highest,:]))
    for j = 1:length(X_train[highest,:])
        if j == change
            percentage = BS.sample(-0.25:0.01:0.25)
            if (X_train[highest,j] + percentage) < 1
                X_high[i,j] = X_train[highest,j] + percentage
            else
                X_high[i,j] = X_train[highest,j]
            end
        else
            X_high[i,j] = X_train[highest,j]
        end
    end
end

y_hat_high = predict(reg,X_high)
histogram(y_hat_high, label=false, yaxis = "Frequency",xaxis = "Predicted RTI",title = "Highest point - Distribution")
y_hat_highest = sort(y_hat_high,rev=true)[1]
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Highest_point_distribution_$data_name.png")





## Norman RI prediction
reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=21)
X_train, X_test, y_train, y_test = train_test_split(desc_temp, RTI, test_size=0.20, random_state=21)
fit!(reg, X_train, y_train)

norm_GR_desc = Matrix(select(norm_GR, selection))
RI_norman_GR = predict(reg, norm_GR_desc)

sp.histogram(RI_norman_GR, label = false, bins=300)
sp.xlabel!("Predicted RTI")
sp.ylabel!("Frequency")
sp.title!("RTIs of the Norman dataset - $data_name")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman_prediction_$data_name.png")

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
## Question: The Applicability Domain should be with all the descriptors?
# The model doesn't contain them all, so why does it make sense?
#
function leverage_dist(X_train, Norman)
    lev = zeros(size(Norman,1))
    z = pinv(transpose(X_train) * X_train)
    for ind = 1:size(Norman,1)
        x = Norman[ind,:]
        lev[ind] = transpose(x) * z * x
        println(ind)
    end
    return lev
end
lev_norman_am = leverage_dist(X_train_am, Matrix(norman_am))
lev_norman_gr = leverage_dist(X_train_gr, Matrix(norman_gr))

df = DataFrame(lev_norman_am = lev_norman_am, lev_norman_gr = lev_norman_gr)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_Norman.csv",df)

## Loading the leverage
df = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_Norman.csv",DataFrame)
lev = Matrix(df)

h_star_am = (3*(1170+1))/1190
h_star_gr = (3*(1170+1))/1452

# Norman - Amide dataset
lev_am = lev[:,1]
assessment_am = convert.(Int64,zeros(length(lev_am)))

for i = 1:length(assessment_am)
    if lev_am[i] <= h_star_am
        assessment_am[i] = 1
    elseif lev_am[i] <= 3*h_star_am
        assessment_am[i] = 2
    else
        assessment_am[i] = 3
    end
end

histogram(lev[:,1], bins=800000, label = false, title="Applicability Domain for the Amide dataset", xaxis="Leverage", xlims = (0,100))
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_histogram_Norman_Amide.png")

histogram(assessment_am, label=false, bins =4, title = "Applicability Domain for the Amide dataset")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_Norman_Amide.png")

assessment_am_1 = assessment_am[assessment_am.==1] # 11.7k out of 95k are ok
assessment_am_2 = assessment_am[assessment_am.==2] # 8.5k out of 95k are meh
assessment_am_3 = assessment_am[assessment_am.==3] # 74.9k out of 95k are NOT ok

# Norman - Greek dataset
lev_gr = lev[:,2]
assessment_gr = convert.(Int64,zeros(length(lev_gr)))

for i = 1:length(assessment_gr)
    if lev_gr[i] <= h_star_gr
        assessment_gr[i] = 1
    elseif lev_gr[i] <= 3*h_star_gr
        assessment_gr[i] = 2
    else
        assessment_gr[i] = 3
    end
end

histogram(lev[:,2], bins=800000, label = false, title="Applicability Domain for the Greek dataset", xaxis="Leverage", xlims = (0,100))
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_histogram_Norman_Greek.png")

histogram(assessment_gr, label=false, bins =4, title = "Applicability Domain for the Greek dataset")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_Norman_Greek.png")

assessment_gr_1 = assessment_gr[assessment_gr.==1] # 26.8k out of 95k are ok
assessment_gr_2 = assessment_gr[assessment_gr.==2] # 27k out of 95k are meh
assessment_gr_3 = assessment_gr[assessment_gr.==3] # 41.3k out of 95k are NOT ok

using BSON
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_amide", assessment_am)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_greek", assessment_gr)



## PCA
@sk_import decomposition: PCA  # We want to run a PCA

#Setup PCA model

# For Amide (all training set) and Norman (all descriptors)
RTI_am = AM[:,2]
desc_am = Matrix(AM[:,6:end])           # Careful! Matrix should have 1170 descriptors

X_train_am, X_test, y_train, y_test = train_test_split(desc_am, RTI_am, test_size=0.20, random_state=21)
norm_am_desc = Matrix(norm_AM[!,3:end])
norm_am = vcat(X_train_am,norm_am_desc)

pca = PCA(n_components = size(norm_am,2))
pca.fit(norm_am)
scatter(1:100, cumsum(pca.explained_variance_ratio_), label = "SVD", ylims=(0,1), legend=:topleft)

pca = PCA(n_components = 2)
pca.fit(norm_am)
loadings_am = pca.components_
scores_am = pca.fit_transform(norm_am)

using BSON
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_amide", judgement_am)

scatter((scores_am[1191:end,1])[judgement_am.==3], (scores_am[1191:end,2])[judgement_am.==3], legend=:topleft,label="Outside",color=:pink,xlabel="PC1",ylabel="PC2")
scatter!((scores_am[1191:end,1])[judgement_am.==2], (scores_am[1191:end,2])[judgement_am.==2], label = "Indecisive", color = :yellow, xlabel = "PC1", ylabel = "PC2")
scatter!((scores_am[1191:end,1])[judgement_am.==1], (scores_am[1191:end,2])[judgement_am.==1], label = "Inside", color = :green, xlabel = "PC1", ylabel = "PC2")
scatter!(scores_am[1:1190,1], scores_am[1:1190,2], label = "Training set", color = :blue, xlabel = "PC1", ylabel = "PC2")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\PCA_Norman_Amide.png")

# For Greek (all training set) and Norman (all descriptors)

CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Greek.csv", GR[!,[:2]])
retention_cor = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Greek.csv", DataFrame, decimal = ',')
RTI_gr = retention_cor[:,1]
desc_gr = Matrix(GR[:,4:end])


X_train_gr, X_test, y_train, y_test = train_test_split(desc_gr, RTI_gr, test_size=0.20, random_state=21)
norm_gr_desc = Matrix(norm_GR[!,3:end])
norm_gr = vcat(X_train_gr,norm_gr_desc)

pca = PCA(n_components = size(norm_gr,2))
pca.fit(norm_gr)
scatter(1:100, cumsum(pca.explained_variance_ratio_), label = "SVD", ylims=(0,1), legend=:topleft)

pca = PCA(n_components = 2)
pca.fit(norm_gr)
loadings_gr = pca.components_
scores_gr = pca.fit_transform(norm_gr)

using BSON
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_greek", judgement_gr)

scatter((scores_gr[1453:end,1])[judgement_gr.==3], (scores_gr[1453:end,2])[judgement_gr.==3], legend=:topleft,label="Outside",color=:pink,xlabel="PC1",ylabel="PC2")
scatter!((scores_gr[1453:end,1])[judgement_gr.==2], (scores_gr[1453:end,2])[judgement_gr.==2], label = "Indecisive", color = :yellow, xlabel = "PC1", ylabel = "PC2")
scatter!((scores_gr[1453:end,1])[judgement_gr.==1], (scores_gr[1453:end,2])[judgement_gr.==1], label = "Inside", color = :green, xlabel = "PC1", ylabel = "PC2")
scatter!(scores_gr[1:1452,1], scores_gr[1:1452,2], label = "Training set", color = :blue, xlabel = "PC1", ylabel = "PC2")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\PCA_Norman_Greek.png")
