using DataFrames
using Statistics
using ScikitLearn
using CSV
using Plots
using LinearAlgebra
using PyCall
using BSON
using Distributions

import StatsBase as BS
import StatsPlots as sp
import PyPlot as plt

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
norm_GR[40271,2]
data = AM
data_name = "n-alkylamide"

#=For the Greek dataset
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
histogram(RTI, bins=90, label=false, xaxis = "Experimental RI", yaxis = "Frequency", title = "RI distribution for the $(data_name) dataset")
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
reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=6, oob_score =true, max_features= MaxFeat)
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
important_desc = names(data[:,6:end])[significant_columns][13]     # For Amide
#important_desc = names(data[:,4:end])[significant_columns]     # For Greek

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

BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Descriptor_names_partial_model_$data_name", selection)


## Model - Selected descriptors
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Descriptor_names_partial_model_$data_name", selection)

desc_temp = Matrix(select(data, selection))
MaxFeat = Int64(ceil(size(desc_temp,2)/3))
reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=21)
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
sp.scatter(y_train,y_hat_train,label="Training set", legend=:topleft, color = :green2)
sp.scatter!(y_test,y_hat_test,label="Test set",legend=:topleft, color=:orange)
sp.plot!([0,1000],[0,1000],label="1:1 line",linecolor ="black",width=2)
sp.xlabel!("Experimental RI")
sp.ylabel!("Predicted RI")
sp.title!("$data_name model using $(size(desc_temp,2)) descriptors")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Regression_Partial_Model_$data_name.png")

histogram(RTI, bins=60, label="Experimental RIs", xaxis = "Experimental RI", yaxis = "Frequency", title = "RTI distribution for the $(data_name) dataset")
histogram2d(vcat(y_hat_test,y_hat_train),vcat(res_test,res_train),label="Predicted RIs",bins = 60,alpha=0.5)

# Correlation matrix
dat = hcat(RTI, desc_temp)
cortemp = cor(dat)
labels = vcat("RI",String.(selection))

heatmap(cortemp, title = "Correlation matrix heatmap - $data_name",size=(950,700),dpi=100,xtickfont=11,tickfont=11,bottom_margin = 5Plots.mm, right_margin = 12Plots.mm)
#yticks!(rotation=0,[1:1:length(labels);], [labels[1],labels[2],labels[3],labels[4],labels[5],labels[6],labels[7],labels[8],labels[9],labels[10],labels[11],labels[12],labels[13],labels[14]])
#xticks!(xflip=false,xrotation=45,[1:1:length(labels);], [labels[1],labels[2],labels[3],labels[4],labels[5],labels[6],labels[7],labels[8],labels[9],labels[10],labels[11],labels[12],labels[13],labels[14]])
yticks!(yflip=true,yrotation=0,1:length(labels),[labels[i] for i in 1:length(labels)])
xticks!(xflip=false,xrotation=45,1:length(labels),[labels[i] for i in 1:length(labels)])

savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Correlation_matrix_heatmap_$data_name.png")

#With a dummy variable
cortemp_dum = cor(hcat(dat,rand(size(dat,1))))
sort(abs.(cortemp_dum[:,15]), rev=true)[2]
labels_dum = vcat(labels,"Dummy")
heatmap(cortemp_dum, title = "Correlation matrix heatmap - $data_name",size=(950,700),dpi=100,xtickfont=11,tickfont=11,bottom_margin = 5Plots.mm, right_margin = 12Plots.mm)
yticks!(yflip=true,1:length(labels_dum),[labels_dum[i] for i in 1:length(labels_dum)])
xticks!(xflip=false,xrotation=45,1:length(labels_dum),[labels_dum[i] for i in 1:length(labels_dum)])
savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Correlation_matrix_heatmap_$data_name with dummy.png")




# Distribution of Descriptors (to find the adjustment range)
quantiles = zeros(size(desc_temp,2),2)
for i=1:size(desc_temp,2)
    quantiles[i,:] .= (quantile(desc_temp[:,i],0.025), quantile(desc_temp[:,i],0.975))
end
range_quant = (5/100).*(quantiles[:,2]-quantiles[:,1])

histogram(desc_temp[:,5],bins=100,legend=false,xlabel="",ylabel="Probability")
a1,a2=(quantile(desc_temp[:,5],0.025), quantile(desc_temp[:,5],0.975))
sp.plot!([a1,a1],[-10,10],label="1:1 line",linewidth=7,linecolor ="black",width=2)
sp.plot!([a2,a2],[-10,10],label="1:1 line",linewidth=4,linecolor ="black",width=2)
sp.plot!([a1,a2],[-10,-10],label="1:1 line",linewidth=1,arrow=:both,linecolor ="black",width=2)

sp.savefig("C:\\Users\\alex_\\Desktop\\Descriptor 95% area.png")


## Hi-Lo Distribution
# Lowest point
sort(y_hat_train)[1]

lowest = sortperm(y_hat_train)[1]
y_hat_train[lowest]

#=  X_low = zeros(5000,length(X_train[lowest,:]))
for i = 1:size(X_low,1)
    change = BS.sample(1:length(X_train[lowest,:]))
    for j = 1:length(X_train[lowest,:])
        if j == change
            small_change = BS.sample(-range_quant[j]:0.0001:range_quant[j])
            X_low[i,j] = X_train[lowest,j] + small_change
        else
            X_low[i,j] = X_train[lowest,j]
        end
    end
end =#

X_low = zeros(10000,length(X_train[lowest,:]))
for i = 1:size(X_low,1)
    for j = 1:length(X_train[lowest,:])
            X_low[i,j] = X_train[lowest,j] + BS.sample(-range_quant[j]:0.0001:range_quant[j])
    end
end

y_hat_low = predict(reg,X_low)
y_hat_lowest = sort(y_hat_low)[1]
histogram(y_hat_low, label=false, yaxis = "Frequency",xaxis = "Predicted RI",title = "Lowest point - Distribution",bins=100)
#sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Lowest_point_distribution_$data_name.png")

boxplot(y_hat_low, label=false,yaxis = "Predicted RI",title = "Lowest point - Distribution")
#sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Lowest_point_boxplot_$data_name.png")

centr_low = y_hat_low.-y_hat_lowest.+0.00001
histogram(centr_low)

d_low = fit(Gamma, centr_low)
lo, hi = quantile.(d_low, [0.001, 0.99])
x = range(lo, hi; length = 10000)
uncertainty_low =quantile(d_low,0.975) - quantile(d_low,0.025)


histogram(y_hat_low, legend=false, yaxis = "Frequency",xaxis = "Predicted RI",title = "Lowest point - Gamma distribution",bins=100)
plot!(x.+y_hat_lowest, 5000*(pdf.(d,x)),label=false,linewidth=4)
plot!([y_hat_lowest,y_hat_lowest+uncertainty_low],[-10,-10],label="1:1 line",linewidth=1,arrow=:both,linecolor ="black",width=2)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Lowest_point_distribution_$data_name.png")


# Highest point
sort(y_hat_train)[end]

highest = sortperm(y_hat_train)[end]
y_hat_train[highest]

X_high = zeros(10000,length(X_train[highest,:]))
for i = 1:size(X_high,1)
    for j = 1:length(X_train[highest,:])
            X_high[i,j] = X_train[highest,j] + BS.sample(-range_quant[j]:0.0001:range_quant[j])
    end
end

y_hat_high = predict(reg,X_high)
y_hat_highest = sort(y_hat_high,rev=true)[1]

histogram(y_hat_high, label=false, yaxis = "Frequency",xaxis = "Predicted RI",title = "Highest point - Distribution")
#sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Highest_point_distribution_$data_name.png")

#boxplot(y_hat_high, label=false, zlims = (950,970),yaxis = "Predicted RTI",title = "Highest point - Distribution")
#sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Highest_point_boxplot_$data_name.png")

centr_high = -(y_hat_high.-y_hat_highest.-0.00001)
histogram(centr_high)

d_high = fit(Gamma, centr_high)
lo, hi = quantile.(d_high, [0.001, 0.99])
x = range(lo, hi; length = 10000)
uncertainty_high =quantile(d_high,0.975) - quantile(d_high,0.025)


histogram(y_hat_high, legend=false, yaxis = "Frequency",xaxis = "Predicted RI",title = "Highest point - Gamma distribution",bins=50)
plot!(-(x.-y_hat_highest), 9000*(pdf.(d,x)),label=false,linewidth=4)
plot!([y_hat_highest,y_hat_highest-uncertainty_high],[-10,-10],label="1:1 line",linewidth=1,arrow=:both,linecolor ="black",width=2)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Highest_point_distribution_$data_name.png")


##RTI acceptance limits
threshold_low = y_hat_lowest + uncertainty_low
threshold_high = y_hat_highest - uncertainty_high

## Norman RI prediction
reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=21)
X_train, X_test, y_train, y_test = train_test_split(desc_temp, RTI, test_size=0.20, random_state=21)
fit!(reg, X_train, y_train)

norm_GR_desc = Matrix(select(norm_GR, selection))
RI_norman_GR = predict(reg, norm_GR_desc)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_GR", RI_norman_GR)


## Acceptance RI vector (for Amide dataset)
RI_assessment_AM = zeros(length(RI_norman_AM))
for i = 1:length(RI_norman_AM)
    if RI_norman_AM[i]>threshold_low && RI_norman_AM[i]<threshold_high
        RI_assessment_AM[i] = 1
    else
        RI_assessment_AM[i] = 2
    end
end

RI_norman_AM[RI_assessment_AM.== 2] # Red region

using BSON
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Assessment_Amide", RI_assessment_AM)

# Plot
histogram(RI_norman_AM[RI_assessment_AM.== 1], c=:lime,label = "Acceptable",legend=:topleft, bins=200,xlabel="Predicted RI", ylabel="Frequency", title="RIs of the Norman dataset - $data_name")
histogram!(RI_norman_AM[RI_assessment_AM.== 2], c=:coral,bins=200,label = "Unacceptable", xlabel="Predicted RI", ylabel="Frequency", title="RIs of the Norman dataset - $data_name")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman_prediction_$data_name with regions.png")
#

## Acceptance RI vector (for Greek dataset)
RI_assessment_GR = zeros(length(RI_norman_GR))
for i = 1:length(RI_norman_GR)
    if RI_norman_GR[i]>threshold_low && RI_norman_GR[i]<threshold_high
        RI_assessment_GR[i] = 1
    else
        RI_assessment_GR[i] = 2
    end
end

RI_norman_GR[RI_assessment_GR.== 2] # Red region

BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Assessment_Greek", RI_assessment_GR)

# Plot
histogram(RI_norman_GR[RI_assessment_GR.== 1], c=:lime,label="Acceptable",legend=:topleft, bins=100,xlabel="Predicted RI", ylabel="Frequency", title="RIs of the Norman dataset - $data_name")
histogram!(RI_norman_GR[RI_assessment_GR.== 2], c=:coral,bins=100,label = "Unacceptable")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman_prediction_Greek with regions.png")
#

##Remarks:
# Increasing max_features shows a better regression
# min_samples_leaf should stay as low as 2
# number of trees has no significant benefit after a certain number (eg.100)

## Leverage
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

# Williams
res_test = y_test - y_hat_test
res_train = y_train - y_hat_train

lev_train = leverage_dist(X_train, X_train)
lev_test = leverage_dist(X_train, X_test)

boxplot(vcat(lev_train,lev_test), label=false, ylabel = "Leverage")
cutoff = 3*std(vcat(res_test,res_train))
h_star_am = (3*(1170+1))/1190
h_star_gr = (3*(1170+1))/1452

o = abs.(vcat(res_test,res_train))
findall(x -> x.>cutoff,o)

findall(x ->x.>h_star_gr,lev_test)

scatter(lev_train,res_train, c=:green2, label="Training set", title="$data_name",xlims=(0,10),xlabel="Leverage", ylabel="Residual")
scatter!(lev_test,res_test, c=:orange, label="Test set")
plot!([h_star_gr,h_star_gr],[minimum(vcat(res_test,res_train)),maximum(vcat(res_test,res_train))],label=false,linecolor ="black",width=2)
plot!([0,maximum(vcat(lev_train,lev_test))],[cutoff,cutoff],label=false,linecolor ="black",width=2)
plot!([0,maximum(vcat(lev_train,lev_test))],[-cutoff,-cutoff],label=false,linecolor ="black",width=2)
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Williams_$data_name.png")


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

BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_amide", assessment_am)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_greek", assessment_gr)

## Total RTI assessment
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Assessment_Amide", RI_assessment_AM)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Assessment_Greek", RI_assessment_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_amide", assessment_am)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_greek", assessment_gr)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_GR", RI_norman_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_AM", RI_norman_AM)


#a = [RI_assessment_AM RI_assessment_GR assessment_am assessment_gr]
#a = DataFrame(RI_AM=RI_assessment_AM,RI_GR=RI_assessment_GR,AD_AM=assessment_am,AD_GR=assessment_gr)
#CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\temp_file.CSV", a)

total_assessment =falses(length(RI_assessment_AM),2)  #convert.(Int64,zeros(length(RI_assessment_AM),2))
for i = 1:length(RI_assessment_AM)
    # Say we have Greek inside AD and RI range (marked as 1)
    if assessment_gr[i]==1 && RI_assessment_GR[i]==1
        total_assessment[i,1] = 1
    end
    # Say we have Amide inside AD and RI range (marked as 1)
    if assessment_am[i]==1 && RI_assessment_AM[i]==1
        total_assessment[i,2] = 1
    end
end
b = DataFrame(OK_AM=total_assessment[:,2],OK_GR=total_assessment[:,1])
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\temp_file.CSV", b)
b[40271,:]
total_sum = (sum(total_assessment,dims=2))[:]
total_sum[40271]
histogram(total_sum, label=false, c=:yellow)
findall(x -> x.==2,total_sum)
#= Contour
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_GR", RI_norman_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_AM", RI_norman_AM)
b = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\temp_file.CSV", DataFrame)
total_sum = (sum(Matrix(b),dims=2))[:].+1

#RIs = hcat(RI_norman_GR,RI_norman_AM)
i = 1:20000
df_= DataFrame(RI_norman_GR=RI_norman_GR[i],RI_norman_AM=RI_norman_AM[i],total_sum=total_sum[i])
df = sort(df_)

z = Diagonal(df.total_sum[i])
=#

#Scatter RI1,RI2, classification
z=zeros(length(RI_norman_AM))
z_sort=zeros(length(RI_norman_AM))

z_col=Vector{String}()
z_mark=Vector{String}()
for i = 1:length(RI_norman_AM)
    push!(z_col,"black")
    push!(z_mark,"circle")
end
assessment_gr[40271]
RI_assessment_GR[40271]
assessment_am[40271]
RI_assessment_AM[40271]
for i = 1:length(RI_norman_AM)
    # Say we have Greek inside AD and RI range (marked as 1821)
    if assessment_gr[i]==1 && RI_assessment_GR[i]==1 && assessment_am[i]==1 && RI_assessment_AM[i]==1
        z[i] = 1
        z_sort[i] = 6860
        z_col[i] = "#e9a3c9"        #In both
        z_mark[i] = "diamond"
    elseif assessment_am[i]==1 && RI_assessment_AM[i]==1
        z[i] = 2
        z_sort[i] = 4776
        z_col[i] = "#4575b4"
        z_mark[i] = "circle"  #In Amide
    elseif assessment_gr[i]==1 && RI_assessment_GR[i]==1
        z[i] = 3
        z_sort[i] = 18952
        z_col[i] = "#c7eae5"
        z_mark[i] = "pentagon"       #In UoA
    elseif assessment_gr[i]==1 && assessment_am[i]!=1
        z[i] = 4
        z_sort[i] = 613
        z_col[i] = "#f1a340"        #Out of Amide AD
        z_mark[i] = "rect"
    elseif assessment_am[i]==1 && assessment_gr[i]!=1
        z[i] = 5
        z_sort[i] = 76
        z_col[i] = "#d73027"        #Out of UoA AD
        z_mark[i] = "octagon"
    elseif assessment_gr[i]!=1 && assessment_am[i]!=1
        z[i] = 6
        z_sort[i] = 63837
        z_col[i] = "#8c510a"        #Out of both ADs
        z_mark[i] = "rect"
    end
end
findall(x -> x.==6,z)
index = Int.(zeros(6,2))
index[:,1]=Int.(collect(1:6))
for i = 1:6
    index[i,2] = length(findall(x -> x == i, z))      #[total_sum.==3843,:]
end

df_=DataFrame(RI_norman_AM=RI_norman_AM, RI_norman_GR=RI_norman_GR,z_col=z_col,z=z,z_sort=z_sort,z_mark=z_mark)
df=sort(df_,[order(:z_sort, rev=true)])
marker = reshape(Symbol.(df[:,6]),length(df[:,6]),1)

scatter(df[:,1], df[:,2], color=df[:,3], legend=false,markershape=marker[:],bottom_margin = 30Plots.px,left_margin = 30Plots.px,
markersize=5,xlabel="n-alkylamide RI", ylabel="UoA RI",dpi=300,size=(1200,800),tickfontsie=2)

sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Scatter-RI1-RI2-ClassificationHD.png")



groupA = findall(x -> x .== 1, z)
groupB = findall(x -> x .== 2, z)
groupC = findall(x -> x .== 3, z)
groupD = findall(x -> x .== 4, z)
groupE = findall(x -> x .== 5, z)
groupF = findall(x -> x .== 6, z)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupA", groupA)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupB", groupB)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupC", groupC)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupD", groupD)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupE", groupE)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupF", groupF)

random_list_f = Vector{String}()
for i = 1:20
    ind = BS.sample(groupF)
    push!(random_list_f,norm_AM[ind,2])
end
list = unique(random_list_f)





norm_AM[40271,2]







plt.clf()
a = plt.contour(df[i,2], df[i,1], z[i,i], 2,cmap = "hot",levels=[1,2,3], alpha=0.7)
a.cmap.set_over("red")
plt.colorbar()
plt.xlabel("n-alkylamide RI")
plt.ylabel("UoA RI")

plt.savefig("contour9-sorted-noextentboth-nonchunk15-20k.png")

# Venn diagram
using Conda
Conda.pip_interop(true)
#Conda.pip("install",["matplotlib_venn"])
#Conda.pip("install",["pyqt5"])
venn = pyimport("matplotlib_venn")

AM_and_GR = length(total_sum[total_sum.==2])
AM_not_GR = (sum(total_assessment,dims=1))[2] - AM_and_GR
GR_not_AM = (sum(total_assessment,dims=1))[1] - AM_and_GR

AMnotGRnotNorman
#= Simple 2 circles Venn
plt.figure(figsize=(4,4))
v2 = venn.venn2(subsets = (AM_not_GR, GR_not_AM, AM_and_GR), set_labels = ("AM_not_GR","GR_not_AM","AM_and_GR"))

v2.get_patch_by_id("100").set_color("orange")
v2.get_patch_by_id("100").set_alpha(0.9)
v2.get_patch_by_id("010").set_color("orange")
v2.get_patch_by_id("010").set_alpha(1.0)
v2.get_patch_by_id("110").set_color("green")
v2.get_patch_by_id("110").set_alpha(0.9)

plt.show()
plt.savefig("simple_two_circles7.png")
=#
# 3 circles Venn
# (Abc, aBc, ABc, abC, AbC, aBC, ABC), where A=AM   B=GR    C=NORMAN
Abc = 0
aBc = 0
ABc = 0
abC = length(total_sum) - AM_not_GR - GR_not_AM - AM_and_GR
AbC = AM_not_GR
aBC = GR_not_AM
ABC = AM_and_GR

plt.clf()
v3 = venn.venn3(subsets = (Abc, aBc, ABc, abC, AbC, aBC, ABC), set_labels = ("n-alkylamide", "UoA", "NORMAN", "NORMAN", "AbC", "aBC", "ABC"))
v3.get_patch_by_id("111").set_color("green")
v3.get_patch_by_id("111").set_alpha(0.8)
v3.get_patch_by_id("111").set_linestyle("solid")
v3.get_patch_by_id("111").set_linewidth(0)

v3.get_patch_by_id("011").set_color("orange")
v3.get_patch_by_id("011").set_alpha(0.8)
v3.get_patch_by_id("011").set_linestyle("solid")
v3.get_patch_by_id("011").set_linewidth(3)

v3.get_patch_by_id("101").set_color("orange")
v3.get_patch_by_id("101").set_alpha(0.8)
v3.get_patch_by_id("101").set_linestyle("solid")
v3.get_patch_by_id("101").set_linewidth(3)

v3.get_patch_by_id("001").set_color("red")
v3.get_patch_by_id("001").set_alpha(0.8)
v3.get_patch_by_id("001").set_linestyle("solid")
v3.get_patch_by_id("001").set_linewidth(3)

plt.savefig("three_circles12.png")
## PCA
@sk_import decomposition: PCA  # We want to run a PCA

#Setup PCA model

# For Amide (full training set) and Norman (all descriptors)
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
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_amide", assessment_am)

scatter((scores_am[1191:end,1])[assessment_am.==3], (scores_am[1191:end,2])[assessment_am.==3], legend=:topleft,label="Outside",color=:pink,xlabel="PC1",ylabel="PC2",title="n-alkylamide")
scatter!((scores_am[1191:end,1])[assessment_am.==2], (scores_am[1191:end,2])[assessment_am.==2], label = "Indecisive", color = :yellow, xlabel = "PC1", ylabel = "PC2")
scatter!((scores_am[1191:end,1])[assessment_am.==1], (scores_am[1191:end,2])[assessment_am.==1], label = "Inside", color = :blue, xlabel = "PC1", ylabel = "PC2")
scatter!(scores_am[1:1190,1], scores_am[1:1190,2], label = "Training set", color = :green2, xlabel = "PC1", ylabel = "PC2")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\PCA_Norman_Amide.png")

#= For Amide (full training set) and Norman (all descriptors) - 2nd try two different PCAs #Not good either
RTI_am = AM[:,2]
desc_am = Matrix(AM[:,6:end])           # Careful! Matrix should have 1170 descriptors

X_train_am, X_test, y_train, y_test = train_test_split(desc_am, RTI_am, test_size=0.20, random_state=21)
norm_am_desc = Matrix(norm_AM[!,3:end])

pcatr = PCA(n_components = 2)
pcatr.fit(X_train_am)
scores_am = pcatr.fit_transform(X_train_am)

pcano = PCA(n_components = 2)
pcano.fit(norm_am_desc)
scores_norm_am = pcano.fit_transform(norm_am_desc)

using BSON
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_amide", assessment_am)

scatter((scores_norm_am[:,1])[assessment_am.==3], (scores_norm_am[:,2])[assessment_am.==3], legend=:topleft,label="Outside",color=:pink,xlabel="PC1",ylabel="PC2")
scatter!((scores_norm_am[:,1])[assessment_am.==2], (scores_norm_am[:,2])[assessment_am.==2], label = "Indecisive", color = :yellow, xlabel = "PC1", ylabel = "PC2")
scatter!((scores_norm_am[:,1])[assessment_am.==1], (scores_norm_am[:,2])[assessment_am.==1], label = "Inside", color = :green, xlabel = "PC1", ylabel = "PC2")
scatter!(scores_am[:,1], scores_am[:,2], label = "Training set", color = :blue, xlabel = "PC1", ylabel = "PC2")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\PCA_Norman_Amide.png")

=#

# For Greek (full training set) and Norman (all descriptors)

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
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_greek", assessment_gr)

scatter((scores_gr[1453:end,1])[assessment_gr.==3], (scores_gr[1453:end,2])[assessment_gr.==3],title="UoA", legend=:topleft,label="Outside",color=:pink,xlabel="PC1",ylabel="PC2")
scatter!((scores_gr[1453:end,1])[assessment_gr.==2], (scores_gr[1453:end,2])[assessment_gr.==2], label = "Indecisive", color = :yellow, xlabel = "PC1", ylabel = "PC2")
scatter!((scores_gr[1453:end,1])[assessment_gr.==1], (scores_gr[1453:end,2])[assessment_gr.==1], label = "Inside", color = :blue, xlabel = "PC1", ylabel = "PC2")
scatter!(scores_gr[1:1452,1], scores_gr[1:1452,2], label = "Training set", color = :green2, xlabel = "PC1", ylabel = "PC2")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\PCA_Norman_Greek.png")
