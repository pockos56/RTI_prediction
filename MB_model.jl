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
mb_GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_mb_(Greek model).csv", DataFrame)
mb_AM = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_mb_(Amide model).csv", DataFrame)
mb_GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_MB_(Greek model).csv", DataFrame)
mb_AM = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_MB_(Amide model).csv", DataFrame)

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
## Model - Selected descriptors
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Descriptor_names_partial_model_$data_name", selection)

desc_temp = Matrix(select(data, selection))
MaxFeat = Int64(ceil(size(desc_temp,2)/3))
reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=21)
X_train, X_test, y_train, y_test = train_test_split(desc_temp, RTI, test_size=0.20, random_state=21)
fit!(reg, X_train, y_train)

############
##RTI acceptance limits
threshold_low = y_hat_lowest + uncertainty_low
threshold_high = y_hat_highest - uncertainty_high



BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\threshold_low_GR", threshold_low)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\threshold_high_GR", threshold_high)
## mb RI prediction
reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=4, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=21)
X_train, X_test, y_train, y_test = train_test_split(desc_temp, RTI, test_size=0.20, random_state=21)
fit!(reg, X_train, y_train)

mb_GR_desc = Matrix(select(mb_GR, selection))
RI_mb_GR = predict(reg, mb_GR_desc)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_GR", RI_mb_GR)


## Acceptance RI vector (for Amide dataset)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\threshold_low_AM", threshold_low)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\threshold_high_AM", threshold_high)

RI_assessment_mb_AM = zeros(length(RI_mb_AM))
for i = 1:length(RI_mb_AM)
    if RI_mb_AM[i]>threshold_low && RI_mb_AM[i]<threshold_high
        RI_assessment_mb_AM[i] = 1
    else
        RI_assessment_mb_AM[i] = 2
    end
end

RI_mb_AM[RI_assessment_mb_AM.== 2] # Red region

using BSON
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_assessment_mb_Amide", RI_assessment_mb_AM)

# Plot
histogram(RI_mb_AM[RI_assessment_mb_AM.== 1], c=:lime,label = "Acceptable",legend=:topleft, bins=200,xlabel="Predicted RI", ylabel="Frequency", title="RIs of the mb dataset - $data_name")
histogram!(RI_mb_AM[RI_assessment_mb_AM.== 2], c=:coral,bins=200,label = "Unacceptable", xlabel="Predicted RI", ylabel="Frequency", title="RIs of the mb dataset - $data_name")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\mb_prediction_$data_name with regions.png")
#

## Acceptance RI vector (for Greek dataset)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\threshold_low_GR", threshold_low)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\threshold_high_GR", threshold_high)

RI_assessment_mb_GR = zeros(length(RI_mb_GR))
for i = 1:length(RI_mb_GR)
    if RI_mb_GR[i]>threshold_low && RI_mb_GR[i]<threshold_high
        RI_assessment_mb_GR[i] = 1
    else
        RI_assessment_mb_GR[i] = 2
    end
end

RI_mb_GR[RI_assessment_mb_GR.== 2] # Red region

BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_assessment_mb_Greek", RI_assessment_mb_GR)

## Leverage
## Question: The Applicability Domain should be with all the descriptors?
# The model doesn't contain them all, so why does it make sense?
#
function leverage_dist(X_train, mb)
    lev = zeros(size(mb,1))
    z = pinv(transpose(X_train) * X_train)
    for ind = 1:size(mb,1)
        x = mb[ind,:]
        lev[ind] = transpose(x) * z * x
        println(ind)
    end
    return lev
end
lev_mb_am = leverage_dist(X_train_AM, Matrix(mb_AM[:,2:end]))
lev_mb_gr = leverage_dist(X_train_GR, Matrix(mb_GR[:,2:end]))

df = DataFrame(lev_mb_am = lev_mb_am, lev_mb_gr = lev_mb_gr)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_mb.csv",df)

## Loading the leverage
df = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_mb.csv",DataFrame)
lev = Matrix(df)

h_star_am = (3*(1170+1))/1190
h_star_gr = (3*(1170+1))/1452

# mb - Amide dataset
lev_am = lev[:,1]
assessment_mb_am = convert.(Int64,zeros(length(lev_am)))

for i = 1:length(assessment_mb_am)
    if lev_am[i] <= h_star_am
        assessment_mb_am[i] = 1
    elseif lev_am[i] <= 3*h_star_am
        assessment_mb_am[i] = 2
    else
        assessment_mb_am[i] = 3
    end
end

histogram(lev[:,1], bins=800000, label = false, title="Applicability Domain for the Amide dataset", xaxis="Leverage", xlims = (0,100))
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_histogram_mb_Amide.png")

histogram(assessment_mb_am, label=false, bins =4, title = "Applicability Domain for the Amide dataset")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_Amide.png")

assessment_mb_am_1 = assessment_mb_am[assessment_mb_am.==1] # 416 out of 2780 are ok
assessment_mb_am_2 = assessment_mb_am[assessment_mb_am.==2] # 215 out of 2780 are meh
assessment_mb_am_3 = assessment_mb_am[assessment_mb_am.==3] # 2149 out of 2780 are NOT ok

# mb - Greek dataset
lev_gr = lev[:,2]
assessment_mb_gr = convert.(Int64,zeros(length(lev_gr)))
for i = 1:length(assessment_mb_gr)
    if lev_gr[i] <= h_star_gr
        assessment_mb_gr[i] = 1
    elseif lev_gr[i] <= 3*h_star_gr
        assessment_mb_gr[i] = 2
    else
        assessment_mb_gr[i] = 3
    end
end

histogram(lev[:,2], bins=800, label = false, title="Applicability Domain for the Greek dataset", xaxis="Leverage", xlims = (0,100))
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Leverage_histogram_mb_Greek.png")

histogram(assessment_mb_gr, label=false, bins =4, title = "Applicability Domain for the Greek dataset")
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_Greek.png")

assessment_mb_gr_1 = assessment_mb_gr[assessment_mb_gr.==1] # 1615 out of 2780 are ok
assessment_mb_gr_2 = assessment_mb_gr[assessment_mb_gr.==2] # 721 out of 2780 are meh
assessment_mb_gr_3 = assessment_mb_gr[assessment_mb_gr.==3] # 444 out of 2780 are NOT ok

BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_amide", assessment_mb_am)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_greek", assessment_mb_gr)

## Total RTI assessment
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Assessment_mb_Amide", RI_assessment_mb_AM)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Assessment_mb_Greek", RI_assessment_mb_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_amide", assessment_mb_am)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_greek", assessment_mb_gr)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_GR", RI_mb_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_AM", RI_mb_AM)

total_assessment_mb =falses(length(RI_assessment_mb_AM),2)  #convert.(Int64,zeros(length(RI_assessment_mb_AM),2))
for i = 1:length(RI_assessment_mb_AM)
    # Say we have Greek inside AD and RI range (marked as 1)
    if assessment_mb_gr[i]==1 && RI_assessment_mb_GR[i]==1
        total_assessment_mb[i,1] = 1
    end
    # Say we have Amide inside AD and RI range (marked as 1)
    if assessment_mb_am[i]==1 && RI_assessment_mb_AM[i]==1
        total_assessment_mb[i,2] = 1
    end
end
b = DataFrame(OK_AM=total_assessment_mb[:,2],OK_GR=total_assessment_mb[:,1])

total_sum_mb = (sum(total_assessment_mb,dims=2))[:]

histogram(total_sum_mb, label=false, c=:yellow)
findall(x -> x.==2,total_sum_mb)



#Scatter RI1,RI2, classification
z=zeros(length(RI_mb_AM))
z_sort=zeros(length(RI_mb_AM))

z_col=Vector{String}()
z_mark=Vector{String}()
for i = 1:length(RI_mb_AM)
    push!(z_col,"black")
    push!(z_mark,"circle")
end

for i = 1:length(RI_mb_AM)
    # Say we have Greek inside AD and RI range
    if assessment_mb_gr[i]==1 && RI_assessment_mb_GR[i]==1 && assessment_mb_am[i]==1 && RI_assessment_mb_AM[i]==1
        z[i] = 1
        z_sort[i] = 255
        z_col[i] = "#e9a3c9"        #In both
        z_mark[i] = "diamond"
    elseif assessment_mb_am[i]==1 && RI_assessment_mb_AM[i]==1
        z[i] = 2
        z_sort[i] = 161
        z_col[i] = "#4575b4"
        z_mark[i] = "circle"  #In Amide
    elseif assessment_mb_gr[i]==1 && RI_assessment_mb_GR[i]==1
        z[i] = 3
        z_sort[i] = 1350
        z_col[i] = "#c7eae5"
        z_mark[i] = "pentagon"       #In UoA
    elseif assessment_mb_gr[i]==1 && assessment_mb_am[i]!=1
        z[i] = 4
        z_sort[i] = 6
        z_col[i] = "#f1a340"        #Out of Amide AD
        z_mark[i] = "rect"
    elseif assessment_mb_am[i]==1 && assessment_mb_gr[i]!=1
        z[i] = 5
        z_sort[i] = 0
        z_col[i] = "#d73027"        #Out of UoA AD
        z_mark[i] = "octagon"
    elseif assessment_mb_gr[i]!=1 && assessment_mb_am[i]!=1
        z[i] = 6
        z_sort[i] = 1008
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

df_=DataFrame(RI_mb_AM=RI_mb_AM, RI_mb_GR=RI_mb_GR,z_col=z_col,z=z,z_sort=z_sort,z_mark=z_mark)
df=sort(df_,[order(:z_sort, rev=true)])
marker = reshape(Symbol.(df[:,6]),length(df[:,6]),1)

scatter(df[:,1], df[:,2], color=df[:,3], legend=false,markershape=marker[:],bottom_margin = 30Plots.px,left_margin = 30Plots.px,
markersize=7,xlabel="n-alkylamide RI", ylabel="UoA RI",dpi=300,size=(1200,800),tickfontsize=12)

sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Scatter-RI1-RI2-ClassificationHD_MB.png")
ACCURACY = ((255+1350)/2780)*100
ACCURACY_2 = ((255+1350+161)/2780)*100
