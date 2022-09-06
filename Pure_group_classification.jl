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
## Importing the data

mb_GR = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_MB_(Greek model).csv", DataFrame)
mb_AM = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_MB_(Amide model).csv", DataFrame)

BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_amide", assessment_mb_am)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_greek", assessment_mb_gr)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_GR", RI_mb_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_AM", RI_mb_AM)

residual_am = 60
residual_gr = 60
residual_am_outsideAD = 120
residual_gr_outsideAD = 120
urea_am = 227
urea_gr = 296.4
#urea_gr = 100

maingroup=Int.(zeros(length(RI_mb_AM)))
for i = 1:length(RI_mb_AM)
    # Say we have Greek inside AD and RI range
    if assessment_mb_am[i]==1 || assessment_mb_gr[i]==1
        maingroup[i]=1
    else maingroup[i]=2
    end
end

#Greek first (53.0% accuracy)
secondgroup=Int.(zeros(length(RI_mb_AM)))
for i = 1:length(RI_mb_AM)
        # In both ADs
    if assessment_mb_am[i]==1 && assessment_mb_gr[i]==1
        if RI_mb_AM[i] > (urea_am + residual_am) && RI_mb_AM[i] < (maximum(RI_mb_AM) - residual_am) && RI_mb_GR[i] > (urea_gr + residual_gr) && RI_mb_GR[i] < (maximum(RI_mb_GR) - residual_gr)
            secondgroup[i] = 1
        else secondgroup[i] = 2
        end
        # In Greek AD
    elseif assessment_mb_am[i]!=1 && assessment_mb_gr[i]==1
        if RI_mb_GR[i] > (urea_gr + residual_gr) && RI_mb_GR[i] < (maximum(RI_mb_GR) - residual_gr)
            secondgroup[i] = 1
        else secondgroup[i] = 2
        end
        # In Amide AD
    elseif assessment_mb_am[i]==1 && assessment_mb_gr[i]!=1
        if RI_mb_AM[i] > (urea_am + residual_am) && RI_mb_AM[i] < (maximum(RI_mb_AM) - residual_am)
            secondgroup[i] = 1
        else secondgroup[i] = 2
        end
        # Outside both ADs
    elseif assessment_mb_am[i]!=1 && assessment_mb_gr[i]!=1
        if RI_mb_AM[i] > (urea_am + residual_am_outsideAD) && RI_mb_AM[i] < (maximum(RI_mb_AM) - residual_am_outsideAD) && RI_mb_GR[i] > (urea_gr + residual_gr_outsideAD) && RI_mb_GR[i] < (maximum(RI_mb_GR) - residual_gr_outsideAD)
            secondgroup[i] = 1
        else secondgroup[i] = 2
        end
    end
end

accepted = secondgroup[secondgroup.==1]
rejected = secondgroup[secondgroup.==2]
accuracy = (length(accepted)*100)/(length(accepted)+length(rejected))
ddd = secondgroup[secondgroup.==5]
(length(rejected)*100)/(length(secondgroup))
