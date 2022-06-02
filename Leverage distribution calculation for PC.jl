using DataFrames
using Statistics
using ScikitLearn
using CSV
using Plots
using LinearAlgebra
using PyCall
using BSON

import StatsBase as BS
import StatsPlots as sp

using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
@sk_import ensemble: RandomForestRegressor
#################################

BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Amide_normalisation_factors", amide_max)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Greek_normalisation_factors", greek_max)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Descriptors", descriptors)

#################################
amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Amide_descriptors1.csv", DataFrame)
amide_ = select(amide_raw, descriptors)
amide = convert.(Float64, amide_[:, :])

greek_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\GreekDataset_+ESI_Descriptors.csv", DataFrame)
greek_ = select(greek_raw, descriptors)
greek = convert.(Float64, greek_[:, :])

Norm_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part1.csv", DataFrame)
for i = 2:20
    Norm_temp = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part$i.csv", DataFrame)
    Norm_raw = vcat(Norm_raw, Norm_temp)
end
norman_ = select(Norm_raw, descriptors)
norman = convert.(Float64, norman_[:, :])
#################################

for j = 1:size(amide,2)
    factor = amide_max[j]
    if factor != 0
        vec_temp = (amide[:,j] ./ abs(factor))
        amide[:,j] = vec_temp
    end
end

for j = 1:size(greek,2)
    factor = greek_max[j]
    if factor != 0
          vec_temp = (greek[:,j] ./ abs(factor))
          greek[:,j] = vec_temp
    end
end

norman_gr = deepcopy(norman)
norman_am = deepcopy(norman)

for j = 1:size(norman_gr,2)
    factor = greek_max[j]
    if factor != 0
          vec_temp = (norman_gr[:,j] ./ abs(factor))
          norman_gr[:,j] = vec_temp
    end
end

for j = 1:size(norman_am,2)
    factor = amide_max[j]
    if factor != 0
          vec_temp = (norman_am[:,j] ./ abs(factor))
          norman_am[:,j] = vec_temp
    end
end
#################################
X_train_am, X_test_am, y_train_am, y_test_am = train_test_split(Matrix(amide), amide_raw[:,2], test_size=0.20, random_state=21);
X_train_gr, X_test_gr, y_train_gr, y_test_gr = train_test_split(Matrix(greek), Vector(greek_raw[:,5]), test_size=0.20, random_state=21);


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
