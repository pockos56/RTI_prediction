using CSV, DataFrames, Statistics, Plots, BSON

## Loading the nice descriptors vector from the Amide dataset
# (Removing high std variation descs, and descriptors with missing values)

BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Nice descriptors", nice_desc)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\bad_comps_amide", bad_comps_amide)

amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Amide_descriptors1.csv", DataFrame)
amide_ = select(amide_raw, nice_desc)
amide = amide_[Not(bad_comps_amide), :]
amide_abs = abs.(amide)
# Maxima for Amide
amide_max = zeros(size(amide_abs,2))
for j = 1:size(amide_abs,2)
    amide_max[j] = maximum(amide_abs[:,j])
end

histogram(amide_max)

## Loading the Greek dataset
greek_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\GreekDataset_+ESI_Descriptors.csv", DataFrame)
greek_ = select(greek_raw, nice_desc)
greek = greek_[Not(688,1043,1807), :]

greek_abs = abs.(greek)

# Maxima for Greek
greek_max = zeros(size(greek_abs,2))
for j = 1:size(greek_abs,2)
    greek_max[j] = maximum(greek_abs[:,j])
end

histogram(greek_max)

## Comparing the maxima of Amide_maxima and Greek_maxima
dmax = abs.(amide_max - greek_max)
ind_dmax = findall(x -> x >= 100,dmax[:])     # The descriptors with high maximum difference
dmax[dmax .> 100]       # The maxima difference
bad_descs1 = names(amide[!,ind_dmax])
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\high_max_descs", bad_descs1)


## Normalization for the Amide Dataset
amide_ref = Matrix(copy(amide))
# nan = count(Matrix(isnan.(amide_ref) .== 1))
# ind_ss = findall(nan)
# amide_ref[isnan.(amide_ref) .== 1]

for j = 1:size(amide_ref,2)
    factor = amide_max[j]
    if factor > 0
          if eltype(amide_ref[:,j]) == Int64
              vec_temp = round.(amide_ref[:,j] ./ abs(factor))
              amide_ref[:,j] = vec_temp
          else
              vec_temp = (amide_ref[:,j] ./ abs(factor))
              amide_ref[:,j] = vec_temp
          end
    elseif factor < 0
        if eltype(amide_ref[:,j]) == Int64
            vec_temp = round.(amide_ref[:,j] ./ abs(factor))
            amide_ref[:,j] = vec_temp
        else
            vec_temp = (amide_ref[:,j] ./ abs(factor))
            amide_ref[:,j] = vec_temp
        end
    end
end

amide_max = zeros(size(amide_ref,2))
for j = 1:size(amide_ref,2)
    amide_max[j] = maximum(amide_ref[:,j])
end
mmmax = findall(x -> x < 1,amide_max[:])     # The descriptors with high maximum difference
z = (amide_max[amide_max .<1])
z[z .!= 0]
sum(amide_max)
