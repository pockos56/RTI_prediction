using CSV, DataFrames, Statistics, Plots, BSON

## Loading the nice descriptors vector from the Amide dataset
# (Removing high std variation descs, and descriptors with missing values)

BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Nice descriptors", nice_desc)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\bad_comps_amide", bad_comps_amide)

amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Amide_descriptors1.csv", DataFrame)
amide_ = select(amide_raw, nice_desc)
amide = amide_[Not(bad_comps_amide), :]

amide_abs = abs.(amide)
amide_max = collect(maximum(eachrow(amide_abs)))
histogram(amide_max)

## Loading the Greek dataset
greek_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\GreekDataset_+ESI_Descriptors.csv", DataFrame)
greek_ = select(greek_raw, nice_desc)
greek = greek_[Not(688,1043,1807), :]

greek_abs = abs.(greek)
greek_max = collect(maximum(eachrow(greek_abs)))

histogram(greek_max)

## Comparing the maxima of Amide_maxima and Greek_maxima
dmax = abs.(amide_max - greek_max)
ind_dmax = findall(x -> x >= 100,dmax[:])     # The descriptors with high maximum difference
dmax[dmax .> 100]       # The maxima difference
bad_descs1 = names(amide[!,ind_dmax])

## Normalization for the Amide Dataset
amide_ref = Matrix(copy(amide))

# nan = count(Matrix(isnan.(amide_ref) .== 1))
# ind_ss = findall(nan)
# amide_ref[isnan.(amide_ref) .== 1]

for j = 1:size(amide_ref,2)
    factor = amide_max[j]
    if factor != 0
      if eltype(amide_ref[:,j]) == Int64
          vec_temp = round.(amide_ref[:,j] ./ factor)
          amide_ref[:,j] = vec_temp
      else
          vec_temp = (amide_ref[:,j] ./ factor)
          amide_ref[:,j] = vec_temp
      end
    end
end
