using CSV, DataFrames, Statistics, Plots, BSON

## Loading the nice descriptors vector from the Amide dataset
# (Removing high std variation descs, and descriptors with missing values)

BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Nice descriptors", nice_desc)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\bad_comps_amide", bad_comps_amide)

## Loading Norman dataset

Norm_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part1.csv", DataFrame)
for i = 2:5   #:20
    Norm_temp = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part$i.csv", DataFrame)
    Norm_raw = vcat(Norm_raw, Norm_temp)
end

Norm_clean = select(Norm_raw, nice_desc)

countmissing = (ismissing.(Matrix(Norm_clean)))
heatmap(countmissing)
x = findall(x -> x >0, countmissing)


bad = " "
threshold = 0
for j = 1:size(countmissing,2)
    if sum(countmissing[:,j]) > threshold
        bad_temp = names(Norm_clean, j)
        bad = vcat(bad,bad_temp)
    end
end
bad = bad[2:length(bad)]
#734 descriptors have a least one missing value

norm = select(Norm_clean,Not(bad))
countmissing = (ismissing.(Matrix(norm)))
heatmap(countmissing)
x = findall(x -> x >0, countmissing)        # No missing values :-)


## Loading the Amide dataset
amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Amide_descriptors1.csv", DataFrame)
amide_ = select(amide_raw, nice_desc)
amide = select(amide_,Not(bad))

#Check for missing
countmissing = (ismissing.(Matrix(amide_)))
x = findall(x -> x >0, countmissing)        # No missing values :-)

## Loading the Greek dataset
greek_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\GreekDataset_+ESI_Descriptors.csv", DataFrame)
greek_ = select(greek_raw, nice_desc)
greek = select(greek_, Not(bad))

#Check for missing
countmissing = (ismissing.(Matrix(greek_)))
x = findall(x -> x >0, countmissing)        # No missing values :-)

## Comparing the maxima of Amide_maxima and Greek_maxima

# Maxima for Amide
amide_abs = abs.(amide)
amide_max = zeros(size(amide_abs,2))
for j = 1:size(amide_abs,2)
    amide_max[j] = maximum(amide_abs[:,j])
end

# Maxima for Greek
greek_abs = abs.(greek)
greek_max = zeros(size(greek_abs,2))
for j = 1:size(greek_abs,2)
    greek_max[j] = maximum(greek_abs[:,j])
end

# Maxima for Norman
norm_abs = abs.(norm)
norm_max = zeros(size(norm_abs,2))
for j = 1:size(norm_abs,2)
    norm_max[j] = maximum(norm_abs[:,j])
end

dmax_AM_GR = abs.(amide_max - greek_max)
dmax_NORM_GR = abs.(norm_max - greek_max)
dmax_NORM_AM = abs.(norm_max - amide_max)
#
#
#
#
#
#
#
#
#

ind_dmax = findall(x -> x > 100,dmax[:])     # The descriptors with high maximum difference
dmax[dmax .> 100]       # The maxima difference
bad_descs1 = names(amide[!,ind_dmax])
#BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\high_max_descs", bad_descs1)


## Normalization for the Amide Dataset
amide_ref_ = deepcopy(amide)
amide_ref = Matrix(select(amide_ref_, Not(bad_descs1)))

# Maxima for Amide
amide_abs = abs.(amide_ref)
amide_max = zeros(size(amide_abs,2))
for j = 1:size(amide_abs,2)
    amide_max[j] = maximum(amide_abs[:,j])
end

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

#amide_max = zeros(size(amide_ref,2))
#for j = 1:size(amide_ref,2)
#    amide_max[j] = maximum(amide_ref[:,j])
#end
#mmmax = findall(x -> x < 1,amide_max[:])     # The descriptors with high maximum difference
#z = (amide_max[amide_max .<1])
#z[z .!= 0]
#sum(amide_max)


## Normalization for the Greek Dataset
greek_ref_ = deepcopy(greek)
greek_ref = Matrix(select(greek_ref_, Not(bad_descs1)))

# Maxima for Greek
greek_abs = abs.(greek_ref)
greek_max = zeros(size(greek_abs,2))
for j = 1:size(greek_abs,2)
    greek_max[j] = maximum(greek_abs[:,j])
end

for j = 1:size(greek_ref,2)
    factor = greek_max[j]
    if factor > 0
          if eltype(greek_ref[:,j]) == Int64
              vec_temp = round.(greek_ref[:,j] ./ abs(factor))
              greek_ref[:,j] = vec_temp
          else
              vec_temp = (greek_ref[:,j] ./ abs(factor))
              greek_ref[:,j] = vec_temp
          end
    elseif factor < 0
        if eltype(greek_ref[:,j]) == Int64
            vec_temp = round.(greek_ref[:,j] ./ abs(factor))
            greek_ref[:,j] = vec_temp
        else
            vec_temp = (greek_ref[:,j] ./ abs(factor))
            greek_ref[:,j] = vec_temp
        end
    end
end
