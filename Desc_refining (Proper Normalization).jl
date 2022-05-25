using CSV, DataFrames, Statistics, Plots, BSON

## Loading the nice descriptors vector from the Amide dataset
# (Removing high std variation descs, and descriptors with missing values)

BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Nice descriptors", nice_desc)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\bad_comps_amide", bad_comps_amide)

## Loading Norman dataset

Norm_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part1.csv", DataFrame)
for i = 2:20
    Norm_temp = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part$i.csv", DataFrame)
    Norm_raw = vcat(Norm_raw, Norm_temp)
    println("$i/20")
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
#734 descriptors have a least one missing value (25% of the dataset used for that Calculation)
#913 descriptors have a least one missing value (100% of the dataset used for that Calculation)


norm = select(Norm_clean,Not(bad))
countmissing = (ismissing.(Matrix(norm)))
heatmap(countmissing)
x = findall(x -> x >0, countmissing)        # No missing values :-)


## Loading the Amide dataset
amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Amide_descriptors1.csv", DataFrame)
amide_ = select(amide_raw, nice_desc)
amide = select(amide_,Not(bad))

#Check for missing
countmissing = (ismissing.(Matrix(amide)))
x = findall(x -> x >0, countmissing)        # No missing values :-)

## Loading the Greek dataset
greek_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\GreekDataset_+ESI_Descriptors.csv", DataFrame)
greek_ = select(greek_raw, nice_desc)
greek = select(greek_, Not(bad))

#Check for missing
countmissing = (ismissing.(Matrix(greek)))
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

dmax_AM_GR = zeros(length(amide_max))
for i = 1:length(amide_max)
    if amide_max[i] > greek_max[i]
        dmax_AM_GR[i] = amide_max[i] / greek_max[i]
    else dmax_AM_GR[i] = amide_max[i] \ greek_max[i]
    end
    if dmax_AM_GR[i] == Inf
        dmax_AM_GR[i] = 0
    end

end

dmax_NORM_GR = zeros(length(norm_max))
for i = 1:length(norm_max)
    if norm_max[i] > greek_max[i]
        dmax_NORM_GR[i] = norm_max[i] / greek_max[i]
    else dmax_NORM_GR[i] = norm_max[i] \ greek_max[i]
    end
    if dmax_NORM_GR[i] == Inf
        dmax_NORM_GR[i] = 0
    end

end

dmax_NORM_AM = zeros(length(amide_max))
for i = 1:length(amide_max)
    if norm_max[i] > amide_max[i]
        dmax_NORM_AM[i] = norm_max[i] / amide_max[i]
    else dmax_NORM_AM[i] = norm_max[i] \ amide_max[i]
    end
    if dmax_NORM_AM[i] == Inf
        dmax_NORM_AM[i] = 0
    end
end

ind_dmax1 = findall(x -> x > 100,dmax_AM_GR[:])     # The descriptors with high maximum difference
ind_dmax2 = findall(x -> x > 100,dmax_NORM_GR[:])     # The descriptors with high maximum difference
ind_dmax3 = findall(x -> x > 100,dmax_NORM_AM[:])     # The descriptors with high maximum difference

dmax_AM_GR[dmax_AM_GR .> 100]       # The maxima difference
dmax_NORM_GR[dmax_NORM_GR .> 100]       # The maxima difference
dmax_NORM_AM[dmax_NORM_AM .> 100]       # The maxima difference

bad_descs1 = names(amide[!,ind_dmax1])
bad_descs2 = names(amide[!,ind_dmax2])
bad_descs3 = names(amide[!,ind_dmax3])
bad_descs = unique(vcat(bad_descs1,bad_descs2,bad_descs3))

#BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\high_max_descs", bad_descs)


## Normalization for the Amide Dataset
amide_ref_ = select(deepcopy(amide), Not(bad_descs))
amide_ref = convert.(Float64, amide_ref_[:, :])

# Maxima for Amide
amide_abs = abs.(amide_ref)
amide_max = zeros(size(amide_abs,2))
for j = 1:size(amide_abs,2)
    amide_max[j] = maximum(amide_abs[:,j])
end
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Amide_normalisation_factors", amide_max)

for j = 1:size(amide_ref,2)
    factor = amide_max[j]
    if factor != 0
        vec_temp = (amide_ref[:,j] ./ abs(factor))
        amide_ref[:,j] = vec_temp
    end
end

#    amide_max = zeros(size(amide_ref,2))
#    for j = 1:size(amide_ref,2)
#        amide_max[j] = maximum(amide_ref[:,j])
#    end
#    mmmax = findall(x -> x < 1,amide_max[:])     # The descriptors with high maximum difference
#z = (amide_max[amide_max .<1])
#z[z .!= 0]
#sum(amide_max)


## Normalization for the Greek Dataset
greek_ref_ = select(deepcopy(greek), Not(bad_descs))
greek_ref = convert.(Float64, greek_ref_[:, :])

# Maxima for Greek
greek_abs = abs.(greek_ref)
greek_max = zeros(size(greek_abs,2))
for j = 1:size(greek_abs,2)
    greek_max[j] = maximum(greek_abs[:,j])
end
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Greek_normalisation_factors", greek_max)

for j = 1:size(greek_ref,2)
    factor = greek_max[j]
    if factor != 0
          vec_temp = (greek_ref[:,j] ./ abs(factor))
          greek_ref[:,j] = vec_temp
    end
end

## Normalization for the Norman Dataset
#
# Questions
# a) Should the normalisation factors be the same for the Norman dataset or should it change depending on the model?
# b) Maybe we should apply the same normalisation factors to everything?
#
norman_ref_ = select(deepcopy(norm), Not(bad_descs))
norman_ref_GR = convert.(Float64, norman_ref_[:, :])
norman_ref_AM = convert.(Float64, norman_ref_[:, :])

#
for j = 1:size(norman_ref_GR,2)
    factor = greek_max[j]
    if factor != 0
          vec_temp = (norman_ref_GR[:,j] ./ abs(factor))
          norman_ref_GR[:,j] = vec_temp
    end
end

for j = 1:size(norman_ref_AM,2)
    factor = amide_max[j]
    if factor != 0
          vec_temp = (norman_ref_AM[:,j] ./ abs(factor))
          norman_ref_AM[:,j] = vec_temp
    end
end

## Saving the refined datasets
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Amide.csv", amide_ref)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Greek.csv", greek_ref)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norman_(Greek model).csv", norman_ref_GR)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_Norman_(Amide model).csv", norman_ref_AM)
