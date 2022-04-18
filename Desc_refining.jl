using CSV, DataFrames, Statistics, Tables
#####################
# Let's check whether our data is the same
a = CSV.read("C:\\Users\\alex_\\Desktop\\Minor project\\Julia\\Amide_descriptors1.csv", DataFrame)
b = CSV.read("C:\\Users\\alex_\\Desktop\\Minor project\\Julia\\Amide_descriptors2.csv", DataFrame)
c = CSV.read("C:\\Users\\alex_\\Desktop\\Minor project\\Julia\\Amide_descriptors3.csv", DataFrame)
data1 = a[:,:]
data2 = b[:,:]
data3 = c[:,:]

## Reporting the column names that show high variation between runs

# fiddling for high variation
#data1[1,10] = 17100000         # Playing with a value
#data1[1,9] = 17100000         # Playing with a value
# fiddling for high variation


z = falses(size(data1,1),size(data1,2))
std_thres = 1
for j = 8:size(data1,2)
    for i = 1:size(data1,1)
        if ((isequal(data1[i,j],data2[i,j]) == false) || (isequal(data2[i,j],data3[i,j])==false) || (isequal(data3[i,j],data1[i,j])==false))
            if ((std(skipmissing([data1[i,j],data2[i,j],data3[i,j]]))) > std_thres)
                z[i,j] = true
            end
        end
    end
end

gg = sum(z, dims=1)
# Now gg has the values of high std. Non-zero values mean the column cannot be used, due to high variation.
bad1 = " "
for i = 1:length(gg)
    if gg[i]>0
        bad1_temp = names(data1, i)
        bad1 = vcat(bad1,bad1_temp)
    end
end
bad1 = bad1[2:length(bad1)]

println("High variation at: $(bad1)")

## Reporting the column names that show no change between the compounds


# fiddling for non changing variable
#for i = 1:size(data1,1)
#    data1[i,10] = 1
#end
# fiddling for non changing variable

z = falses(size(data1,1),size(data1,2))
for j = 8:size(data1,2)
    for i = 1:size(data1,1)
        z[i,j] = (skipmissing(data1[1,j])==skipmissing(data1[i,j]))
    end
end
gg = sum(z, dims=1)

bad2 = " "
for i = 1:length(gg)
    if gg[i] == size(data1,1)
        bad2_temp = names(data1, i)
        bad2 = vcat(bad2,bad2_temp)
    end
end
bad2 = bad2[2:length(bad2)]
println("Nothing changes at: $(bad2)")
# Let's keep them...
## Refining the columns

data1_not_bad1 = select(data1, Not(bad1))            # Removing the high variation descriptors
#data1_not_bad2 = select(data1_not_bad1, Not(bad2))    # Removing the descriptors which not change (!!) We decided not to ignore those descriptors,
                                                       # as they would be of value to investigate the applicability domain of other compounds outside the data

refined_data = data1_not_bad1

## Exctracting the useful descriptors
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Refined_GreekDataset_+ESI.csv", refined_data)

using BSON
nice_desc = names(refined_data)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Nice descriptors", nice_desc)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Nice descriptors", nice_desc)


## Normalization
# Descriptors 0<desc<10 are OK
# Above 1000 we divide them by 1000
