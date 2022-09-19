using DataFrames
using Statistics
using CSV
using Plots
using BSON
import StatsBase as BS

#Grid creation
up_gr = collect(5:5:150)
down_gr = up_gr
prcnt_increase_gr = collect(120:10:300)
up_am = collect(5:5:150)
down_am = up_am
prcnt_increase_am = collect(120:10:300)

min_gr = 30
max_gr = 910
min_am = 255
max_am = 969

# Positive controls
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_GR", RI_mb_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_AM", RI_mb_AM)

BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_amide", assessment_mb_am)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_greek", assessment_mb_gr)
assessment_mb_am[assessment_mb_am .== 3] .= 2
assessment_mb_gr[assessment_mb_gr .== 3] .= 2

positive_cntrls_gr = [RI_mb_GR assessment_mb_gr]
positive_cntrls_am = [RI_mb_AM assessment_mb_am]

# Negative controls
Norm_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part1.csv", DataFrame)
for i = 2:20
    Norm_temp = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part$i.csv", DataFrame)
    Norm_raw = vcat(Norm_raw, Norm_temp)
    println("$i/20")
end
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_GR", RI_norman_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_AM", RI_norman_AM)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupF", groupF)

# Negative controls - Selected MWs
#=
F_ = (Norm_raw[groupF,:])[:,[:SMILES, :MW, :CrippenLogP]]
F_.RI_AM = RI_norman_AM[groupF]
F_.RI_GR = RI_norman_GR[groupF]
F_sort = sort(F_,[order(:MW)])

f_test_ = F_[4,:]
f_test_[1]

definitely_F = [4,5,638,2022,12767,14683,32876,60100]           # Please add more to this list!!!

negative_cntrls_gr = F_sort[definitely_F,:RI_GR]
negative_cntrls_am = F_sort[definitely_F,:RI_AM]
=#
# Negative controls - CrippenLogP threshold <-4 , >+6
F_ = (Norm_raw[groupF,:])[:,[:SMILES, :MW, :CrippenLogP]]
F_.RI_AM = RI_norman_AM[groupF]
F_.RI_GR = RI_norman_GR[groupF]
F_sort = sort(F_,[order(:CrippenLogP)])

definitely_F = vcat(findall(x -> x < -4, F_sort.CrippenLogP), findall(x -> x > 6, F_sort.CrippenLogP))

negative_cntrls_gr = F_sort[definitely_F,:RI_GR]
negative_cntrls_am = F_sort[definitely_F,:RI_AM]
#
# Notes on meeting 19/09/2022
#Filter to the MW range of the training set
#<-4 and >+6 CrippenLogP -> pick 200 compounds for training the validation parameters set and 100 compounds for test set
# ROC

iterations = 10000
data = zeros(iterations,6)
for i = 1:iterations
    x1_gr = BS.sample(up_gr)
    x2_gr =  (BS.sample(prcnt_increase_gr)/100) * x1_gr
    x1_am = BS.sample(up_am)
    x2_am =  (BS.sample(prcnt_increase_am)/100) * x1_am

    range_gr = [min_gr+x1_gr max_gr-x1_gr;min_gr+x2_gr max_gr-x2_gr]
    range_am = [min_am+x1_am max_am-x1_am;min_am+x2_am max_am-x2_am]

    # Calculation of positive controls
    z_pos = Int.(zeros(size(positive_cntrls_gr,1),2))
    for j = 1:size(positive_cntrls_gr,1)
            # Outside the AD of UoA
        if positive_cntrls_gr[j,2] == 2
            if positive_cntrls_gr[j,1]>range_gr[2,1] && positive_cntrls_gr[j,1]<range_gr[2,2]
                z_pos[j,1] = 1
            end
            # Inside the AD of UoA
        elseif positive_cntrls_gr[j,2] == 1
            if positive_cntrls_gr[j,1]>range_gr[1,1] && positive_cntrls_gr[j,1]<range_gr[1,2]
                z_pos[j,1] = 1
            end
        end
            # Outside the AD of n-alkylamide
        if positive_cntrls_am[j,2] == 2
            if positive_cntrls_am[j,1]>range_am[2,1] && positive_cntrls_am[j,1]<range_am[2,2]
                z_pos[j,2] = 1
            end
            # Inside the AD of n-alkylamide
        elseif positive_cntrls_am[j,2] == 1
            if positive_cntrls_am[j,1]>range_am[1,1] && positive_cntrls_am[j,1]<range_am[1,2]
                z_pos[j,2] = 1
            end
        end
    end

    # Calculation of negative controls
    z_neg = Int.(zeros(size(negative_cntrls_gr,1),2))
    for k = 1:size(negative_cntrls_gr,1)
        # Outside the AD of UoA
        if negative_cntrls_gr[k]>range_gr[2,1] && negative_cntrls_gr[k]<range_gr[2,2]
            z_neg[k,1] = 1
        end
        # Outside the AD of n-alkylamide
        if negative_cntrls_am[k]>range_am[2,1] && negative_cntrls_am[k]<range_am[2,2]
            z_neg[k,2] = 1
        end
    end

# True positives, false positive Calculation
true_positives = count(==(2),sum(z_pos,dims=2)[:])
data[i,1] = true_positives/(size(z_pos,1))

false_positives = count(==(2),sum(z_neg,dims=2)[:])
data[i,2] = false_positives/(size(z_neg,1))

data[i,3] = x1_gr
data[i,4] = x2_gr
data[i,5] = x1_am
data[i,6] = x2_am

end

# Picking the max points
df_data = DataFrame(data, :auto)
unique_false_positives = sort(unique(df_data[:,2]))

max_points = zeros(length(unique_false_positives),6)
for i = 1:5:length(unique_false_positives)
    f_pos = unique_false_positives[i]
    df_data_temp = filter(row -> row.x2 == f_pos,df_data)
    df_data_temp_sorted = sort(df_data_temp, :x1, rev=true)
    max_points[i,:] = Vector(df_data_temp_sorted[1,:])
end

# ROC
plot(max_points[:,2], max_points[:,1], legend=false, xlims=[0,1], ylims=[0,1], xaxis="False positives",yaxis="True positives")
scatter!(max_points[:,2], max_points[:,1], title="ROC Curve")
savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\ROC.png")

# The whole ROC
scatter(data[:,2], data[:,1], legend=false, xlims=[0,1], ylims=[0,1], xaxis="False positives",yaxis="True positives", title="ROC 'curve'")
