using DataFrames
using Statistics
using CSV
using Plots
using BSON
import StatsBase as BS
using ScikitLearn.CrossValidation: train_test_split

#Grid creation
up_gr = collect(5:10:250)
down_gr = up_gr
prcnt_increase_gr = collect(120:10:300)
up_am = collect(5:10:250)
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

pos_gr = [RI_mb_GR assessment_mb_gr]
pos_am = [RI_mb_AM assessment_mb_am]

ind_train, ind_test = train_test_split(collect(1:size(pos_gr,1)), test_size=0.20, random_state=21)

positive_cntrls_gr = pos_gr[ind_train,:]
positive_cntrls_am = pos_am[ind_train,:]

positive_cntrls_gr_test = pos_gr[ind_test,:]
positive_cntrls_am_test = pos_am[ind_test,:]

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

# Negative controls - CrippenLogP threshold <-4 , >+6
F_ = (Norm_raw[groupF,:])[:,[:SMILES, :MW, :CrippenLogP]]
F_.RI_AM = RI_norman_AM[groupF]
F_.RI_GR = RI_norman_GR[groupF]
F_sort = sort(F_,[order(:CrippenLogP)])

definitely_F = vcat(findall(x -> x < -4, F_sort.CrippenLogP), findall(x -> x > 6, F_sort.CrippenLogP))
definitely_F_train, definitely_F_test = train_test_split(definitely_F, test_size=0.20, random_state=21)

negative_cntrls_gr = F_sort[definitely_F_train,:RI_GR]
negative_cntrls_am = F_sort[definitely_F_train,:RI_AM]

negative_cntrls_gr_test = F_sort[definitely_F_test,:RI_GR]
negative_cntrls_am_test = F_sort[definitely_F_test,:RI_AM]

#
# Notes on meeting 19/09/2022
#Filter to the MW range of the training set
#<-4 and >+6 CrippenLogP -> pick 200 compounds for training the validation parameters set and 100 compounds for test set
# ROC

iterations = 20000
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
for i = 1:1:size(unique_false_positives,1)
    f_pos = unique_false_positives[i]
    df_data_temp = filter(row -> row.x2 == f_pos,df_data)
    df_data_temp_sorted = sort(df_data_temp, :x1, rev=true)
    max_points[i,:] = Vector(df_data_temp_sorted[1,:])
end

# Picking the max points (2nd try)
interval = 0.01

max_points_ = zeros(length(collect(0:interval:1)),6)
for i = 1:size(max_points_,1)
    j = (collect(0:interval:1))[i]
    k = j + interval
    ind1 = findall(x -> x >= j, df_data.x2)
    ind2 = findall(x -> x < k, df_data.x2)
    ind = intersect(ind1,ind2)
    if isempty(ind)==false
        max_ = findmax(df_data[ind,:x1])
        max_points_[i,:] = Vector((df_data[ind,:])[(findmax(df_data[ind,:x1])[2]),:])
    end
end
max_points_sorted = sort!(max_points_,dims=1)

#Picking the optimal point
#Accuracy as true positive ratio - false positive ratio
accuracy = df_data[:,1] .- df_data[:,2]
optimal_point = df_data[findmax(accuracy)[2],:]

# ROC
scatter(max_points[:,2], max_points[:,1], legend=false, xlims=[0,1],title="ROC Curve",ylims=[0,1], xaxis="False positives",yaxis="True positives")
savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\ROC7.png")

# ROC (2nd try)
plot(max_points_sorted[:,2], max_points_sorted[:,1],c=:black,linewidth=4, legend=false, xlims=[0,1],title="ROC Curve",ylims=[0,1.05], xaxis="False positives",yaxis="True positives")
scatter!(max_points_sorted[:,2], max_points_sorted[:,1], c=:cyan,markershape=:diamond,markersize=3, markerstrokewidth=1)
scatter!([optimal_point.x2],[optimal_point.x1], markershape=:star, c=:gold, markersize=6)
savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\ROC8.png")

# The whole ROC
scatter(data[:,2], data[:,1], legend=false, xlims=[0,1], ylims=[0,1], xaxis="False positives",yaxis="True positives", title="ROC 'curve'")


# Assessment validation with test sets

x1_gr = Vector(optimal_point)[3]
x2_gr = Vector(optimal_point)[4]
x1_am = Vector(optimal_point)[5]
x2_am = Vector(optimal_point)[6]
range_gr = [min_gr+x1_gr max_gr-x1_gr;min_gr+x2_gr max_gr-x2_gr]
range_am = [min_am+x1_am max_am-x1_am;min_am+x2_am max_am-x2_am]
# Calculation of positive test controls
z_pos = Int.(zeros(size(positive_cntrls_gr_test,1),2))
for j = 1:size(positive_cntrls_gr_test,1)
        # Outside the AD of UoA
    if positive_cntrls_gr_test[j,2] == 2
        if positive_cntrls_gr_test[j,1]>range_gr[2,1] && positive_cntrls_gr_test[j,1]<range_gr[2,2]
            z_pos[j,1] = 1
        end
        # Inside the AD of UoA
    elseif positive_cntrls_gr_test[j,2] == 1
        if positive_cntrls_gr_test[j,1]>range_gr[1,1] && positive_cntrls_gr_test[j,1]<range_gr[1,2]
            z_pos[j,1] = 1
        end
    end
        # Outside the AD of n-alkylamide
    if positive_cntrls_am_test[j,2] == 2
        if positive_cntrls_am_test[j,1]>range_am[2,1] && positive_cntrls_am_test[j,1]<range_am[2,2]
            z_pos[j,2] = 1
        end
        # Inside the AD of n-alkylamide
    elseif positive_cntrls_am_test[j,2] == 1
        if positive_cntrls_am_test[j,1]>range_am[1,1] && positive_cntrls_am_test[j,1]<range_am[1,2]
            z_pos[j,2] = 1
        end
    end
end
# Calculation of negative controls
z_neg = Int.(zeros(size(negative_cntrls_gr_test,1),2))
for k = 1:size(negative_cntrls_gr_test,1)
    # Outside the AD of UoA
    if negative_cntrls_gr_test[k]>range_gr[2,1] && negative_cntrls_gr_test[k]<range_gr[2,2]
        z_neg[k,1] = 1
    end
    # Outside the AD of n-alkylamide
    if negative_cntrls_am_test[k]>range_am[2,1] && negative_cntrls_am_test[k]<range_am[2,2]
        z_neg[k,2] = 1
    end
end
# True positives, false positive Calculation
true_positives = count(==(2),sum(z_pos,dims=2)[:])
TPR = true_positives/(size(z_pos,1))
false_positives = count(==(2),sum(z_neg,dims=2)[:])
FPR = false_positives/(size(z_neg,1))


ConfusionMatrix = DataFrame(Confusion_Matrix=["Predicted Positive","Predicted Negative"])
ConfusionMatrix.True_Positive = [true_positives, (size(z_pos,1))-true_positives]
ConfusionMatrix.True_Negative = [false_positives,(size(z_neg,1))-false_positives]
show(ConfusionMatrix)

accuracy = (100*(ConfusionMatrix[1,2]+ConfusionMatrix[2,3]))/(ConfusionMatrix[1,2]+ConfusionMatrix[1,3]+ConfusionMatrix[2,2]+ConfusionMatrix[2,3])


# Norman
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_GR", RI_norman_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_Norman_AM", RI_norman_AM)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_amide", assessment_am)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_greek", assessment_gr)
assessment_am[assessment_am .== 3] .= 2
assessment_gr[assessment_gr .== 3] .= 2

norman_gr = hcat(RI_norman_GR, assessment_gr)
norman_am = hcat(RI_norman_AM, assessment_am)

#= In case you don't want to run the previous code
optimal_point = [0.77, 0.3, 95, 275.5, 15, 25.5]
=#
x1_gr = Vector(optimal_point)[3]
x2_gr = Vector(optimal_point)[4]
x1_am = Vector(optimal_point)[5]
x2_am = Vector(optimal_point)[6]
range_gr = [min_gr+x1_gr max_gr-x1_gr;min_gr+x2_gr max_gr-x2_gr]
range_am = [min_am+x1_am max_am-x1_am;min_am+x2_am max_am-x2_am]


z_norman = Int.(zeros(size(norman_gr,1),2))
for j = 1:size(norman_gr,1)
        # Outside the AD of UoA
    if norman_gr[j,2] == 2
        if norman_gr[j,1]>range_gr[2,1] && norman_gr[j,1]<range_gr[2,2]
            z_norman[j,1] = 1
        end
        # Inside the AD of UoA
    elseif norman_gr[j,2] == 1
        if norman_gr[j,1]>range_gr[1,1] && norman_gr[j,1]<range_gr[1,2]
            z_norman[j,1] = 1
        end
    end
        # Outside the AD of n-alkylamide
    if norman_am[j,2] == 2
        if norman_am[j,1]>range_am[2,1] && norman_am[j,1]<range_am[2,2]
            z_norman[j,2] = 1
        end
        # Inside the AD of n-alkylamide
    elseif norman_am[j,2] == 1
        if norman_am[j,1]>range_am[1,1] && norman_am[j,1]<range_am[1,2]
            z_norman[j,2] = 1
        end
    end
end

pos_norm = count(==(2),sum(z_norman,dims=2)[:])
neg_norm = size(norman_gr,1) - pos_norm
col = sum(z_norman,dims=2)[:]

scatter!((norman_am[:,1])[col .== 2],(norman_gr[:,1])[col .== 2],c=:green4, label="Accepted",legend=:topleft, xaxis="n-alkylamide RI", yaxis = "UoA RI")
scatter((norman_am[:,1])[col .== 1],(norman_gr[:,1])[col .== 1], c=:orange3,label = "Maybe")
scatter!((norman_am[:,1])[col .== 0],(norman_gr[:,1])[col .== 0],c=:red3, label="Rejected")
savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Classification(3 groups)2.png")



using Conda
using PyCall
import PyPlot as plt

Conda.pip_interop(true)
#Conda.pip("install",["matplotlib_venn"])
#Conda.pip("install",["pyqt5"])
venn = pyimport("matplotlib_venn")

GR_not_AM = count(==(1),(z_norman[:,1]-z_norman[:,2]))
AM_not_GR = count(==(1),(z_norman[:,2]-z_norman[:,1]))
AM_and_GR = count(==(2),sum(z_norman,dims=2)[:])


Abc = 0
aBc = 0
ABc = 0
abC = size(z_norman,1) - AM_and_GR
AbC = 0
aBC = 0
ABC = AM_and_GR


plt.figure(figsize=(4,4))
v2 = venn.venn2(subsets = (abC, 0, AM_and_GR), set_labels = ("NORMAN"," "," "))

v2.get_patch_by_id("100").set_color("red")
v2.get_patch_by_id("100").set_alpha(0.9)
v2.get_label_by_id("010").set_text(" ")
v2.get_patch_by_id("110").set_color("green")
v2.get_patch_by_id("110").set_alpha(0.9)

plt.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Venn_simple_3.png")








# three venn
Abc = 0
aBc = 0
ABc = 0
abC = size(z_norman,1) - AM_not_GR - GR_not_AM - AM_and_GR
AbC = AM_not_GR
aBC = GR_not_AM
ABC = AM_and_GR

Abc = 0
aBc = 0
ABc = 0
abC = size(z_norman,1) - AM_and_GR
AbC = 0
aBC = 0
ABC = AM_and_GR

plt.clf()
v3 = venn.venn3(subsets = (Abc, aBc, ABc, abC, AbC, aBC, ABC), set_labels = (" ", " ", "NORMAN", "NORMAN", "AbC", "aBC", "ABC"))
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

plt.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Venn_5.png")


#AM_and_GR = length(total_sum[total_sum.==2])
#AM_not_GR = (sum(total_assessment,dims=1))[2] - AM_and_GR
#GR_not_AM = (sum(total_assessment,dims=1))[1] - AM_and_GR
