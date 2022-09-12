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

#Load data
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_GR", RI_mb_GR)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\RI_mb_AM", RI_mb_AM)

BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_amide", assessment_mb_am)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\AD_category_mb_greek", assessment_mb_gr)
assessment_mb_am[assessment_mb_am .== 3] .= 2
assessment_mb_gr[assessment_mb_gr .== 3] .= 2

positive_cntrls_gr = [RI_mb_GR assessment_mb_gr]
positive_cntrls_am = [RI_mb_AM assessment_mb_am]

maximum(RI_mb_AM)

# ROC
for i = 100

    x1_gr = BS.sample(up_gr)
    x2_gr =  (BS.sample(prcnt_increase_gr)/100) * x1_gr
    x1_am = BS.sample(up_am)
    x2_am =  (BS.sample(prcnt_increase_am)/100) * x1_am

    range_gr = [min_gr+x1_gr max_gr-x1_gr;min_gr+x2_gr max_gr-x2_gr]
    range_am = [min_am+x1_am max_am-x1_am;min_am+x2_am max_am-x2_am]

    z = zeros(aaa)



end
