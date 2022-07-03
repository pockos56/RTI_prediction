using Plots
using BSON
using Distributions
using CSV
using DataFrames
import StatsBase as BS
import StatsPlots as sp
import PyPlot as plt

z=zeros(6)
z_sort=zeros(6)
z_col=Vector{String}()
z_mark=Vector{String}()
for i = 1:6
    push!(z_col,"black")
    push!(z_mark,"circle")
end

for i = 1:6
    if i==1
        z[i] = 3
        z_sort[i] = 5948
        z_col[i] = "#e9a3c9"        #In both
        z_mark[i] = "diamond"
    elseif i==2
        z[i] = 2
        z_sort[i] = 4057
        z_col[i] = "#4575b4"
        z_mark[i] = "circle"  #In Amide
    elseif i==3
        z[i] = 1
        z_sort[i] = 20732
        z_col[i] = "#c7eae5"
        z_mark[i] = "pentagon"       #In UoA
    elseif i==4
        z[i] = 3
        z_sort[i] = 19
        z_col[i] = "#f1a340"        #Out of Amide AD
        z_mark[i] = "rect"
    elseif i==5
        z[i] = 2
        z_sort[i] = 521
        z_col[i] = "#d73027"        #Out of UoA AD
        z_mark[i] = "octagon"
    elseif i==6
        z[i] = 1
        z_sort[i] = 63837
        z_col[i] = "#8c510a"        #Out of both ADs
        z_mark[i] = "rect"
    end
end
marker = reshape(Symbol.(z_mark),length(z_mark),1)


scatter(vcat(zeros(3),(ones(3).-0.4)),z, color=z_col, legend=false,markershape=marker[:],markersize=35,
ticks=false,xlims=(-0.1,1.0),ylims=(0.45,3.6),grid=false,xaxis=false,yaxis=false)
annotate!(0.38, 1, text("Group C", :black, :right, 25))
annotate!(0.38, 2, text("Group B", :black, :right, 25))
annotate!(0.38, 3, text("Group A", :black, :right, 25))
annotate!(0.98, 1, text("Group F", :black, :right, 25))
annotate!(0.98, 2, text("Group E", :black, :right, 25))
annotate!(0.98, 3, text("Group D", :black, :right, 25))
sp.savefig("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Legend for classification scatter plot.png")





### Plot for 6-subplot histograms
Norm_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part1.csv", DataFrame)
for i = 2:20
    Norm_temp = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman big database\\Norm_descriptors_part$i.csv", DataFrame)
    Norm_raw = vcat(Norm_raw, Norm_temp)
    println("$i/20")
end


BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupA", groupA)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupB", groupB)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupC", groupC)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupD", groupD)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupE", groupE)
BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Index_GroupF", groupF)
Norm_raw_=deepcopy(Norm_raw)

inch = Norm_raw_[:,3]
inch[ismissing.(inch) .== 1] .= "WeMissYou"
Norm = convert.(String, inch)

com = Int.(zeros(20))
com[1] = findall(x -> x .== "RYYVLZVUVIJVGH-UHFFFAOYSA-N",Norm)[1]      #Caffeine
com[2] = findall(x -> x .== "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",Norm)[1]      #Aspirin
com[3] = findall(x -> x .== "LNEPOXFFQSENCJ-UHFFFAOYSA-N",Norm)[1]      #Haloperidol
com[4] = findall(x -> x .== "VGKDLMBJGBXTGI-SJCJKPOMSA-N",Norm)[1]      #Sertraline
com[5] = findall(x -> x .== "KAAZGXDPUNNEFN-UHFFFAOYSA-N",Norm)[1]      # Clothiapine
com[6] = findall(x -> x .== "HSUGRBWQSSZJOP-RTWAWAEBSA-N",Norm)[1]      # Diltiazem
com[7] = findall(x -> x .== "KWOLFJPFCHCOCG-UHFFFAOYSA-N",Norm)[1]      # Acetophenone
com[8] = findall(x -> x .== "YXFVVABEGXRONW-UHFFFAOYSA-N",Norm)[1]      # Toluene
com[9] = findall(x -> x .== "YNQLUTRBYVCPMQ-UHFFFAOYSA-N",Norm)[1]      # Ethyl benzene
com[10] = findall(x -> x .== "ZUOUZKKEUPVFJK-UHFFFAOYSA-N",Norm)[1]     # Biphenyl
com[11] = findall(x -> x .== "HPNMFZURTQLUMO-UHFFFAOYSA-N",Norm)[1]     # Diethylamine
com[12] = findall(x -> x .== "QGZKDVFQNNGYKY-UHFFFAOYSA-N",Norm)[1]     # Ammonia
com[13] = findall(x -> x .== "QORAVNMWUNPXAO-UHFFFAOYSA-N",Norm)[1]     # 2,2',4,4'-Tetrachlorobiphenyl
com[14] = findall(x -> x .== "YPZRWBKMTBYPTK-BJDJZHNGSA-N",Norm)[1]     # Glutathione disulfide
com[15] = findall(x -> x .== "GVGLGOZIDCSQPN-PVHGPHFFSA-N",Norm)[1]     # Heroin
com[16] = findall(x -> x .== "QIVBCDIJIAJPQS-VIFPVBQESA-N",Norm)[1]     # Tryptophan
com[17] = findall(x -> x .== "OCJBOOLMMGQPQU-UHFFFAOYSA-N",Norm)[1]     # 1,4-dichlorobenzen
com[18] = findall(x -> x .== "VHBFFQKBGNRLFZ-UHFFFAOYSA-N",Norm)[1]     # Flavone
com[19] = findall(x -> x .== "LNTHITQWFMADLM-UHFFFAOYSA-N",Norm)[1]     # Gallic acid
com[20] = findall(x -> x .== "IKGXIBQEEMLURG-NVPNHPEKSA-N",Norm)[1]     # Rutin

z = zeros(length(com),6)
for i = 1:length(com)
    z[i,1] = length(findall(x -> x .== com[i],groupA))
    z[i,2] = length(findall(x -> x .== com[i],groupB))
    z[i,3] = length(findall(x -> x .== com[i],groupC))
    z[i,4] = length(findall(x -> x .== com[i],groupD))
    z[i,5] = length(findall(x -> x .== com[i],groupE))
    z[i,6] = length(findall(x -> x .== com[i],groupF))
end
z_sum = Int.(sum(z,dims=1))

A_ = sort(Norm_raw[groupA,:],[order(:MW)])[:,[:SMILES, :MW]]
a1 = A_[Int(round(size(A_,1)*0.05)),:]      # No
a2 = A_[Int(round(size(A_,1)*0.45)),:]      # Yes
a3 = A_[Int(round(size(A_,1)*0.50)),:]      # Yes 10.1016/j.jep.2019.112371
a4 = A_[Int(round(size(A_,1)*0.56)),:]      # Yes 10.1080/00365510310000529
a5 = A_[Int(round(size(A_,1)*0.954)),:]     # No
B_ = sort(Norm_raw[groupB,:],[order(:MW)])[:,[:SMILES, :MW]]
b1 = B_[Int(round(size(B_,1)*0.05)),:]       # No, RPLC additive
b2 = B_[Int(round(size(B_,1)*0.453)),:]      # No
b3 = B_[Int(round(size(B_,1)*0.512)),:]      # No
b4 = B_[Int(round(size(B_,1)*0.553)),:]      # No
b5 = B_[Int(round(size(B_,1)*0.952)),:]      # Yes
C_ = sort(Norm_raw[groupC,:],[order(:MW)])[:,[:SMILES, :MW]]
c1 = C_[Int(round(size(C_,1)*0.052)),:]       # No
c2 = C_[Int(round(size(C_,1)*0.452)),:]       # Yes 10.1007/BF00262588
c3 = C_[Int(round(size(C_,1)*0.501)),:]       # Yes pubchem
c4 = C_[Int(round(size(C_,1)*0.553)),:]       # Yes pubchem
c5 = C_[Int(round(size(C_,1)*0.953)),:]       # No
D_ = sort(Norm_raw[groupD,:],[order(:MW)])[:,[:SMILES, :MW]]
d1 = D_[Int(round(size(D_,1)*0.05)),:]       #
d2 = D_[Int(round(size(D_,1)*0.45)),:]
d3 = D_[Int(round(size(D_,1)*0.50)),:]
d4 = D_[Int(round(size(D_,1)*0.55)),:]
d5 = D_[Int(round(size(D_,1)*0.95)),:]
d1[1]
E_ = sort(Norm_raw[groupE,:],[order(:MW)])[:,[:SMILES, :MW]]
e1 = E_[Int(round(size(E_,1)*0.05)),:]       #
e2 = E_[Int(round(size(E_,1)*0.45)),:]
e3 = E_[Int(round(size(E_,1)*0.50)),:]
e4 = E_[Int(round(size(E_,1)*0.55)),:]
e5 = E_[Int(round(size(E_,1)*0.95)),:]

F_ = sort(Norm_raw[groupF,:],[order(:MW)])[:,[:SMILES, :MW]]
f1 = F_[Int(round(size(F_,1)*0.06)),:]       #
f2 = F_[Int(round(size(F_,1)*0.45)),:]
f3 = F_[Int(round(size(F_,1)*0.50)),:]
f4 = F_[Int(round(size(F_,1)*0.55)),:]
f5 = F_[Int(round(size(F_,1)*0.95)),:]

a_mw = plot(Norm_raw[groupA,:].MW,seriestype=stephist,fill=true,title="Group A",bins=:auto)
plot!([a1[2],a1[2]],[0,202],c=:red,linewidth=2)
plot!([a2[2],a2[2]],[0,510],c=:red,linewidth=2)
plot!([a3[2],a3[2]],[0,433],c=:red,linewidth=2)
plot!([a4[2],a4[2]],[0,370],c=:red,linewidth=2)
plot!([a5[2],a5[2]],[0,52],c=:red,linewidth=2)
b_mw = plot(Norm_raw[groupB,:].MW,seriestype=stephist,fill=true,title="Group B",bins=50)
plot!([b1[2],b1[2]],[0,213],c=:red,linewidth=2)
plot!([b2[2],b2[2]],[0,192],c=:red,linewidth=2)
plot!([b3[2],b3[2]],[0,180],c=:red,linewidth=2)
plot!([b4[2],b4[2]],[0,133],c=:red,linewidth=2)
plot!([b5[2],b5[2]],[0,47],c=:red,linewidth=2)
c_mw = plot(Norm_raw[groupC,:].MW,seriestype=stephist,fill=true,title="Group C",bins=80,xlims=(50,600))
plot!([c1[2],c1[2]],[0,1040],c=:red,linewidth=2)
plot!([c2[2],c2[2]],[0,1720],c=:red,linewidth=2)
plot!([c3[2],c3[2]],[0,1480],c=:red,linewidth=2)
plot!([c4[2],c4[2]],[0,1480],c=:red,linewidth=2)
plot!([c5[2],c5[2]],[0,345],c=:red,linewidth=2)
d_mw = plot(Norm_raw[groupD,:].MW,seriestype=stephist,fill=true,title="Group D",bins=60)
plot!([d1[2],d1[2]],[0,17],c=:red,linewidth=2)
plot!([d2[2],d2[2]],[0,44],c=:red,linewidth=2)
plot!([d3[2],d3[2]],[0,44],c=:red,linewidth=2)
plot!([d4[2],d4[2]],[0,35],c=:red,linewidth=2)
plot!([d5[2],d5[2]],[0,51],c=:red,linewidth=2)
e_mw = plot(Norm_raw[groupE,:].MW,seriestype=stephist,fill=true,title="Group E",bins=70)
plot!([e1[2],e1[2]],[0,3],c=:red,linewidth=2)
plot!([e2[2],e2[2]],[0,1],c=:red,linewidth=2)
plot!([e3[2],e3[2]],[0,3],c=:red,linewidth=2)
plot!([e4[2],e4[2]],[0,3],c=:red,linewidth=2)
plot!([e5[2],e5[2]],[0,2],c=:red,linewidth=2)
f_=(Norm_raw[groupF,:].MW)
f_[ismissing.(f_) .==1] .= -500
f_mw = plot(f_,seriestype=stephist,fill=true,xlims=(0,1300),title="Group F",bins=180)
plot!([f1[2],f1[2]],[0,2250],c=:red,linewidth=2)
plot!([f2[2],f2[2]],[0,2490],c=:red,linewidth=2)
plot!([f3[2],f3[2]],[0,2515],c=:red,linewidth=2)
plot!([f4[2],f4[2]],[0,2535],c=:red,linewidth=2)
plot!([f5[2],f5[2]],[0,330],c=:red,linewidth=2)
a_logp = plot(Norm_raw[groupA,:].CrippenLogP,seriestype=stephist,fill=true,title="Group A",bins=:auto)
b_logp = plot(Norm_raw[groupB,:].CrippenLogP,seriestype=stephist,fill=true,title="Group B",bins=:auto)
c_logp = plot(Norm_raw[groupC,:].CrippenLogP,seriestype=stephist,fill=true,title="Group C",bins=:auto)
d_logp = plot(Norm_raw[groupD,:].CrippenLogP,seriestype=stephist,fill=true,title="Group D",bins=:auto)
e_logp = plot(Norm_raw[groupE,:].CrippenLogP,seriestype=stephist,fill=true,title="Group E",bins=70)
f_logp = plot(Norm_raw[groupF,:].CrippenLogP,seriestype=stephist,fill=true,title="Group F",bins=:auto)
l1 = @layout [z{0.1h};a b;c d; e f]
l2 = @layout [z{0.1h} z{0.1h};a b;c d; e f]
title1 = plot(title="Molecular weight",grid = false, showaxis = false,tick=false,bottom_margin = -50Plots.px)
title2 = plot(title="Crippen LogP",grid = false, showaxis = false,tick=false,bottom_margin = -50Plots.px)

plot(title1,title2,a_mw,a_logp,b_mw,b_logp,c_mw,c_logp,legend=false, c=:blue3, layout=l2,dpi=600,size=(900,600))
plot(title1,title2,d_mw,d_logp,e_mw,e_logp,f_mw,f_logp,legend=false, c=:blue3, layout=l2,dpi=600,size=(900,600))

plot(title1,a_mw,b_mw,c_mw,d_mw,e_mw,f_mw,legend=false, c=:blue3, layout=l1,dpi=600,size=(900,600))
plot(title2,a_logp,b_logp,c_logp,d_logp,e_logp,f_logp,legend=false, c=:blue3, layout=l1,dpi=600,size=(900,600))
