using CSV
using DataFrames
using ScikitLearn

using Plots
using LinearAlgebra

#using Clustering
using Distributions
import StatsBase as BS
import StatsPlots as sp
import PyPlot as pt
import Random as r

# using Conda
# Conda.pip("install","catboost")
using PyCall

using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
@sk_import ensemble: RandomForestRegressor
@sk_import ensemble: RandomForestClassifier
#@sk_import ensemble: ExtraTreesClassifier

@sk_import cluster: KMeans
jl = pyimport("joblib")

# cb=  pyimport("catboost")

#######################################
# Import the data

data = DataFrame(CSV.File("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/toxicity_data_fish_desc.csv"))

describe(data)
tox = hcat(data[!,"LC50[-LOG(mol/L)]"],data[!,"MONOISOTOPIC_MASS"])

des = Matrix(data[:,8:end])
#####################################
# Exploring the tox data

sp.histogram(tox[:,1],bins = 10,normalized=true,label=false)
sp.xlabel!("LC50(-log(mol/L))")
sp.ylabel!("Probability")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/hist.png")

#

sp.scatter(tox[:,2],tox[:,1],label=false)
sp.ylabel!("LC50(-log(mol/L))")
sp.xlabel!("Monoisotopic mass")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/scat.png")

#############################################################
# data cleanup
des[ismissing.(des) .== 1] .=0 # removing the missing values
ss = sum(des,dims=1)

ind_ss = findall(x -> x > 0,ss[:])


des = des[:,ss[:] .> 0]

m = mean(des,dims=1)
ind_m = findall(x -> x >= 100,m[:])
ind_mn = findall(x -> x < 100,m[:])

des1 = des[:,ind_m ] ./100
des2 = des[:,ind_mn ]
desf = hcat(des1,des2)

##############################################
# estimate the outcome of each model for the NORMAN chemicals


data_n = DataFrame(CSV.File("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/Norman_descriptors_raw.csv"))
des_n = Matrix(data_n[:,7:end])

des_n[ismissing.(des_n) .== 1] .=0 # removing the missing values

des_n = des_n[:,ind_ss]

des_n1 = des_n[:,ind_m ] ./100
des_n2 = des_n[:,ind_mn ]
des_nf = hcat(des_n1,des_n2)

#######################################
# Dividing the tox data

tox_ = hcat(tox[:,1],tox[:,2] ./100 )

mdl = KMeans(n_clusters=4, random_state=0).fit(tox_)
# jl.dump(mdl,"/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/K_means_clf.joblib")

a = predict(mdl,tox_)


#
sp.scatter(tox[a .==3,2],tox[a .==3,1],label="Very low toxicity",c=:green)
sp.scatter!(tox[a .==0,2],tox[a .==0,1],label="Low toxicity",c=:yellow)
sp.scatter!(tox[a .==2,2],tox[a .==2,1],label="Moderate toxicity",c=:darkorange)
sp.scatter!(tox[a .==1,2],tox[a .==1,1],label="High toxicity",c=:red)
sp.title!("Fish Toxicity")
sp.ylabel!("LC50(-log(mol/L))")
sp.xlabel!("Monoisotopic mass")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/scat_cat.png")

#

bns = BS.fit(BS.Histogram,tox[:,1])

pri_x = fit_mle(Gamma,tox[:,1])

#sp.plot!(0.5:1:9.5,bns.weights)
#sp.plot!(pri_x,label ="Gamma Distribution")

targ = 0:0.1:10
probs = cdf.(pri_x,targ)

tr = [0.1573,0.4986,0.8399,0.95]



ht = findfirst(x -> x>= tr[1],probs)
mt = findfirst(x -> x>= tr[2],probs)
lt = findfirst(x -> x>= tr[3],probs)
vlt = findfirst(x -> x>= tr[4],probs)

tox_cats = [targ[ht],targ[mt],targ[lt],targ[vlt]]

sp.scatter(tox[tox[:,1] .<= tox_cats[1],2],tox[tox[:,1] .<= tox_cats[1],1],label="High toxicity",c=:red)
sp.scatter!(tox[findall(x -> targ[ht] < x <= targ[mt],tox[:,1]),2],
tox[findall(x -> targ[ht] < x <= targ[mt],tox[:,1]),1],label="Moderate toxicity",c=:darkorange)

sp.scatter!(tox[findall(x -> targ[mt] < x <= targ[lt],tox[:,1]),2],
tox[findall(x -> targ[mt] < x <= targ[lt],tox[:,1]),1],label="Low toxicity",c=:yellow)

sp.scatter!(tox[findall(x -> targ[lt] < x <= targ[vlt],tox[:,1]),2],
tox[findall(x -> targ[lt] < x <= targ[vlt],tox[:,1]),1],label="Very low toxicity",c=:green)

sp.title!("Fish Toxicity")
sp.ylabel!("LC50(-log(mol/L))")
sp.xlabel!("Monoisotopic mass")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/scat_cat_gamma.png")


#########################################################
# Regerssion modeling

X = deepcopy(desf)
Y = tox[:,1]

reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=7,verbose =1,
 oob_score =true, max_features= "auto",n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42);

fit!(reg, X_train, y_train)
accuracy = score(reg, X_train, y_train)

# jl.dump(reg,"/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/reg_model.joblib",compress=5)
y_h = predict(reg,X_test);
y_h_tr = predict(reg,X_train);
cross_val_score(reg, X_train, y_train; cv=5)

fi_r_i = sortperm(reg.feature_importances_, rev=true)
fi_r = 100 .* sort(reg.feature_importances_, rev=true)
sig_f_r = fi_r_i[fi_r .>=1]

#
# sp.scatter(y_train,y_h_tr,label="training set", legend=:topleft)
# sp.scatter!(y_test,y_h,label="test set")

#

# # test catboost
# train_pool = cb.Pool(X_train,y_train)
# test_pool = cb.Pool(X_test,y_test)

# model = cb.CatBoostRegressor(iterations=500,learning_rate=0.4,loss_function="RMSE")
# model.fit(X_train,y_train,eval_set=(X_test, y_test))

# preds = model.predict(X_test)


# # plot cat

# sp.scatter(y_test,preds)




#
desf_ref_reg = desf[:,sig_f_r]

reg1 = RandomForestRegressor(n_estimators=600, min_samples_leaf=7,verbose =1,
 oob_score =true, max_features= size(desf_ref_reg,2),n_jobs=-1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(desf_ref_reg, Y, test_size=0.10, random_state=42);

fit!(reg1, X_train1, y_train1)
accuracy = score(reg1, X_train1, y_train1)

y_h1 = predict(reg1,X_test1);
y_h_tr1 = predict(reg1,X_train1);
cross_val_score(reg1, X_train1, y_train1; cv=5)

y_h_reg = predict(reg1,desf_ref_reg)

tox_h = hcat(y_h_reg,tox[:,2] ./100)

a_h = predict(mdl,tox_h)

dif_a = a - a_h

##




####
sp.scatter(tox[dif_a .== 0,2],tox[dif_a .== 0,1],label="Correct class", legend=:topleft)
sp.scatter!(tox[dif_a .!= 0,2],tox[dif_a .!= 0,1],label="Wrong class", c = :magenta)
sp.xlabel!("Monoisotopic mass")
sp.ylabel!("LC50")
sp.title!("Fish toxicity based on regression")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/clf_model_reg.png")



#

sp.scatter(tox[a_h .==3,2],tox[a_h .==3,1],label="Very low toxicity",c=:green)
sp.scatter!(tox[a_h .==0,2],tox[a_h .==0,1],label="Low toxicity",c=:yellow)
sp.scatter!(tox[a_h .==2,2],tox[a_h .==2,1],label="Moderate toxicity",c=:darkorange)
sp.scatter!(tox[a_h .==1,2],tox[a_h .==1,1],label="High toxicity",c=:red)

sp.title!("Fish Toxicity based on regression")
sp.ylabel!("LC50(-log(mol/L))")
sp.xlabel!("Monoisotopic mass")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/scat_cat_hat.png")


#
sp.scatter(y_train,y_h_tr,label="Training set",legend=:topleft)
sp.scatter!(y_test,y_h,label="Test set")
sp.plot!([0,10],[0,10],label="1:1 line",linecolor ="black")
sp.xlabel!("Measured LC50")
sp.ylabel!("Predicted LC50")
sp.title!("Fish toxicity regression full descriptors")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/reg_model.png")


#
sp.scatter(y_train1,y_h_tr1,label="Training set",legend=:topleft)
sp.scatter!(y_test1,y_h1,label="Test set")
sp.plot!([0,10],[0,10],label="1:1 line",linecolor ="black")
sp.xlabel!("Measured LC50")
sp.ylabel!("Predicted LC50")
sp.title!("Fish toxicity regression 11 descriptors")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/reg_model_1.png")

#


sp.bar(fi_r[1:11],xticks = (1:11 , names(data)[8:end][sig_f_r]),xrotation = 60,label=false)
sp.ylabel!("Precentage importance")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/reg_model_features.png")



####
# Classification model

Y_cl = deepcopy(a)

clf = RandomForestClassifier(n_estimators=1200, min_samples_leaf=5,verbose =1,
 oob_score =true, max_features= "auto",n_jobs=-1)

X_train, X_test, y_train_cl, y_test_cl = train_test_split(X, Y_cl, test_size=0.10, random_state=42);

fit!(clf, X_train, y_train_cl)

accuracy_cl = score(clf, X_train, y_train_cl)

y_h_cl = predict(clf,X_test);

cross_val_score(clf, X_train, y_train_cl; cv=5)

fi_c_i = sortperm(clf.feature_importances_, rev=true)
fi_c = 100 .* sort(clf.feature_importances_, rev=true)
sig_f_c = fi_c_i[fi_c .>=0.1]

#
y_h_tr_cl = predict(clf,X);
y_h_tr_cl_p = predict_proba(clf,X)

tr_clf = abs.(Y_cl - y_h_tr_cl)
#tr_clf[tr_clf .> 0] .= 1

sp.scatter(tox[tr_clf .== 0,2],tox[tr_clf .== 0,1],label="Correct class", legend=:topleft)
sp.scatter!(tox[tr_clf .!= 0,2],tox[tr_clf .!= 0,1],label="Wrong class", c = :magenta)
sp.xlabel!("Monoisotopic mass")
sp.ylabel!("LC50")
sp.title!("Fish toxicity based on descriptors")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/clf_model.png")


#


sp.scatter(tox[y_h_tr_cl .==3,2],tox[y_h_tr_cl .==3,1],label="Very low toxicity",c=:green)

sp.scatter!(tox[y_h_tr_cl .==0,2],tox[y_h_tr_cl .==0,1],label="Low toxicity",c=:yellow)
sp.scatter!(tox[y_h_tr_cl .==2,2],tox[y_h_tr_cl .==2,1],label="Moderate toxicity",c=:darkorange)
sp.scatter!(tox[y_h_tr_cl .==1,2],tox[y_h_tr_cl .==1,1],label="High toxicity",c=:red)
sp.title!("Fish Toxicity based on classification")
sp.ylabel!("LC50(-log(mol/L))")
sp.xlabel!("Monoisotopic mass")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/scat_cat_hat_clf.png")

##############################################
# Leverage calculations

function leverage_dist(X_train,itr)

    lev = zeros(itr)


    for i =1:itr
        ind = BS.sample(1:size(X_train,1))
        x = X_train[ind,:]
        lev[i] = transpose(x) * pinv(transpose(X_train) * X_train) * x
        println(i)
    end

    return lev


end

itr = 1000
lev = leverage_dist(X_train,itr)



################
# tox predict

function tox_pred(des_nf,X_train)
    tox_h = zeros(size(data_n,1),2)

    for i=1:size(data_n,1)
        try
            tv_x = zeros(2,size(des_nf,2))
            tv_x[1,:] = des_nf[i,:]
            tox_h[i,1] = predict(reg,tv_x)[1]; # regression model with full descriptors
            tox_h[i,2] = transpose(tv_x[1,:]) * pinv(transpose(X_train) * X_train) * tv_x[1,:]
            println(i)
        catch
            println("Something went wrong")
            println(i)
            continue
        end
    end

    return tox_h

end

y_h_norman = tox_pred(des_nf,X_train)


sp.scatter(y_h_norman[:,2],label=false)
ylims!(0,20)

rep_table = hcat(data_n[:,1:6],DataFrame(LC_50 =y_h_norman[:,1],Leverage = y_h_norman[:,2] ))

CSV.write("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/rep_table.csv",rep_table)

#

sp.histogram(lev,label=false)
sp.xlabel!("Leverages")
sp.ylabel!("Frequency")
sp.savefig("/Volumes/SAER HD/Data/QSAR_data/toxicity_fish_data/leverages.png")
