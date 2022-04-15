using Conda
using CSV
using DataFrames
using PyCall
using ScikitLearn

################################################################
# Installing PubChemPy

Conda.pip_interop(true)
Conda.pip("install",["pubchempy","padelpy"])
Pkg.build("PyCall")
# Conda.pip("install",["statsmodels"])
# stats = pyimport("statsmodels")

pcp = pyimport("pubchempy")

pd = pyimport("padelpy")



function padel_desc(rep)

    desc_p = DataFrame(pd.from_smiles(rep[1,4],fingerprints=true))
    desc_p = hcat( DataFrame(NAME = rep[1,2], RI = rep[1,5], SMILES = rep[1,4], FORMULA = rep[1,3]), desc_p )
    for i = 2:size(rep,2)

        try
            desc_p_temp = DataFrame(pd.from_smiles(rep[i,4],fingerprints=true))
            desc_p_temp = hcat( DataFrame(NAME = rep[i,2], RI = rep[i,5], SMILES = rep[i,4], FORMULA = rep[i,3]), desc_p_temp )
            desc_p = vcat(desc_p,desc_p_temp)
            println(i)

        catch
            continue
        end
    end

    return desc_p
end

rep = CSV.read("C:\\Users\\alex_\\Desktop\\Minor project\\Julia\\GreekDataset_+ESI.csv", DataFrame)

desc_p = padel_desc(rep[:,1:5])
CSV.write("C:\\Users\\alex_\\Desktop\\Minor project\\Julia\\GreekDataset_+ESI_Descriptors.csv", desc_p)
