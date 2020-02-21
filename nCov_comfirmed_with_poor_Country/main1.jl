using DataFrames
using CSV

A=DataFrame(CSV.read("nCov_comfirmed_with_poor_Country\\DXYArea.csv"))
describe(A)
