using Pkg

pkg"activate ."

packages = [
  "Agents",
  "AlgebraicPetri",
  "ApproxBayes",
  "BenchmarkTools",
  "BlackBoxOptim",
  "Bridge",
  "Catalyst",
  "Catlab",
  "DataFrames",
  "DiffEqCallbacks",
  "DiffEqJump",
  "DiffEqParamEstim",
  "DiffEqSensitivity",
  "DifferentialEquations",
  "Distances",
  "Distributions",
  "DrWatson",
  "ForwardDiff",
  "GpABC",
  "LabelledArrays",
  "LinearAlgebra",
  "MCMCChains",
  "ModelingToolkit",
  "NestedSamplers",
  "NLopt",
  "Optim",
  "OrdinaryDiffEq",
  "Petri",
  "PyCall",
  "Plots",
  "Random",
  "ResumableFunctions",
  "SimJulia",
  "SimpleDiffEq",
  "Soss",
  "SparseArrays",
  "StaticArrays",
  "StatsBase",
  "StatsPlots",
  "StochasticDiffEq",
  "Turing"
]

for p in packages
  Pkg.add(p)
end

unregistered = [
   ("https://github.com/sdwfrost/Gillespie.jl","master"),
   ("https://github.com/augustinas1/MomentClosure.jl","main")
]

for u in unregistered
  Pkg.add(PackageSpec(url=u[1], rev=u[2]))
end

pkg"instantiate"
