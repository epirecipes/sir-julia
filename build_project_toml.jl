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
  "IJulia",
  "LabelledArrays",
  "Latexify",
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
  "Tables",
  "Turing",
  "Weave"
]

for p in packages
  Pkg.add(p)
end

pkg"instantiate"
