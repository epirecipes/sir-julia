# Stochastic differential equation model using AlgebraicPetri.jl
Micah Halter (@mehalter), 2020-07-13

## Introduction

This implementation defines the model as the composition of two interactions defined at domain-level semantics, transmission and recovery, simulated using `StochasticDiffEq.jl`.

## Libraries

````julia
using AlgebraicPetri.Epidemiology
using Petri
using Catlab.Theories
using Catlab.CategoricalAlgebra.ShapeDiagrams
using Catlab.Graphics
using StochasticDiffEq
using Random
using DataFrames
using StatsPlots
using BenchmarkTools
import Base: ≤

# helper function to visualize categorical representation
display_wd(ex) = to_graphviz(ex, orientation=LeftToRight, labels=true);
````


````
display_wd (generic function with 1 method)
````





## Transitions

Using the categorical framework provided by the AlgebraicJulia environment, we
can think of building models as the combination of known building block open
models.  For example we have $transmission: S \otimes I \rightarrow I$ and
$recovery: I \rightarrow R$ interactions defined in the Epidemiology module of
AlgebraicPetri which can be visualized as the following Petri nets.

Transmission (where $S_1 = S$ and $S_2 = I$):

````julia
Graph(decoration(F_epi(transmission)))
````


![](figures/sde_algebraicpetri_2_1.svg)



Recovery (where $S_1 = I$ and $S_2 = R$):

````julia
Graph(decoration(F_epi(recovery)))
````


![](figures/sde_algebraicpetri_3_1.svg)



In this approach we can think of an sir model as the composition of transmission
and recovery. This allows us to define the relationship that the infected
population coming out of the transmission interaction is the same as population
of infected in the recovery interaction.

````julia
sir_wiring_diagram = transmission ⋅ recovery
display_wd(sir_wiring_diagram)
````


![](figures/sde_algebraicpetri_4_1.svg)



using the function `F_epi` provided by `AlgebraicPetri.Epidemiology`, we can
convert this categorical definition of SIR to the Petri net representation and
visualize the newly created model (where $S_1 = S$, $S_2 = I$, and $S_3 = R$).

````julia
sir_model = decoration(F_epi(sir_wiring_diagram));
Graph(sir_model)
````


![](figures/sde_algebraicpetri_5_1.svg)



## Time domain

````julia
δt = 0.1
tmax = 40.0
tspan = (0.0,tmax)
t = 0.0:δt:tmax;
````


````
0.0:0.1:40.0
````





## Initial conditions

````julia
u0 = [990.0,10.0,0.0]; # S,I,R
````


````
3-element Array{Float64,1}:
 990.0
  10.0
   0.0
````





## Parameter values

````julia
p = [0.05*10.0/sum(u0),0.25]; # β*c/N,γ
````


````
2-element Array{Float64,1}:
 0.0005
 0.25
````





## Random number seed

````julia
Random.seed!(1234);
````


````
Random.MersenneTwister(UInt32[0x000004d2], Random.DSFMT.DSFMT_state(Int32[-
1393240018, 1073611148, 45497681, 1072875908, 436273599, 1073674613, -20437
16458, 1073445557, -254908435, 1072827086  …  -599655111, 1073144102, 36765
5457, 1072985259, -1278750689, 1018350124, -597141475, 249849711, 382, 0]),
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, 0.0, 0.0], UInt128[0x00000000000000000000000000000000, 
0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x0
0000000000000000000000000000000, 0x00000000000000000000000000000000, 0x0000
0000000000000000000000000000, 0x00000000000000000000000000000000, 0x0000000
0000000000000000000000000, 0x00000000000000000000000000000000, 0x0000000000
0000000000000000000000  …  0x00000000000000000000000000000000, 0x0000000000
0000000000000000000000, 0x00000000000000000000000000000000, 0x0000000000000
0000000000000000000, 0x00000000000000000000000000000000, 0x0000000000000000
0000000000000000, 0x00000000000000000000000000000000, 0x0000000000000000000
0000000000000, 0x00000000000000000000000000000000, 0x0000000000000000000000
0000000000], 1002, 0)
````





## Running the model

````julia
prob_sde,cb = SDEProblem(sir_model,u0,tspan,p)
````


````
(SDEProblem with uType Array{Float64,1} and tType Float64. In-place: true
timespan: (0.0, 40.0)
u0: [990.0, 10.0, 0.0], DiffEqBase.CallbackSet{Tuple{DiffEqBase.ContinuousC
allback{Petri.var"#4#6"{Int64},Petri.var"#5#7"{Int64},Petri.var"#5#7"{Int64
},typeof(DiffEqBase.INITIALIZE_DEFAULT),Float64,Int64,Nothing,Int64},DiffEq
Base.ContinuousCallback{Petri.var"#4#6"{Int64},Petri.var"#5#7"{Int64},Petri
.var"#5#7"{Int64},typeof(DiffEqBase.INITIALIZE_DEFAULT),Float64,Int64,Nothi
ng,Int64},DiffEqBase.ContinuousCallback{Petri.var"#4#6"{Int64},Petri.var"#5
#7"{Int64},Petri.var"#5#7"{Int64},typeof(DiffEqBase.INITIALIZE_DEFAULT),Flo
at64,Int64,Nothing,Int64}},Tuple{}}((DiffEqBase.ContinuousCallback{Petri.va
r"#4#6"{Int64},Petri.var"#5#7"{Int64},Petri.var"#5#7"{Int64},typeof(DiffEqB
ase.INITIALIZE_DEFAULT),Float64,Int64,Nothing,Int64}(Petri.var"#4#6"{Int64}
(1), Petri.var"#5#7"{Int64}(1), Petri.var"#5#7"{Int64}(1), DiffEqBase.INITI
ALIZE_DEFAULT, nothing, true, 10, Bool[1, 1], 1, 2.220446049250313e-15, 0),
 DiffEqBase.ContinuousCallback{Petri.var"#4#6"{Int64},Petri.var"#5#7"{Int64
},Petri.var"#5#7"{Int64},typeof(DiffEqBase.INITIALIZE_DEFAULT),Float64,Int6
4,Nothing,Int64}(Petri.var"#4#6"{Int64}(2), Petri.var"#5#7"{Int64}(2), Petr
i.var"#5#7"{Int64}(2), DiffEqBase.INITIALIZE_DEFAULT, nothing, true, 10, Bo
ol[1, 1], 1, 2.220446049250313e-15, 0), DiffEqBase.ContinuousCallback{Petri
.var"#4#6"{Int64},Petri.var"#5#7"{Int64},Petri.var"#5#7"{Int64},typeof(Diff
EqBase.INITIALIZE_DEFAULT),Float64,Int64,Nothing,Int64}(Petri.var"#4#6"{Int
64}(3), Petri.var"#5#7"{Int64}(3), Petri.var"#5#7"{Int64}(3), DiffEqBase.IN
ITIALIZE_DEFAULT, nothing, true, 10, Bool[1, 1], 1, 2.220446049250313e-15, 
0)), ()))
````



````julia
sol_sde = solve(prob_sde,SRA1(),callback=cb);
````


````
retcode: Success
Interpolation: 1st order linear
t: 2108-element Array{Float64,1}:
  0.0
  0.0013173816394377667
  0.0015838115019839506
  0.0018835450973484075
  0.0022207453921334213
  0.002576889681198567
  0.002576889681198567
  0.0029562400128317077
  0.003240279361828909
  0.003240279361828909
  ⋮
 39.79281141697257
 39.822382942705346
 39.851969227514395
 39.881457484413836
 39.91087533608831
 39.940318226269966
 39.969758810674364
 39.99927648619817
 40.0
u: 2108-element Array{Array{Float64,1},1}:
 [990.0, 10.0, 0.0]
 [990.0859345461865, 9.95321033902613, -0.03914488521257496]
 [990.0514533513923, 9.993085229668788, -0.04453858106104694]
 [990.0406143750769, 10.023496650540741, -0.06411102561763854]
 [989.9730420405374, 10.093050784941472, -0.0660928254789496]
 [989.9783822174255, 10.021617782574411, -2.067857295679603e-17]
 [989.9783822174255, 10.021617782574411, 0.0]
 [989.9357790738744, 10.046993369070398, 0.017227557055194503]
 [989.934906879975, 10.06509312002507, 7.436012914865576e-18]
 [989.934906879975, 10.06509312002507, 0.0]
 ⋮
 [186.71420001316457, 14.109251581947674, 799.1765484048881]
 [186.53810812577032, 14.36193825734913, 799.0999536168808]
 [186.12186399727423, 14.492659032500303, 799.3854769702258]
 [186.08661580654177, 14.410003683365685, 799.5033805100928]
 [186.11122861731818, 13.919652186746255, 799.9691191959358]
 [185.6828440226982, 14.06454484850224, 800.2526111287998]
 [185.7675904261989, 13.575216287484897, 800.6571932863164]
 [186.09146752621405, 13.335342152787822, 800.5731903209983]
 [186.05372644027543, 13.447440794629566, 800.4988327650951]
````





## Post-processing

We can convert the output to a dataframe for convenience.

````julia
df_sde = DataFrame(sol_sde(t)')
df_sde[!,:t] = t;
````


````
0.0:0.1:40.0
````





## Plotting

We can now plot the results.

````julia
@df df_sde plot(:t,
    [:x1 :x2 :x3],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")
````


![](figures/sde_algebraicpetri_13_1.png)



## Benchmarking

````julia
@benchmark solve(prob_sde,SRA1(),callback=cb)
````


````
BenchmarkTools.Trial: 
  memory estimate:  1.42 MiB
  allocs estimate:  73699
  --------------
  minimum time:     4.031 ms (0.00% GC)
  median time:      13.809 ms (0.00% GC)
  mean time:        14.411 ms (3.46% GC)
  maximum time:     28.718 ms (29.98% GC)
  --------------
  samples:          347
  evals/sample:     1
````




## Appendix
### Computer Information
```
Julia Version 1.4.2
Commit 44fa15b150* (2020-05-23 18:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
Environment:
  JULIA_HOME = /home/micah/.local/share/julia
  JULIA_LOAD_PATH = :
  JULIA_DEPOT_PATH = /home/micah/Documents/git/sir-julia/env/.julia
  SPACESHIP_JULIA_SYMBOL = ∴

```

### Package Information

```
Status `~/Documents/git/sir-julia/Project.toml`
[46ada45e-f475-11e8-01d0-f70cc89e6671] Agents 3.2.1
[4f99eebe-17bf-4e98-b6a1-2c4f205a959b] AlgebraicPetri 0.3.1
[b19378d9-d87a-599a-927f-45f220a2c452] ArrayFire 1.0.6
[c52e3926-4ff0-5f6e-af25-54175e0327b1] Atom 0.12.16
[6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf] BenchmarkTools 0.5.0
[be33ccc6-a3ff-5ff2-a52e-74243cff1e17] CUDAnative 3.2.0
[134e5e36-593f-5add-ad60-77f754baafbe] Catlab 0.7.1
[3a865a2d-5b23-5a0f-bc46-62713ec82fae] CuArrays 2.2.2
[717857b8-e6f2-59f4-9121-6e50c889abd2] DSP 0.6.7
[2445eb08-9709-466a-b3fc-47e12bd697a2] DataDrivenDiffEq 0.2.0
[a93c6f00-e57d-5684-b7b6-d8193f3e46c0] DataFrames 0.21.4
[459566f4-90b8-5000-8ac3-15dfb0a30def] DiffEqCallbacks 2.13.3
[aae7a2af-3d4f-5e19-a356-7da93b79d9d0] DiffEqFlux 1.17.0
[c894b116-72e5-5b58-be3c-e6d8d4ac2b12] DiffEqJump 6.9.3
[41bf760c-e81c-5289-8e54-58b1f1f8abe2] DiffEqSensitivity 6.23.0
[6d1b261a-3be8-11e9-3f2f-0b112a9a8436] DiffEqTutorials 0.1.0
[0c46a032-eb83-5123-abaf-570d42b7fbaa] DifferentialEquations 6.15.0
[31c24e10-a181-5473-b8eb-7969acd0382f] Distributions 0.23.4
[634d3b9d-ee7a-5ddf-bec9-22491ea816e1] DrWatson 1.14.3
[587475ba-b771-5e3f-ad9e-33799f191a9c] Flux 0.10.4
[0c68f7d7-f131-5f86-a1c3-88cf8149b2d7] GPUArrays 3.4.1
[28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71] GR 0.50.1
[523d8e89-b243-5607-941c-87d699ea6713] Gillespie 0.1.0
[7073ff75-c697-5162-941a-fcdaad2a7d2a] IJulia 1.21.2
[e5e0dc1b-0480-54bc-9374-aad01c23163d] Juno 0.8.2
[961ee093-0014-501f-94e3-6117800e7a78] ModelingToolkit 3.13.0
[429524aa-4258-5aef-a3af-852621145aeb] Optim 0.22.0
[1dea7af3-3e70-54e6-95c3-0bf5283fa5ed] OrdinaryDiffEq 5.41.0
[4259d249-1051-49fa-8328-3f8ab9391c33] Petri 1.1.0
[91a5bcdd-55d7-5caf-9e0b-520d859cae80] Plots 1.5.4
[e6cf234a-135c-5ec9-84dd-332b85af5143] RandomNumbers 1.4.0
[c5292f4c-5179-55e1-98c5-05642aab7184] ResumableFunctions 0.5.1
[428bdadb-6287-5aa5-874b-9969638295fd] SimJulia 0.8.0
[05bca326-078c-5bf0-a5bf-ce7c7982d7fd] SimpleDiffEq 1.1.0
[2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91] StatsBase 0.33.0
[f3b207a7-027a-5e70-b257-86293d7955fd] StatsPlots 0.14.6
[789caeaf-c7a9-5a7d-9973-96adeb23e2a0] StochasticDiffEq 6.24.0
[44d3d7a6-8a23-5bf8-98c5-b353f8df5ec9] Weave 0.10.2
[37e2e46d-f89d-539d-b4ee-838fcccc9c8e] LinearAlgebra
[cf7118a7-6976-5b1a-9a39-7adc72f591a4] UUIDs
```
