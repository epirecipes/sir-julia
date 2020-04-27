
````julia
using Gillespie
using Random
using Plots
using BenchmarkTools
````



````julia
function sir_rates(x,parms)
  (S,I,R) = x
  (β,γ) = parms
  N = S+I+R
  infection = β*S*I/N
  recovery = γ*I
  [infection,recovery]
end
sir_transitions = [[-1 1 0];[0 -1 1]]
````


````
2×3 Array{Int64,2}:
 -1   1  0
  0  -1  1
````



````julia
u0 = [999,1,0]
p = [0.5,0.25]
Random.seed!(1235)
tf = 50.0
````


````
50.0
````



````julia
sir_result = ssa(u0,sir_rates,sir_transitions,p,tf)
data = ssa_data(sir_result)
````


````
1509×4 DataFrames.DataFrame
│ Row  │ time     │ x1    │ x2    │ x3    │
│      │ Float64  │ Int64 │ Int64 │ Int64 │
├──────┼──────────┼───────┼───────┼───────┤
│ 1    │ 0.0      │ 999   │ 1     │ 0     │
│ 2    │ 0.236916 │ 998   │ 2     │ 0     │
│ 3    │ 1.07532  │ 997   │ 3     │ 0     │
│ 4    │ 2.16728  │ 997   │ 2     │ 1     │
│ 5    │ 2.25891  │ 997   │ 1     │ 2     │
│ 6    │ 2.96813  │ 996   │ 2     │ 2     │
│ 7    │ 3.38675  │ 995   │ 3     │ 2     │
⋮
│ 1502 │ 49.3619  │ 238   │ 22    │ 740   │
│ 1503 │ 49.415   │ 237   │ 23    │ 740   │
│ 1504 │ 49.485   │ 237   │ 22    │ 741   │
│ 1505 │ 49.559   │ 236   │ 23    │ 741   │
│ 1506 │ 49.6047  │ 235   │ 24    │ 741   │
│ 1507 │ 49.702   │ 235   │ 23    │ 742   │
│ 1508 │ 49.9326  │ 235   │ 22    │ 743   │
│ 1509 │ 50.0064  │ 235   │ 21    │ 744   │
````



````julia
plot(data[:,1],data[:,2])
plot!(data[:,1],data[:,3])
plot!(data[:,1],data[:,4])
````


![](figures/sir_continuous_jump_gillespie_5_1.png)

````julia
@benchmark ssa(u0,sir_rates,sir_transitions,p,tf)
````


````
BenchmarkTools.Trial: 
  memory estimate:  1.13 KiB
  allocs estimate:  18
  --------------
  minimum time:     499.000 ns (0.00% GC)
  median time:      162.399 μs (0.00% GC)
  mean time:        186.481 μs (11.98% GC)
  maximum time:     20.668 ms (97.53% GC)
  --------------
  samples:          10000
  evals/sample:     1
````




## Appendix
 
Computer Information:
```
Julia Version 1.4.0
Commit b8e9a9ecc6 (2020-03-21 16:36 UTC)
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
Environment:
  JULIA_EDITOR = "C:\Users\sdwfr\AppData\Local\atom\app-1.45.0\atom.exe"  -a
  JULIA_NUM_THREADS = 4

```

Package Information:

```
Status `~\.julia\environments\v1.4\Project.toml`
[46ada45e-f475-11e8-01d0-f70cc89e6671] Agents 3.0.0
[b19378d9-d87a-599a-927f-45f220a2c452] ArrayFire 1.0.6
[c52e3926-4ff0-5f6e-af25-54175e0327b1] Atom 0.12.10
[6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf] BenchmarkTools 0.5.0
[be33ccc6-a3ff-5ff2-a52e-74243cff1e17] CUDAnative 3.0.4
[3a865a2d-5b23-5a0f-bc46-62713ec82fae] CuArrays 2.0.1
[717857b8-e6f2-59f4-9121-6e50c889abd2] DSP 0.6.6
[2445eb08-9709-466a-b3fc-47e12bd697a2] DataDrivenDiffEq 0.2.0
[a93c6f00-e57d-5684-b7b6-d8193f3e46c0] DataFrames 0.20.2
[aae7a2af-3d4f-5e19-a356-7da93b79d9d0] DiffEqFlux 1.8.1
[41bf760c-e81c-5289-8e54-58b1f1f8abe2] DiffEqSensitivity 6.13.0
[6d1b261a-3be8-11e9-3f2f-0b112a9a8436] DiffEqTutorials 0.1.0
[0c46a032-eb83-5123-abaf-570d42b7fbaa] DifferentialEquations 6.13.0
[31c24e10-a181-5473-b8eb-7969acd0382f] Distributions 0.23.2
[634d3b9d-ee7a-5ddf-bec9-22491ea816e1] DrWatson 1.10.2
[587475ba-b771-5e3f-ad9e-33799f191a9c] Flux 0.10.4
[0c68f7d7-f131-5f86-a1c3-88cf8149b2d7] GPUArrays 3.1.0
[28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71] GR 0.48.0
[523d8e89-b243-5607-941c-87d699ea6713] Gillespie 0.1.0
[7073ff75-c697-5162-941a-fcdaad2a7d2a] IJulia 1.21.2
[e5e0dc1b-0480-54bc-9374-aad01c23163d] Juno 0.8.1
[961ee093-0014-501f-94e3-6117800e7a78] ModelingToolkit 3.0.2
[429524aa-4258-5aef-a3af-852621145aeb] Optim 0.20.6
[1dea7af3-3e70-54e6-95c3-0bf5283fa5ed] OrdinaryDiffEq 5.34.1
[91a5bcdd-55d7-5caf-9e0b-520d859cae80] Plots 1.0.12
[e6cf234a-135c-5ec9-84dd-332b85af5143] RandomNumbers 1.4.0
[c5292f4c-5179-55e1-98c5-05642aab7184] ResumableFunctions 0.5.1
[428bdadb-6287-5aa5-874b-9969638295fd] SimJulia 0.8.0
[05bca326-078c-5bf0-a5bf-ce7c7982d7fd] SimpleDiffEq 1.1.0
[f3b207a7-027a-5e70-b257-86293d7955fd] StatsPlots 0.14.5
[789caeaf-c7a9-5a7d-9973-96adeb23e2a0] StochasticDiffEq 6.19.2
[44d3d7a6-8a23-5bf8-98c5-b353f8df5ec9] Weave 0.9.4
[37e2e46d-f89d-539d-b4ee-838fcccc9c8e] LinearAlgebra
[cf7118a7-6976-5b1a-9a39-7adc72f591a4] UUIDs
```
