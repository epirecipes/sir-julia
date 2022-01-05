# Discrete event simulation using SimJulia
Simon Frost (@sdwfrost), 2020-04-27

## Libraries

```julia
using ResumableFunctions
using SimJulia
using Distributions
using DataFrames
using Random
using StatsPlots
using BenchmarkTools
```




## Utility functions

```julia
function increment!(a::Array{Int64})
    push!(a,a[length(a)]+1)
end

function decrement!(a::Array{Int64})
    push!(a,a[length(a)]-1)
end

function carryover!(a::Array{Int64})
    push!(a,a[length(a)])
end;
```




## Transitions

```julia
mutable struct SIRPerson
    id::Int64 # numeric ID
    status::Symbol # :S,I,R
end;
```


```julia
mutable struct SIRModel
    sim::Simulation
    β::Float64
    c::Float64
    γ::Float64
    ta::Array{Float64}
    Sa::Array{Int64}
    Ia::Array{Int64}
    Ra::Array{Int64}
    allIndividuals::Array{SIRPerson}
end;
```




These functions update the state of the 'world' when either an infection or recovery occurs.

```julia
function infection_update!(sim::Simulation,m::SIRModel)
    push!(m.ta,now(sim))
    decrement!(m.Sa)
    increment!(m.Ia)
    carryover!(m.Ra)
end;
```


```julia
function recovery_update!(sim::Simulation,m::SIRModel)
    push!(m.ta,now(sim))
    carryover!(m.Sa)
    decrement!(m.Ia)
    increment!(m.Ra)
end;
```




The following is the main simulation function. It's not efficient, as it involves activating a process for all susceptibles; a more efficient algorithm would involve just considering infected individuals, and activating each susceptible individual when infection occurs. This however requires more bookkeeping and detracts from the ability to easily compare between implementations.

```julia
@resumable function live(sim::Simulation, individual::SIRPerson, m::SIRModel)
  while individual.status==:S
      # Wait until next contact
      @yield timeout(sim,rand(Distributions.Exponential(1/m.c)))
      # Choose random alter
      alter=individual
      while alter==individual
          N=length(m.allIndividuals)
          index=rand(Distributions.DiscreteUniform(1,N))
          alter=m.allIndividuals[index]
      end
      # If alter is infected
      if alter.status==:I
          infect = rand(Distributions.Uniform(0,1))
          if infect < m.β
              individual.status=:I
              infection_update!(sim,m)
          end
      end
  end
  if individual.status==:I
      # Wait until recovery
      @yield timeout(sim,rand(Distributions.Exponential(1/m.γ)))
      individual.status=:R
      recovery_update!(sim,m)
  end
end;
```


```julia
function MakeSIRModel(u0,p)
    (S,I,R) = u0
    N = S+I+R
    (β,c,γ) = p
    sim = Simulation()
    allIndividuals=Array{SIRPerson,1}(undef,N)
    for i in 1:S
        p=SIRPerson(i,:S)
        allIndividuals[i]=p
    end
    for i in (S+1):(S+I)
        p=SIRPerson(i,:I)
        allIndividuals[i]=p
    end
    for i  in (S+I+1):N
        p=SIRPerson(i,:R)
        allIndividuals[i]=p
    end
    ta=Array{Float64,1}(undef,0)
    push!(ta,0.0)
    Sa=Array{Int64,1}(undef,0)
    push!(Sa,S)
    Ia=Array{Int64,1}(undef,0)
    push!(Ia,I)
    Ra=Array{Int64,1}(undef,0)
    push!(Ra,R)
    SIRModel(sim,β,c,γ,ta,Sa,Ia,Ra,allIndividuals)
end;
```


```julia
function activate(m::SIRModel)
     [@process live(m.sim,individual,m) for individual in m.allIndividuals]
end;
```


```julia
function sir_run(m::SIRModel,tf::Float64)
    SimJulia.run(m.sim,tf)
end;
```


```julia
function out(m::SIRModel)
    result = DataFrame()
    result[!,:t] = m.ta
    result[!,:S] = m.Sa
    result[!,:I] = m.Ia
    result[!,:R] = m.Ra
    result
end;
```




## Time domain

```julia
tmax = 40.0;
```




## Initial conditions

```julia
u0 = [990,10,0];
```




## Parameter values

```julia
p = [0.05,10.0,0.25];
```




## Random number seed

```julia
Random.seed!(1234);
```




## Running the model

```julia
des_model = MakeSIRModel(u0,p)
activate(des_model)
sir_run(des_model,tmax)
```




## Postprocessing

```julia
data_des=out(des_model);
```




## Plotting

```julia
@df data_des plot(:t,
    [:S :I :R],
    labels = ["S" "I" "R"],
    xlab="Time",
    ylab="Number")
```

![](figures/des_18_1.png)



## A model with a fixed delay

One of the advantages of a discrete event framework is that it is straightforward to include different distributions for the transitions between states. We can update the above example to include a fixed delay using the following code.


```julia
@resumable function live_delay(sim::Simulation, individual::SIRPerson, m::SIRModel)
  while individual.status==:S
      # Wait until next contact
      @yield timeout(sim,rand(Distributions.Exponential(1/m.c)))
      # Choose random alter
      alter=individual
      while alter==individual
          N=length(m.allIndividuals)
          index=rand(Distributions.DiscreteUniform(1,N))
          alter=m.allIndividuals[index]
      end
      # If alter is infected
      if alter.status==:I
          infect = rand(Distributions.Uniform(0,1))
          if infect < m.β
              individual.status=:I
              infection_update!(sim,m)
          end
      end
  end
  # The following is modified to capture a fixed infection delay
  if individual.status==:I
      # Wait until recovery, here fixed
      @yield timeout(sim,1/m.γ)
      individual.status=:R
      recovery_update!(sim,m)
  end
end;
```


```julia
function activate_delay(m::SIRModel)
     [@process live_delay(m.sim,individual,m) for individual in m.allIndividuals]
end;
```


```julia
des_delay_model = MakeSIRModel(u0,p)
activate_delay(des_delay_model)
sir_run(des_delay_model,tmax)
```


```julia
data_delay_des=out(des_delay_model);
```


```julia
@df data_delay_des plot(:t,
    [:S :I :R],
    labels = ["S" "I" "R"],
    xlab="Time",
    ylab="Number")
```

![](figures/des_23_1.png)



## Benchmarking

```julia
@benchmark begin
    des_model = MakeSIRModel(u0,p)
    activate(des_model)
    sir_run(des_model,tmax)
end
```

```
BenchmarkTools.Trial: 11 samples with 1 evaluation.
 Range (min … max):  399.768 ms … 501.322 ms  ┊ GC (min … max): 4.59% … 3.6
7%
 Time  (median):     470.421 ms               ┊ GC (median):    3.84%
 Time  (mean ± σ):   459.710 ms ±  34.598 ms  ┊ GC (mean ± σ):  2.18% ± 2.1
3%

  █          █  █          █      █         █  █   █     █  █ █  
  █▁▁▁▁▁▁▁▁▁▁█▁▁█▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁█▁▁█▁▁▁█▁▁▁▁▁█▁▁█▁█ ▁
  400 ms           Histogram: frequency by time          501 ms <

 Memory estimate: 71.04 MiB, allocs estimate: 2506315.
```


