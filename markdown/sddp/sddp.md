# Optimal control of an SIR epidemic with a non-pharmaceutical intervention using SDDP.jl
Sean L. Wu (@slwu89) and Simon Frost (@sdwfrost), 2023-5-9

## Introduction

[SDDP.jl](https://odow.github.io/SDDP.jl/stable/) (stochastic dual dynamic programming) is a package designed
to solve optimal policies in multi-stage (time or world state) linear programming problems with exogeneous stochasticity.
We can use it to optimize policy for a non-pharmaceutical intervention which decreses the transmission rate.

Because SDDP.jl solves an optimization problem for each node in a graph of nodes (which may represent the passage of time, or other changes
in world state), the model we solve is a discretization of following ODEs ($\upsilon$ is the intensity of intervention). 

$$
\begin{align*}
\dfrac{\mathrm dS}{\mathrm dt} &= -\beta (1 - \upsilon(t)) S I, \\
\dfrac{\mathrm dI}{\mathrm dt} &= \beta (1 - \upsilon(t)) S I - \gamma I,\\ 
\dfrac{\mathrm dC}{\mathrm dt} &= \beta (1 - \upsilon(t)) S I\\
\end{align*}
$$

The minimization objective at each node (time point) is a linear combination of cumulative intervention applied,
and cumulative cases. The total cumulative intervention force applied cannot exceed some maximum value.
The decision variable is the intensity of the intervention at each time point (node).

## Libraries

```julia
using SDDP, JuMP, HiGHS, Plots;
```




## Parameters

We set the parameters, which includes the maximum intervention level at any node, `υ_max`, and the cost, which is the integral of the intervention level over time, `υ_total`.

```julia
β = 0.5 # infectivity rate
γ = 0.25 # recovery rate
υ_max = 0.5 # maximum intervention
υ_total = 10.0; # maximum cost
```




## Time domain

We set the time horizon to be long enough for the system to settle down to an equilibrium. We use a grid of timepoints fine enough to capture a wide variety of policy shapes, but coarse enough to keep the number of policy parameters to optimize low.

```julia
tmax = 100.0
δt = 1.0
nsteps = Int(tmax / δt);
```




## Initial conditions

We set the initial conditions for the proportion of susceptibles and infecteds.

```julia
u0 = [0.99, 0.01]; # S,I
```




## Model setup

We specify a model using `SDDP.LinearPolicyGraph`. Because the nodes in the policy graph represent the
passage of time, we use a linear policy graph. We set the `optimizer` to the one from the `Ipopt`.

We set `S`, `I`, and `C` to be `SDDP.State` variables, meaning the values from the previous node in the policy
graph will be available to the current node. We specify 2 constraints on the intervention. While the second
constraint is mathematically the same as specifying `υ_cumulative.out ≤ υ_total` we must write it in
the form shown so that `υ` appears in the constraint.

We then set up the differences as non-linear expressions and the update rules as non-linear constraints.
Finally, we use `@stageobjective` to set the minimization objective for this node to be a linear combination
of total intervention pressure and cumulative cases.

```julia
model = SDDP.LinearPolicyGraph(
    stages = nsteps,
    sense = :Min,
    lower_bound = 0,
    optimizer = HiGHS.Optimizer,
) do sp, t

    @variable(sp, 0 ≤ S, SDDP.State, initial_value = u0[1])
    @variable(sp, 0 ≤ I, SDDP.State, initial_value = u0[2])
    @variable(sp, 0 ≤ C, SDDP.State, initial_value = 0)

    @variable(sp, 0 ≤ υ_cumulative, SDDP.State, initial_value = 0)
    @variable(sp, 0 ≤ υ ≤ υ_max)

    # constraints on control    
    @constraint(sp, υ_cumulative.out == υ_cumulative.in + (δt * υ))
    @constraint(sp, υ_cumulative.in + (δt * υ) ≤ υ_total)

    # expressions to simplify the state updates
    @NLexpression(sp, infection, (1-exp(-(1 - υ) * β * I.in * δt)) * S.in)
    @NLexpression(sp, recovery, (1-exp(-γ*δt)) * I.in)

    # state updating rules
    @NLconstraint(sp, S.out == S.in - infection)
    @NLconstraint(sp, I.out == I.in + infection - recovery)
    @NLconstraint(sp, C.out == C.in + infection)

    # linear weighting of objectives
    @stageobjective(sp, υ_cumulative.out + 40*C.out)

end;
```




## Running the model

We train the model for 100 iterations. SDDP.jl needs to iterate between forwards passes over the policy
graph where the policy is optimized given an approximation of the overall objective for each node,
and backwards passes to improve the approximation.

```julia
SDDP.train(model; iteration_limit = 100);
```

```
---------------------------------------------------------------------------
---
                      SDDP.jl (c) Oscar Dowson, 2017-21

Problem
  Nodes           : 100
  State variables : 4
  Scenarios       : 1.00000e+00
  Existing cuts   : false
  Subproblem structure                           : (min, max)
    Variables                                    : (10, 10)
    JuMP.VariableRef in MOI.GreaterThan{Float64} : (6, 6)
    JuMP.AffExpr in MOI.LessThan{Float64}        : (1, 1)
    JuMP.VariableRef in MOI.LessThan{Float64}    : (1, 2)
    JuMP.AffExpr in MOI.EqualTo{Float64}         : (1, 1)
Options
  Solver          : serial mode
  Risk measure    : SDDP.Expectation()
  Sampling scheme : SDDP.InSampleMonteCarlo

Numerical stability report
  Non-zero Matrix range     [1e+00, 1e+00]
  Non-zero Objective range  [1e+00, 4e+01]
  Non-zero Bounds range     [5e-01, 5e-01]
  Non-zero RHS range        [1e+01, 1e+01]
No problems detected

 Iteration    Simulation       Bound         Time (s)    Proc. ID   # Solve
s
Error: UndefVarError: libhighs not defined
```





## Plotting

After the model has been trained, we can simulate from the model under the final optimal policy.
The second argument is the number of trajectories to draw (because the model is deterministic, a single
trajectory will suffice). The third argument is the variables to record during simulation.

```julia
sims = SDDP.simulate(model, 1, [:S,:I, :C, :υ, :υ_cumulative]);
```

```
Error: UndefVarError: libhighs not defined
```





We can use the plotting utilities of SDDP.jl to show the optimal policy and state variables.

```julia
Plots.plot(
    SDDP.publication_plot(sims, title = "S") do data
        return data[:S].out
    end,
    SDDP.publication_plot(sims, title = "I") do data
        return data[:I].out
    end,
    SDDP.publication_plot(sims, title = "C") do data
        return data[:C].out
    end,
    SDDP.publication_plot(sims, title = "Control") do data
        return data[:υ]
    end,
    SDDP.publication_plot(sims, title = "Cumulative control") do data
        return data[:υ_cumulative].out
    end;
    xlabel = "Time"
)
```

```
Error: UndefVarError: sims not defined
```


