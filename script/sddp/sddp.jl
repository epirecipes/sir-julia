
using SDDP, JuMP, Ipopt, Plots;


β = 0.5 # infectivity rate
γ = 0.25 # recovery rate
υ_max = 0.5 # maximum intervention
υ_total = 10.0; # maximum cost


tmax = 100.0
δt = 1.0
nsteps = Int(tmax / δt);


u0 = [0.99, 0.01]; # S,I


model = SDDP.LinearPolicyGraph(
    stages = nsteps,
    sense = :Min,
    lower_bound = 0,
    optimizer = Ipopt.Optimizer,
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


SDDP.train(model; iteration_limit = 100);


sims = SDDP.simulate(model, 1, [:S,:I, :C, :υ, :υ_cumulative]);


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

