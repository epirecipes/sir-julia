using POMDPs, POMDPTools, QuickPOMDPs
using MCTS
using Distributions
using Plots

@inline function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;

# u0 is initial conditions, s0 includes the
# cumulative intervention force (υ_cumulative)
u0 = (990,10,0);
s0 = (u0...,0.0)

δt = 0.1
β = 0.5 # infectivity rate
γ = 0.25 # recovery rate
υ_max = 0.5 # maximum intervention
υ_total = 10.0; # maximum cost

𝒜 = collect(0:0.05:υ_max)

sir_mdp = QuickMDP(
    actiontype = Float64,
    actions = function(s)
        if s[end] >= υ_total
            return [0.0]
        else
            return 𝒜
        end
    end,
    reward = function(s, a, sp)
        return -sp[3]
    end,
    transition = function(s, a)
        ImplicitDistribution() do rng
            (S,I,C) = s[1:3]
            υ_cumulative = s[end]
            N = sum(u0)
            ifrac = rate_to_proportion((1-a)*β*I/N,δt)
            rfrac = rate_to_proportion(γ,δt)
            infection=rand(rng,Binomial(S,ifrac))
            recovery=rand(rng,Binomial(I,rfrac))
            return (S-infection,I+infection-recovery,C+infection,υ_cumulative+(a*δt))
        end
    end,
    initialstate = Deterministic(s0),
    isterminal = s -> s[2] == 0,
    discount = 0.95
)

# solver = MCTSSolver(n_iterations=500,depth=100)
solver = DPWSolver(n_iterations=500,depth=20,show_progress=true)
planner = solve(solver, sir_mdp)

# action0, info0 = action_info(planner, s0)

hr = HistoryRecorder(max_steps=10000)
h = simulate(hr, sir_mdp, planner)

state_trajectory = transpose(hcat([collect(h[i][:s][1:3]) for i in eachindex(h)]…))
action_trajectory = [h[i][:a] for i in eachindex(h)]

plot(state_trajectory)
plot(action_trajectory)
