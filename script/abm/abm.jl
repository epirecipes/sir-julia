
using Agents
using Random
using Distributions
using DrWatson: @dict
using Plots
using BenchmarkTools


function rate_to_proportion(r::Float64,t ::Float64)
    1 - exp(-r * t)
end;


@agent struct Person(NoSpaceAgent)
    status::Symbol
end;


function agent_step!(agent, model)
    transmit!(agent, model)
    recover!(agent, model)
end;


function transmit!(agent, model)
    # If I'm not susceptible, I return
    agent.status != :S && return
    ncontacts = rand(Poisson(model.c))
    for i in 1:ncontacts
        # Choose random individual
        alter = random_agent(model)
        if alter.status == :I && (rand() ≤ model.β)
            # An infection occurs
            agent.status = :I
            break
        end
    end
end;


function recover!(agent, model)
    agent.status != :I && return
    if rand() ≤ model.γ
            agent.status = :R
    end
end;


susceptible(x) = count(i == :S for i in x)
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x);


function init_model(β::Float64, c::Float64, γ::Float64, N::Int64, I0::Int64, rng::AbstractRNG=Random.GLOBAL_RNG)
    properties = @dict(β,c,γ)
    model = StandardABM(Person; agent_step!, properties, rng)
    for i in 1:N
        if i <= I0
            s = :I
        else
            s = :S
        end
        p = Person(;id=i,status=s)
        p = add_agent!(p,model)
    end
    return model
end;


δt = 0.1
nsteps = 400
tf = nsteps * δt
t = 0:δt:tf;


β = 0.05
c = 10.0 * δt
γ = rate_to_proportion(0.25, δt);


N = 1000
I0 = 10;


seed = 1234
rng = Random.Xoshiro(seed);


abm_model = init_model(β, c, γ, N, I0, rng);


to_collect = [(:status, f) for f in (susceptible, infected, recovered)]
abm_data, _ = run!(abm_model, nsteps; adata = to_collect);


abm_data[!, :t] = t;


plot(t, abm_data[:,2], label="S", xlab="Time", ylabel="Number")
plot!(t, abm_data[:,3], label="I")
plot!(t, abm_data[:,4], label="R")


@benchmark begin
abm_model = init_model(β, c, γ, N, I0, rng)
abm_data, _ = run!(abm_model, nsteps; adata = to_collect)
end

