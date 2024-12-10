
using Agents
using Random
using Distributions
using DrWatson: @dict
using Plots
using BenchmarkTools


@agent struct Person(NoSpaceAgent)
    status::Symbol
end;


function transmit!(agent, model)
    # Choose random individual
    alter = random_agent(model)
    if alter.status == :I && (rand() ≤ model.β)
        # An infection occurs
        agent.status = :I
    end
end;


function recover!(agent, model)
    agent.status = :R
end;


function transmit_propensity(agent, model)
    if agent.status == :S
        return model.c
    else
        return 0.0
    end
end


function recovery_propensity(agent, model)
    if agent.status == :I
        return model.γ
    else
        return 0.0
    end
end;


transmit_event = AgentEvent(action! = transmit!, propensity = transmit_propensity)
recovery_event = AgentEvent(action! = recover!, propensity = recovery_propensity);


events = (transmit_event, recovery_event);


susceptible(x) = count(i == :S for i in x)
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x);


function init_model(β::Float64, c::Float64, γ::Float64, N::Int64, I0::Int64, rng::AbstractRNG=Random.GLOBAL_RNG)
    properties = @dict(β,c,γ)
    model = EventQueueABM(Person, events; properties, rng)
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


tf = 40.0;


β = 0.05
c = 10.0
γ = 0.25;


N = 1000
I0 = 10;


seed = 1234
rng = Random.Xoshiro(seed);


abm_model = init_model(β, c, γ, N, I0, rng);


to_collect = [(:status, f) for f in (susceptible, infected, recovered)]
abm_data, _ = run!(abm_model, tf; adata = to_collect);


plot(abm_data[:,1], abm_data[:,2], label="S", xlab="Time", ylabel="Number")
plot!(abm_data[:,1], abm_data[:,3], label="I")
plot!(abm_data[:,1], abm_data[:,4], label="R")


@benchmark begin
abm_model = init_model(β, c, γ, N, I0, rng)
abm_data, _ = run!(abm_model, tf; adata = to_collect)
end

