
using Agents
using Random
using DataFrames
using Distributions
using DrWatson
using StatsPlots
using BenchmarkTools


function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;


mutable struct Person <: AbstractAgent
    id::Int64
    status::Symbol
end


function init_model(β::Float64,c::Float64,γ::Float64,N::Int64,I0::Int64)
    properties = @dict(β,c,γ)
    model = ABM(Person; properties=properties)
    for i in 1:N
        if i <= I0
            s = :I
        else
            s = :S
        end
        p = Person(i,s)
        p = add_agent!(p,model)
    end
    return model
end;


function agent_step!(agent, model)
    transmit!(agent, model)
    recover!(agent, model)
end;


function transmit!(agent, model)
    # If I'm not susceptible, I return
    agent.status != :S && return
    ncontacts = rand(Poisson(model.properties[:c]))
    for i in 1:ncontacts
        # Choose random individual
        alter = random_agent(model)
        if alter.status == :I && (rand() ≤ model.properties[:β])
            # An infection occurs
            agent.status = :I
            break
        end
    end
end;


function recover!(agent, model)
    agent.status != :I && return
    if rand() ≤ model.properties[:γ]
            agent.status = :R
    end
end;


susceptible(x) = count(i == :S for i in x)
infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x);


δt = 0.1
nsteps = 400
tf = nsteps*δt
t = 0:δt:tf;


β = 0.05
c = 10.0*δt
γ = rate_to_proportion(0.25,δt);


N = 1000
I0 = 10;


Random.seed!(1234);


abm_model = init_model(β,c,γ,N,I0)


to_collect = [(:status, f) for f in (susceptible, infected, recovered)]
abm_data, _ = run!(abm_model, agent_step!, nsteps; adata = to_collect);


abm_data[!,:t] = t;


plot(t,abm_data[:,2],label="S",xlab="Time",ylabel="Number")
plot!(t,abm_data[:,3],label="I")
plot!(t,abm_data[:,4],label="R")


@benchmark begin
abm_model = init_model(β,c,γ,N,I0)
abm_data, _ = run!(abm_model, agent_step!, nsteps; adata = to_collect)
end

