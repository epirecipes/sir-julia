
using Distributions
using StatsBase
using Random
using DataFrames
using StatsPlots
using BenchmarkTools


function rate_to_proportion(r::Float64,t::Float64)
    1-exp(-r*t)
end;


@enum InfectionStatus Susceptible Infected Recovered


function sir_abm(u,p,t)
    du = deepcopy(u)
    (β,c,γ,δt) = p
    N = length(u)
    for i in 1:N # loop through agents
        # If recovered
        if u[i]==Recovered continue
        # If susceptible
        elseif u[i]==Susceptible
            ncontacts = rand(Poisson(c*δt))
            idx = sample(1:N,ncontacts,replace=false)
            for j in 1:length(idx)
                if j==i continue end
                a = u[idx[j]]
                if a==Infected && rand() < β
                    du[i] = Infected
                    break
                end
            end
        # If infected
    else u[i]==Infected
            if rand() < γ
                du[i] = Recovered
            end
        end
    end
    du
end;


function sir_abm!(du,u,p,t)
    (β,c,γ,δt) = p
    N = length(u)
    for i in 1:N # loop through agents
        # If recovered
        if u[i]==Recovered
            du[i] = u[i]
        # If susceptible
        elseif u[i]==Susceptible
            ncontacts = rand(Poisson(c*δt))
            idx = sample(1:N,ncontacts,replace=false)
            for j in 1:length(idx)
                if j==i continue end
                a = u[idx[j]]
                if a==Infected && rand() < β
                    du[i] = Infected
                    break
                end
            end
        # If infected
    else u[i]==Infected
            if rand() < γ
                du[i] = Recovered
            else
                du[i] = u[i]
            end
        end
    end
    nothing
end;


δt = 0.1
nsteps = 400
tf = nsteps*δt
tspan = (0.0,nsteps)
t = 0:δt:tf;


β = 0.05
c = 10.0
γ = rate_to_proportion(0.25,δt)
p = [β,c,γ,δt]


N = 1000
I0 = 10
u0 = Array{InfectionStatus}(undef,N)
for i in 1:N
    if i <= I0
        s = Infected
    else
        s = Susceptible
    end
    u0[i] = s
end


Random.seed!(1234);


susceptible(x) = count(i == Susceptible for i in x)
infected(x) = count(i == Infected for i in x)
recovered(x) = count(i == Recovered for i in x);


function sim(u0,nsteps,dt)
    u = copy(u0)
    t = 0.0
    ta = []
    Sa = []
    Ia = []
    Ra =[]
    push!(ta,t)
    push!(Sa,susceptible(u))
    push!(Ia,infected(u))
    push!(Ra,recovered(u))
    for i in 1:nsteps
        u=sir_abm(u,p,t)
        t = t + dt
        push!(ta,t)
        push!(Sa,susceptible(u))
        push!(Ia,infected(u))
        push!(Ra,recovered(u))
    end
    DataFrame(t=ta,S=Sa,I=Ia,R=Ra)
end


function sim!(u0,nsteps,dt)
    u = copy(u0)
    du = copy(u0)
    t = 0.0
    ta = []
    Sa = []
    Ia = []
    Ra =[]
    push!(ta,t)
    push!(Sa,susceptible(u))
    push!(Ia,infected(u))
    push!(Ra,recovered(u))
    for i in 1:nsteps
        sir_abm!(du,u,p,t)
        u,du = du,u
        t = t + dt
        push!(ta,t)
        push!(Sa,susceptible(u))
        push!(Ia,infected(u))
        push!(Ra,recovered(u))
    end
    DataFrame(t=ta,S=Sa,I=Ia,R=Ra)
end


df_abm = sim(u0,nsteps,δt);


df_abm! = sim!(u0,nsteps,δt);


@df df_abm plot(:t,
    [:S :I :R],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number",
    title="New state")


@df df_abm! plot(:t,
    [:S :I :R],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number",
    title="In-place")


@benchmark sim(u0,nsteps,δt);


@benchmark sim!(u0,nsteps,δt);


include(joinpath(@__DIR__,"tutorials","appendix.jl"))
appendix()

