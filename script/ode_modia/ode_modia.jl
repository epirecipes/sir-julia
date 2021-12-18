
using Modia
using Tables
using DataFrames
using StatsPlots
using BenchmarkTools


SIR = Model(
    equations = :[
        N = S + I + R
        der(S) = -β*c*I/N*S
        der(I) = β*c*I/N*S - γ*I
        der(R) = γ*I
    ]
);


δt = 0.1
tmax = 40.0;


u0 = Map(S = Var(init=990.0),
         I = Var(init=10.0),
         R = Var(init=0.0));


p = Map(β = 0.05,
        c = 10.0,
        γ = 0.25);


model = SIR | u0 | p;


sir = @instantiateModel(model);


simulate!(sir,Tsit5(),stopTime=tmax,interval=δt);


result = sir.result_x.u # extract result
result = hcat(result...) # concatenate vectors
result = result' # transpose
result = Tables.table(result) # convert to table
df_modia = DataFrame(result) # convert to DataFrame
rename!(df_modia,["S","I","R"]) # rename
df_modia[!,:t] = sir.result_x.t; # add in time


@df df_modia plot(:t,
    [:S :I :R],
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number")


@benchmark simulate!(sir,Tsit5(),stopTime=tmax,interval=δt)

