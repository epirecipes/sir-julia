
using Random
using Plots
using Distributions
using Fleck


function sir_vas(β, c, γ)
    take = [
        1 0;
        1 1;
        0 0;
    ]
    give = [
        0 0;
        2 0;
        0 1;
    ]
    rates = [
             (state) -> Exponential(1.0/(β*c*state[2]/sum(state)*state[1])),
             (state) -> Exponential(1.0/(state[2] * γ))
             ]
    (take, give, rates)
end;


tmax = 40.0;


u0 = [990, 10, 0]; # S, I, R


p = [0.05, 10.0, 0.25]; # β, c, γ


seed = 1234
rng = MersenneTwister(seed);


take, give, rates = sir_vas(p...);
vas = VectorAdditionSystem(take, give, rates);


smplr = DirectCall{Int}();
# smplr = FirstReaction{Int}();


fsm = VectorAdditionFSM(vas, vas_initial(vas, u0), smplr, rng);


t = Vector{Float64}(undef, u0[2] + 2*u0[1] + 1) # time is Float64
u = Matrix{Int}(undef, length(u0), u0[2] + 2*u0[1] + 1) # states are Ints
# Store initial conditions
t[1] = 0.0
u[1:end, 1] = u0
let event_cnt = 1 # number of events; this format is used to avoid soft scope errors
    while true
        when, next_transition = simstep!(fsm)
        if ((next_transition === nothing) | (when > tmax))
            break
        end
        event_cnt = event_cnt + 1
        t[event_cnt] = fsm.state.when
        u[1:end, event_cnt] = fsm.state.state
    end
    global total_events = event_cnt
end;


plot(
    t[1:total_events],
    u[1:end, 1:total_events]',
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number"
)

