
using InfiniteArrays
using Distributions
using DifferentialEquations
using Random
using Plots


# find index of last nonzero element
function find_end(I)
    findfirst(x -> isequal(x, sum(I)), cumsum(I))
end

# find indices of nonzero elements
function find_nonzero(I)
    last = find_end(I)
    findall(>(0), I[1:last])
end

struct SIR_struct
    I::AbstractArray
    R::AbstractArray
end

SIR_struct(I0) = SIR_struct(I0, zeros(Int64, ∞))


function infection_rate(u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    N = S+I+R
    β*c*I/N*S
end

function infection!(integrator, SIR::SIR_struct)

    I_elements = find_nonzero(SIR.I)
    infector_bin = wsample(I_elements, SIR.I[I_elements], 1)[1]

    # infector increases their count of infections by one
    SIR.I[infector_bin] -= 1
    SIR.I[infector_bin + 1] += 1

    # add a 0-infections infector
    SIR.I[1] += 1

    # update S and I
    integrator.u[1] -= 1
    integrator.u[2] = sum(SIR.I)

end

const infection_jump = ConstantRateJump(infection_rate, (integrator) -> infection!(integrator, SIR))


function recovery_rate(u,p,t)
    (S,I,R) = u
    (β,c,γ) = p
    γ*I
end

function recovery!(integrator, SIR::SIR_struct)

    I_elements = find_nonzero(SIR.I)
    recovery_bin = wsample(I_elements, SIR.I[I_elements], 1)[1]

    SIR.I[recovery_bin] -= 1
    SIR.R[recovery_bin] += 1

    integrator.u[2] = sum(SIR.I)
    integrator.u[3] = sum(SIR.R)
end

const recovery_jump = ConstantRateJump(recovery_rate, (integrator) -> recovery!(integrator, SIR))


tmax = 40.0
tspan = (0.0,tmax);


δt = 0.1
t = 0:δt:tmax;


u0 = [990,10,0]; # S,I,R

I0 = zeros(Int64, ∞)
I0[1] = u0[2]

SIR = SIR_struct(I0)


p = [0.05,10.0,0.25]; # β,c,γ


Random.seed!(1234);


prob_discrete = DiscreteProblem(u0,tspan,p);


prob_jump = JumpProblem(prob_discrete,Direct(),infection_jump,recovery_jump);


sol_jump = solve(prob_jump,SSAStepper());


out_jump = sol_jump(t);


plot(
    out_jump,
    label=["S" "I" "R"],
    xlabel="Time",
    ylabel="Number"
)


infectors = find_nonzero(SIR.R)
infectors_counts = zeros(Int64, infectors[end])
infectors_counts[infectors] = SIR.R[infectors]

plot(
    infectors_counts, 
    seriestype = :bar, 
    xlabel="Number",
    ylabel="Frequency", 
    color = 1:length(infectors_counts), 
    legend = false,
    xticks = 1:length(infectors_counts),
    xformatter = x -> Int(x - 1)
)

