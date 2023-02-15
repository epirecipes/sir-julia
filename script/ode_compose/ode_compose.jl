
using ModelingToolkit
using OrdinaryDiffEq
using Plots


@parameters t
D = Differential(t)


@variables S(t) λ(t)
@named seqn = ODESystem([D(S) ~ -λ*S])


@variables I(t)
@parameters γ
@named ieqn = ODESystem([D(I) ~ λ*S - γ*I])


@parameters β
@named λeqn = ODESystem([λ ~ β*I])


sys = compose(ODESystem([
                            ieqn.S ~ seqn.S,
                            seqn.λ ~ λeqn.λ,
                            ieqn.λ ~ λeqn.λ,
                            λeqn.I ~ ieqn.I,
                        ],
                        t,
                        [S, I, λ],
                        [β, γ],
                        defaults = [λeqn.β => β,
                                    ieqn.γ => γ],
                        name = :sir),
              seqn,
              ieqn,
              λeqn)


simpsys = structural_simplify(sys);


equations(simpsys)


tspan = (0.0, 40.0)
u₀ = [seqn.S => 0.99, ieqn.I => 0.01]
p = [β => 0.5, γ => 0.25];


prob = ODEProblem(simpsys, u₀, tspan, p, jac = true)
sol = solve(prob, Tsit5());


plot(sol)

