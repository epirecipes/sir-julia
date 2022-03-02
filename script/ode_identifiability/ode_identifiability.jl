
using StructuralIdentifiability
using ModelingToolkit


@parameters b c g
@variables t S(t) I(t) R(t) N(t)
N = S + I + R
D = Differential(t);


sir_eqs1 = [
  D(S) ~ -b*S*I,
  D(I) ~ b*S*I-g*I,
  D(R) ~ g*I
]
sir_ode1 = ODESystem(sir_eqs1, t, name=:SIR1)


sir_eqs2 = [
  D(S) ~ -b*c*S*I,
  D(I) ~ b*c*S*I-g*I,
  D(R) ~ g*I
]
sir_ode2 = ODESystem(sir_eqs2, t, name=:SIR2)


sir_eqs3 = [
  D(S) ~ -b*c*S*I/N,
  D(I) ~ b*c*S*I/N-g*I,
  D(R) ~ g*I
]
sir_ode3 = ODESystem(sir_eqs3, t, name=:SIR3)


@variables y(t)
measured_quantities1 = [y ~ b*S*I];


funcs_to_check1 = [b,g,b/g]
ident1 = assess_identifiability(sir_ode1; measured_quantities= measured_quantities1, funcs_to_check = funcs_to_check1)


funcs_to_check2 = [b,c,g,b*c]
measured_quantities2 = [y ~ b*c*S*I]
ident2 = assess_identifiability(sir_ode2; measured_quantities= measured_quantities2, funcs_to_check = funcs_to_check2)


funcs_to_check3 = [b,c,g,b*c]
measured_quantities3 = [y ~ b*c*S*I/N]
ident3 = assess_identifiability(sir_ode3; measured_quantities= measured_quantities3, funcs_to_check = funcs_to_check3)


@variables n(t)
measured_quantities3a = [y ~ b*c*S*I/N, n ~ N]
ident3a = assess_identifiability(sir_ode3; measured_quantities= measured_quantities3a, funcs_to_check = funcs_to_check3)


exp_eqs = [
  D(I) ~ b*S*I-g*I,
  D(R) ~ g*I
]
exp_ode = ODESystem(exp_eqs, t, name=:EXP)
assess_identifiability(exp_ode;
                       measured_quantities= [y ~ b*S*I],
                       funcs_to_check = [b,g])

