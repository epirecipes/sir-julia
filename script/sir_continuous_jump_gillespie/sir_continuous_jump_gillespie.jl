
using Gillespie
using Random
using Plots
using BenchmarkTools


function sir_rates(x,parms)
  (S,I,R) = x
  (β,γ) = parms
  N = S+I+R
  infection = β*S*I/N
  recovery = γ*I
  [infection,recovery]
end
sir_transitions = [[-1 1 0];[0 -1 1]]


u0 = [999,1,0]
p = [0.5,0.25]
Random.seed!(1235)
tf = 50.0


sir_result = ssa(u0,sir_rates,sir_transitions,p,tf)
data = ssa_data(sir_result)


plot(data[:,1],data[:,2])
plot!(data[:,1],data[:,3])
plot!(data[:,1],data[:,4])


@benchmark ssa(u0,sir_rates,sir_transitions,p,tf)


using Weave, Pkg, InteractiveUtils, IJulia
function tutorial_footer(folder=nothing, file=nothing; remove_homedir=true)
    display("text/markdown", """
    ## Appendix
     """)
    display("text/markdown", "Computer Information:")
    vinfo = sprint(InteractiveUtils.versioninfo)
    display("text/markdown",  """
    ```
    $(vinfo)
    ```
    """)

    ctx = Pkg.API.Context()
    pkgs = Pkg.Display.status(Pkg.API.Context(), use_as_api=true);
    projfile = ctx.env.project_file
    remove_homedir && (projfile = replace(projfile, homedir() => "~"))

    display("text/markdown","""
    Package Information:
    """)

    md = ""
    md *= "```\nStatus `$(projfile)`\n"

    for pkg in pkgs
        if !isnothing(pkg.old) && pkg.old.ver !== nothing
          md *= "[$(string(pkg.uuid))] $(string(pkg.name)) $(string(pkg.old.ver))\n"
        else
          md *= "[$(string(pkg.uuid))] $(string(pkg.name))\n"
        end
    end
    md *= "```"
    display("text/markdown", md)
end
tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])

