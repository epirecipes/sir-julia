using Weave, Pkg, InteractiveUtils, IJulia

function appendix()
    display("text/markdown","## Appendix")
    display("text/markdown", "### Computer Information")
    vinfo = sprint(InteractiveUtils.versioninfo)
    vinfo = replace(vinfo,r"  JULIA_EDITOR(.*)\n"  => s"")
    display("text/markdown",  """
    ```
    $(vinfo)
    ```
    """)

    ctx = Pkg.API.Context()
    pkgs = Pkg.Display.status(Pkg.API.Context(), use_as_api=true);
    projfile = ctx.env.project_file
    projfile = replace(projfile, homedir() => "~")

    display("text/markdown","""
    ### Package Information
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
