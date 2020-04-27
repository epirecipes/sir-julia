using Weave, Pkg, InteractiveUtils, IJulia

repo_directory = joinpath(@__DIR__)
cssfile = joinpath(@__DIR__, "templates", "skeleton_css.css")
latexfile = joinpath(@__DIR__, "templates", "julia_tex.tpl")

function weave_file(folder,file,build_list=(:script,:html,:pdf,:github,:notebook); kwargs...)
  tmp = joinpath(repo_directory,"tutorials",folder,file)
  args = Dict{Symbol,String}(:folder=>folder,:file=>file)
  if :script ∈ build_list
    println("Building Script")
    dir = joinpath(repo_directory,"script",folder)
    isdir(dir) || mkdir(dir)
    args[:doctype] = "script"
    tangle(tmp;out_path=dir)
  end
  if :html ∈ build_list
    println("Building HTML")
    dir = joinpath(repo_directory,"html",folder)
    isdir(dir) || mkdir(dir)
    args[:doctype] = "html"
    weave(tmp,doctype = "md2html",out_path=dir,args=args; fig_ext=".svg", css=cssfile, kwargs...)
  end
  if :pdf ∈ build_list
    println("Building PDF")
    dir = joinpath(repo_directory,"pdf",folder)
    isdir(dir) || mkdir(dir)
    args[:doctype] = "pdf"
    weave(tmp,doctype="md2pdf",out_path=dir,args=args; template=latexfile, kwargs...)
  end
  if :github ∈ build_list
    println("Building Github Markdown")
    dir = joinpath(repo_directory,"markdown",folder)
    isdir(dir) || mkdir(dir)
    args[:doctype] = "github"
    weave(tmp,doctype = "github",out_path=dir,args=args; kwargs...)
  end
  if :notebook ∈ build_list
    println("Building Notebook")
    dir = joinpath(repo_directory,"notebook",folder)
    isdir(dir) || mkdir(dir)
    args[:doctype] = "notebook"
    Weave.convert_doc(tmp,joinpath(dir,file[1:end-4]*".ipynb"))
  end
end

function weave_all()
  for folder in readdir(joinpath(repo_directory,"tutorials"))
    folder == "test.jmd" && continue
    weave_folder(folder)
  end
end

function weave_folder(folder)
  for file in readdir(joinpath(repo_directory,"tutorials",folder))
    println("Building $(joinpath(folder,file)))")
    try
      weave_file(folder,file)
    catch
    end
  end
end

weave_all()
