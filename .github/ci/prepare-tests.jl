using Pkg
using TOML
using UUIDs

function prepare_tests(; root=dirname(dirname(@__DIR__)))
    root = realpath(root)

    # Never reuse the development manifest or its sibling-checkout overrides.
    project = TOML.parsefile(joinpath(root, "test", "Project.toml"))
    pop!(project, "sources", nothing)
    environment = mktempdir(; cleanup=false)
    open(joinpath(environment, "Project.toml"), "w") do io
        return TOML.print(io, project; sorted=true)
    end

    Pkg.Registry.add("General")
    Pkg.Registry.update()
    Pkg.activate(environment)
    Pkg.develop(Pkg.PackageSpec(; path=root))
    Pkg.instantiate()

    dependencies = Pkg.dependencies()
    for name in ("H2Factory",)
        info = dependencies[UUID(project["deps"][name])]
        expected = root
        info.is_tracking_path && realpath(info.source) == expected ||
            error("$name did not resolve to the intended checkout")
        println("$name: ", info.source)
    end
    for name in ("BEAST", "CompScienceMeshes", "H2Trees", "H2FactoryTestUtils")
        info = dependencies[UUID(project["deps"][name])]
        info.is_tracking_registry ||
            error("$name must resolve from a registry, not a checkout")
        println("$name: registry version ", info.version, ", tree ", info.tree_hash)
    end
    return environment
end
