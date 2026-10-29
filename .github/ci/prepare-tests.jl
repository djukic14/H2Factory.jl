using Pkg
using TOML
using UUIDs

# Restore the pinned TestUtils setup below after H2FactoryTestUtils is registered.
# function prepare_tests(testutils; revision, root=dirname(dirname(@__DIR__)))
function prepare_tests(; root=dirname(dirname(@__DIR__)))
    root = realpath(root)
    # testutils = realpath(testutils)
    # actual = readchomp(`git -C $testutils rev-parse HEAD`)
    # actual == revision ||
    #     error("TestUtils revision mismatch: expected $revision, got $actual")
    # isempty(readchomp(`git -C $testutils status --porcelain`)) ||
    #     error("The pinned TestUtils checkout has local changes")

    # Never reuse the development manifest or its sibling-checkout overrides.
    project = TOML.parsefile(joinpath(root, "test", "Project.toml"))
    pop!(project, "sources", nothing)
    environment = mktempdir(; cleanup=false)
    open(joinpath(environment, "Project.toml"), "w") do io
        return TOML.print(io, project; sorted=true)
    end

    Pkg.Registry.add("General")
    Pkg.activate(environment)
    # Pkg.develop([Pkg.PackageSpec(; path=root), Pkg.PackageSpec(; path=testutils)])
    Pkg.develop(Pkg.PackageSpec(; path=root))
    Pkg.instantiate()

    dependencies = Pkg.dependencies()
    # for name in ("H2Factory", "H2FactoryTestUtils")
    for name in ("H2Factory",)
        info = dependencies[UUID(project["deps"][name])]
        # expected = name == "H2Factory" ? root : testutils
        expected = root
        info.is_tracking_path && realpath(info.source) == expected ||
            error("$name did not resolve to the intended checkout")
        println("$name: ", info.source)
    end
    for name in ("BEAST", "CompScienceMeshes", "H2Trees")
        info = dependencies[UUID(project["deps"][name])]
        info.is_tracking_registry ||
            error("$name must resolve from a registry, not a checkout")
        println("$name: registry version ", info.version, ", tree ", info.tree_hash)
    end
    return environment
end
