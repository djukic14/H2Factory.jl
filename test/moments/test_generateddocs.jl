module TestGeneratedDocs

using Test
using H2Factory

# THE GENERATED DOCSTRINGS HAVE TO RESOLVE THEIR INTERPOLATIONS.
#
# The moment macros build their docstrings as string literals inside `quote`/`@eval` blocks, so the
# `$` naming the concrete type is interpolated when the GENERATED code runs, not when the macro
# expands. One backslash too many turns that into literal source text: the reader is shown
# `m = $(MType){2,Float64}((10,))` instead of `m = H2Factory.Moment{2,Float64}((10,))`.
#
# That is exactly what happened. JuliaFormatter 2.11.3 began escaping `$` inside fenced code blocks
# in docstrings, rewriting `$(MType)` to `\$(MType)`, and 28 rendered lines degraded without a single
# test noticing, because the source keeps compiling and the escape looks deliberate.
# `test/Project.toml` pins the formatter below that version; this file is the other half, so
# unpinning it fails loudly rather than quietly degrading the docs again.

# WHAT COUNTS AS UNRESOLVED, and why this is not "no dollar signs in documentation".
#
# A docstring has every right to contain a `$`: inline maths, a shell example, a currency amount, or
# prose explaining Julia interpolation itself. Banning the character package-wide would trade one
# silent defect for a standing obstacle, so the test keys on interpolation SHAPE instead.
#
#   * `$(Identifier)` is unambiguous. Parenthesised interpolation is not a shape that maths, shell
#     or currency ever takes, so in rendered output it is always a defect.
#   * `$Identifier` is ambiguous: `$k$` is maths and `$HOME` is a shell variable. Only the macro
#     placeholders actually used by this package are flagged in that bare form.
#
# `MACROPLACEHOLDERS` is deliberately a closed list, not a guess at every possible name: a NEW
# placeholder introduced by a future macro will not be caught in its bare form until it is added
# here. The parenthesised rule needs no such maintenance, and is the form the macros use today.
const MACROPLACEHOLDERS = ("MType", "MColType", "stringname", "AbstractType")

const PARENTHESISEDINTERPOLATION = r"\$\([A-Za-z_][A-Za-z0-9_!]*\)"
const BAREPLACEHOLDER = Regex("\\\$(" * join(MACROPLACEHOLDERS, "|") * ")\\b")

function isunresolved(line::AbstractString)
    return occursin(PARENTHESISEDINTERPOLATION, line) || occursin(BAREPLACEHOLDER, line)
end

"""
    unresolvedinterpolations(mod::Module) -> Vector{String}

Every RENDERED docstring line in `mod` (and its submodules) still carrying interpolation-shaped text.

Rendered, not source: an escape level that is correct and one that is off by one look equally
plausible in the source, and only evaluating the string tells them apart.
"""
function unresolvedinterpolations(mod::Module)
    offenders = String[]
    for m in submodules(mod)
        meta = Base.Docs.meta(m; autoinit=false)
        meta === nothing && continue
        for (binding, multidoc) in meta
            for (_, docstr) in multidoc.docs
                docstr isa Base.Docs.DocStr || continue
                for line in split(join(docstr.text, ""), '\n')
                    isunresolved(line) &&
                        push!(offenders, string(m, ".", binding.var, ": ", strip(line)))
                end
            end
        end
    end
    return offenders
end

function submodules(root::Module)
    found = Module[root]
    for n in names(root; all=true)
        isdefined(root, n) || continue
        m = getfield(root, n)
        m isa Module && m !== root && parentmodule(m) === root && push!(found, m)
    end
    return found
end

# Docstrings whose `$` IS escaped, so the rendered text keeps the literal. Defined at file scope
# because `module` needs top level.
module BrokenDocsFixture

    """
    Example: `m = \$(SomeType){2,Float64}((3, 3))`
    """
    parenthesised() = nothing

    """
    Example: `m = \$stringname{2,Float64}((3, 3))`
    """
    bareplaceholder() = nothing

end

# The other side of the same coin: docstrings that legitimately contain `$` and must be left alone.
module LegitimateDollarsFixture

    """
    The wavenumber \$k\$ enters the kernel as \$e^{-jkr}\$.
    """
    inlinemaths() = nothing

    """
    Run it with `\$HOME/bin/julia`, costing roughly \$5 of compute.
    """
    shellandcurrency() = nothing

    """
    Julia interpolates a variable into a string with \$name, which this package's macros rely on.
    """
    proseaboutinterpolation() = nothing

end

@testset "generated docstrings resolve their interpolations" begin
    # NEGATIVE CONTROL. Without it, an empty result below is indistinguishable from a scan pointed
    # at the wrong docs database, which would pass forever while checking nothing.
    caught = unresolvedinterpolations(BrokenDocsFixture)
    @test length(caught) == 2
    @test any(o -> occursin("\$(SomeType)", o), caught)
    @test any(o -> occursin("\$stringname", o), caught)

    # POSITIVE CONTROL, and the reason the detector is shaped rather than character-based: maths,
    # shell, currency and prose about interpolation all have to survive untouched.
    @test isempty(unresolvedinterpolations(LegitimateDollarsFixture))

    @test isempty(unresolvedinterpolations(H2Factory))
end

"""
    malformedparameters(signature::AbstractString) -> Vector{String}

Arguments in a rendered docstring signature that are not a valid parameter form.

Merely PARSING the signature is not enough, and assuming otherwise is how the two defects below
sat in the rendered docs unnoticed: `mul!(momentH2Factory.Moment{D,T1}, ...)` (the `::` dropped)
parses happily as a field access on a variable named `momentH2Factory`, and `β:Number` (one colon)
parses as a range. Both are syntactically fine and semantically nonsense in an argument position,
so the check has to be on the FORM each argument takes, not on whether the line parses.
"""
function malformedparameters(signature::AbstractString)
    parsed = Meta.parse(signature; raise=false)
    # `f(x::T) where {T}` parses as `Expr(:where, call, ...)`, and a signature may carry more than
    # one `where`, so peel them off before asking for the call.
    while parsed isa Expr && parsed.head === :where
        parsed = parsed.args[1]
    end
    parsed isa Expr && parsed.head === :call || return ["not a call: $signature"]
    bad = String[]
    for arg in parsed.args[2:end]
        arg isa Symbol && continue                                   # bare name
        if arg isa Expr && arg.head in (:(::), :kw, :parameters, :...)
            continue                                                 # typed, default, keyword, splat
        end
        push!(bad, string(arg))
    end
    return bad
end

@testset "generated signatures declare well-formed parameters" begin
    # Self-check on the two historical defects, so a future weakening of `malformedparameters`
    # shows up here rather than silently passing everything.
    @test malformedparameters("mul!(momentH2Factory.Moment{D,T1}, y::AbstractArray)") ==
        ["momentH2Factory.Moment{D, T1}"]
    @test malformedparameters("mul!(moment::M, y::AbstractArray, α::Number, β:Number)") ==
        ["β:Number"]
    @test isempty(malformedparameters("mul!(moment::M, y::AbstractArray, α::Number)"))

    checked = 0
    for (binding, multidoc) in Base.Docs.meta(H2Factory)
        string(binding.var) in ("mul!", "conjmul!") || continue
        for (_, docstr) in multidoc.docs
            sig = strip(first(split(join(docstr.text, ""), '\n')))
            isempty(sig) && continue
            @test isempty(malformedparameters(sig))
            checked += 1
        end
    end
    # Non-vacuity: the loop above asserts nothing at all if it never finds a generated signature.
    @test checked > 0
end

"""
    unconstructibleexamples(mod::Module) -> Vector{String}

Every `T{...}(...)` construction in a rendered docstring of `mod` whose `T` cannot actually be
constructed: undefined in that module, or ABSTRACT.

Resolving an interpolation is not the same as producing a runnable example, and this package has had
both failures independently. The generated `getindex` docstring read
`m = AbstractMoment{2,Float64}((3, 3))`: the interpolation was correct, the rendered text was exactly
what the source asked for, and the line still could not be run by anybody, because an abstract type
has no constructor. The escaping check above passes that line without complaint.

Resolution is attempted in the module the docstring belongs to, so an unqualified name that is not
exported is caught too. `Moment{2,Float64}(...)` is unrunnable for a reader who typed
`using H2Factory`, and that reader is who an example is for.
"""
function unconstructibleexamples(mod::Module)
    offenders = String[]
    pattern = r"^\s*(?:@show\s+)?(?:\w+\s*=\s*)?([A-Za-z_][\w.]*)\{[^}]*\}\("
    for m in submodules(mod)
        meta = Base.Docs.meta(m; autoinit=false)
        meta === nothing && continue
        for (binding, multidoc) in meta
            for (_, docstr) in multidoc.docs
                docstr isa Base.Docs.DocStr || continue
                for line in split(join(docstr.text, ""), '\n')
                    matched = match(pattern, line)
                    matched === nothing && continue
                    name = matched.captures[1]
                    # Types the example DEFINES on an earlier line (`MyMoment` from a
                    # `@createmoment`, `NewAbstractMoment` from an `abstract type`) are
                    # constructible by the example itself and out of scope here.
                    (startswith(name, "My") || startswith(name, "New")) && continue
                    resolved = try
                        Core.eval(m, Meta.parse(name))
                    catch
                        nothing
                    end
                    if resolved === nothing
                        push!(offenders, "$(m).$(binding.var): `$(name)` does not resolve")
                    elseif resolved isa Type && isabstracttype(resolved)
                        push!(offenders, "$(m).$(binding.var): `$(name)` is abstract")
                    end
                end
            end
        end
    end
    return offenders
end

# The defect this check exists for, in both of its shapes.
module UnconstructibleFixture

    using H2Factory

    """
    # Example

    ```julia
    m = H2Factory.AbstractMoment{2,Float64}((3, 3))
    ```
    """
    abstractexample() = nothing

    """
    # Example

    ```julia
    m = NoSuchMomentType{2,Float64}((3, 3))
    ```
    """
    undefinedexample() = nothing

end

@testset "generated examples construct something real" begin
    # NEGATIVE CONTROL, both shapes. Without it an empty result below is indistinguishable from a
    # pattern that matches nothing at all.
    caught = unconstructibleexamples(UnconstructibleFixture)
    @test length(caught) == 2
    @test any(o -> occursin("is abstract", o), caught)
    @test any(o -> occursin("does not resolve", o), caught)

    @test isempty(unconstructibleexamples(H2Factory))

    # Non-vacuity, and the specific line this fixed: both generated `getindex` examples name a
    # CONCRETE, QUALIFIED type, and that type really does construct and index as the example claims.
    examples = String[]
    for (binding, multidoc) in Base.Docs.meta(H2Factory)
        string(binding.var) == "getindex" || continue
        for (_, docstr) in multidoc.docs
            for line in split(join(docstr.text, ""), '\n')
                occursin("{2,Float64}((3, 3))", line) && push!(examples, strip(line))
            end
        end
    end
    @test length(examples) == 2
    @test any(e -> occursin("H2Factory.Moment{2,Float64}((3, 3))", e), examples)
    @test any(e -> occursin("H2Factory.MomentCollection{2,Float64}((3, 3))", e), examples)
    # `===`, matching the example. `==` would be flaky here for the reason the example now states:
    # storage is uninitialized, and `NaN == NaN` is false.
    for C in (H2Factory.Moment, H2Factory.MomentCollection)
        m = C{2,Float64}((3, 3))
        @test m[1] === m.moments[1]
    end
end

end # module TestGeneratedDocs
