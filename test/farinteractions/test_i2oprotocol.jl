module TestI2OProtocol

using Test
using LinearAlgebra
using H2Factory

# Unit tests for the additive I2O protocol foundation (step 1 of the I2O protocol & modulation refactor):
# the `I2OTranslation` composition object, the centralized `couplingkey` direction convention, the
# `NoModulation` no-op hooks, and that `applyi2o!` routes to a coupling collection's `applycoupling!`
# with the direction applied at the coupling boundary. No traversal, no B-spline, no iFMM -- tiny
# hand-checkable coupling data, which is exactly what makes this the safe first consumer of the protocol.

ctx() = H2Factory.TranslationContext(H2Factory.SerialScheduler(), nothing)

# A minimal dense coupling collection: a `(recv, transl) => matrix` dict. It participates exactly as a
# production collection does -- declares the two answers `applycoupling!` asks of every
# `AbstractI2OCouplingCollection`, then supplies a KERNEL -- so this fake drives the SAME dispatch
# production uses, including the guard. Deliberately NON-symmetric complex matrices so a wrong
# direction or a missing key-swap cannot masquerade as correct.
#
# THE CONVENTION IS WRITTEN OUT BY HAND HERE, deliberately: `_fakeblock` spells out the key swap and
# the boundary transform instead of calling `H2Factory.couplingkey`/`directedoperation`. Reusing those
# would make this fake agree with the library by construction, which is not a test. The real
# dense/diagonal/Toeplitz kernels do resolve through them -- see the real-collection regression testset
# below.
struct FakeDenseCoupling{D} <: H2Factory.AbstractI2OCouplingCollection
    blocks::D
end

# Its own storage shape: a plain `(recv, transl) => matrix` dict, distinct from any of H2Factory's.
struct FakeDenseCouplingTrait <: H2Factory.AbstractI2OTranslationTrait end

# A coupling collection by TYPE that declares nothing. Composition accepts it (it is one) and the
# application entry refuses it (it has not opted in) -- the two contracts the composition testset
# separates.
struct UndeclaredI2OCoupling <: H2Factory.AbstractI2OCouplingCollection end

struct RecordingModulation <: H2Factory.AbstractI2OModulation
    calls::Vector{Any}
end

struct ModulationMap{T,C,M}
    translation::T
    channel::C
    trait::M
end

H2Factory.i2otranslator(A::ModulationMap) = A.translation
H2Factory.i2ochannel(A::ModulationMap) = A.channel
H2Factory.maptrait(A::ModulationMap) = A.trait

function H2Factory.modulate_in!(
    moment,
    level::Int,
    trait,
    policy::RecordingModulation,
    coupling,
    channel,
    scheduler,
    buffer,
)
    push!(policy.calls, (:in, moment, level, trait, coupling, channel, scheduler, buffer))
    moment .*= 2
    return moment
end

function H2Factory.modulate_out!(
    moment,
    level::Int,
    trait,
    policy::RecordingModulation,
    coupling,
    channel,
    scheduler,
    buffer,
)
    push!(policy.calls, (:out, moment, level, trait, coupling, channel, scheduler, buffer))
    moment ./= 2
    return moment
end

@testset "Map modulation adapters forward to policy hooks" begin
    @test !isdefined(H2Factory, :i2omodulation!)
    @test !isdefined(H2Factory, :reversei2omodulation!)
    for trait in (H2Factory.isForward(), H2Factory.isTranspose(), H2Factory.isAdjoint())
        policy = RecordingModulation(Any[])
        coupling = FakeDenseCoupling(Dict{NTuple{2,Int},Matrix{ComplexF64}}())
        translation = H2Factory.I2OTranslation(coupling, policy)
        channel, buffer = Ref(:channel), Ref(:buffer)
        A = ModulationMap(translation, channel, trait)
        scheduler = H2Factory.SerialScheduler()
        moment = ComplexF64[1 + im, 2 - im]
        original = copy(moment)

        @test H2Factory.modulate_in!(moment, 3, A, scheduler, buffer) === moment
        @test moment == 2original
        @test only(policy.calls) ===
            (:in, moment, 3, trait, coupling, channel, scheduler, buffer)

        @test H2Factory.modulate_out!(moment, 3, A, scheduler, buffer) === moment
        @test moment == original
        @test length(policy.calls) == 2
        @test policy.calls[2] ===
            (:out, moment, 3, trait, coupling, channel, scheduler, buffer)
    end
end

# Declared, not inherited. A block dict carries no symmetry tags, so none can be left unconsumed.
H2Factory.assertnosymmetry(::FakeDenseCoupling) = nothing
H2Factory.i2otrait(::FakeDenseCoupling) = FakeDenseCouplingTrait()

# forward: the `(recv, transl)` block directly. transpose/adjoint: the SWAPPED `(transl, recv)` block,
# transposed or adjointed at the boundary.
_fakeblock(c::FakeDenseCoupling, recv, transl, ::H2Factory.ForwardTranslation) =
    c.blocks[(recv, transl)]
_fakeblock(c::FakeDenseCoupling, recv, transl, ::H2Factory.TransposeTranslation) =
    transpose(c.blocks[(transl, recv)])
_fakeblock(c::FakeDenseCoupling, recv, transl, ::H2Factory.AdjointTranslation) =
    adjoint(c.blocks[(transl, recv)])

function H2Factory.applycouplingkernel!(
    out,
    in,
    c::FakeDenseCoupling,
    ::FakeDenseCouplingTrait,
    recv::Int,
    transl::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    ::H2Factory.TranslationContext,
)
    LinearAlgebra.mul!(out, _fakeblock(c, recv, transl, direction), in)
    return out
end
function H2Factory.applycouplingkernel!(
    out,
    in,
    c::FakeDenseCoupling,
    ::FakeDenseCouplingTrait,
    recv::Int,
    transl::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    ::H2Factory.TranslationContext,
)
    LinearAlgebra.mul!(out, _fakeblock(c, recv, transl, direction), in, true, true)
    return out
end

@testset "couplingkey: forward keeps order, transpose/adjoint swap" begin
    @test H2Factory.couplingkey(3, 7, H2Factory.ForwardTranslation()) == (3, 7)
    @test H2Factory.couplingkey(3, 7, H2Factory.TransposeTranslation()) == (7, 3)
    @test H2Factory.couplingkey(3, 7, H2Factory.AdjointTranslation()) == (7, 3)
end

@testset "I2OTranslation: default NoModulation + accessors" begin
    c = FakeDenseCoupling(Dict{NTuple{2,Int},Matrix{ComplexF64}}())
    t = H2Factory.I2OTranslation(c)
    @test H2Factory.coupling(t) === c
    @test H2Factory.modulation(t) === H2Factory.NoModulation()

    t2 = H2Factory.I2OTranslation(c, H2Factory.NoModulation())
    @test H2Factory.modulation(t2) === H2Factory.NoModulation()
end

@testset "I2OTranslation composes only declared coupling collections" begin
    # THE COUPLING HALF IS CONSTRAINED, so an invalid composition is rejected at CONSTRUCTION. While
    # the protocol still accepted any object as a coupling this field was unconstrained, and a wrong
    # composition survived until a matvec, where it surfaced as a `MethodError` naming some inner
    # generic rather than the composition that was actually wrong.
    #
    # A bare matrix is the realistic mistake, not a contrived one: the coupling for a single node pair
    # IS a matrix, so passing the block where the collection belongs is the obvious slip.
    for notacollection in (42, randn(2, 2), Dict((1, 2) => randn(2, 2)), nothing)
        @test_throws MethodError H2Factory.I2OTranslation(notacollection)
        @test_throws MethodError H2Factory.I2OTranslation(
            notacollection, H2Factory.NoModulation()
        )
    end

    # Positive: everything that IS a declared coupling collection composes -- a production collection,
    # the dummy, and a test fake. All three reach the same constructor.
    production = H2Factory.I2OTranslationCollection(
        Dict((2, 9) => randn(ComplexF64, 2, 2)), H2Factory.DenseTranslation()
    )
    for collection in (
        production,
        H2Factory.DummyI2OTranslationCollection(),
        FakeDenseCoupling(Dict((2, 9) => randn(ComplexF64, 2, 2))),
    )
        @test collection isa H2Factory.AbstractI2OCouplingCollection
        t = H2Factory.I2OTranslation(collection)
        @test H2Factory.coupling(t) === collection
        @test H2Factory.modulation(t) === H2Factory.NoModulation()

        # A bare collection is its own coupling and carries no modulation -- the defaults that used to
        # be `coupling(x) = x` for ANY `x` and now answer only for declared collections.
        @test H2Factory.coupling(collection) === collection
        @test H2Factory.modulation(collection) === H2Factory.NoModulation()
    end

    # Those defaults no longer answer for anything else. Before narrowing, every one of these was
    # silently "its own coupling, no modulation".
    for notacollection in (42, randn(2, 2), nothing)
        @test_throws MethodError H2Factory.coupling(notacollection)
        @test_throws MethodError H2Factory.modulation(notacollection)
    end

    # TWO DIFFERENT CONTRACTS, and the constructor only enforces the first. Being an
    # `AbstractI2OCouplingCollection` is a TYPE claim and is what composition checks; having declared
    # `assertnosymmetry`/`i2otrait` is a BEHAVIOURAL claim and is what the `applycoupling!` entry
    # checks, on first use. A subtype that declares neither composes fine and then refuses loudly,
    # which is the right split: construction cannot know what a type will declare later.
    @test H2Factory.I2OTranslation(UndeclaredI2OCoupling()) isa H2Factory.I2OTranslation
    @test_throws "assertnosymmetry is not declared" H2Factory.applyi2o!(
        nothing,
        nothing,
        UndeclaredI2OCoupling(),
        2,
        9,
        H2Factory.ForwardTranslation(),
        H2Factory.Overwrite(),
        ctx(),
    )
end

@testset "NoModulation modulate_in!/modulate_out! are no-ops (return the moment unchanged)" begin
    # Signature matches the per-node traversal hook: (moment, level, maptrait, modulation, coupling,
    # channel, scheduler, buffer). For an ordinary (non-Toeplitz) moment NoModulation touches nothing.
    moment = ComplexF64[1 + 1im, 2 - 3im]
    sched = H2Factory.SerialScheduler()
    out_in = H2Factory.modulate_in!(
        moment,
        2,
        H2Factory.isForward(),
        H2Factory.NoModulation(),
        :coupling,
        nothing,
        sched,
        nothing,
    )
    out_out = H2Factory.modulate_out!(
        moment,
        2,
        H2Factory.isForward(),
        H2Factory.NoModulation(),
        :coupling,
        nothing,
        sched,
        nothing,
    )
    @test out_in === moment          # aliases, no conversion
    @test out_out === moment
    @test moment == ComplexF64[1 + 1im, 2 - 3im]   # untouched
end

@testset "applyi2o!: routes to applycoupling!, direction applied at coupling boundary" begin
    # Distinct non-symmetric complex blocks for the forward key (recv, transl) and the swapped key
    # (transl, recv), so forward vs transpose/adjoint resolve genuinely different data AND apply a
    # different linear map. A single symmetric block would hide both a wrong direction and a missing swap.
    recv, transl = 2, 9
    Mforward = ComplexF64[1+2im 3-1im; 0 4+0im]      # used by forward:  key (recv, transl)
    Mswapped = ComplexF64[5+0im 0; 6-2im 7+1im]      # used by t/a:      key (transl, recv)
    blocks = Dict((recv, transl) => Mforward, (transl, recv) => Mswapped)
    t = H2Factory.I2OTranslation(FakeDenseCoupling(blocks))

    x = ComplexF64[1 - 1im, 2 + 3im]

    for (dir, expected) in (
        (H2Factory.ForwardTranslation(), Mforward * x),
        (H2Factory.TransposeTranslation(), transpose(Mswapped) * x),
        (H2Factory.AdjointTranslation(), adjoint(Mswapped) * x),
    )
        out = zeros(ComplexF64, 2)
        H2Factory.applyi2o!(out, x, t, recv, transl, dir, H2Factory.Overwrite(), ctx())
        @test out ≈ expected

        # Accumulate adds onto a prefilled target.
        pre = ComplexF64[10 + 0im, -4 + 1im]
        outacc = copy(pre)
        H2Factory.applyi2o!(outacc, x, t, recv, transl, dir, H2Factory.Accumulate(), ctx())
        @test outacc ≈ pre .+ expected
    end
end

@testset "applyi2o!: transpose and adjoint differ for complex coupling, agree for real" begin
    recv, transl = 4, 8
    Mc = ComplexF64[1+2im 3-1im; 0 4+1im]            # complex, non-symmetric
    tc = H2Factory.I2OTranslation(FakeDenseCoupling(Dict((transl, recv) => Mc)))
    x = ComplexF64[5 - 1im, 2 + 3im]

    outT = zeros(ComplexF64, 2)
    outA = zeros(ComplexF64, 2)
    H2Factory.applyi2o!(
        outT,
        x,
        tc,
        recv,
        transl,
        H2Factory.TransposeTranslation(),
        H2Factory.Overwrite(),
        ctx(),
    )
    H2Factory.applyi2o!(
        outA,
        x,
        tc,
        recv,
        transl,
        H2Factory.AdjointTranslation(),
        H2Factory.Overwrite(),
        ctx(),
    )
    @test !(outT ≈ outA)
    @test outA ≈ adjoint(Mc) * x

    Mr = ComplexF64[1 3; 0 4]                         # real values, non-symmetric
    tr = H2Factory.I2OTranslation(FakeDenseCoupling(Dict((transl, recv) => Mr)))
    outTr = zeros(ComplexF64, 2)
    outAr = zeros(ComplexF64, 2)
    H2Factory.applyi2o!(
        outTr,
        x,
        tr,
        recv,
        transl,
        H2Factory.TransposeTranslation(),
        H2Factory.Overwrite(),
        ctx(),
    )
    H2Factory.applyi2o!(
        outAr,
        x,
        tr,
        recv,
        transl,
        H2Factory.AdjointTranslation(),
        H2Factory.Overwrite(),
        ctx(),
    )
    @test outTr ≈ outAr
end

@testset "applyi2o!: bare coupling collection (no I2OTranslation wrapper) also routes" begin
    recv, transl = 1, 2
    M = ComplexF64[2+0im 1-1im; 0 3+0im]
    c = FakeDenseCoupling(Dict((recv, transl) => M))
    x = ComplexF64[1 + 0im, 1 + 1im]
    out = zeros(ComplexF64, 2)
    H2Factory.applyi2o!(
        out,
        x,
        c,
        recv,
        transl,
        H2Factory.ForwardTranslation(),
        H2Factory.Overwrite(),
        ctx(),
    )
    @test out ≈ M * x
end

# Regression test for the COMPLETED `couplingkey` wiring, on the REAL dense/diagonal
# `I2OTranslationCollection` kernels. The `FakeDenseCoupling` tests above pin the PROTOCOL contract
# (forward keys `(recv,transl)`, transpose/adjoint key `(transl,recv)`); this pins the CONCRETE kernels,
# which now resolve the block via `couplingkey(recv, transl, <direction>())` (no longer a hardcoded
# tuple), with the direction's transform (transpose/adjoint) still at the coupling boundary. DISTINCT
# complex non-symmetric blocks at the two keys make the swap discriminating:
#   * a MISSING swap (transpose/adjoint reading the forward block) fails numerically,
#   * a DOUBLE swap (forward reading the swapped block) fails numerically,
#   * a transpose-vs-adjoint collapse fails on the complex, non-Hermitian block.
# Verified through BOTH the raw kernels AND the public `applyi2o!(I2OTranslation(coll, NoModulation()))`
# path, so a future change to the swap/direction handling has a fixed, discriminating contract to match.
# `Moment1D` (hand-checkable, no traversal), matching the real kernels' moment API.
@testset "real Dense/Diagonal kernels: non-symmetric swap (couplingkey acceptance gate)" begin
    recv, transl = 3, 8
    mkm(v) = (m=H2Factory.Moment1D{ComplexF64}((2,)); m[1].=v; m)
    x = ComplexF64[1 - 1im, 2 + 3im]

    # --- DenseTranslation: forward applies the (recv,transl) block; transpose/adjoint apply the
    #     transpose/adjoint of the SWAPPED (transl,recv) block. ---
    Drt = ComplexF64[1+2im 3-1im; 0+0im 4+0im]      # forward key (recv, transl)
    Dtr = ComplexF64[5+0im 0+1im; 6-2im 7+1im]      # swapped key (transl, recv); non-Hermitian
    densecoll = H2Factory.I2OTranslationCollection(
        Dict((recv, transl) => Drt, (transl, recv) => Dtr), H2Factory.DenseTranslation()
    )
    denset = H2Factory.I2OTranslation(densecoll)
    for (dir, expected) in (
        (H2Factory.ForwardTranslation(), Drt * x),
        (H2Factory.TransposeTranslation(), transpose(Dtr) * x),
        (H2Factory.AdjointTranslation(), adjoint(Dtr) * x),
    )
        # The KERNEL, below the guard: what a collection implements.
        oraw = mkm(zeros(ComplexF64, 2))
        H2Factory.applycouplingkernel!(
            oraw,
            mkm(x),
            densecoll,
            H2Factory.i2otrait(densecoll),
            recv,
            transl,
            dir,
            H2Factory.Overwrite(),
            ctx(),
        )
        @test oraw[1] ≈ expected

        opub = mkm(zeros(ComplexF64, 2))
        H2Factory.applyi2o!(
            opub, mkm(x), denset, recv, transl, dir, H2Factory.Overwrite(), ctx()
        )
        @test opub[1] ≈ expected
    end
    # discriminating-data guards: forward vs swapped block differ, and transpose vs adjoint do not
    # collapse -- so a future swap or direction bug cannot pass unnoticed.
    @test !(transpose(Dtr) * x ≈ transpose(Drt) * x)
    @test !(transpose(Dtr) * x ≈ adjoint(Dtr) * x)

    # --- DiagonalTranslation: transpose applies the SWAPPED coeffs untransformed; adjoint conjugates. ---
    drt = ComplexF64[2 + 1im, 3 - 2im]              # forward key (recv, transl)
    dtr = ComplexF64[5 - 1im, 1 + 4im]              # swapped key (transl, recv)
    diagcoll = H2Factory.I2OTranslationCollection(
        Dict((recv, transl) => drt, (transl, recv) => dtr), H2Factory.DiagonalTranslation()
    )
    diagt = H2Factory.I2OTranslation(diagcoll)
    for (dir, expected) in (
        (H2Factory.ForwardTranslation(), drt .* x),
        (H2Factory.TransposeTranslation(), dtr .* x),
        (H2Factory.AdjointTranslation(), conj.(dtr) .* x),
    )
        oraw = mkm(zeros(ComplexF64, 2))
        H2Factory.applycouplingkernel!(
            oraw,
            mkm(x),
            diagcoll,
            H2Factory.i2otrait(diagcoll),
            recv,
            transl,
            dir,
            H2Factory.Overwrite(),
            ctx(),
        )
        @test oraw[1] ≈ expected

        opub = mkm(zeros(ComplexF64, 2))
        H2Factory.applyi2o!(
            opub, mkm(x), diagt, recv, transl, dir, H2Factory.Overwrite(), ctx()
        )
        @test opub[1] ≈ expected
    end
    @test !(dtr .* x ≈ drt .* x)
    @test !(dtr .* x ≈ conj.(dtr) .* x)
end

end # module TestI2OProtocol
