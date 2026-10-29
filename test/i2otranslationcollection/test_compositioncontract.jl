module TestI2OCompositionContract

using Test
using H2Factory

# THE I2O COMPOSITION CONTRACT, MADE EXECUTABLE.
#
# An `I2OTranslation` is exactly a COUPLING plus a MODULATION, and the traversal reaches those two
# halves only through the `coupling` and `modulation` accessors. A custom composite wrapper must be
# able to override BOTH and be driven unchanged.
#
# WHAT THIS FILE PINNED BEFORE #180, AND WHAT IT PINS NOW. The defaults used to be
# `coupling(x) = x` / `modulation(x) = NoModulation()` for ANY `x`, so a wrapper that forgot to
# override `coupling` did not fail -- it was applied AS a coupling, quietly standing in for the
# collection it held. This file existed to make that silent failure visible.
#
# #180 narrowed both defaults to `AbstractI2OCouplingCollection`, which removes that failure for any
# wrapper that is not itself a coupling collection: it is now a `MethodError`, not a wrong answer. The
# middle testset below therefore pins the LOUD behaviour, and a third case pins what genuinely
# remains -- a composite that IS a coupling collection still resolves to itself.
#
# The fixture uses DIFFERENT field names from `I2OTranslation`'s, so nothing can satisfy it by
# accident.

struct RenamedComposite{C,M}
    thecoupling::C
    themodulation::M
end

H2Factory.coupling(t::RenamedComposite) = t.thecoupling
H2Factory.modulation(t::RenamedComposite) = t.themodulation

# A coupling, so the bare-coupling defaults apply to it. It has to say so now: the defaults answer for
# declared coupling collections rather than for anything at all.
struct MarkerCoupling <: H2Factory.AbstractI2OCouplingCollection end
struct MarkerModulation end

# A composite wrapper that forgets to override `coupling`, and is NOT a coupling collection.
struct UnforwardedComposite{C,M}
    thecoupling::C
    themodulation::M
end

# The same mistake made by a type that IS a coupling collection. This one still resolves to itself.
struct UnforwardedCouplingCollection{C,M} <: H2Factory.AbstractI2OCouplingCollection
    thecoupling::C
    themodulation::M
end

@testset "a composite may expose its parts under its own field names" begin
    wrapper = RenamedComposite(MarkerCoupling(), MarkerModulation())

    @test H2Factory.coupling(wrapper) === MarkerCoupling()
    @test H2Factory.modulation(wrapper) === MarkerModulation()

    # NON-VACUITY, and the reason the fields are renamed: `I2OTranslation` reads `.coupling` and
    # `.modulation`, which this type does not have, so a field-reading implementation could not
    # satisfy the assertions above rather than merely returning something wrong.
    @test !hasfield(RenamedComposite, :coupling)
    @test !hasfield(RenamedComposite, :modulation)

    # Both overrides land on the `Interfaces` bindings, not same-named locals.
    @test H2Factory.coupling === H2Factory.Interfaces.coupling
    @test H2Factory.modulation === H2Factory.Interfaces.modulation
    @test which(H2Factory.coupling, Tuple{RenamedComposite}).module === @__MODULE__
    @test which(H2Factory.modulation, Tuple{RenamedComposite}).module === @__MODULE__
end

@testset "a non-collection wrapper that forgets the override now fails loudly" begin
    # THE FORMER SILENT FAILURE, now an error. While the defaults were `coupling(x) = x` for any `x`,
    # this wrapper resolved to ITSELF -- no error, just the wrapper standing in for the coupling it
    # holds, and handed to coupling application in that role. #180 narrowed the defaults to
    # `AbstractI2OCouplingCollection`, so a wrapper that is not one gets a `MethodError` instead.
    bare = UnforwardedComposite(MarkerCoupling(), MarkerModulation())

    @test !(bare isa H2Factory.AbstractI2OCouplingCollection)
    @test_throws MethodError H2Factory.coupling(bare)
    @test_throws MethodError H2Factory.modulation(bare)

    # It cannot be composed either, which is the second barrier: `I2OTranslation` constrains its
    # coupling half, so this never reaches a matvec to fail inside.
    @test_throws MethodError H2Factory.I2OTranslation(bare)

    # NON-VACUITY: the overriding wrapper with the same shape is fine, so the failures above are
    # about the missing override rather than about the fixture being unusable.
    @test H2Factory.coupling(RenamedComposite(MarkerCoupling(), MarkerModulation())) ===
        MarkerCoupling()
end

@testset "a composite that IS a coupling collection still resolves to itself" begin
    # WHAT REMAINS, pinned honestly rather than left to be rediscovered. The narrowing removed the
    # hazard for non-collections; it cannot remove it for a composite that subtypes
    # `AbstractI2OCouplingCollection`, because being one is exactly the claim the default answers.
    bare = UnforwardedCouplingCollection(MarkerCoupling(), MarkerModulation())

    @test H2Factory.coupling(bare) === bare
    @test H2Factory.coupling(bare) !== MarkerCoupling()

    # And it is the DEFAULT doing that, not a specific method -- so this catches any coupling
    # collection that forgets to override, not just this fixture.
    resolved = which(H2Factory.coupling, Tuple{UnforwardedCouplingCollection})
    @test resolved.sig == Tuple{
        typeof(H2Factory.Interfaces.coupling),H2Factory.AbstractI2OCouplingCollection
    }
    @test resolved.module === H2Factory

    # THE RESIDUAL IS NARROWER THAN IT LOOKS, though, and this is why it is acceptable rather than a
    # bug to fix: reaching a matvec at all now requires `assertnosymmetry` and `i2otrait` as well, and
    # this type declares neither. A silently wrong composition therefore needs a type that declared
    # both and forgot only `coupling` -- not merely one that forgot everything.
    @test_throws "assertnosymmetry is not declared" H2Factory.applyi2o!(
        nothing,
        nothing,
        bare,
        3,
        8,
        H2Factory.ForwardTranslation(),
        H2Factory.Overwrite(),
        H2Factory.TranslationContext(H2Factory.SerialScheduler(), nothing),
    )
end

@testset "the bare-coupling defaults still hold for a declared collection" begin
    # The contract must not have cost the default that makes a mock coupling usable unwrapped.
    @test MarkerCoupling() isa H2Factory.AbstractI2OCouplingCollection
    @test H2Factory.coupling(MarkerCoupling()) === MarkerCoupling()
    @test H2Factory.modulation(MarkerCoupling()) === H2Factory.NoModulation()
end

end # module TestI2OCompositionContract
