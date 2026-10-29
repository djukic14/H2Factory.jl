module TestH2BEASTWrappers

using Test
using BEAST
using H2Factory

@testset "EFIE wrappers" begin
    @efie efie

    k = 2.0
    γ = k * im
    α = 3.0

    op = efie(; gamma=γ, alpha=α)
    opcomparison = Maxwell3D.singlelayer(; gamma=γ, alpha=Complex(α), beta=α / γ^2)

    @test op.operator.α == opcomparison.α
    @test op.operator.β == opcomparison.β
    @test BEAST.gamma(op) == BEAST.gamma(opcomparison)
    @test BEAST.Integrand(op, 1.0, 2.0, 3.0, 4.0) ==
        BEAST.Integrand(opcomparison, 1.0, 2.0, 3.0, 4.0)
    @test BEAST.scalartype(op) == BEAST.scalartype(opcomparison)
    @test BEAST.regularpart(op) == BEAST.regularpart(opcomparison)
    @test BEAST.singularpart(op) == BEAST.singularpart(opcomparison)

    op2 = efie(opcomparison)
    @test op == op2
    invalid = Maxwell3D.singlelayer(; gamma=γ, alpha=1.0, beta=1.0)
    @test_throws ArgumentError efie(invalid)

    @test H2Factory.isgalerkinsymmetric(op)
    @test H2Factory.isgalerkinsymmetric(opcomparison)
end

@testset "MWHyperSingular3D wrappers" begin
    @mw3Dhypersingular mw3Dhypersingular

    k = 2.0
    γ = k * im
    β = 3.0

    op = mw3Dhypersingular(; gamma=γ, beta=β)
    opcomparison = Maxwell3D.singlelayer(; gamma=γ, alpha=Complex(0.0), beta=Complex(β))

    @test op.operator.α == opcomparison.α
    @test op.operator.β == opcomparison.β
    @test BEAST.gamma(op) == BEAST.gamma(opcomparison)
    @test BEAST.Integrand(op, 1.0, 2.0, 3.0, 4.0) ==
        BEAST.Integrand(opcomparison, 1.0, 2.0, 3.0, 4.0)
    @test BEAST.scalartype(op) == BEAST.scalartype(opcomparison)
    @test BEAST.regularpart(op) == BEAST.regularpart(opcomparison)
    @test BEAST.singularpart(op) == BEAST.singularpart(opcomparison)

    op2 = mw3Dhypersingular(opcomparison)
    @test op == op2
    invalid_alpha = Maxwell3D.singlelayer(; gamma=γ, alpha=1.0, beta=β)
    invalid_beta = Maxwell3D.singlelayer(; gamma=γ, alpha=0.0, beta=0.0)
    @test_throws ArgumentError mw3Dhypersingular(invalid_alpha)
    @test_throws ArgumentError mw3Dhypersingular(invalid_beta)

    @test H2Factory.isgalerkinsymmetric(op)
    @test H2Factory.isgalerkinsymmetric(opcomparison)
end
@testset "MWWeaklySingular3D wrappers" begin
    @mw3Dweaklysingular mw3Dweaklysingular

    k = 2.0
    γ = k * im
    α = 3.0

    op = mw3Dweaklysingular(; gamma=γ, alpha=α)
    opcomparison = Maxwell3D.singlelayer(; gamma=γ, alpha=Complex(α), beta=Complex(0.0))

    @test op.operator.α == opcomparison.α
    @test op.operator.β == opcomparison.β
    @test BEAST.gamma(op) == BEAST.gamma(opcomparison)
    @test BEAST.Integrand(op, 1.0, 2.0, 3.0, 4.0) ==
        BEAST.Integrand(opcomparison, 1.0, 2.0, 3.0, 4.0)
    @test BEAST.scalartype(op) == BEAST.scalartype(opcomparison)
    @test BEAST.regularpart(op) == BEAST.regularpart(opcomparison)
    @test BEAST.singularpart(op) == BEAST.singularpart(opcomparison)

    op2 = mw3Dweaklysingular(opcomparison)
    @test op == op2
    invalid_alpha = Maxwell3D.singlelayer(; gamma=γ, alpha=0.0, beta=0.0)
    invalid_beta = Maxwell3D.singlelayer(; gamma=γ, alpha=α, beta=1.0)
    @test_throws ArgumentError mw3Dweaklysingular(invalid_alpha)
    @test_throws ArgumentError mw3Dweaklysingular(invalid_beta)

    @test H2Factory.isgalerkinsymmetric(op)
    @test H2Factory.isgalerkinsymmetric(opcomparison)
end

@testset "MWDoubleLayer3D wrappers" begin
    @mw3Ddoublelayer mw3Ddoublelayerwrapper
    k = 2.0
    γ = k * im
    α = 1.0

    op = mw3Ddoublelayerwrapper(; gamma=γ, alpha=α)
    opcomparison = Maxwell3D.doublelayer(; gamma=γ, alpha=α)

    @test op.operator.alpha == opcomparison.alpha
    @test BEAST.gamma(op) == BEAST.gamma(opcomparison)
    @test BEAST.Integrand(op, 1.0, 2.0, 3.0, 4.0) ==
        BEAST.Integrand(opcomparison, 1.0, 2.0, 3.0, 4.0)
    @test BEAST.scalartype(op) == BEAST.scalartype(opcomparison)
    @test BEAST.regularpart(op) == BEAST.regularpart(opcomparison)
    @test BEAST.singularpart(op) == BEAST.singularpart(opcomparison)

    op2 = mw3Ddoublelayerwrapper(opcomparison)
    @test op == op2

    @test H2Factory.isgalerkinsymmetric(op)
    @test H2Factory.isgalerkinsymmetric(opcomparison)
end

@testset "HH3DDoubleLayerFDBIO wrappers" begin
    @hh3Ddoublelayer hh3Ddoublelayer

    k = 2.0
    γ = k * im
    α = 1.0

    op = hh3Ddoublelayer(; gamma=γ, alpha=α)
    opcomparison = Helmholtz3D.doublelayer(; gamma=γ, alpha=α)

    @test op.operator.alpha == opcomparison.alpha
    @test BEAST.gamma(op) == BEAST.gamma(opcomparison)
    @test BEAST.Integrand(op, 1.0, 2.0, 3.0, 4.0) ==
        BEAST.Integrand(opcomparison, 1.0, 2.0, 3.0, 4.0)
    @test BEAST.scalartype(op) == BEAST.scalartype(opcomparison)
    @test BEAST.regularpart(op) == BEAST.regularpart(opcomparison)
    @test BEAST.singularpart(op) == BEAST.singularpart(opcomparison)

    op2 = hh3Ddoublelayer(opcomparison)
    @test op == op2

    @test !H2Factory.isgalerkinsymmetric(op)
    @test !H2Factory.isgalerkinsymmetric(opcomparison)
end

@testset "HH3DDoubleLayerTransposedFDBIO wrappers" begin
    @hh3Ddoublelayer_transposed hh3Ddoublelayer_transposed

    k = 2.0
    γ = k * im
    α = 1.0

    op = hh3Ddoublelayer_transposed(; gamma=γ, alpha=α)
    opcomparison = Helmholtz3D.doublelayer_transposed(; gamma=γ, alpha=α)

    @test op.operator.alpha == opcomparison.alpha
    @test BEAST.gamma(op) == BEAST.gamma(opcomparison)
    @test BEAST.Integrand(op, 1.0, 2.0, 3.0, 4.0) ==
        BEAST.Integrand(opcomparison, 1.0, 2.0, 3.0, 4.0)
    @test BEAST.scalartype(op) == BEAST.scalartype(opcomparison)
    @test BEAST.regularpart(op) == BEAST.regularpart(opcomparison)
    @test BEAST.singularpart(op) == BEAST.singularpart(opcomparison)

    op2 = hh3Ddoublelayer_transposed(opcomparison)
    @test op == op2

    @test !H2Factory.isgalerkinsymmetric(op)
    @test !H2Factory.isgalerkinsymmetric(opcomparison)
end

@testset "Remaining Isgalerkinsymmetric" begin
    @test H2Factory.isgalerkinsymmetric(BEAST.HH3DSingleLayerFDBIO)
    @test H2Factory.isgalerkinsymmetric(BEAST.HH3DHyperSingularFDBIO)
    @test H2Factory.isgalerkinsymmetric(BEAST.MWSingleLayer3D)
    @test !H2Factory.isgalerkinsymmetric(Float64)
end

end # module TestH2BEASTWrappers
