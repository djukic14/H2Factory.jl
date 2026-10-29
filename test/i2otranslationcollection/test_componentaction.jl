module TestComponentAction

using Test
using FFTW
using LinearAlgebra
using StaticArrays
using H2Factory, H2Trees

# THE COMPONENT HALF OF THE SYMMETRY ACTION.
#
# `toeplitzpermutation` handles the SAMPLE index and is the same for every operator. A VECTOR or
# DYADIC kernel also transforms its FIELD COMPONENTS, and that part is a property of the kernel:
#
#     scalar   g        no component action
#     vector   ∇g       ∇g(Sx) == S ∇g(x)        -> component i from perm[i], scaled by signs[i]
#     dyadic   G        G(Sx)  == S G(x) Sᵀ      -> component (i,j) from (perm[i], perm[j]),
#                                                   scaled by signs[i]·signs[j]
#
# `componentsource` states which canonical component supplies which image component and with what
# scalar, so a consumer supplies only WHICH of the three shapes its operator is. This file checks that
# statement against kernels built independently at `S d`, and checks that dropping the signs breaks it
# -- a sign-free version looks plausible and passes on the identity element.

# An isotropic dyad, the general form of an EFIE-style Green dyad:
#     G(x) = I f(|x|) + (x xᵀ / |x|²) h(|x|)
# Its isotropy is the hypothesis `G(Sx) == S G(x) Sᵀ` needs; the parts are complex so a dropped
# conjugation would also show.
f(r) = exp(1im * 1.1 * r) / (r + 0.4)
h(r) = exp(1im * 1.1 * r) * (0.7 - 0.3im) / (r + 0.9)

function dyad(x)
    r = norm(x)
    return [
        (i == j ? f(r) : zero(ComplexF64)) + x[i] * x[j] / r^2 * h(r) for i in 1:3, j in 1:3
    ]
end

gradient(x) = [x[i] / norm(x) * h(norm(x)) for i in 1:3]

_wraporder(n) = vcat(0:(n - 1), (-(n - 1)):-1)

# Component `(i,j)` of the difference tensor at displacement `d`.
function dyadtensor(d, spacing, n, i, j)
    m = _wraporder(n)
    return [dyad(d + spacing * SVector(a, b, c))[i, j] for a in m, b in m, c in m]
end

function gradienttensor(d, spacing, n, i)
    m = _wraporder(n)
    return [gradient(d + spacing * SVector(a, b, c))[i] for a in m, b in m, c in m]
end

const CHIRAL = H2Trees.LatticeSymmetry((2, 3, 1), (1, -1, 1))

@testset "componentsource states the signed permutation of S v and S A Sᵀ" begin
    # Against the matrices themselves, before any tensor is involved.
    signs = (1, -1, 1)
    perm = (2, 3, 1)
    S = zeros(Float64, 3, 3)
    for i in 1:3
        S[i, perm[i]] = signs[i]
    end
    @test S * transpose(S) ≈ I
    @test det(S) ≈ -1                     # a reflection: exactly where dropped signs bite

    x = SVector(1.7, -0.6, 2.3)
    Sx = SVector((S * x)...)

    @test dyad(Sx) ≈ S * dyad(x) * transpose(S)
    @test gradient(Sx) ≈ S * gradient(x)

    A = dyad(x)
    for i in 1:3, j in 1:3
        sourcei, sourcej, scale = H2Factory.componentsource(CHIRAL, i, j)
        @test (S * A * transpose(S))[i, j] ≈ scale * A[sourcei, sourcej]
    end

    v = gradient(x)
    for i in 1:3
        source, scale = H2Factory.componentsource(CHIRAL, i)
        @test (S * v)[i] ≈ scale * v[source]
    end
end

@testset "Dyadic: the full action reproduces a tensor built independently at S d" begin
    n = 3
    L = 2n - 1
    spacing = 1.0
    d = SVector(5.0, 2.0, -3.0)
    samplepermutation = H2Factory.toeplitzpermutation((L, L, L), CHIRAL)

    worst = 0.0
    scale_ = 0.0
    for i in 1:3, j in 1:3
        image = dyadtensor(
            SVector(H2Trees.applysymmetry(CHIRAL, Tuple(d))), spacing, n, i, j
        )
        sourcei, sourcej, scale = H2Factory.componentsource(CHIRAL, i, j)
        canonical = dyadtensor(d, spacing, n, sourcei, sourcej)
        view = H2Factory.ScaledPermutedTranslation(canonical, samplepermutation, scale)
        worst = max(worst, maximum(abs, image - collect(view)))
        scale_ = max(scale_, maximum(abs, image))
    end
    @test worst / scale_ < 1e-13

    # AND IN THE FREQUENCY DOMAIN, which is what a Toeplitz collection actually stores.
    worstfreq = 0.0
    for i in 1:3, j in 1:3
        image = fft(
            dyadtensor(SVector(H2Trees.applysymmetry(CHIRAL, Tuple(d))), spacing, n, i, j)
        )
        sourcei, sourcej, scale = H2Factory.componentsource(CHIRAL, i, j)
        canonical = fft(dyadtensor(d, spacing, n, sourcei, sourcej))
        view = H2Factory.ScaledPermutedTranslation(canonical, samplepermutation, scale)
        worstfreq = max(worstfreq, maximum(abs, image - collect(view)))
    end
    @test worstfreq < 1e-12
end

@testset "Dropping the signs breaks it, and dropping the component permutation does too" begin
    # NON-VACUITY, and specifically for the two pieces that look like bookkeeping. `CHIRAL` is a
    # reflection, so a sign-free version is wrong by O(1) -- but it would pass on the identity element,
    # which is why the control uses this one.
    n = 3
    L = 2n - 1
    d = SVector(5.0, 2.0, -3.0)
    samplepermutation = H2Factory.toeplitzpermutation((L, L, L), CHIRAL)

    signdropped = 0.0
    permdropped = 0.0
    reference = 0.0
    for i in 1:3, j in 1:3
        image = dyadtensor(SVector(H2Trees.applysymmetry(CHIRAL, Tuple(d))), 1.0, n, i, j)
        sourcei, sourcej, scale = H2Factory.componentsource(CHIRAL, i, j)
        reference = max(reference, maximum(abs, image))

        nosign = H2Factory.ScaledPermutedTranslation(
            dyadtensor(d, 1.0, n, sourcei, sourcej), samplepermutation, 1
        )
        signdropped = max(signdropped, maximum(abs, image - collect(nosign)))

        noperm = H2Factory.ScaledPermutedTranslation(
            dyadtensor(d, 1.0, n, i, j), samplepermutation, scale
        )
        permdropped = max(permdropped, maximum(abs, image - collect(noperm)))
    end
    @test signdropped / reference > 1e-2
    @test permdropped / reference > 1e-2
end

@testset "Vector: the same statement with one index" begin
    n = 3
    L = 2n - 1
    d = SVector(5.0, 2.0, -3.0)
    samplepermutation = H2Factory.toeplitzpermutation((L, L, L), CHIRAL)

    worst = 0.0
    reference = 0.0
    for i in 1:3
        image = gradienttensor(SVector(H2Trees.applysymmetry(CHIRAL, Tuple(d))), 1.0, n, i)
        source, scale = H2Factory.componentsource(CHIRAL, i)
        view = H2Factory.ScaledPermutedTranslation(
            gradienttensor(d, 1.0, n, source), samplepermutation, scale
        )
        worst = max(worst, maximum(abs, image - collect(view)))
        reference = max(reference, maximum(abs, image))
    end
    @test worst / reference < 1e-13
end

@testset "ScaledPermutedTranslation is a plain lazy array" begin
    canonical = ComplexF64[1, 2, 3, 4]
    permutation = Int32[4, 3, 2, 1]
    view = H2Factory.ScaledPermutedTranslation(canonical, permutation, -2)
    @test size(view) == size(canonical)
    @test Base.IndexStyle(typeof(view)) === IndexLinear()
    @test collect(view) == -2 .* [4, 3, 2, 1]
    @test eltype(view) == ComplexF64

    # The length check is the same guard `PermutedTranslation` has, and it is what turns a
    # wrong-shaped wrapping into a named error rather than a plausible result.
    @test_throws DimensionMismatch H2Factory.ScaledPermutedTranslation(
        canonical, Int32[1, 2], 1
    )
end

end # module TestComponentAction
