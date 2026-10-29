module TestAssembleMomentsPotential

using Test
using LinearAlgebra
using StaticArrays
using CompScienceMeshes
using BEAST
using H2Factory

# Tier 2: the H2BEAST driver computing a REAL BEAST quantity, self-contained (no iFMM). We assemble a
# Maxwell far field per box THROUGH the driver -- fake leaf clusters, a callback that reuses BEAST's
# own `farfieldlocal!` (so the kernel/convention are BEAST's, not a reimplementation) and contracts
# the assembly data to global basis functions -- then sum the boxes and compare to
# `BEAST.potential(MWFarField3D, ...)`. Both paths use the same quadrature strategy, so agreement is
# machine-precision, not convergence-limited: any mismatch is a driver traversal/mapping bug.

@testset "far-field potential assembled through the driver == BEAST.potential (Maxwell)" begin
    k = 0.7
    m = meshrectangle(1.0, 1.0, 0.5)
    X = raviartthomas(m)
    nf = numfunctions(X)

    ffop = BEAST.MWFarField3D(; wavenumber=k)
    directions = [
        normalize(SVector(1.0, 0.2, 0.3)),
        normalize(SVector(-0.5, 1.0, 0.1)),
        normalize(SVector(0.0, 0.0, 1.0)),
    ]

    # Reference: BEAST's own far-field, over the whole basis, for a random density.
    coeffs = randn(ComplexF64, nf)
    ffref = BEAST.potential(
        ffop,
        directions,
        coeffs,
        X;
        type=SVector{3,ComplexF64},
        quadstrat=BEAST.SingleNumQStrat(3),
    )
    ffref_flat = ComplexF64[ffref[p][d] for p in eachindex(directions) for d in 1:3]

    # --- assemble the same far field, box by box, through the driver ------------------------------
    rs = BEAST.refspace(X)
    geo = geometry(X)
    nrefs = BEAST.numfunctions(rs, CompScienceMeshes.domain(chart(geo, first(geo))))

    # Fake leaf clusters: partition the basis functions into disjoint groups. Disjoint so each column
    # of `moments` is written by exactly one cluster (one task) -- no accumulation race under `@tasks`.
    clusters = [collect(g) for g in Iterators.partition(1:nf, 4)]
    clusterdata = fill(nothing, length(clusters))

    # moments[direction-component, function] = far-field contribution of that basis function.
    moments = zeros(ComplexF64, 3 * length(directions), nf)

    # Per-cluster scratch: one `zlocal` (length = refspace functions per element) the callback reuses.
    ffbuffer(op, basis, refspace, cdatum, type) = zeros(SVector{3,ComplexF64}, nrefs)

    function ffcallback(
        store,
        op,
        basis,
        fids,
        refspace,
        zlocal,
        elements,
        active,
        ad,
        qdata,
        qs,
        cdatum,
        cluster,
    )
        for localq in eachindex(active)
            a = active[localq]              # global active-element index (indexes elements, ad)
            el = elements[a]
            qr = qdata[1, localq]           # Far-field quadrule; qdata is over elements[active]
            for (pidx, y) in enumerate(directions)
                fill!(zlocal, zero(SVector{3,ComplexF64}))
                BEAST.farfieldlocal!(zlocal, op, refspace, y, el, qr)   # BEAST's own kernel
                for r in eachindex(zlocal)
                    for (n, b) in ad[a, r]  # (global basis function, coefficient) on this element
                        n in fids || continue
                        for d in 1:3
                            moments[3 * (pidx - 1) + d, n] += zlocal[r][d] * b
                        end
                    end
                end
            end
        end
        return nothing
    end

    H2Factory.assemblemoments!(
        nothing,
        ffop,
        X,
        clusters,
        clusterdata,
        Val(SVector{3,ComplexF64}),
        BEAST.SingleNumQStrat(3),
        ffcallback,
        ffbuffer,
    )

    ffdriver = moments * coeffs

    # Machine-precision agreement (same kernel, same quadrature): a driver traversal/mapping bug would
    # break this; a mere quadrature difference cannot.
    @test ffdriver ≈ ffref_flat rtol = 1e-10

    # Guard against a vacuous pass: the far field must be genuinely nonzero.
    @test norm(ffref_flat) > 1e-6
end

end # module TestAssembleMomentsPotential
