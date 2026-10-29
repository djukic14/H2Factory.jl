# Shared geometry and kernel fixtures for the component-level far/near interaction harnesses.
# Keep assertions in the individual test files; this file only defines reusable test data.

b4points() = [0.0, 0.1, 5.0, 5.1, 40.0, 40.1, 45.0, 45.1]

function b4tree()
    pts = [SVector(x, 0.0, 0.0) for x in b4points()]
    return H2Trees.buildtree(
        pts; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=2.0, minvalues=0, root=1)
    ),
    length(pts)
end

uniformpts(N) = [(i + 0.5) / N for i in 0:(N - 1)]

nonuniformpoints() = vcat([0.001 * i for i in 0:127], [5.0, 10.0, 20.0, 40.0, 80.0])

function nonuniformtree()
    pts = [SVector(x, 0.0, 0.0) for x in nonuniformpoints()]
    return H2Trees.buildtree(
        pts; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.0005, root=1, minvalues=1)
    ),
    length(pts)
end

function nodelevel(tree, node)
    l = 1
    n = node
    while n != H2Trees.root(tree)
        n = H2Trees.parent(tree, n)
        l += 1
    end
    return l
end
leaflevels(tree) = sort(unique(nodelevel(tree, lf) for lf in H2Trees.leaves(tree)))

function manyclusters(k, n; gap=100.0, spread=0.1)
    pts = SVector{3,Float64}[]
    cid = Int[]
    for c in 1:k
        for j in 0:(n - 1)
            push!(pts, SVector((c - 1) * gap + spread * j / (n - 1), 0.0, 0.0))
            push!(cid, c)
        end
    end
    return pts, cid
end

petrovunb(base=0.0) = vcat(
    [base + 0.002 * i for i in 0:31],
    [base + 0.3, base + 0.45, base + 0.6, base + 0.75, base + 0.9, base + 1.1, base + 1.4],
)

mkpts(v) = [(p, 0.0, 0.0) for p in v]

function dense_petrov_coordinate_configs()
    u = uniformpts(128)
    b = petrovunb()
    return (
        ("balanced coincident", u, u, 0, 0, 0.006),
        ("balanced disjoint-far", u, 100.0 .+ u, 0, 0, 0.006),
        ("balanced overlapping", u, 0.5 .+ u, 0, 0, 0.006),
        ("unbalanced coincident", b, b, 1, 1, 0.001),
        ("unbalanced disjoint-far", b, 100.0 .+ b, 1, 1, 0.001),
        ("unbalanced overlapping", b, 0.5 .+ b, 1, 1, 0.001),
        ("many-test same-region", uniformpts(256), uniformpts(16), 1, 1, 0.004),
        ("many-test overlapping", uniformpts(256), 0.5 .+ uniformpts(16), 1, 1, 0.004),
        ("few-test many-trial", uniformpts(16), uniformpts(256), 1, 1, 0.004),
    )
end

dense_petrov_point_configs() = (
    (name, mkpts(testpts), mkpts(trialpts), tv, rv, mhs) for
    (name, testpts, trialpts, tv, rv, mhs) in dense_petrov_coordinate_configs()
)

function toeplitz_petrov_coordinate_configs()
    u = uniformpts(64)
    b = petrovunb()
    return (
        ("balanced coincident", u, u, 0, 0, 0.02),
        ("balanced disjoint-far", u, 100.0 .+ u, 0, 0, 0.02),
        ("balanced overlapping", u, 0.5 .+ u, 0, 0, 0.02),
        ("unbalanced coincident", b, b, 1, 1, 0.002),
        ("unbalanced disjoint-far", b, 100.0 .+ b, 1, 1, 0.002),
        ("unbalanced overlapping", b, 0.5 .+ b, 1, 1, 0.002),
        ("many-test same-region", uniformpts(128), uniformpts(16), 1, 1, 0.008),
        ("many-test overlapping", uniformpts(128), 0.5 .+ uniformpts(16), 1, 1, 0.008),
        ("few-test many-trial", uniformpts(16), uniformpts(128), 1, 1, 0.008),
    )
end

toeplitz_petrov_point_configs() = (
    (name, mkpts(testpts), mkpts(trialpts), tv, rv, mhs) for
    (name, testpts, trialpts, tv, rv, mhs) in toeplitz_petrov_coordinate_configs()
)

real_affine_kernel(x, y) = 1.0 + 0.5x - 0.25y
complex_affine_kernel(x, y) = 1.0 + 0.3im + (0.5 - 0.2im) * x - (0.25 + 0.4im) * y
affine_kernel_cases() =
    (("real affine", real_affine_kernel), ("complex affine", complex_affine_kernel))

toeplitz_real_affine(z) = 1.0 + 0.5z
toeplitz_complex_affine(z) = 1.0 + 0.3im + (0.4 - 0.2im) * z
toeplitz_affine_kernel_cases() =
    (("real affine", toeplitz_real_affine), ("complex affine", toeplitz_complex_affine))

toeplitz_realkernel(z) = 1 / (1 + z^2)
toeplitz_complexkernel(z) = 1 / (1 + z^2) + 0.3im * exp(-(z^2))
toeplitz_kernel_cases() =
    (("real", toeplitz_realkernel), ("complex", toeplitz_complexkernel))

function fullkernelmatrix(testcoords, trialcoords, kernel, ::Type{T}) where {T}
    return [T(kernel(x, y)) for x in testcoords, y in trialcoords]
end
