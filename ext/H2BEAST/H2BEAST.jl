module H2BEAST
using BEAST
using H2Factory
using SparseArrays
using CompScienceMeshes
using LinearAlgebra
import H2Factory: @efie, @mw3Dweaklysingular, @mw3Dhypersingular, @mw3Ddoublelayer
import H2Factory: @hh3Ddoublelayer, @hh3Ddoublelayer_transposed
import H2Factory: progressbar, isgalerkinsymmetric
import H2Trees: TwoNTree
import ClusterTrees.BlockTrees: BlockTree
using ProgressMeter
using StaticArrays
using OhMyThreads

include("moments/assemblemoments.jl")
include("nearinteractions/sparseassembly.jl")
include("nearinteractions/randomnearinteractions.jl")
include("nearinteractions/sparsenearinteractions.jl")

include("wrappers/beastwrappers.jl")
include("wrappers/maxwell/efie.jl")
include("wrappers/maxwell/mwhypersingular.jl")
include("wrappers/maxwell/mwweaklysingular.jl")
include("wrappers/maxwell/mwdoublelayer.jl")
include("wrappers/helmholtz/hhdoublelayer.jl")
include("wrappers/helmholtz/hhdoublelayertransposed.jl")

# Helmholtz symmetry traits

function isgalerkinsymmetric(::Type{<:BEAST.HH3DSingleLayerFDBIO})
    return true
end

function isgalerkinsymmetric(::Type{<:BEAST.HH3DHyperSingularFDBIO})
    return true
end

# Maxwell symmetry traits

function isgalerkinsymmetric(::Type{<:BEAST.MWSingleLayer3D})
    return true
end

function isgalerkinsymmetric(::Type{<:BEAST.MWDoubleLayer3D})
    return true
end

end # module H2BEAST
