module TestMomentCreationModule

using H2Factory
using Test
using StaticArrays
using SparseArrays

a = [spzeros(2, 2)]
@test typeof(H2Factory.getpointers(a)) == Vector{Nothing}
a = [zeros(2, 2)]
@test typeof(H2Factory.getpointers(a)) == Vector{Ptr{Float64}}

abstract type AbstractTestCreationMomentType{D,T} <: H2Factory.AbstractMoment{D,T} end

@createmoment TestMoment AbstractTestCreationMomentType
@test isdefined(TestMomentCreationModule, :TestMoment)
@test TestMoment <: AbstractTestCreationMomentType

for T in [Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64]
    for i in 1:10
        for j in 1:10
            @test hasmethod(TestMoment{i,T}, Tuple{NTuple{j,Int}})

            @test hasmethod(
                TestMoment{i,AbstractArray{T,j}}, Tuple{SVector{i,AbstractArray{T,j}}}
            )

            @test hasmethod(TestMoment{i}, Tuple{SVector{i,AbstractArray{T,j}}})

            @test hasmethod(TestMoment, Tuple{SVector{i,AbstractArray{T,j}}})

            @test hasmethod(
                TestMoment{i,AbstractArray{T,j}}, Tuple{TestMoment{i,AbstractArray{T,j}}}
            )
            @test hasmethod(TestMoment{i}, Tuple{TestMoment{i,AbstractArray{T,j}}})
            @test hasmethod(TestMoment, Tuple{TestMoment{i,AbstractArray{T,j}}})
        end
    end
end

# create moment from tuple
m = TestMoment{2,Float64}((3, 3))
@test typeof(m) <: TestMoment{2,Matrix{Float64},Base.Ptr{Float64}}

m = TestMoment{3,Float64}((2, 2, 2))
@test typeof(m) <: TestMoment{3,Array{Float64,3},Base.Ptr{Float64}}

m = TestMoment{3,Float64,Base.Ptr{Float64}}((2, 2, 2))
@test typeof(m) <: TestMoment{3,Array{Float64,3},Base.Ptr{Float64}}

# create moment from SVector
m2 = TestMoment{3,Array{Float64,3}}(m.moments)
@test typeof(m2) == typeof(m)

m2 = TestMoment{3}(m.moments)
@test typeof(m2) == typeof(m)

m2 = TestMoment(m.moments)
@test typeof(m2) == typeof(m)

m2 = TestMoment{3,Float64,Base.Ptr{Float64}}(m.moments)
@test typeof(m2) == typeof(m)

# create moment from moment
m2 = TestMoment{3,Array{Float64,3}}(m)
@test typeof(m2) == typeof(m)

m2 = TestMoment{3}(m)
@test typeof(m2) == typeof(m)

m2 = TestMoment(m)
@test typeof(m2) == typeof(m)

for i in eachindex(m2)
    @test m2[i] === m2.moments[i]
end

end

module TestCollectionCreationModule

using H2Factory
using Test
using StaticArrays

abstract type AbstractTestCreationMomentCollectionType{D,T} <:
              H2Factory.AbstractMomentCollection{D,T} end
@createmomentcollection TestMomentCollection AbstractTestCreationMomentCollectionType

@test isdefined(TestCollectionCreationModule, :TestMomentCollection)
@test TestMomentCollection <: AbstractTestCreationMomentCollectionType

for T in [Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64]
    for i in 1:10
        j = 2
        @test hasmethod(TestMomentCollection{i,T}, Tuple{NTuple{j,Int}})

        @test hasmethod(
            TestMomentCollection{i,AbstractArray{T,j}}, Tuple{SVector{i,AbstractArray{T,j}}}
        )

        @test hasmethod(TestMomentCollection{i}, Tuple{SVector{i,AbstractArray{T,j}}})

        @test hasmethod(TestMomentCollection, Tuple{SVector{i,AbstractArray{T,j}}})

        @test hasmethod(
            TestMomentCollection{i,AbstractArray{T,j}},
            Tuple{TestMomentCollection{i,AbstractArray{T,j}}},
        )
        @test hasmethod(
            TestMomentCollection{i}, Tuple{TestMomentCollection{i,AbstractArray{T,j}}}
        )
        @test hasmethod(
            TestMomentCollection, Tuple{TestMomentCollection{i,AbstractArray{T,j}}}
        )
    end
end

# create collection from tuple
m = TestMomentCollection{3,Float64}((3, 3))
@test typeof(m) == TestMomentCollection{3,Matrix{Float64}}

@test_throws MethodError TestMomentCollection{3,Float64}((2, 2, 2))

# create collection from SVector
m2 = TestMomentCollection{3,Array{Float64,2}}(m.moments)
@test typeof(m2) == typeof(m)

m2 = TestMomentCollection{3}(m.moments)
@test typeof(m2) == typeof(m)

m2 = TestMomentCollection(m.moments)
@test typeof(m2) == typeof(m)

# create collection from collection
m2 = TestMomentCollection{3,Array{Float64,2}}(m)
@test typeof(m2) == typeof(m)

m2 = TestMomentCollection{3}(m)
@test typeof(m2) == typeof(m)

m2 = TestMomentCollection(m)
@test typeof(m2) == typeof(m)

for i in eachindex(m2)
    @test m2[i] === m2.moments[i]
end
end

module TestLinkedCreationModule

using H2Factory
using Test
using StaticArrays
using LinearAlgebra

abstract type AbstractTestCreationMomentCollectionType{D,T} <:
              H2Factory.AbstractMomentCollection{D,T} end

abstract type AbstractTestCreationMomentType{D,T} <: H2Factory.AbstractMoment{D,T} end

@createlinked TestMoment TestMomentCollection AbstractTestCreationMomentType AbstractTestCreationMomentCollectionType

@test isdefined(TestLinkedCreationModule, :TestMoment)
@test isdefined(TestLinkedCreationModule, :TestMomentCollection)

@test TestMoment <: AbstractTestCreationMomentType
@test TestMomentCollection <: AbstractTestCreationMomentCollectionType

# THESE ASK ABOUT CONCRETE TYPES, because that is what `@linkmomenttocollection` generates for. The
# generated methods are `M<:MType{D,T1}, MCol<:MColType{D,T2}`, which the UNPARAMETERIZED `TestMoment`
# cannot satisfy for any concrete `D, T1` -- so asking about `TestMoment` matched nothing generated.
# Four of these sat at `@test_broken` for that reason, and two more passed only because
# `conjmul!` carried a universal `(Any, Any, Any)` fallback that answered `true` for everything. One
# root cause, two symptoms that read as unrelated. The fallback is gone and all six now pass honestly.
#
# BUILT, NOT SPELLED. The concrete types come from `typeof` on representative values rather than being
# written out, which exercises the public constructors and the generated methods together and keeps the
# test off storage internals: `M` here is
# `TestMoment{2, Vector{ComplexF64}, Ptr{ComplexF64}, Memory{ComplexF64}}`, and a hand-written
# signature would both be unreadable and break on any change to the pointer/memory representation.
m = TestMoment{2,ComplexF64}((3,))
mcol = TestMomentCollection{2,ComplexF64}((3, 3))

M = typeof(m)
MC = typeof(mcol)

@test hasmethod(LinearAlgebra.mul!, Tuple{M,MC,AbstractArray})
@test hasmethod(LinearAlgebra.mul!, Tuple{M,MC,AbstractArray,Number,Number})

@test hasmethod(H2Factory.conjmul!, Tuple{M,MC,AbstractArray{<:Complex}})
@test hasmethod(H2Factory.conjmul!, Tuple{M,MC,AbstractArray{<:Real}})
@test hasmethod(H2Factory.conjmul!, Tuple{M,MC,AbstractArray{<:Complex},Number,Number})
@test hasmethod(H2Factory.conjmul!, Tuple{M,MC,AbstractArray{<:Real},Number,Number})

@test hasmethod(LinearAlgebra.mul!, Tuple{AbstractArray,MC,M})
@test hasmethod(LinearAlgebra.mul!, Tuple{AbstractArray,MC,M,Number,Number})

# AN UNSUPPORTED PAIR IS UNSUPPORTED, both when asked and when called. The first assertion is the one
# the removed fallback made impossible: while it existed, `conjmul!` advertised support for every
# 3-arg combination, so a caller could not distinguish a generated pair from one nothing generates
# for. The second says dispatch agrees -- the static answer and the runtime behaviour now match, where
# before they disagreed and only the runtime one was right.
@test !hasmethod(H2Factory.conjmul!, Tuple{Int,Int,Int})
@test_throws MethodError H2Factory.conjmul!(1, 1, 1)

# NON-VACUITY for the pair above: the unparameterized type is still not a match, so the concrete
# assertions are load-bearing rather than true of anything.
@test !hasmethod(
    H2Factory.conjmul!, Tuple{TestMoment,TestMomentCollection,AbstractArray{<:Complex}}
)

end
