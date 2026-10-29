"""
    abstract type AbstractMoment{D,T}

Base type for a moment with `D` component arrays of type `T`.
"""
abstract type AbstractMoment{D,T} end

"""
    AbstractToeplitzMoment{D,T}

Base type for a moment with `D` components and separate padded arrays for Toeplitz products.
"""
abstract type AbstractToeplitzMoment{D,T} <: AbstractMoment{D,T} end

# Pointer-backed array views are kept in a separate file.
include("unsafe_wraps.jl")

"""
    LinearAlgebra.mul!(resultmoment::AbstractMoment, matrix::AbstractMatrix, moment::AbstractMoment)

Apply `matrix` to each flattened component of `moment`, writing into `resultmoment`.
Return `resultmoment`.

```julia
m = H2Factory.Moment{2,Float64}((10, 10))
matrix = rand(100, 100)
resultmoment = typeof(m)(m)
LinearAlgebra.mul!(resultmoment, matrix, m)
```
"""
function LinearAlgebra.mul!(
    resultmoment::M1, matrix::A, moment::M2
) where {
    D,TM1,TM2,TA,M1<:AbstractMoment{D,TM1},M2<:AbstractMoment{D,TM2},A<:AbstractMatrix{TA}
}
    for i in eachindex(resultmoment)
        LinearAlgebra.mul!(wrapvector(resultmoment, i), matrix, wrapvector(moment, i))
    end
    return resultmoment
end

"""
    LinearAlgebra.mul!(resultmoment::AbstractMoment, matrix::AbstractMatrix, moment::AbstractMoment, α, β)

For each flattened component, compute
`resultmoment[i] = α * matrix * moment[i] + β * resultmoment[i]`.
Return `resultmoment`.
"""
function LinearAlgebra.mul!(
    resultmoment::M1, matrix::A, moment::M2, α, β
) where {
    D,TM1,TM2,TA,M1<:AbstractMoment{D,TM1},M2<:AbstractMoment{D,TM2},A<:AbstractMatrix{TA}
}
    for i in eachindex(resultmoment)
        LinearAlgebra.mul!(wrapvector(resultmoment, i), matrix, wrapvector(moment, i), α, β)
    end
    return resultmoment
end

function LinearAlgebra.mul!(
    resultmoment::M1, tensor::A, moment::M2
) where {
    D,
    TM1,
    TM2,
    TA<:AbstractArray,
    M1<:AbstractMoment{D,TM1},
    M2<:AbstractMoment{D,TM2},
    A<:AbstractMatrix{TA},
}
    for i in eachindex(resultmoment)
        for j in eachindex(moment)
            tf = tensor[i, j]
            if j == first(eachindex(moment))
                LinearAlgebra.mul!(wrapvector(resultmoment, i), tf, wrapvector(moment, j))
            else
                LinearAlgebra.mul!(
                    wrapvector(resultmoment, i), tf, wrapvector(moment, j), true, true
                )
            end
        end
    end
    return resultmoment
end

function LinearAlgebra.mul!(
    resultmoment::M1, tensor::A, moment::M2, α::Bool, β::Bool
) where {
    D,
    TM1,
    TM2,
    TA<:AbstractArray,
    M1<:AbstractMoment{D,TM1},
    M2<:AbstractMoment{D,TM2},
    A<:AbstractMatrix{TA},
}
    for i in eachindex(resultmoment)
        for j in eachindex(moment)
            tf = tensor[i, j]
            if j == first(eachindex(moment))
                LinearAlgebra.mul!(
                    wrapvector(resultmoment, i), tf, wrapvector(moment, j), α, β
                )
            else
                LinearAlgebra.mul!(
                    wrapvector(resultmoment, i), tf, wrapvector(moment, j), true, true
                )
            end
        end
    end
    return resultmoment
end

function LinearAlgebra.mul!(
    resultmoment::M1, tensor::A, moment::M2
) where {
    D,
    TM1,
    TM2,
    TA<:AbstractArray,
    M1<:AbstractMoment{D,TM1},
    M2<:AbstractMoment{D,TM2},
    A<:Vector{TA},
}
    for i in eachindex(resultmoment)
        LinearAlgebra.mul!(wrapvector(resultmoment, i), tensor[i], wrapvector(moment, i))
    end
    return resultmoment
end

function LinearAlgebra.mul!(
    resultmoment::M1, tensor::A, moment::M2, α::Bool, β::Bool
) where {
    D,
    TM1,
    TM2,
    TA<:AbstractArray,
    M1<:AbstractMoment{D,TM1},
    M2<:AbstractMoment{D,TM2},
    A<:Vector{TA},
}
    for i in eachindex(resultmoment)
        LinearAlgebra.mul!(
            wrapvector(resultmoment, i), tensor[i], wrapvector(moment, i), α, β
        )
    end
    return resultmoment
end

function LinearAlgebra.mul!(
    resultmoment::M1, tensor::A, moment::M2, α::Bool, β::Bool
) where {
    D1,
    D2,
    TM1,
    TM2,
    TA<:AbstractArray,
    M1<:AbstractMoment{D1,TM1},
    M2<:AbstractMoment{D2,TM2},
    A<:AbstractArray{TA},
}
    if D1 == 1
        for i in eachindex(moment)
            if i == 1
                LinearAlgebra.mul!(
                    wrapvector(resultmoment, 1), tensor[i], wrapvector(moment, i), α, β
                )
            else
                LinearAlgebra.mul!(
                    wrapvector(resultmoment, 1),
                    tensor[i],
                    wrapvector(moment, i),
                    true,
                    true,
                )
            end
        end

    elseif D2 == 1
        for i in eachindex(resultmoment)
            LinearAlgebra.mul!(
                wrapvector(resultmoment, i), tensor[i], wrapvector(moment, 1), α, β
            )
        end
    else
        error(
            "Dimensions of moments $(D1) and $(D2) are not supported. One of the dimensions has to be 1.",
        )
    end
    return resultmoment
end

"""
    abstract type AbstractMomentCollection{D,T}

Base type for a collection with `D` moment matrices of type `T`.
"""
abstract type AbstractMomentCollection{D,T} end

"""
    subset(momencollection::AbstractMomentCollection{D,T}, indices::AbstractVector{<:Integer}) where {D,T}

Copy the selected columns of each component matrix into a new collection of the same type.
`indices` are local column indices.

```julia
m = H2Factory.MomentCollection{2,Float64}((10, 10))
msubset = H2Factory.subset(m, [1, 2, 3])
```
"""
function subset(
    momencollection::MCol, indices::AbstractVector{<:Integer}
) where {D,T,MCol<:AbstractMomentCollection{D,T}}
    matrixsize = (size(momencollection[1], 1), length(indices))

    splitmoments = [T(undef, matrixsize) for _ in 1:D]

    for i in eachindex(momencollection)
        splitmoments[i] .= momencollection[i][:, indices]
    end

    return MCol(SVector{D}(splitmoments))
end

struct ElementWiseMulFunctor{F}
    transform::F
end
ElementWiseMulFunctor() = ElementWiseMulFunctor(identity)
const elementwisefunctorconst = ElementWiseMulFunctor()

function (f::ElementWiseMulFunctor)(x, y, z)
    return x + y * f.transform(z)
end

function (f::ElementWiseMulFunctor)(y, z)
    return y * f.transform(z)
end

## Declare common functions for AbstractMoment and AbstractMomentCollection ################

# Use qualified strings in examples: these concrete types are defined later and are not exported.
# A symbol inside `$($ConcreteName)` would be treated as a variable at eval time.
for (AbstractType, ConcreteName) in [
    (AbstractMoment, "H2Factory.Moment"),
    (AbstractMomentCollection, "H2Factory.MomentCollection"),
]
    eval(
        quote
            """
            moments(m::$($AbstractType){D,T}) where {D,T}

        Return the component arrays in `m`.
        """
            function moments(m::$AbstractType{D,T}) where {D,T}
                return m.moments
            end

            """
            eachindex(m::$($AbstractType){D,T}) where {D,T}

        Return an iterator over the indices of the arrays in `m`.
        """
            function Base.eachindex(m::$AbstractType{D,T}) where {D,T}
                return Base.eachindex(moments(m))
            end

            """
            Base.getindex(m::M, i::Int) where {M<:$($AbstractType)}

        Return the `i`-th array of `m`.

        # Example

        ```julia
        m = $($ConcreteName){2,Float64}((3, 3))

        # Indexing returns the array itself. Use identity because fresh storage is uninitialized.
        @show m[1] === m.moments[1]

        # Iterate over the component arrays.
        for i in eachindex(m)
            println(m[i])
        end
        ```
        """
            function Base.getindex(m::M, i::Int) where {M<:$AbstractType}
                return moments(m)[i]
            end

            """
            dim(m::$($AbstractType){D,T}) where {D,T}

        Return the component count, e.g. three for x, y, and z components.
        """
            function dim(m::$AbstractType)
                return dim(typeof(m))
            end

            function dim(::Type{A}) where {D,A<:$AbstractType{D}}
                return D
            end

            function Base.eltype(::Type{A}) where {D,T,A<:$AbstractType{D,T}}
                return T
            end

            """
            eltype(m::$($AbstractType){D,T}) -> T

        Return the component array type, usually an `AbstractArray` subtype.
        """
            function Base.eltype(m::$AbstractType)
                return eltype(typeof(m))
            end

            """
            add!(momentA::M, momentB::M) where {D,M<:$($AbstractType){D}}

        Add each component of `momentB` into `momentA` and return `momentA`.
        """
            function add!(momentA::$AbstractType{D}, momentB::$AbstractType{D}) where {D}
                for i in eachindex(momentA)
                    momentA[i] .+= momentB[i]
                end
                return momentA
            end

            function add!(
                momentA::$AbstractType{D}, momentB::$AbstractType{D}, α, β
            ) where {D}
                for i in eachindex(momentA)
                    @simd for j in eachindex(momentA[i])
                        momentA[i][j] *= β
                        momentA[i][j] += α * momentB[i][j]
                    end
                end
                return momentA
            end

            """
            Base.conj!(moment::M) where {D,T,M<:$($AbstractType){D,T}}

        Conjugate each element `moment` object in-place.
        """
            function Base.conj!(moment::M) where {D,T,M<:$AbstractType{D,T}}
                for i in eachindex(moment)
                    conj!(moment[i])
                end
                return moment
            end

            """
            Base.conj(moment::M) where {D,T,M<:$($AbstractType){D,T}}

        Return a conjugated version of `moment`.
        """
            function Base.conj(moment::M) where {D,T,M<:$AbstractType{D,T}}
                result = deepcopy(moment)
                conj!(result)
                return result
            end

            """
            copyto!(storagemoment::M, moment::M) where {M<:$($AbstractType)}

        Copy the values from `moment` to `storagemoment` element-wise.
        """
            function Base.copyto!(
                storagemoment::$AbstractType{D}, moment::$AbstractType{D}
            ) where {D}
                for i in eachindex(storagemoment)
                    Base.copyto!(storagemoment[i], moment[i])
                end

                return storagemoment
            end

            """
        LinearAlgebra.mul!(storagemoment::$($AbstractType){D}, moment::$($AbstractType){D}, coefficient::Number) where {D}

        Multiply each array in `moment` by `coefficient` and store the result in
        `storagemoment`.
        """
            function LinearAlgebra.mul!(
                storagemoment::$AbstractType{D},
                moment::$AbstractType{D},
                coefficient::Number,
            ) where {D}
                for i in eachindex(storagemoment)
                    LinearAlgebra.mul!(storagemoment[i], moment[i], coefficient)
                end
                return storagemoment
            end

            """
        LinearAlgebra.mul!(storagemoment::$($AbstractType){D}, moment::$($AbstractType){D}, coefficient::Number, α::Number, β::Number) where {D}

        Compute `storagemoment[i] = α * coefficient * moment[i] + β * storagemoment[i]`
        for each component. Return `storagemoment`.
        """
            function LinearAlgebra.mul!(
                storagemoment::$AbstractType{D},
                moment::$AbstractType{D},
                coefficient::Number,
                α,
                β,
            ) where {D}
                for i in eachindex(storagemoment)
                    LinearAlgebra.mul!(storagemoment[i], moment[i], coefficient, α, β)
                end
                return storagemoment
            end

            """
            elementwisemul!(storagemoment::$($AbstractType), moment::$($AbstractType), coefficients::AbstractArray; transform=identity)

        Multiply each component elementwise by `transform.(coefficients)` and return `storagemoment`.
        Apply `transform` inline, without allocating a transformed copy.
        """
            function elementwisemul!(
                storagemoment::$AbstractType{D},
                moment::$AbstractType{D},
                coefficients::AbstractArray;
                threading=nothing,
                transform=identity,
            ) where {D}
                f = ElementWiseMulFunctor(transform)
                for i in eachindex(storagemoment)
                    map!(f, storagemoment[i], moment[i], coefficients)
                end
                return storagemoment
            end

            """
            elementwisemul!(storagemoment::$($AbstractType), moment::$($AbstractType), coefficients::AbstractArray, α::Bool, β::Bool; transform=identity)

        Compute `storagemoment[i] = α * moment[i] .* transform.(coefficients) + β * storagemoment[i]`.
        Apply `transform` inline. Return `storagemoment`.
        """
            function elementwisemul!(
                storagemoment::$AbstractType{D},
                moment::$AbstractType{D},
                coefficients::AbstractArray,
                α::Bool,
                β::Bool;
                threading=nothing,
                transform=identity,
            ) where {D}
                if β == false
                    for i in eachindex(storagemoment)
                        storagemoment[i] .= β * storagemoment[i]
                    end
                end

                α == false && return storagemoment

                f = ElementWiseMulFunctor(transform)

                for i in eachindex(storagemoment)
                    map!(f, storagemoment[i], storagemoment[i], moment[i], coefficients)
                end
                return storagemoment
            end

            """
            elementwisemul!(storagemoment::$($AbstractType){D}, moment::$($AbstractType){D}, coefficients::SVector{D}) where {D}

        Multiply component `i` by `coefficients[i]` and return `storagemoment`.
        Unlike the array-coefficient form, this method has no `transform` keyword.
        """
            function elementwisemul!(
                storagemoment::$AbstractType{D},
                moment::$AbstractType{D},
                coefficients::SVector{D};
                threading=nothing,
            ) where {D}
                for i in eachindex(storagemoment)
                    storagemoment[i] .= moment[i] .* coefficients[i]
                end
                return storagemoment
            end

            """
            elementwisemul!(storagemoment::$($AbstractType){D}, moment::$($AbstractType){D}, coefficients::SVector{D}, α::Bool, β::Bool) where {D}

        Compute `storagemoment[i] = α * moment[i] .* coefficients[i] + β * storagemoment[i]`.
        Return `storagemoment`. This method has no `transform` keyword.
        """
            function elementwisemul!(
                storagemoment::$AbstractType{D},
                moment::$AbstractType{D},
                coefficients::SVector{D},
                α::Bool,
                β::Bool;
                threading=nothing,
            ) where {D}
                if β == false
                    for i in eachindex(storagemoment)
                        storagemoment[i] .= β
                    end
                end

                α == false && return storagemoment
                f = ElementWiseMulFunctor()

                for i in eachindex(storagemoment)
                    map!(f, storagemoment[i], storagemoment[i], moment[i], coefficients[i])
                end
                return storagemoment
            end
        end,
    )
end

function numberofentries(moment::A) where {A<:AbstractMoment}
    numentries = 0
    for i in eachindex(moment)
        @assert moment[i] isa DenseArray
        numentries += length(moment[i])
    end
    return numentries
end
