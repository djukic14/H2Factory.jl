# Generate methods only for linked moment/collection pairs. Unsupported pairs have no fallback.
# Both types must be available in the calling module.
"""
    @linkmomenttocollection MomentType MomentCollectionType

Define multiplication methods between an existing moment type and moment collection type.
Both types must be defined before calling this macro. Other pairs remain unsupported.
"""
macro linkmomenttocollection(MomentType, MomentCollectionType)
    quote
        local mod = @__MODULE__

        local MType = try
            $MomentType
        catch
            mod.$MomentType
        end

        local MColType = try
            $MomentCollectionType
        catch
            mod.$MomentCollectionType
        end

        """
            LinearAlgebra.mul!(moment::$(MType){D,T1}, momentcollection::$(MColType){D,T2}, y::AbstractArray{<:Number})

        Calculate the superposition of the moments inside `momentcollection` according to
        the array `y`. The result is stored in `moment`. If the arrays of `moment` are
        interpreted as a vector, the operation is equivalent to the matrix-vector product

        ```
        moment[i] = momentcollection[i] * y, for i in 1:D
        ```

        # Arguments

          - `moment::$(MType){D,T1}`: A moment.
          - `momentcollection::$(MColType){D,T2}`: A momentcollection.
          - `y::AbstractArray{<:Number}`: An abstract array of numbers.

        # Example

        ```julia
        m = $(MType){2,Float64}((10,))
        mcol = $(MColType){2,Float64}((10, 10))
        y = zeros(Float64, 10)

        LinearAlgebra.mul!(m, mcol, y)
        ```

        # Returns

          - `moment::$(MType){D,T1}`: The updated moment after multiplication.
        """
        function LinearAlgebra.mul!(
            moment::M, momentcollection::MCol, y::A
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2},A<:AbstractArray}
            for i in eachindex(moment)
                LinearAlgebra.mul!(H2Factory.wrapvector(moment, i), momentcollection[i], y)
            end
            return moment
        end

        """
            LinearAlgebra.mul!(moment::$(MType){D,T1}, momentcollection::$(MColType){D,T2}, y::AbstractArray{<:Number}, α::Number, β::Number)

        Calculate the superposition of the moments inside `momentcollection` according to
        the array `y` and the factors `α` and `β`. The result is stored in `moment`. If the
        arrays of `moment` are interpreted as a vector, the operation is equivalent to the
        matrix-vector product

        ```
        moment[i] = α * momentcollection[i] * conj(y) + β * moment[i], for i in 1:D
        ```

        # Arguments

          - `moment::$(MType){D,T1}`: A moment.
          - `momentcollection::$(MColType){D,T2}`: A momentcollection.
          - `y::AbstractArray{<:Number}`: An abstract array of numbers.

        # Example

        ```julia
        m = $(MType){2,Float64}((3, 3))
        mcol = $(MColType){2,Float64}((9, 9))
        y = ones(Float64, 9)

        LinearAlgebra.mul!(m, mcol, y, 1.0, 2.0)
        ```

        # Returns

          - `moment::$(MType){D,T1}`: The updated moment.
        """
        function LinearAlgebra.mul!(
            moment::M, momentcollection::MCol, y::A, α, β
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2},A<:AbstractArray}
            for i in eachindex(moment)
                LinearAlgebra.mul!(
                    H2Factory.wrapvector(moment, i), momentcollection[i], y, α, β
                )
            end
            return moment
        end

        # real y do not need any conjugation
        function H2Factory.conjmul!(
            moment::M, momentcollection::MCol, y::A
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2},A<:AbstractArray{<:Real}}
            return LinearAlgebra.mul!(moment, momentcollection, y)
        end

        # real y do not need any conjugation
        function H2Factory.conjmul!(
            moment::M,
            momentcollection::MCol,
            y::A,
            α::Union{Number,AbstractVector},
            β::Number,
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2},A<:AbstractArray{<:Real}}
            return LinearAlgebra.mul!(moment, momentcollection, y, α, β)
        end

        """
            conjmul!(moment::$(MType){D,T1}, momentcollection::$(MColType){D,T2}, y::AbstractArray{<:Number})

        Calculate the superposition of the moments inside `momentcollection` according to
        the conjugated factors of the array `y`. The result is stored in `moment`. If the
        arrays of `moment` are interpreted as a vector, the operation is equivalent to the
        matrix-vector product

        ```
        moment[i] = momentcollection[i] * conj(y), for i in 1:D
        ```

        # Arguments

          - `moment::$(MType){D,T1}`: A moment.
          - `momentcollection::$(MColType){D,T2}`: A momentcollection.
          - `y::AbstractArray{<:Number}`: An abstract array of numbers.

        # Example

        ```julia
        m = $(MType){2,ComplexF64}((10,))
        mcol = $(MColType){2,ComplexF64}((10, 10))
        y = Array(1:10) * im

        H2Factory.conjmul!(m, mcol, y)
        ```

        # Returns

          - `moment::$(MType){D,T1}`: The updated moment after conjugate multiplication.
        """
        function H2Factory.conjmul!(
            moment::M, momentcollection::MCol, y::A
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2},A<:AbstractArray{<:Complex}}
            # create views to the real and imaginary part of y separately
            yreal, yimag = H2Factory._viewrealimag(y)
            αimag = -im
            βimag = one(eltype(y))

            for i in eachindex(moment)
                LinearAlgebra.mul!(
                    H2Factory.wrapvector(moment, i), momentcollection[i], yreal
                )
                LinearAlgebra.mul!(
                    H2Factory.wrapvector(moment, i),
                    momentcollection[i],
                    yimag,
                    αimag,
                    βimag,
                )
            end
            return moment
        end

        """
            conjmul!(moment::$(MType){D,T1}, momentcollection::$(MColType){D,T2}, y::AbstractArray{<:Number}, α::Number, β::Number)

        Calculate the superposition of the moments inside `momentcollection` according to
        the conjugated factors of the array `y` and the factors `α` and `β`. The result is
        stored in `moment`. If the arrays of `moment` are interpreted as a vector, the
        operation is equivalent to the matrix-vector product

        ```
        moment[i] = α * momentcollection[i] * conj(y) + β * moment[i], for i in 1:D
        ```

        # Arguments

          - `moment::$(MType){D,T1}`: A moment.
          - `momentcollection::$(MColType){D,T2}`: A momentcollection.
          - `y::AbstractArray{<:Number}`: An abstract array of numbers.

        # Example

        ```julia
        m = $(MType){2,ComplexF64}((3, 3))
        mcol = $(MColType){2,ComplexF64}((9, 9))
        y = Array(1:9) * im

        H2Factory.conjmul!(m, mcol, y, 1.0, 2.0)
        ```

        # Returns

          - `moment::$(MType){D,T1}`: The updated moment after conjugate multiplication.
        """
        function H2Factory.conjmul!(
            moment::M, momentcollection::MCol, y::A, α::Number, β::Number
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2},A<:AbstractArray{<:Complex}}
            # create views to the real and imaginary part of y separately
            yreal, yimag = H2Factory._viewrealimag(y)
            αimag = -im * α
            βimag = one(eltype(y))

            for i in eachindex(moment)
                LinearAlgebra.mul!(
                    H2Factory.wrapvector(moment, i), momentcollection[i], yreal, α, β
                )
                LinearAlgebra.mul!(
                    H2Factory.wrapvector(moment, i),
                    momentcollection[i],
                    yimag,
                    αimag,
                    βimag,
                )
            end
            return moment
        end

        function LinearAlgebra.mul!(
            result::AbstractArray, momentcollection::MCol, moment::M
        ) where {T1,T2,M<:MType{1,T1},MCol<:MColType{1,T2}}
            LinearAlgebra.mul!(
                result, transpose(momentcollection[1]), H2Factory.wrapvector(moment, 1)
            )
            return result
        end

        function LinearAlgebra.mul!(
            result::AbstractArray, momentcollection::MCol, moment::M, α, β
        ) where {T1,T2,M<:MType{1,T1},MCol<:MColType{1,T2}}
            LinearAlgebra.mul!(
                result,
                transpose(momentcollection[1]),
                H2Factory.wrapvector(moment, 1),
                α,
                β,
            )
            return result
        end

        function LinearAlgebra.mul!(
            result::AbstractArray, momentcollection::MCol, moment::M, α::AbstractVector, β
        ) where {T1,T2,M<:MType{1,T1},MCol<:MColType{1,T2}}
            LinearAlgebra.mul!(
                result,
                transpose(momentcollection[1]),
                H2Factory.wrapvector(moment, 1),
                α[1],
                β,
            )
            return result
        end

        function LinearAlgebra.mul!(
            result::AbstractArray,
            momentcollection::MCol,
            moment::M,
            α::AbstractVector,
            β::AbstractVector,
        ) where {T1,T2,M<:MType{1,T1},MCol<:MColType{1,T2}}
            LinearAlgebra.mul!(
                result,
                transpose(momentcollection[1]),
                H2Factory.wrapvector(moment, 1),
                α[1],
                β[1],
            )
            return result
        end

        function LinearAlgebra.mul!(
            result::AbstractArray, momentcollection::MCol, moment::M, α, β::AbstractVector
        ) where {T1,T2,M<:MType{1,T1},MCol<:MColType{1,T2}}
            LinearAlgebra.mul!(
                result,
                transpose(momentcollection[1]),
                H2Factory.wrapvector(moment, 1),
                α,
                β[1],
            )
            return result
        end

        """
            LinearAlgebra.mul!(result::AbstractArray{<:Number}, momentcollection::$(MColType){D,T2}, moment::$(MType){D,T1})

        Transpose operation to `LinearAlgebra.mul!(moment, momentcollection, result)`. If
        the arrays of `moment` are interpreted as a vector, the operation is equivalent to
        the matrix-vector product

        ```
        result .= 0
        result += transpose(momentcollection[i]) * moment[i], for i in 1:D
        ```

        # Arguments

          - `result::AbstractArray{<:Number}`: An abstract array of numbers.
          - `moment::$(MType){D,T1}`: A moment.
          - `momentcollection::$(MColType){D,T2}`: A momentcollection.

        # Example

        ```julia
        m = $(MType){2,ComplexF64}((10,))
        mcol = $(MColType){2,ComplexF64}((10, 10))
        y = Array(1:10) * im

        LinearAlgebra.mul!(y, mcol, m)
        ```

        # Returns

          - `result::AbstractArray{<:Number}`: Updated array after the multiplication.
        """
        function LinearAlgebra.mul!(
            result::AbstractArray, momentcollection::MCol, moment::M
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2}}
            LinearAlgebra.mul!(
                result, transpose(momentcollection[1]), H2Factory.wrapvector(moment, 1)
            )

            αnew = one(eltype(result))
            βnew = one(eltype(result))
            for i in eachindex(moment)[2:end]
                LinearAlgebra.mul!(
                    result,
                    transpose(momentcollection[i]),
                    H2Factory.wrapvector(moment, i),
                    αnew,
                    βnew,
                )
            end
            return result
        end

        """
            LinearAlgebra.mul!(result::AbstractArray{<:Number}, momentcollection::$(MColType){D,T2}, moment::$(MType){D,T1}, α::Number, β::Number)

        Transpose operation to `LinearAlgebra.mul!(moment, momentcollection, result, α, β)`. If
        the arrays of `moment` are interpreted as a vector, the operation is equivalent to
        the matrix-vector product

        ```
        result *= α
        result += β * transpose(momentcollection[i]) * moment[i], for i in 1:D
        ```

        # Arguments

          - `result::AbstractArray{<:Number}`: An abstract array of numbers.
          - `moment::$(MType){D,T1}`: A moment.
          - `momentcollection::$(MColType){D,T2}`: A momentcollection.

        # Example

        ```julia
        m = $(MType){2,ComplexF64}((3, 3))
        mcol = $(MColType){2,ComplexF64}((9, 9))
        y = Array(1:9) * im

        LinearAlgebra.mul!(y, mcol, m, 1.0, 2.0)
        ```

        # Returns

          - `result::AbstractArray{<:Number}`: Updated array after the multiplication.
        """
        function LinearAlgebra.mul!(
            result::AbstractArray, momentcollection::MCol, moment::M, α, β
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2}}
            LinearAlgebra.mul!(
                result,
                transpose(momentcollection[1]),
                H2Factory.wrapvector(moment, 1),
                α,
                β,
            )

            βnew = one(eltype(result))
            firstindex = first(eachindex(moment))
            for i in eachindex(moment)
                i == firstindex && continue
                LinearAlgebra.mul!(
                    result,
                    transpose(momentcollection[i]),
                    H2Factory.wrapvector(moment, i),
                    α,
                    βnew,
                )
            end

            return result
        end

        function LinearAlgebra.mul!(
            result::AbstractArray, momentcollection::MCol, moment::M, α::AbstractVector, β
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2}}
            LinearAlgebra.mul!(
                result,
                transpose(momentcollection[1]),
                H2Factory.wrapvector(moment, 1),
                α[1],
                β,
            )

            βnew = one(eltype(result))
            firstindex = first(eachindex(moment))
            for i in eachindex(moment)
                i == firstindex && continue
                LinearAlgebra.mul!(
                    result,
                    transpose(momentcollection[i]),
                    H2Factory.wrapvector(moment, i),
                    α[i],
                    βnew,
                )
            end

            return result
        end

        """
            conjmul!(result, momentcollection::$(MColType), moment::$(MType))

        Store the conjugate of the transposed moment-collection product in `result`.
        """
        function H2Factory.conjmul!(
            result::AbstractArray, momentcollection::MCol, moment::M
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2}}
            LinearAlgebra.mul!(result, momentcollection, moment)
            conj!(result)
            return result
        end

        """
            conjmul!(result, momentcollection::$(MColType), moment::$(MType), α::Number, β::Number)

        Scale the transposed product by `α`, add `β * result`, and conjugate the result.
        """
        function H2Factory.conjmul!(
            result::AbstractArray, momentcollection::MCol, moment::M, α::Number, β::Number
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2}}
            conj!(result)
            LinearAlgebra.mul!(result, momentcollection, moment, α, β)
            conj!(result)
            return result
        end

        """
            conjmul!(result, momentcollection::$(MColType), moment::$(MType), α::AbstractVector, β::Number)

        Apply one coefficient from `α` to each moment component, add `β * result`, and
        store the conjugated product in `result`.
        """
        function H2Factory.conjmul!(
            result::AbstractArray,
            momentcollection::MCol,
            moment::M,
            α::AbstractVector,
            β::Number,
        ) where {D,T1,T2,M<:MType{D,T1},MCol<:MColType{D,T2}}
            conj!(result)
            conj!(α)
            LinearAlgebra.mul!(result, momentcollection, moment, α, β)
            conj!(α)
            conj!(result)
            return result
        end
    end
end

function _viewrealimag(z::AbstractArray{T}) where {T<:Complex}
    zreinterpreted = reinterpret(real(T), z)
    N = length(zreinterpreted)
    zreal = view(zreinterpreted, 1:2:(N - 1))
    zimag = view(zreinterpreted, 2:2:N)

    return zreal, zimag
end
