
macro efie(name, AbstractType)
    quote
        local mod = @__MODULE__
        local AbstractType = try
            $AbstractType
        catch
            mod.$AbstractType
        end

        struct $name{T,U} <: AbstractType{T,U}
            operator::MWSingleLayer3D{T,U}

            function $name(singlelayer)
                isapprox(singlelayer.α, singlelayer.gamma^2 * singlelayer.β) ||
                    throw(ArgumentError("EFIE requires α = βγ²"))
                return new{typeof(singlelayer.gamma),typeof(singlelayer.α)}(singlelayer)
            end

            function $name(; gamma=nothing, wavenumber=nothing, alpha=nothing)
                γ, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)
                α = if isnothing(alpha)
                    -γ
                else
                    alpha
                end

                beta = α / γ^2
                alphabetatype = promote_type(typeof(α), typeof(beta))
                α = convert(alphabetatype, α)
                beta = convert(alphabetatype, beta)
                singlelayer = MWSingleLayer3D(γ, α, beta)
                return $name(singlelayer)
            end

            function H2Factory.isgalerkinsymmetric(::Type{<:$name})
                return true
            end
        end

        stringname = string(mod.$name)

        """
            struct $stringname{T,U} <: $AbstractType{T,U}

        Wrap a `BEAST.MWSingleLayer3D` with ``α = βγ²``. Accepts a compatible operator or the
        keywords `gamma`/`wavenumber` and `alpha`.
        """
        $name
    end
end

macro efie(name)
    quote
        @efie $name Maxwell3DWrapper

        function H2Factory.operator(op::$(esc(name)))
            return op.operator
        end
    end
end
