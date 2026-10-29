macro hh3Ddoublelayer(name, AbstractType)
    quote
        local mod = @__MODULE__
        local AbstractType = try
            $AbstractType
        catch
            mod.$AbstractType
        end

        struct $name{T,U} <: AbstractType{T,U}
            operator::BEAST.HH3DDoubleLayerFDBIO{T,U}

            function $name(doublelayer)
                return new{typeof(doublelayer.alpha),typeof(doublelayer.gamma)}(doublelayer)
            end

            function $name(; gamma=nothing, wavenumber=nothing, alpha=nothing)
                α, γ = BEAST.operator_parameter_handler(alpha, gamma, wavenumber)
                doublelayer = BEAST.HH3DDoubleLayerFDBIO(α, γ)
                return $name(doublelayer)
            end
        end

        stringname = string(mod.$name)

        """
            struct $stringname{T,U} <: $AbstractType{T,U}

        Wrap a `BEAST.HH3DDoubleLayerFDBIO`. Accepts an existing operator or the keywords
        `gamma`/`wavenumber` and `alpha`.
        """
        $name
    end
end

macro hh3Ddoublelayer(name)
    quote
        @hh3Ddoublelayer $name Helmholtz3DWrapper

        function H2Factory.operator(op::$(esc(name)))
            return op.operator
        end
    end
end
