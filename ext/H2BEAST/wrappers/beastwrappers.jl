"""
    Maxwell3DWrapper{T,K}

Base type for wrapped Maxwell operators.
"""
abstract type Maxwell3DWrapper{T,K} <: BEAST.MaxwellOperator3D{T,K} end

"""
    Helmholtz3DWrapper{T,K}

Base type for wrapped Helmholtz operators.
"""
abstract type Helmholtz3DWrapper{T,K} <: BEAST.Helmholtz3DOp{T,K} end

const MWHH3DWrapper = Union{Maxwell3DWrapper,Helmholtz3DWrapper}

function BEAST.gamma(op::MWHH3DWrapper)
    return BEAST.gamma(H2Factory.operator(op))
end

function BEAST.Integrand(op::MWHH3DWrapper, x, y, f, g)
    return BEAST.Integrand(H2Factory.operator(op), x, y, f, g)
end

function BEAST.scalartype(op::MWHH3DWrapper)
    return BEAST.scalartype(H2Factory.operator(op))
end

function BEAST.regularpart(op::MWHH3DWrapper)
    return BEAST.regularpart(H2Factory.operator(op))
end

function BEAST.singularpart(op::MWHH3DWrapper)
    return BEAST.singularpart(H2Factory.operator(op))
end
