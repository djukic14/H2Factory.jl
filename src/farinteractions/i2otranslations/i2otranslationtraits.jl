"""
    AbstractI2OTranslationTrait

Base type for I2O coupling-storage traits.
"""
abstract type AbstractI2OTranslationTrait end

"""
    DiagonalTranslation()

Trait for elementwise diagonal couplings.
"""
struct DiagonalTranslation <: AbstractI2OTranslationTrait end

"""
    DenseTranslation()

Trait for dense matrix couplings.
"""
struct DenseTranslation <: AbstractI2OTranslationTrait end

"""
    AbstractToeplitzI2OTranslationTrait

Base type for Toeplitz coupling-storage traits.
"""
abstract type AbstractToeplitzI2OTranslationTrait <: AbstractI2OTranslationTrait end

"""
    ToeplitzI2OTranslationTrait()

Trait for Toeplitz couplings applied with FFT-based multiplication.
"""
struct ToeplitzI2OTranslationTrait <: AbstractToeplitzI2OTranslationTrait end

"""
    DummyTranslation()

Trait for [`DummyI2OTranslationCollection`](@ref), which applies no coupling.
"""
struct DummyTranslation <: AbstractI2OTranslationTrait end
