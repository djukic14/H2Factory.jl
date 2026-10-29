"""
    LevelTranslators(i2itranslator, o2otranslator)

Hold the I2I and O2O translators for transfers within a tree. These are shared data; their scratch
buffers belong to each workspace.
"""
struct LevelTranslators{I2I,O2O}
    i2itranslator::I2I
    o2otranslator::O2O
end

"""
    SidedLevelTranslators(test, trial)

Hold one set of [`LevelTranslators`](@ref) for each side of a Petrov map. The test and trial trees
each use their own transfers. Map accessors select the side for the current direction.
"""
struct SidedLevelTranslators{T,R}
    test::T
    trial::R
end

"""
    Moments(test, trial)

Hold the test and trial moments used during multiplication. These are mutable workspace data.
"""
struct Moments{Test,Trial}
    test::Test
    trial::Trial
end

"""
    MomentCollections(test, trial)

Hold the test and trial moment collections for the leaves of a Petrov map. Unlike [`Moments`](@ref),
these collections are built once and shared.
"""
struct MomentCollections{Test,Trial}
    test::Test
    trial::Trial
end

"""
    VectorBuffers(test, trial)

Hold the test and trial vector buffers used for leaf multiplication in a Petrov workspace.
"""
struct VectorBuffers{Test,Trial}
    test::Test
    trial::Trial
end
