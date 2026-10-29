"""
    @createlinked MomentName CollectionName [AbstractMomentType AbstractCollectionType]

Create a moment type and a matching collection type, then link their multiplication methods.
Without parent types, the new types subtype `H2Factory.AbstractMoment` and
`H2Factory.AbstractMomentCollection`.
"""
macro createlinked(
    momentname, momentcollectionname, AbstractTypeMoment, AbstractTypeMomentCollection
)
    return quote
        @createmoment $momentname $AbstractTypeMoment
        @createmomentcollection $momentcollectionname $AbstractTypeMomentCollection
        @linkmomenttocollection $momentname $momentcollectionname
    end
end

macro createlinked(momentname, momentcollectionname)
    return quote
        @createlinked $momentname $momentcollectionname H2Factory.AbstractMoment H2Factory.AbstractMomentCollection
    end
end
