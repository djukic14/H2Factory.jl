# Text output for diagnostic reports.

function Base.show(io::IO, ::MIME"text/plain", s::H2Summary)
    println(io, "H2 map summary: ", s.maptype, "  (", s.treetype, ")")
    println(io, "  size            ", s.size[1], " × ", s.size[2], "  (", s.eltype, ")")
    println(
        io,
        "  tree            ",
        s.numlevels,
        " levels, ",
        s.numleaves,
        " leaves, ",
        s.numnodes,
        " nodes",
    )
    println(
        io,
        "  leaf DOFs       min ",
        s.minleafdofs,
        " / mean ",
        round(s.meanleafdofs; digits=2),
        " / max ",
        s.maxleafdofs,
    )
    println(
        io,
        "  far field       ",
        s.farpairs,
        " pairs, ",
        s.uniquetranslations,
        " unique translations",
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", t::TranslationCounts)
    print(
        io,
        "translations: ",
        t.farpairs,
        " far pairs / ",
        t.uniquetranslations,
        " unique = ",
        round(t.reuse; digits=2),
        "× reuse",
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", r::SanityReport)
    if isempty(r)
        print(io, "SanityReport(:", r.target, "): OK")
        return nothing
    end
    println(io, "SanityReport(:", r.target, "): ", length(r.issues), " issue(s)")
    for i in r.issues
        println(io, "  [", i.severity, "] ", i.code, ": ", i.message)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", r::TimingReport)
    println(io, "TimingReport  ", r.metadata.maptype, "  mode=", r.metadata.mode)
    println(io, "  size ", r.metadata.size, "  threads ", r.metadata.nthreads, "  records:")
    for rec in r.records
        lvl = rec.level === missing ? "" : "  level=$(rec.level)"
        println(
            io,
            "    ",
            rpad(string(rec.operation, "/", rec.phase), 20),
            lpad(string(round(rec.seconds * 1e6; digits=1), " µs"), 14),
            lvl,
        )
    end
    return nothing
end
