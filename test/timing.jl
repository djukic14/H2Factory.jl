module TestTiming

function timed(f, label; io=stderr)
    println(io, "[test] START ", label)
    flush(io)
    started = time_ns()
    succeeded = false
    try
        result = f()
        succeeded = true
        return result
    finally
        seconds = round((time_ns() - started) / 1e9; digits=3)
        status = succeeded ? "DONE" : "ERROR"
        println(io, "[test] ", status, " ", label, " elapsed=", seconds, "s")
        flush(io)
    end
end

end
