# ThreadedASAP Timing

Nominal threaded runs had CEM solve p50/p95/p99 of 36.57/38.27/41.96 ms and
end-to-end p50/p95/p99 of 44.92/49.04/51.94 ms. The planner rate was
25.30 +/- 0.26 Hz. The 100 Hz execution loop had p95/p99 periods of
10.093/10.158 ms, with 12 deadline misses in 34,640 ticks (0.0346%).

This is Python soft real-time, not hard real time. The CEM worker is slower
than the command period, while the execution loop maintains the nominal control
schedule for almost all ticks.
