"""Strategy agent — asks Claude for candidate strategies and runs them through
the backtest API (Phase 1, then Phase 2), feeding results back for revision.

It never runs the final test on the held-back data, never queues for paper and
never touches the live bot: a strategy that passes both phases ends with a
printed scripts/submit_strategy.sh command for a human to run.
"""
