"""Strategy agent — asks Claude for candidate strategies and runs them through
the backtest API (Phase 1, then Phase 2), feeding results back for revision.

It never touches the live bot: the only hand-off is a printed paper-add
command, or ``--queue-paper`` which calls ``paper-add`` without ``--force``.
"""
