"""Local checks on a strategy file before it is sent anywhere.

Cheap and strict: the file must parse, follow the two-in-one format, import
only data libraries, and keep its tunable constants few. Failures are returned
as plain strings so the loop can hand them straight back to Claude.
"""

import ast
import re

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")  # same rule as the backtest API

ALLOWED_IMPORTS = {"pandas", "numpy", "pandas_ta", "freqtrade", "typing", "math"}
BANNED_CALLS = {"exec", "eval", "compile", "open", "__import__", "breakpoint", "input"}

MAX_PARAMS = 6  # distinct tunable numeric literals in the file; the overfitting guard
EXEMPT_LITERALS = {0, 1, -1}
# IStrategy class attributes Freqtrade needs or that config overrides (stoploss and the
# trailing stop come from config/backtest.json); numbers in their values are not knobs.
# minimal_roi is deliberately not here: the config does not override it, so its values tune exits.
EXEMPT_CLASS_ATTRS = {
    "INTERFACE_VERSION", "timeframe", "startup_candle_count", "can_short", "stoploss",
    "trailing_stop", "trailing_stop_positive", "trailing_stop_positive_offset",
    "trailing_only_offset_is_reached", "process_only_new_candles",
}


def strategy_name_from_code(code: str) -> str | None:
    """The name of the IStrategy subclass, or None if there is not exactly one."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and _is_istrategy(n)]
    return classes[0].name if len(classes) == 1 else None


def validate_strategy(code: str, name: str, max_params: int = MAX_PARAMS) -> list[str]:
    """Return a list of problems; empty means the file is good to submit."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"File does not parse: {e}"]

    problems = []
    if not NAME_RE.match(name):
        problems.append(f"Strategy name {name!r} must be an identifier of at most 64 characters")

    problems += _check_imports(tree)
    problems += _check_calls(tree)
    problems += _check_generate_signals(tree)
    problems += _check_class(tree, name)
    problems += _check_param_count(tree, max_params)
    return problems


def _root(module: str) -> str:
    return module.split(".")[0]


def _check_imports(tree: ast.AST) -> list[str]:
    bad = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root(alias.name) not in ALLOWED_IMPORTS:
                    bad.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or _root(node.module or "") not in ALLOWED_IMPORTS:
                bad.add(node.module or ".")
    return [f"Import of {m!r} is not allowed; use only {sorted(ALLOWED_IMPORTS)}" for m in sorted(bad)]


# Calls whose period argument reads the future when negative (pandas defaults are all 1).
PERIOD_METHODS = {"shift", "diff", "pct_change"}
BACKFILL_METHODS = {"bfill", "backfill"}


def _check_calls(tree: ast.Module) -> list[str]:
    """Banned builtins plus the look-ahead patterns that can be spotted in the source.

    Full-series operations (normalising by the whole column's max/mean, iloc[-1]
    broadcast to every bar, ...) cannot be caught statically; the backtest server
    runs a behavioural prefix check for those, so it is not duplicated here.
    """
    constants = _module_constants(tree)
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
            problems.append(f"Call to {func.id}() is not allowed")
        if not isinstance(func, ast.Attribute):
            continue
        method = func.attr
        line = f"line {node.lineno}"
        if method in PERIOD_METHODS:
            args = list(node.args[:1]) + [kw.value for kw in node.keywords if kw.arg == "periods"]
            for arg in args:
                value = _period_value(arg, constants)
                if value is None:
                    problems.append(
                        f"{line}: {method}() period must be a non-negative number or a module-level constant "
                        f"holding one; anything else may look ahead")
                elif value < 0:
                    problems.append(f"{line}: {method}() with a negative period looks ahead; not allowed")
        elif method == "rolling":
            for kw in node.keywords:
                if kw.arg == "center" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                    problems.append(f"{line}: rolling(center=True) uses future bars; not allowed")
        elif method in BACKFILL_METHODS:
            problems.append(f"{line}: {method}() fills gaps from future bars; use ffill() instead")
        elif method == "fillna":
            for kw in node.keywords:
                if kw.arg == "method" and isinstance(kw.value, ast.Constant) and kw.value.value in BACKFILL_METHODS:
                    problems.append(f"{line}: fillna(method={kw.value.value!r}) fills from future bars; not allowed")
    return problems


def _module_constants(tree: ast.Module) -> dict[str, float]:
    """Module-level NAME = <number> bindings whose name is assigned nowhere else in the file."""
    stores: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores[node.id] = stores.get(node.id, 0) + 1
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, _number(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value = node.target.id, _number(node.value)
        else:
            continue
        if value is not None and stores.get(target) == 1:
            constants[target] = value
    return constants


def _number(node: ast.AST) -> float | None:
    """The value of a numeric literal, including a unary minus; None for anything else."""
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return sign * node.value
    return None


def _period_value(arg: ast.AST, constants: dict[str, float]) -> float | None:
    if isinstance(arg, ast.Name):
        return constants.get(arg.id)
    return _number(arg)


def _check_generate_signals(tree: ast.Module) -> list[str]:
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_signals"]
    if not fns:
        return ["Missing module-level function generate_signals(df)"]
    if len(fns[0].args.args) != 1:
        return ["generate_signals must take exactly one argument (the OHLCV DataFrame)"]
    return []


def _is_istrategy(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "IStrategy":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "IStrategy":
            return True
    return False


def _check_class(tree: ast.Module, name: str) -> list[str]:
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and _is_istrategy(n)]
    if len(classes) != 1:
        return [f"Expected exactly one class deriving from IStrategy, found {len(classes)}"]
    cls = classes[0]
    problems = []
    if cls.name != name:
        problems.append(f"The IStrategy class must be named {name!r} (found {cls.name!r})")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    for required in ("populate_indicators", "populate_entry_trend", "populate_exit_trend"):
        if required not in methods:
            problems.append(f"Class {cls.name} is missing {required}()")
    return problems


def _check_param_count(tree: ast.Module, max_params: int) -> list[str]:
    """Count distinct tunable numeric literals anywhere in the file — the strategy's knobs.

    Every number counts wherever it appears: module constants of any case, annotated
    or tuple-unpacked assignments, containers, arithmetic, and literals inline in
    function bodies and call arguments (rolling(20), > 1.5, ewm(span=200)). A unary
    minus is part of the value. Exempt: 0, 1 and -1 (signal values, axis, booleans as
    ints) and the right-hand side of the Freqtrade boilerplate attributes in
    EXEMPT_CLASS_ATTRS inside the IStrategy class.
    """
    exempt_nodes = set()
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and _is_istrategy(cls):
            for stmt in cls.body:
                targets = stmt.targets if isinstance(stmt, ast.Assign) else (
                    [stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
                if targets and all(isinstance(t, ast.Name) and t.id in EXEMPT_CLASS_ATTRS for t in targets):
                    exempt_nodes.update(id(n) for n in ast.walk(stmt))

    seen: dict[float, int] = {}  # value -> first line
    negated = set()
    for node in ast.walk(tree):
        if id(node) in exempt_nodes:
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = _number(node)
            if value is not None:
                negated.add(id(node.operand))
                seen.setdefault(value, node.lineno)
            continue
        if id(node) in negated:
            continue
        value = _number(node)
        if value is not None:
            seen.setdefault(value, node.lineno)
    # ast.walk visits parents before children, so a negated operand is always marked first.
    tunable = {v: line for v, line in seen.items() if v not in EXEMPT_LITERALS}
    if len(tunable) > max_params:
        listed = ", ".join(f"{v!r} (line {line})" for v, line in sorted(tunable.items(), key=lambda kv: kv[1]))
        return [f"{len(tunable)} distinct tunable numbers; keep it to {max_params} or fewer to limit overfitting. "
                f"Every numeric literal anywhere in the file counts except 0, 1 and -1 and the Freqtrade "
                f"boilerplate attributes. Reuse values or drop parameters. Found: {listed}"]
    return []
