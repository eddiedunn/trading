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

MAX_PARAMS = 6  # module-level numeric constants; the overfitting guard


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


def _check_calls(tree: ast.AST) -> list[str]:
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
            problems.append(f"Call to {func.id}() is not allowed")
        # df.shift(-n) reads the future; Phase 1 would reward it and live trading cannot do it.
        if isinstance(func, ast.Attribute) and func.attr == "shift":
            for arg in list(node.args) + [kw.value for kw in node.keywords if kw.arg == "periods"]:
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    problems.append("shift() with a negative period looks ahead; not allowed")
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
                    problems.append("shift() with a negative period looks ahead; not allowed")
    return problems


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
    """Count module-level UPPER_CASE = <number> constants — the strategy's knobs."""
    count = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, (int, float)) and not isinstance(node.value.value, bool):
                if all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                    count += 1
    if count > max_params:
        return [f"{count} tunable constants; keep it to {max_params} or fewer to limit overfitting"]
    return []
