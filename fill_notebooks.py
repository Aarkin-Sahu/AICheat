#!/usr/bin/env python3
"""
fill_notebooks.py - fill the "YOUR CODE HERE" blanks in .ipynb files with a
local Ollama model. Works offline. Uses only the Python standard library.

Usage (run from the folder that holds the notebooks):
    python fill_notebooks.py                   # every .ipynb in this folder
    python fill_notebooks.py Problems          # every .ipynb in ./Problems
    python fill_notebooks.py my_notebook.ipynb # a single notebook

Options:
    --model NAME   Ollama model to use      (default: qwen2.5-coder:7b)
    --out DIR      where filled notebooks go (default: filled)
    --check        after filling, run each notebook and report if it executes
    --force        redo notebooks that already exist in the output folder

Filled notebooks are saved one by one, so you can stop with Ctrl+C and
re-run later; finished notebooks are skipped.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST

DEFAULT_MARKER = r"^\s*#+\s*(>>>\s*)?YOUR CODE HERE\b"
START_RE = re.compile(DEFAULT_MARKER)
END_RE = re.compile(r"^\s*#+\s*=+\s*#*\s*$")
MAX_TRIES = 3
TEMPERATURES = [0.1, 0.3, 0.5]


# --------------------------------------------------------------------------
# Talking to Ollama
# --------------------------------------------------------------------------
def ollama_ask(prompt, model, temperature, num_ctx):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": 1024},
    }).encode("utf-8")
    req = urllib.request.Request(
        HOST + "/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["response"]


def check_ollama(model):
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        sys.exit("Cannot reach Ollama at %s.\n"
                 "Start the Ollama app (llama icon in the system tray) or run "
                 "'ollama serve' in another terminal, then try again." % HOST)
    if not any(n == model or n.startswith(model + ":") or n == model + ":latest" for n in names):
        sys.exit("Model '%s' is not installed. Installed models: %s\n"
                 "Run:  ollama pull %s" % (model, ", ".join(names) or "none", model))


# --------------------------------------------------------------------------
# Finding blanks inside a cell
# --------------------------------------------------------------------------
def indent_of(line):
    return len(line) - len(line.lstrip())


def find_blanks(lines):
    """Return [(start, end, has_end_marker)] for every blank in the cell.

    start = index of the 'YOUR CODE HERE' line.
    end   = index just past the blank. If an '=====' closing marker exists it
            is the last line of the blank (and is kept in the output).
    Handles the four formats found in the assignment notebooks:
      A) '### >>> YOUR CODE HERE <<< ###' ... pass ... '### ====== ###'
      B) '### >>> YOUR CODE HERE <<<'     ... pass ... '### ====== ###'
      C) '### YOUR CODE HERE ###' followed by 'pass' (no closing marker)
      D) '### YOUR CODE HERE ###' + comments + raise NotImplementedError(...)
    """
    blanks = []
    for i, line in enumerate(lines):
        if not START_RE.search(line):
            continue
        ind = indent_of(line)
        j = i + 1
        last_content = i
        has_end = False
        while j < len(lines):
            l = lines[j]
            if not l.strip():
                j += 1
                continue
            if END_RE.match(l) and indent_of(l) >= ind:
                last_content = j
                has_end = True
                break
            if indent_of(l) < ind:
                break
            last_content = j
            j += 1
        blanks.append((i, last_content + 1, has_end))
    return blanks



def repair_indented_method(text, last_class):
    """Some notebooks ship a method indented inside its own cell, which is an
    IndentationError. Dedent it into a normal function and attach it to the
    class defined in an earlier cell (the same pattern other notebooks use)."""
    try:
        ast.parse(text)
        return text, False
    except IndentationError:
        pass
    except SyntaxError:
        return text, False
    lines = text.split("\n")
    k = 0
    while k < len(lines) and (not lines[k].strip() or lines[k].lstrip().startswith("#")) \
            and indent_of(lines[k]) == 0:
        k += 1
    head, rest = lines[:k], lines[k:]
    if not rest or indent_of(rest[0]) == 0:
        return text, False
    rest = textwrap.dedent("\n".join(rest)).split("\n")
    m = re.match(r"def\s+(\w+)\s*\(", rest[0])
    if not m:
        return text, False
    fixed = head + rest
    if last_class:
        fixed += ["", "%s.%s = %s" % (last_class, m.group(1), m.group(1))]
    new = "\n".join(fixed)
    try:
        ast.parse(new)
    except SyntaxError:
        return text, False
    return new, True

# --------------------------------------------------------------------------
# Prompt + cleaning the model's answer
# --------------------------------------------------------------------------
def notebook_context(nb):
    parts = []
    for k, c in enumerate(nb["cells"], start=1):
        text = "".join(c["source"]).rstrip()
        parts.append("[Cell %d - %s]\n%s" % (k, c["cell_type"], text))
    return "\n\n".join(parts)


def build_prompt(context, cell_text, error=None):
    p = (
        "You are completing a university programming assignment notebook.\n"
        "Below is the ENTIRE notebook (markdown and code cells) so you can see every\n"
        "constant, class, formula and helper that already exists.\n\n"
        "=== NOTEBOOK START ===\n%s\n=== NOTEBOOK END ===\n\n"
        "Now complete the blank inside this cell:\n\n"
        "=== CELL ===\n%s\n=== END CELL ===\n\n"
        "Replace everything after the 'YOUR CODE HERE' marker (the `pass` line or the\n"
        "`raise NotImplementedError` line) with working code.\n\n"
        "Rules:\n"
        "- Follow the docstring and comments exactly. Use the exact variable names,\n"
        "  constants and formulas given in the notebook.\n"
        "- Output ONLY the Python statements that go in the blank. No `def` or `class`\n"
        "  line, no docstring, no explanation, no markdown fences.\n"
        "- Start the first line at column 0 (it will be indented for you); indent\n"
        "  nested blocks with 4 spaces.\n"
        "- Include `return` statements wherever the docstring says to return a value.\n"
        "- Only use libraries that are already imported in the notebook.\n"
    ) % (context, cell_text)
    if error:
        p += "\nYour previous attempt was rejected: %s\nTry again and fix that.\n" % error
    return p


def extract_body(resp):
    """Turn the raw model answer into a clean block of statements."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", resp, re.S)
    if m:
        resp = m.group(1)
    resp = resp.replace("```", "").strip("\n")
    resp = textwrap.dedent(resp)

    # If the model repeated the whole function, keep only its body.
    first = next((l for l in resp.splitlines() if l.strip()), "")
    if first.lstrip().startswith(("def ", "class ")):
        try:
            tree = ast.parse(resp)
            node = tree.body[0]
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]  # drop docstring
            if body and isinstance(node, ast.FunctionDef):
                src = resp.splitlines()[body[0].lineno - 1: node.end_lineno]
                resp = textwrap.dedent("\n".join(src))
        except SyntaxError:
            pass
    return resp.strip("\n")


def apply_blank(lines, start, end, has_end, body):
    ind = lines[start][: indent_of(lines[start])]
    inner_end = end - 1 if has_end else end
    kept_comments = [l for l in lines[start + 1: inner_end] if l.strip().startswith("#")]
    new_body = [(ind + l if l.strip() else "") for l in body.splitlines()]
    out = lines[: start + 1] + kept_comments + new_body
    if has_end:
        out.append(lines[end - 1])
    return out + lines[end:]


def est_ctx(prompt):
    need = len(prompt) // 3 + 1300
    for size in (4096, 8192, 12288, 16384, 24576, 32768):
        if need <= size:
            return size
    return 32768


# --------------------------------------------------------------------------
# Filling one notebook
# --------------------------------------------------------------------------
def fill_notebook(src, dst, ask):
    nb = json.load(open(src, encoding="utf-8"))
    context = notebook_context(nb)
    total = failed = 0
    last_class = None

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        original = "".join(cell["source"])
        original, repaired = repair_indented_method(original, last_class)
        if repaired:
            print("    note: cell %d was an indented method; converted to a function"
                  " and attached to class %s" % (idx + 1, last_class))
        cm = re.findall(r"^class\s+(\w+)", original, re.M)
        if cm:
            last_class = cm[-1]
        lines = original.split("\n")
        blanks = find_blanks(lines)
        if not blanks:
            if repaired:
                cell["source"] = original.splitlines(keepends=True)
            continue

        # Work bottom-to-top so earlier line numbers stay valid.
        for (start, end, has_end) in reversed(blanks):
            total += 1
            title = original.strip().splitlines()[0][:70]
            t0 = time.time()
            error = None
            result = None
            for attempt in range(MAX_TRIES):
                prompt = build_prompt(context, "\n".join(lines), error)
                resp = ask(prompt, TEMPERATURES[attempt], est_ctx(prompt))
                body = extract_body(resp)
                if not body.strip() or body.strip() == "pass":
                    error = "the answer was empty or just `pass`."
                    continue
                candidate = apply_blank(lines, start, end, has_end, body)
                try:
                    ast.parse("\n".join(candidate))
                except SyntaxError as e:
                    error = "Python syntax error: %s (line %s)." % (e.msg, e.lineno)
                    continue
                result = candidate
                break
            dt = time.time() - t0
            if result is None:
                failed += 1
                print("    ! FAILED after %d tries (%s): %s" % (MAX_TRIES, error, title))
            else:
                lines = result
                print("    ok  (%.0fs, try %d) %s" % (dt, attempt + 1, title))

        new_text = "\n".join(lines)
        cell["source"] = new_text.splitlines(keepends=True)

    if total == 0:
        return 0, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return total, failed


# --------------------------------------------------------------------------
# Optional: run the filled notebook to see if it executes
# --------------------------------------------------------------------------
def run_check(path):
    nb = json.load(open(path, encoding="utf-8"))
    code = ["import matplotlib\nmatplotlib.use('Agg')"]
    code += ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    try:
        r = subprocess.run([sys.executable, "-c", "\n\n".join(code)],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "timed out after 10 minutes"
    if r.returncode == 0:
        tail = "\n".join(r.stdout.strip().splitlines()[-4:])
        return True, tail
    return False, "\n".join(r.stderr.strip().splitlines()[-6:])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--out", default="filled")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--marker", default=None,
                    help="regex for the line where a blank starts. Default matches\n"
                         "'YOUR CODE HERE'. Examples:  --marker \"TODO\"   "
                         "--marker \"WRITE CODE BELOW\"")
    a = ap.parse_args()
    if a.marker:
        global START_RE
        START_RE = re.compile(a.marker)

    p = Path(a.path)
    out_dir = Path(a.out)
    if p.is_dir():
        files = sorted(f for f in p.glob("*.ipynb") if out_dir.resolve() not in f.resolve().parents)
    else:
        files = [p]
    if not files:
        sys.exit("No .ipynb files found in %s" % p)

    check_ollama(a.model)
    ask = lambda prompt, temp, ctx: ollama_ask(prompt, a.model, temp, ctx)

    print("Model: %s | %d notebook(s) | output -> %s/\n" % (a.model, len(files), out_dir))
    summary = []
    for n, f in enumerate(files, 1):
        dst = out_dir / f.name
        print("[%d/%d] %s" % (n, len(files), f.name))
        if dst.exists() and not a.force:
            print("    already done, skipping (use --force to redo)")
            summary.append((f.name, "skipped", ""))
            continue
        try:
            total, failed = fill_notebook(f, dst, ask)
        except urllib.error.URLError as e:
            sys.exit("Lost connection to Ollama: %s" % e)
        if total == 0:
            print("    WARNING: no blanks found in this notebook.\n"
                  "    The script looks for a comment line containing 'YOUR CODE HERE'.\n"
                  "    If your notebook uses other wording, pass it with --marker, e.g.\n"
                  "        python fill_notebooks.py %s --marker \"TODO\"" % f.name)
            summary.append((f.name, "NO BLANKS FOUND (nothing written)", ""))
            continue
        status = "%d/%d blanks filled" % (total - failed, total)
        if a.check and total:
            ok, info = run_check(dst)
            status += " | runs: %s" % ("YES" if ok else "NO")
            if info:
                print("    " + info.replace("\n", "\n    "))
        summary.append((f.name, status, ""))

    print("\n=== Summary ===")
    for name, status, _ in summary:
        print("%-32s %s" % (name, status))
    print("\nFilled notebooks are in the '%s' folder. Review each one before submitting." % out_dir)


if __name__ == "__main__":
    main()
