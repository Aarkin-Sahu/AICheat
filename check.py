#!/usr/bin/env python3
"""
check.py - ask a local Ollama model to EXPLAIN a (filled-in) assignment notebook.
Works offline. Uses only the Python standard library.

Usage:
    python check.py Category01_Problem01.ipynb
    python check.py Category01_Problem01            (".ipynb" is added for you)
    python check.py filled\\Category01_Problem01.ipynb

If the file is not found in the current folder, the "filled" folder is tried too.

What you get, printed live and saved to explanations\\<name>_explanation.txt:
    1. THE PROBLEM   - the general scenario, goal, and method
    2. THE WORKFLOW  - how all the cells connect, step by step
    3. YOUR CELLS    - a deep dive on each student cell (Cell 4, Cell 5, ...):
                       purpose, line-by-line walkthrough, a worked example,
                       a check against the requirements, and common mistakes

Options:
    --model NAME      Ollama model to use                (default: qwen2.5-coder:7b)
    --only PART ...   run only some parts: problem workflow cells
    --cells 4,5       explain these cells in depth (default: auto-detect the
                      cells with 'YOUR CODE HERE' / 'STUDENT IMPLEMENTATION')
    --out DIR         where the text file is saved       (default: explanations)
    --no-save         print only, do not write a file

Press Ctrl+C at any time: what was printed so far is still saved.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST

NUM_PREDICT = 1400
TEMPERATURE = 0.3

PREAMBLE = (
    "You are a patient programming tutor. A student is studying an assignment "
    "notebook. The code in it is already completed, including the cells the "
    "student had to write. Explain in simple, clear English for someone who "
    "knows basic Python but is new to this topic. Use plain text only: put "
    "section headings in CAPITAL LETTERS on their own line, use numbered lists "
    "and '-' bullets, and no markdown tables or code fences. When you quote "
    "code, put it on its own indented line.\n\n"
)

TASK_PROBLEM = (
    "TASK: Explain the PROBLEM this notebook is about, in general terms. Do not "
    "walk through the code line by line.\n"
    "Cover, in this order:\n"
    "1. THE SCENARIO - what real-world situation is being modelled (3-4 sentences).\n"
    "2. THE GOAL - what should be achieved and how success is measured.\n"
    "3. THE PIECES - the key ingredients (states, actions, rewards, constraints, "
    "parameters, whatever applies) with the actual values from the notebook.\n"
    "4. THE METHOD - which technique solves it (name it) and the intuition behind "
    "it, using an everyday analogy.\n"
    "5. WHAT THE STUDENT IMPLEMENTS - one sentence per student task.\n"
    "Keep it under about 450 words."
)

TASK_WORKFLOW = (
    "TASK: Explain the WORKFLOW of the whole notebook: how the cells connect and "
    "what happens when it runs from top to bottom.\n"
    "Cover, in this order:\n"
    "1. CELL BY CELL - for each cell, 1-2 sentences: what it does and what it "
    "hands to the next cell. Say which cells are locked and which are student cells.\n"
    "2. ONE EPISODE STEP BY STEP - follow one pass of the main simulation loop, "
    "naming the functions called in order and what data moves between them.\n"
    "3. WHAT COMES OUT - what the printed numbers and plots mean, and what a "
    "good result looks like.\n"
    "Keep it under about 550 words."
)

TASK_CELL = (
    "TASK: Explain Cell {n} in depth. This is code the student wrote, so it "
    "matters most.\n"
    "Cover, in this order:\n"
    "1. PURPOSE - what this code is for in the big picture, and which cell and "
    "line call it.\n"
    "2. INPUTS AND OUTPUTS - what goes in, and what must be returned or updated.\n"
    "3. LINE BY LINE - go through the code in order. Quote each line (or small "
    "group of lines) on its own indented line, then explain it.\n"
    "4. TINY EXAMPLE - a small worked example with concrete numbers showing the "
    "code in action.\n"
    "5. REQUIREMENTS CHECK - compare the code with the requirements written in "
    "the notebook (docstring, comments, markdown). If a requirement is missed, or "
    "the code uses a name that is not defined anywhere in the notebook, point to "
    "the exact line. If everything matches, say so. Do not invent problems.\n"
    "6. COMMON MISTAKES - 2 or 3 mistakes students often make here.\n"
    "Keep it under about 650 words.\n\n"
    "Cell {n} is:\n{code}"
)


# --------------------------------------------------------------------------
# Finding and reading the notebook
# --------------------------------------------------------------------------
def resolve(name):
    base = [Path(name)]
    if not name.lower().endswith(".ipynb"):
        base.append(Path(name + ".ipynb"))
    cands = base + [Path("filled") / p.name for p in base]
    for c in cands:
        if c.is_file():
            return c
    here = sorted(p.name for p in Path(".").glob("*.ipynb"))
    filled = sorted("filled\\" + p.name for p in Path("filled").glob("*.ipynb")) \
        if Path("filled").is_dir() else []
    msg = ["Cannot find '%s'." % name,
           "Type the exact filename including .ipynb (press Tab to autocomplete)."]
    if here or filled:
        msg.append("Notebooks I can see from here:")
        msg += ["    " + n for n in here + filled]
    else:
        msg.append("There are no .ipynb files in this folder. Open PowerShell in the "
                   "folder that holds them.")
    sys.exit("\n".join(msg))


def notebook_context(nb):
    parts = []
    for k, c in enumerate(nb["cells"], start=1):
        text = "".join(c["source"]).rstrip()
        parts.append("[Cell %d - %s]\n%s" % (k, c["cell_type"], text))
    return "\n\n".join(parts)


def pick_student_cells(nb, forced):
    n = len(nb["cells"])
    if forced:
        cells = [int(x) for x in re.split(r"[,\s]+", forced.strip()) if x]
        bad = [c for c in cells if not 1 <= c <= n]
        if bad:
            sys.exit("--cells: this notebook has %d cells, so %s is out of range." % (n, bad))
        return cells
    found = []
    for k, c in enumerate(nb["cells"], start=1):
        if c["cell_type"] != "code":
            continue
        if re.search(r"YOUR CODE HERE|STUDENT IMPLEMENTATION", "".join(c["source"]), re.I):
            found.append(k)
    if found:
        return found
    return [k for k in (4, 5) if k <= n]


# --------------------------------------------------------------------------
# Talking to Ollama (streaming, so you see words as they appear)
# --------------------------------------------------------------------------
def check_ollama(model):
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        sys.exit("Cannot reach Ollama at %s.\n"
                 "Start the Ollama app (llama icon near the clock) or run "
                 "'ollama serve' in another terminal, then try again." % HOST)
    if not any(n == model or n == model + ":latest" or n.startswith(model + ":") for n in names):
        sys.exit("Model '%s' is not installed. Installed: %s\nRun:  ollama pull %s"
                 % (model, ", ".join(names) or "none", model))


def est_ctx(prompt):
    need = len(prompt) // 3 + NUM_PREDICT + 300
    for size in (4096, 8192, 12288, 16384, 24576, 32768):
        if need <= size:
            return size
    return 32768


def stream_answer(prompt, model):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": TEMPERATURE, "num_ctx": est_ctx(prompt),
                    "num_predict": NUM_PREDICT},
    }).encode("utf-8")
    req = urllib.request.Request(HOST + "/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    out = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                raw = raw.strip()
                if not raw:
                    continue
                part = json.loads(raw)
                if "error" in part:
                    sys.exit("\nOllama error: %s" % part["error"])
                tok = part.get("response", "")
                out.append(tok)
                print(tok, end="", flush=True)
                if part.get("done"):
                    break
    except urllib.error.URLError as e:
        sys.exit("\nLost connection to Ollama: %s" % e)
    print()
    return "".join(out).strip()


# --------------------------------------------------------------------------
def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("file", help="the notebook, e.g. Category01_Problem01.ipynb")
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--only", nargs="+", choices=["problem", "workflow", "cells"])
    ap.add_argument("--cells", default=None, help='e.g. "4,5"')
    ap.add_argument("--out", default="explanations")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    path = resolve(a.file)
    nb = json.load(open(path, encoding="utf-8"))
    check_ollama(a.model)

    context = notebook_context(nb)
    student = pick_student_cells(nb, a.cells)
    parts = a.only or ["problem", "workflow", "cells"]

    # Same beginning for every request, so Ollama can reuse the notebook it
    # has already read and the later parts start faster.
    head = PREAMBLE + "=== NOTEBOOK START ===\n" + context + "\n=== NOTEBOOK END ===\n\n"

    print("Notebook: %s | model: %s | student cells: %s" % (path, a.model, student))
    for k in student:
        code = "".join(nb["cells"][k - 1]["source"])
        if re.search(r"^\s*pass\s*$|raise NotImplementedError", code, re.M):
            print("WARNING: Cell %d still has an empty placeholder (pass / "
                  "NotImplementedError). Did you give me the original notebook "
                  "instead of the filled one?" % k)
    print("On CPU the first words of each part can take a minute or two. "
          "Press Ctrl+C to stop.\n")

    report = ["Explanation of %s" % path.name, "Model: %s" % a.model, ""]

    def run(title, task):
        banner = "=" * 12 + " " + title + " " + "=" * 12
        print(banner)
        t0 = time.time()
        text = stream_answer(head + task, a.model)
        print("\n(%.0fs)\n" % (time.time() - t0))
        report.extend([banner, text, ""])

    try:
        if "problem" in parts:
            run("1. THE PROBLEM", TASK_PROBLEM)
        if "workflow" in parts:
            run("2. THE WORKFLOW", TASK_WORKFLOW)
        if "cells" in parts:
            for k in student:
                code = "".join(nb["cells"][k - 1]["source"])
                run("3. YOUR CODE: CELL %d" % k, TASK_CELL.format(n=k, code=code))
    except KeyboardInterrupt:
        print("\n\nStopped. Saving what was finished so far.")

    if not a.no_save and len(report) > 3:
        out_dir = Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / (path.stem + "_explanation.txt")
        dst.write_text("\n".join(report), encoding="utf-8")
        print("Saved to %s" % dst)


if __name__ == "__main__":
    main()
