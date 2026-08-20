#!/usr/bin/env bash
# Regenerate compile_commands.json for the gemm_ar_blackwell build only.
#
# Two steps:
#   1. `bear` intercepts the real nvcc invocation from `make`, so the defines
#      and include paths in the DB are exactly what the compiler saw.
#   2. nvcc-only flags are stripped, because clangd drives clang. Flags that
#      take a separate argument (-gencode, -ccbin, ...) must have that argument
#      dropped too, or clang treats it as a second input file.
#
# clang-side flags (-xcuda, --cuda-path, the ~/.config/mkernel-clangd shims)
# live in .clangd, not here.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/shawnwei/miniconda3/envs/mkernel/bin/python}

bear --output compile_commands.json -- \
    make GPU=blackwell PYTHON="$PYTHON" -B gemm_ar_blackwell

python3 - <<'PY'
import json, shlex

DROP_WITH_ARG = {"-gencode", "--generate-code", "-ccbin", "--compiler-bindir",
                 "-Xptxas", "-Xcompiler", "-Xlinker", "--compiler-options",
                 "--ptxas-options", "-o"}
DROP_EXACT = {"--use_fast_math", "--extended-lambda", "--expt-relaxed-constexpr",
              "--expt-extended-lambda", "-lineinfo", "-shared", "-dc", "--device-c"}
DROP_PREFIX = ("-arch=", "--gpu-architecture", "--ptxas-options=", "-rdc=",
               "-fdevice-sanitize", "-l", "-L", "--compiler-options=")

def clean(argv):
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if i == 0:
            out.append(a); continue
        if a in DROP_WITH_ARG:
            skip = True; continue
        if a in DROP_EXACT:
            continue
        if a.startswith(DROP_PREFIX):
            continue
        out.append(a)
    return out

db = json.load(open("compile_commands.json"))
for e in db:
    argv = e.pop("arguments", None) or shlex.split(e.pop("command"))
    e["arguments"] = clean(argv)
json.dump(db, open("compile_commands.json", "w"), indent=2)
print(f"{len(db)} entry/entries written")
PY
