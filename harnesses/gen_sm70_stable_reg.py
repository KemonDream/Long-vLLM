#!/usr/bin/env python3
"""Generate stable-ABI registrations for the SM70 TurboMind/Marlin op surface.

Why this exists
---------------
The 1Cat-vLLM fork registers its SM70 ops in a plain (non-stable)
`TORCH_LIBRARY_EXPAND(_C, ...)` block.  This tree ports the kernels but only
builds the *stable ABI* extension `_C_stable_libtorch`, and the ported TUs
guard their own registration blocks behind `#ifdef VLLM_QPN4_STANDALONE`
(nvfp4_qpn4 / fp8_qpn8 / glm53 / h3).  Result before this script: the symbols
were linked into `_C_stable_libtorch.abi3.so` but *zero* of the 116 SM70 ops
declared in `csrc/ops.h` were reachable from Python, so every SM70 fast path
silently fell back to emulation.

This generator reads three ground truths and emits one C++ TU:

  1. symbol signatures  <- `nm -C` of the built .so (torch::stable::Tensor
     parameter types as actually compiled),
  2. op schemas         <- `ops.def("...")` strings already in the tree
     (1Cat central bindings + this tree's own standalone fragments),
  3. op name list       <- intersection of (1) and (2); everything else is an
     internal helper and must not be registered.

Usage:
  scripts/gen_sm70_stable_reg.py \
     --so vllm/vllm/_C_stable_libtorch.abi3.so \
     --tree vllm \
     --ref /path/to/1Cat-vLLM \
     --out vllm/csrc/libtorch_stable/sm70_stable_registry.cpp
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_DEF = re.compile(r'ops\.def\(\s*((?:"[^"]*"\s*)+)\)\s*;', re.S)
SCHEMA_STR = re.compile(r'"([^"]*)"')
FRAGMENT = re.compile(
    r"\b(?:STABLE_)?TORCH_LIBRARY_FRAGMENT\(\s*\w+\s*,\s*\w+\s*\)\s*\{(.*?)\n\}",
    re.S,
)


def collect_schemas(paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in paths:
        try:
            src = Path(f).read_text(errors="ignore")
        except OSError:
            continue
        blocks = [m.group(1) for m in FRAGMENT.finditer(src)]
        blocks.append(src)  # flat files (e.g. 1Cat central bindings)
        for block in blocks:
            for m in SCHEMA_DEF.finditer(block):
                body = "".join(SCHEMA_STR.findall(m.group(1)))
                name = body.split("(", 1)[0].strip()
                if name:
                    out.setdefault(name, body)
    return out


def collect_signatures(so: str) -> dict[str, str]:
    nm = subprocess.run(
        ["nm", "-C", so], capture_output=True, text=True, check=True
    ).stdout
    out: dict[str, str] = {}
    for line in nm.splitlines():
        m = re.match(r"^[0-9a-f]+ [tT] ([A-Za-z0-9_]+)\((.*)\)$", line.strip())
        if not m or "[clone .cold]" in line:
            continue
        out.setdefault(m.group(1), m.group(2))
    return out


def split_args(argstr: str) -> list[str]:
    args, depth, cur = [], 0, ""
    for ch in argstr:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


SCHEMA_TO_CPP = {
    "Tensor": "torch::stable::Tensor",
    "Tensor?": "std::optional<torch::stable::Tensor>",
    "int": "int64_t",
    "SymInt": "int64_t",
    "float": "double",
    "bool": "bool",
    "str": "std::string",
    "int[]": "std::vector<int64_t>",
    "float[]": "std::vector<double>",
    "ScalarType": "int64_t",
    "Device": "int64_t",
    "Generator?": "std::optional<int64_t>",
}


def split_rets(retstr: str) -> list[str]:
    retstr = retstr.strip()
    if retstr in ("", "()"):
        return []
    if retstr.startswith("(") and retstr.endswith(")"):
        return [r.strip() for r in retstr[1:-1].split(") ,")] if False else [
            r.strip() for r in split_args(retstr[1:-1])
        ]
    if retstr.endswith("[]"):
        return [retstr]
    return [retstr]


def cpp_return(rets: list[str]) -> str | None:
    if not rets:
        return "void"
    cpp: list[str] = []
    for r in rets:
        if r.endswith("[]"):
            inner = SCHEMA_TO_CPP.get(r[:-2].strip())
            if inner is None:
                return None
            cpp.append(f"std::vector<{inner}>")
        else:
            t = SCHEMA_TO_CPP.get(r)
            if t is None:
                return None
            cpp.append(t)
    return cpp[0] if len(cpp) == 1 else "std::tuple<" + ", ".join(cpp) + ">"


def parse_schema(schema: str) -> tuple[str, list[str], list[str]]:
    name, rest = schema.split("(", 1)
    args_str, ret_str = rest.rsplit(")", 1)
    ret_str = ret_str.split("->", 1)[1] if "->" in ret_str else "()"
    args = []
    for a in split_args(args_str):
        if not a:
            continue
        # strip names: "Tensor(a!) out" / "int group_size"
        parts = a.split()
        args.append(parts[0])
    return name.strip(), args, split_rets(ret_str)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", required=True)
    ap.add_argument("--tree", required=True, help="vllm package root (has csrc/)")
    ap.add_argument("--ref", help="1Cat-vLLM checkout for central schemas")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing-schema", action="store_true", default=True)
    args = ap.parse_args()

    tree = Path(args.tree)
    srcs = sorted(
        glob.glob(str(tree / "csrc/**/*.cu"), recursive=True)
        + glob.glob(str(tree / "csrc/**/*.cpp"), recursive=True)
    )
    schema_paths = srcs + ([str(Path(args.ref) / "csrc/torch_bindings.cpp")] if args.ref else [])
    schemas = collect_schemas(schema_paths)
    sigs = collect_signatures(args.so)

    selected: dict[str, dict] = {}
    skipped: list[str] = []
    for name, argstr in sorted(sigs.items()):
        # SM70 op surface only: plausible op names, no device stubs/templates.
        if "sm70" not in name or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            continue
        schema = schemas.get(name)
        if schema is None:
            skipped.append(name)
            continue
        sname, sargs, srets = parse_schema(schema)
        ret = cpp_return(srets)
        if ret is None:
            skipped.append(f"{name} (unsupported return {srets})")
            continue
        nargs = split_args(argstr) if argstr else []
        if len(nargs) != len(sargs):
            skipped.append(
                f"{name} (arity mismatch: compiled {len(nargs)} vs schema {len(sargs)})"
            )
            continue
        bad = [t for t in nargs if t not in ("torch::stable::Tensor", "long", "double", "bool")
               and not t.startswith("std::optional<torch::stable::Tensor>")]
        if bad:
            skipped.append(f"{name} (unsupported param types {bad})")
            continue
        selected[name] = {
            "schema": schema,
            "params": nargs,
            "ret": ret,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("// GENERATED by scripts/gen_sm70_stable_reg.py -- do not edit by hand.")
    lines.append("//")
    lines.append("// Registers the SM70 TurboMind / Marlin op surface into the stable ABI")
    lines.append("// extension `_C` so torch.ops._C.<op> resolves (the ported kernels are")
    lines.append("// otherwise linked but unreachable, which silently forces every SM70")
    lines.append("// fast path onto the slow emulation fallback).")
    lines.append("//")
    lines.append(f"// ops registered: {len(selected)}; skipped: {len(skipped)}")
    for s in skipped[:60]:
        lines.append(f"//   skipped (internal helper or unsupported signature): {s}")
    lines.append("")
    lines.append("#include <torch/csrc/stable/library.h>")
    lines.append("#include <torch/csrc/stable/ops.h>")
    lines.append("#include <torch/csrc/stable/tensor.h>")
    lines.append("")
    lines.append("#include <optional>")
    lines.append("#include <tuple>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append("#ifdef ENABLE_SM70_TURBOMIND")
    lines.append("")
    # Declarations must stay at global scope: the kernels are defined at global
    # scope in their own TUs, and an anonymous namespace here would declare a
    # *different* internal symbol and fail to link.
    for name, info in selected.items():
        params = ", ".join(f"{t} a{i}" for i, t in enumerate(info["params"]))
        lines.append(f"{info['ret']} {name}({params});")
    lines.append("")
    lines.append("STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {")
    for name, info in selected.items():
        schema = info["schema"]
        # re-flow long schema strings across lines
        chunk = f'      "{schema}");'
        if len(chunk) <= 88:
            lines.append("  ops.def(")
            lines.append(chunk)
        else:
            lines.append("  ops.def(")
            words = schema.split(", ")
            acc = '      "'
            pieces = []
            for w in words:
                frag = w + (", " if w is not words[-1] else "")
                if len(acc) + len(frag) > 80:
                    pieces.append(acc + '"')
                    acc = '      "' + frag
                else:
                    acc += frag
            pieces.append(acc + '");')
            lines.extend(pieces)
    lines.append("}")
    lines.append("")
    lines.append("STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {")
    for name in selected:
        lines.append(f'  ops.impl("{name}", TORCH_BOX(&{name}));')
    lines.append("}")
    lines.append("")
    lines.append("#endif  // ENABLE_SM70_TURBOMIND")
    out.write_text("\n".join(lines) + "\n")

    print(f"registered {len(selected)} ops -> {out}")
    print(f"skipped {len(skipped)}")
    for s in skipped:
        print("  skip:", s)
    json.dump(selected, open("/tmp/sm70_selected.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
