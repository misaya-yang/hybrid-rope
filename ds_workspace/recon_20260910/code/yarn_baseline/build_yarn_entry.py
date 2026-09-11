"""Add an official-YaRN arm to an olmo_fast_screen prepared dir (CPU only).

The reviewer asked for official YaRN on the SAME panel the BM/MrRoPE numbers came
from.  The harness is table-agnostic (runtime.install takes a table dict), so this
needs only a table entry, not new evaluation code.

Convention, verified against the stored MrPro table before use:
    values_float32[j] = theta^{-j/K} * 4^{-m_j}      (m=0 native, m=1 one full /4)
i.e. m_j = log4(native_j / stored_j).  MrPro's stored array reproduces this with
m = 0 on j<=14 and m = +1 on j>=32 -- the expected three-band shape.
"""
import json, pathlib, shutil, sys, numpy as np

SRC = pathlib.Path(sys.argv[1])          # prepared_natural_01
DST = pathlib.Path(sys.argv[2])          # new dir
YARN_M = pathlib.Path(sys.argv[3])       # exact_yarn_olmo.json
sys.path.insert(0, sys.argv[4])          # .../code_bias_01
from scripts.experiments.cross_audit.tables import tensor_sha  # noqa: E402

TH, K = 5e5, 64
j = np.arange(K)
native = TH ** (-j / K)

t = json.loads((SRC / "tables.json").read_text())

# --- guard: the convention must reproduce the stored MrPro table -------------
mr = np.asarray(t["MrPro"]["values_float32"], float)
m_mr = np.log(native / mr) / np.log(4)
print("MrPro reconstructed m: j<=14 max|m|=%.2e ; j>=32 max|m-1|=%.2e ; sum_m=%.4f"
      % (np.abs(m_mr[:15]).max(), np.abs(m_mr[32:] - 1).max(), m_mr.sum()))
assert np.abs(m_mr[:15]).max() < 1e-6 and np.abs(m_mr[32:] - 1).max() < 1e-6, \
    "convention check FAILED -- refusing to build"

m_yarn = np.asarray(json.loads(YARN_M.read_text())["m"], float)
assert m_yarn.size == K
values = (native * 4.0 ** (-m_yarn)).astype(np.float32)

DST.mkdir(parents=True, exist_ok=True)
for f in SRC.iterdir():
    if f.name != "tables.json":
        shutil.copy2(f, DST / f.name)
out = dict(t)
out["YaRN"] = {
    "values_float32": [float(x) for x in values],
    "tensor_sha256": tensor_sha(values),
    "gain": float(t["MrPro"]["gain"]),          # matched amplitude, as for BM
    "construction": {"method": "official_yarn", "alpha": 1, "beta": 32, "scale": 4,
                     "source": str(YARN_M), "sum_m": float(m_yarn.sum())},
}
(DST / "tables.json").write_text(json.dumps(out, indent=1))
print("tables in new dir:", list(out.keys()))
print("YaRN: gain=%.6f sum_m=%.4f sha=%s" % (out["YaRN"]["gain"], m_yarn.sum(),
                                             out["YaRN"]["tensor_sha256"][:16]))
