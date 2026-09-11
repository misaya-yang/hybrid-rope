import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
f = d / "spec.json"
s = json.loads(f.read_text())
lens = {k: len(v) for k, v in s["methods"].items()}
print("before:", lens)
n = max(lens.values())
assert all(v == n for v in lens.values()), "ragged method lists -- refusing"
if "YaRN" in s["methods"]:
    print("YaRN already present")
else:
    s["methods"]["YaRN"] = ["YaRN"] * n
    f.write_text(json.dumps(s, indent=1))
    print("added YaRN x", n)
print("after :", {k: len(v) for k, v in json.loads(f.read_text())["methods"].items()})
print("tables:", list(json.loads((d / "tables.json").read_text()).keys()))
