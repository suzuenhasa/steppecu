# steppe

GPU/CUDA-13 reimplementation of ADMIXTOOLS 2 + qpAdm: f-statistics (f2 block tensor) on
the GPU and a device-resident qpAdm / qpWave fit engine.

steppe is a **GPU product** — a CUDA-capable GPU is required at runtime. The Python
bindings (`steppe._core`, via nanobind) expose `read_f2`, `qpadm`, `qpwave`, and
`qpadm_search` over a pre-built f2-blocks directory, returning pandas-friendly results and
the f2 tensor as a NumPy float64 array.

```python
import steppe
f2 = steppe.read_f2("my_f2_dir/")
res = steppe.qpadm(f2, target="England_BellBeaker",
                   left=["Czechia_EBA_CordedWare", "Turkey_N"],
                   right=["Mbuti", "Israel_Natufian", "Iran_GanjDareh_N",
                          "Han", "Papuan", "Karitiana"])
print(res.weights)   # a pandas DataFrame [target, left, weight, se, z]
```
