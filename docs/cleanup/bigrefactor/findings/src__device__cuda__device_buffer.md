# Review findings — src__device__cuda__device_buffer

Files: /home/suzunik/steppe/src/device/cuda/device_buffer.cuh

## Group 4 — Type & numeric

- [4.7][LOW] device_buffer.cuh:95-96 — `data()` / `const data()` hand out a raw `T*` that is a *device* pointer with no host-vs-device space distinction, so a caller can pass it to a host-side `memcpy`/deref without a compile error. Suggested: acceptable per steppe design (raw device ptr for cuBLAS/kernel args); a thin `DevicePtr<T>` wrapper would make the wrong space unpassable, but this is hygiene, not a bug.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.
