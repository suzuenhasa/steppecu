// src/core/qpadm/model_search.hpp
//
// M(fit-6) S8 ROTATION orchestrator (design §1.3 / the FROZEN CONTRACT §1.3, §3).
// HOST-PURE, CUDA-FREE: it names only the CUDA-free ComputeBackend seam + the
// CUDA-free Resources, exactly like qpadm_fit.cpp. It owns:
//   * the base ComputeBackend::fit_models_batched DEFAULT body (the per-model fit
//     over the backend's OWN five device virtuals — inherited by BOTH backends);
//   * the two public run_qpadm_search entries (device-resident + host-oracle);
//   * the multi-GPU model-shard fan-out (jthread-per-device, pre-sized-slot re-sort,
//     lowest-index rethrow) + the one-time f2 replication broadcast.
#ifndef STEPPE_CORE_QPADM_MODEL_SEARCH_HPP
#define STEPPE_CORE_QPADM_MODEL_SEARCH_HPP

#include <span>
#include <vector>

#include "device/backend.hpp"  // steppe::ComputeBackend
#include "steppe/qpadm.hpp"     // QpAdmModel/QpAdmResult/QpAdmOptions + run_qpadm_search decls

namespace steppe::core::qpadm {

/// The shared per-model fit body the base ComputeBackend::fit_models_batched default
/// delegates to: assemble_f4(device-resident) → run_impl (S4→S6→S7 + rankdrop +
/// popdrop), for ONE model, on `be`'s device. Both the CudaBackend (device-resident
/// batched kernels) and the CpuBackend (the scalar oracle) reach it through the SAME
/// five virtuals run_qpadm uses, so the device batched path and the host oracle are
/// bit-comparable per model. The popdrop full-model row's feasibility rides
/// QpAdmResult::popdrop_feasible (qpadm.hpp), not a parameter here.
[[nodiscard]] QpAdmResult fit_one_model_device(ComputeBackend& be,
                                               const device::DeviceF2Blocks& f2,
                                               const QpAdmModel& model,
                                               const QpAdmOptions& opts);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_MODEL_SEARCH_HPP
