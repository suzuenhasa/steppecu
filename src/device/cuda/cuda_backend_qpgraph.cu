// src/device/cuda/cuda_backend_qpgraph.cu
//
// CudaBackend's qpGraph fleet methods: host-side orchestration that uploads the
// fit inputs, fires one kernel launch, and reduces the small result arrays. The
// optimizer math itself lives in the companion kernel TU (qpgraph_fit_kernels.cu).
// A CUDA TU private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_qpgraph.cu.md
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/qpgraph_fit_kernels.cuh"
#include "device/cuda/check.cuh"

#include <stdexcept>
#include <string>
#include <vector>

namespace steppe::device {

// Fit one topology: the single-topology full fit — reference §5
QpGraphFleet CudaBackend::qpgraph_fit_fleet(const QpGraphTopoArena& topo,
                                             std::span<const double> f_obs,
                                             std::span<const double> qinv,
                                             int numstart, int maxit, double tol,
                                             const Precision& precision) {
    (void)precision;
    guard_device();
    QpGraphFleet out;
    const int D = topo.nadmix;
    if (D > kMaxThetaDev)
        throw std::runtime_error(
            "steppe::device::CudaBackend::qpgraph_fit_fleet: topology nadmix=" +
            std::to_string(D) + " exceeds the per-thread theta-stack cap kMaxThetaDev=" +
            std::to_string(kMaxThetaDev) + ".");
    const int npair = topo.npair, ne = topo.nedge_norm;

    DeviceBuffer<double> dFobs(static_cast<std::size_t>(npair));
    DeviceBuffer<double> dQinv(static_cast<std::size_t>(npair) * static_cast<std::size_t>(npair));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dFobs.data(), f_obs.data(), f_obs.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), qinv.data(), qinv.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    DeviceBuffer<double> dPwts0(topo.pwts0.size());
    DeviceBuffer<int> dPeEdge(topo.pe_edge.size() ? topo.pe_edge.size() : 1);
    DeviceBuffer<int> dPeLeaf(topo.pe_leaf.size() ? topo.pe_leaf.size() : 1);
    DeviceBuffer<int> dPePath(topo.pe_path.size() ? topo.pe_path.size() : 1);
    DeviceBuffer<int> dPaePath(topo.pae_path.size() ? topo.pae_path.size() : 1);
    DeviceBuffer<int> dPaeEdge(topo.pae_admixedge.size() ? topo.pae_admixedge.size() : 1);
    DeviceBuffer<int> dCmb1(topo.cmb1.size());
    DeviceBuffer<int> dCmb2(topo.cmb2.size());
    auto up_d = [&](DeviceBuffer<double>& d, const std::vector<double>& h) {
        if (!h.empty()) STEPPE_CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.size() * sizeof(double),
                                                          cudaMemcpyHostToDevice, stream_.get()));
    };
    auto up_i = [&](DeviceBuffer<int>& d, const std::vector<int>& h) {
        if (!h.empty()) STEPPE_CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.size() * sizeof(int),
                                                          cudaMemcpyHostToDevice, stream_.get()));
    };
    up_d(dPwts0, topo.pwts0);
    up_i(dPeEdge, topo.pe_edge); up_i(dPeLeaf, topo.pe_leaf); up_i(dPePath, topo.pe_path);
    up_i(dPaePath, topo.pae_path); up_i(dPaeEdge, topo.pae_admixedge);
    up_i(dCmb1, topo.cmb1); up_i(dCmb2, topo.cmb2);

    QpGraphDeviceTopo dt{};
    dt.npop = topo.npop; dt.nedge_norm = ne; dt.nadmix = D; dt.npair = npair;
    dt.npath = topo.npath; dt.base_leaf = topo.base_leaf;
    dt.n_pe = static_cast<int>(topo.pe_edge.size());
    dt.n_pae = static_cast<int>(topo.pae_path.size());
    dt.constrained = topo.constrained ? 1 : 0;
    dt.fudge = topo.fudge;
    dt.pwts0 = dPwts0.data();
    dt.pe_edge = dPeEdge.data(); dt.pe_leaf = dPeLeaf.data(); dt.pe_path = dPePath.data();
    dt.pae_path = dPaePath.data(); dt.pae_admixedge = dPaeEdge.data();
    dt.cmb1 = dCmb1.data(); dt.cmb2 = dCmb2.data();

    const int ns = numstart > 0 ? numstart : 1;
    DeviceBuffer<double> dTheta(static_cast<std::size_t>(ns) * static_cast<std::size_t>(D > 0 ? D : 1));
    DeviceBuffer<double> dScore(static_cast<std::size_t>(ns));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    if (D == 0) {
        DeviceBuffer<double> dBl0(static_cast<std::size_t>(ne > 0 ? ne : 1));
        DeviceBuffer<double> dF30(static_cast<std::size_t>(npair > 0 ? npair : 1));
        DeviceBuffer<double> dSc0(1);
        launch_qpgraph_eval_at_theta(dt, /*d_theta=*/nullptr, dFobs.data(),
                                           dQinv.data(), dBl0.data(), dF30.data(),
                                           dSc0.data(), stream_.get());
        std::vector<double> bl(static_cast<std::size_t>(ne), 0.0);
        std::vector<double> fit(static_cast<std::size_t>(npair), 0.0);
        double sc = std::numeric_limits<double>::infinity();
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(bl.data(), dBl0.data(),
                                          static_cast<std::size_t>(ne) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(fit.data(), dF30.data(),
                                          static_cast<std::size_t>(npair) * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&sc, dSc0.data(), sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        const bool ok = std::isfinite(sc) && sc < 1e30;
        out.score = sc; out.edge_length = ok ? bl : std::vector<double>(static_cast<std::size_t>(ne), 0.0);
        out.f3_fit = ok ? fit : std::vector<double>(static_cast<std::size_t>(npair), 0.0);
        out.status = ok ? Status::Ok : Status::NonSpdCovariance;
        return out;
    }

    launch_qpgraph_fleet(dt, ns, maxit, tol, dFobs.data(), dQinv.data(),
                               dTheta.data(), dScore.data(), stream_.get());

    std::vector<double> h_theta(static_cast<std::size_t>(ns) * static_cast<std::size_t>(D));
    std::vector<double> h_score(static_cast<std::size_t>(ns));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_theta.data(), dTheta.data(), h_theta.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_score.data(), dScore.data(), h_score.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    double best = std::numeric_limits<double>::infinity();
    double smin = best, smax = -best;
    int best_i = 0;
    std::vector<double> thmin(static_cast<std::size_t>(D), 1.0), thmax(static_cast<std::size_t>(D), 0.0);
    for (int i = 0; i < ns; ++i) {
        const double s = h_score[static_cast<std::size_t>(i)];
        if (std::isfinite(s)) { if (s < smin) smin = s; if (s > smax) smax = s; }
        for (int d = 0; d < D; ++d) {
            const double v = h_theta[static_cast<std::size_t>(i) * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)];
            if (v < thmin[static_cast<std::size_t>(d)]) thmin[static_cast<std::size_t>(d)] = v;
            if (v > thmax[static_cast<std::size_t>(d)]) thmax[static_cast<std::size_t>(d)] = v;
        }
        if (s < best) { best = s; best_i = i; }
    }
    out.score = best;
    out.restart_spread = (std::isfinite(smax) && std::isfinite(smin)) ? (smax - smin) : 0.0;
    out.theta.assign(static_cast<std::size_t>(D), 0.0);
    for (int d = 0; d < D; ++d)
        out.theta[static_cast<std::size_t>(d)] = h_theta[static_cast<std::size_t>(best_i) * static_cast<std::size_t>(D) + static_cast<std::size_t>(d)];
    out.theta_lo = thmin; out.theta_hi = thmax;

    DeviceBuffer<double> dThetaBest(static_cast<std::size_t>(D));
    DeviceBuffer<double> dBl(static_cast<std::size_t>(ne > 0 ? ne : 1));
    DeviceBuffer<double> dF3(static_cast<std::size_t>(npair > 0 ? npair : 1));
    DeviceBuffer<double> dScFinal(1);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dThetaBest.data(), out.theta.data(),
                                      static_cast<std::size_t>(D) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    launch_qpgraph_eval_at_theta(dt, dThetaBest.data(), dFobs.data(), dQinv.data(),
                                       dBl.data(), dF3.data(), dScFinal.data(), stream_.get());
    std::vector<double> bl(static_cast<std::size_t>(ne), 0.0);
    std::vector<double> fit(static_cast<std::size_t>(npair), 0.0);
    double sc_final = std::numeric_limits<double>::infinity();
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(bl.data(), dBl.data(),
                                      static_cast<std::size_t>(ne) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(fit.data(), dF3.data(),
                                      static_cast<std::size_t>(npair) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(&sc_final, dScFinal.data(), sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (std::isfinite(sc_final) && sc_final < 1e30) { out.edge_length = bl; out.f3_fit = fit; out.status = Status::Ok; }
    else { out.edge_length.assign(static_cast<std::size_t>(ne), 0.0); out.f3_fit.assign(static_cast<std::size_t>(npair), 0.0); out.status = Status::NonSpdCovariance; }
    return out;
}

// Screen many topologies at once: best score per candidate — reference §8
QpGraphFleetBatch CudaBackend::qpgraph_fit_fleet_batch(
    const std::vector<QpGraphTopoArena>& topos, std::span<const double> f_obs,
    std::span<const double> qinv, int numstart, int maxit, double tol,
    const Precision& precision) {
    (void)precision;
    guard_device();
    QpGraphFleetBatch out;
    const int G = static_cast<int>(topos.size());
    if (G == 0) { out.status = Status::Ok; return out; }
    const int ns = numstart > 0 ? numstart : 1;
    const int npair = topos.front().npair;

    DeviceBuffer<double> dFobs(static_cast<std::size_t>(npair));
    DeviceBuffer<double> dQinv(static_cast<std::size_t>(npair) * static_cast<std::size_t>(npair));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dFobs.data(), f_obs.data(), f_obs.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQinv.data(), qinv.data(), qinv.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));

    std::vector<double> h_pwts0;
    std::vector<int> h_pe_edge, h_pe_leaf, h_pe_path, h_pae_path, h_pae_edge, h_cmb1, h_cmb2;
    std::vector<QpGraphDeviceTopoView> h_views(static_cast<std::size_t>(G));
    int max_npop = 0, max_ne = 0, max_npair = 0, max_npath = 0;
    for (int g = 0; g < G; ++g) {
        const QpGraphTopoArena& t = topos[static_cast<std::size_t>(g)];
        if (t.nadmix > kMaxThetaDev)
            throw std::runtime_error(
                "steppe::device::CudaBackend::qpgraph_fit_fleet_batch: candidate topology "
                + std::to_string(g) + " has nadmix=" + std::to_string(t.nadmix) +
                " exceeding the per-thread theta-stack cap kMaxThetaDev=" +
                std::to_string(kMaxThetaDev) + ".");
        QpGraphDeviceTopoView v{};
        v.npop = t.npop; v.nedge_norm = t.nedge_norm; v.nadmix = t.nadmix;
        v.npair = t.npair; v.npath = t.npath; v.base_leaf = t.base_leaf;
        v.n_pe = static_cast<int>(t.pe_edge.size());
        v.n_pae = static_cast<int>(t.pae_path.size());
        v.constrained = t.constrained ? 1 : 0;
        v.fudge = t.fudge;
        v.off_pwts0 = static_cast<long>(h_pwts0.size());
        v.off_pe = static_cast<long>(h_pe_edge.size());
        v.off_pae = static_cast<long>(h_pae_path.size());
        v.off_cmb = static_cast<long>(h_cmb1.size());
        h_pwts0.insert(h_pwts0.end(), t.pwts0.begin(), t.pwts0.end());
        h_pe_edge.insert(h_pe_edge.end(), t.pe_edge.begin(), t.pe_edge.end());
        h_pe_leaf.insert(h_pe_leaf.end(), t.pe_leaf.begin(), t.pe_leaf.end());
        h_pe_path.insert(h_pe_path.end(), t.pe_path.begin(), t.pe_path.end());
        h_pae_path.insert(h_pae_path.end(), t.pae_path.begin(), t.pae_path.end());
        h_pae_edge.insert(h_pae_edge.end(), t.pae_admixedge.begin(), t.pae_admixedge.end());
        h_cmb1.insert(h_cmb1.end(), t.cmb1.begin(), t.cmb1.end());
        h_cmb2.insert(h_cmb2.end(), t.cmb2.begin(), t.cmb2.end());
        h_views[static_cast<std::size_t>(g)] = v;
        max_npop = std::max(max_npop, t.npop);
        max_ne = std::max(max_ne, t.nedge_norm);
        max_npair = std::max(max_npair, t.npair);
        max_npath = std::max(max_npath, t.npath);
    }
    const ScratchLayout Lmax = make_layout(max_npop, max_ne, max_npair, max_npath);

    auto up_d = [&](const std::vector<double>& h) {
        DeviceBuffer<double> d(h.size() ? h.size() : 1);
        if (!h.empty())
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.size() * sizeof(double),
                                              cudaMemcpyHostToDevice, stream_.get()));
        return d;
    };
    auto up_i = [&](const std::vector<int>& h) {
        DeviceBuffer<int> d(h.size() ? h.size() : 1);
        if (!h.empty())
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(d.data(), h.data(), h.size() * sizeof(int),
                                              cudaMemcpyHostToDevice, stream_.get()));
        return d;
    };
    DeviceBuffer<double> dPwts0 = up_d(h_pwts0);
    DeviceBuffer<int> dPeEdge = up_i(h_pe_edge), dPeLeaf = up_i(h_pe_leaf), dPePath = up_i(h_pe_path);
    DeviceBuffer<int> dPaePath = up_i(h_pae_path), dPaeEdge = up_i(h_pae_edge);
    DeviceBuffer<int> dCmb1 = up_i(h_cmb1), dCmb2 = up_i(h_cmb2);
    DeviceBuffer<QpGraphDeviceTopoView> dViews(static_cast<std::size_t>(G));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dViews.data(), h_views.data(),
                                      h_views.size() * sizeof(QpGraphDeviceTopoView),
                                      cudaMemcpyHostToDevice, stream_.get()));

    const std::size_t threads = static_cast<std::size_t>(G) * static_cast<std::size_t>(ns);
    DeviceBuffer<double> dGdbl(threads * static_cast<std::size_t>(Lmax.dbl_total));
    DeviceBuffer<int> dGint(threads * static_cast<std::size_t>(Lmax.int_total > 0 ? Lmax.int_total : 1));
    DeviceBuffer<double> dScore(threads);
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    launch_qpgraph_fleet_batch(
        dViews.data(), G, ns, maxit, tol, Lmax.dbl_total, Lmax.int_total, dPwts0.data(),
        dPeEdge.data(), dPeLeaf.data(), dPePath.data(), dPaePath.data(), dPaeEdge.data(),
        dCmb1.data(), dCmb2.data(), dFobs.data(), dQinv.data(), dGdbl.data(), dGint.data(),
        dScore.data(), stream_.get());

    std::vector<double> h_score(threads);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_score.data(), dScore.data(), threads * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    out.best_score.assign(static_cast<std::size_t>(G), std::numeric_limits<double>::infinity());
    out.restart_spread.assign(static_cast<std::size_t>(G), 0.0);
    for (int g = 0; g < G; ++g) {
        double smin = std::numeric_limits<double>::infinity(), smax = -smin;
        for (int r = 0; r < ns; ++r) {
            const double s = h_score[static_cast<std::size_t>(g) * static_cast<std::size_t>(ns) + static_cast<std::size_t>(r)];
            if (std::isfinite(s)) { if (s < smin) smin = s; if (s > smax) smax = s; }
        }
        out.best_score[static_cast<std::size_t>(g)] = smin;
        out.restart_spread[static_cast<std::size_t>(g)] =
            (std::isfinite(smax) && std::isfinite(smin)) ? (smax - smin) : 0.0;
    }
    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
