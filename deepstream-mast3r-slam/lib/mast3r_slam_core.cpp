/*
 * mast3r_slam_core.cpp — C++/libtorch port of the MASt3R-SLAM front-end.
 *
 * Data flow (per frame):
 *   encoder tensors (feat,pos)  ── from gst-nvinfer, wrapped as CUDA tensors
 *     -> decoder + DPT heads     ── TensorRT engine (run for the i<->j pair)
 *     -> matching                ── reuses iter_proj_cuda / refine_matches_cuda
 *     -> Sim3 pose tracking      ── Gauss-Newton (ray mode), port of tracker.py
 *     -> keyframe management + pose out
 *   on finish(): TUM trajectory + colored .ply (lib/io)
 *
 * The custom CUDA kernels (gn_kernels.cu / matching_kernels.cu) are compiled
 * into this library by CMake and reused unchanged; we forward-declare the
 * `*_cuda` entry points (same signatures as backend/include/gn.h) to avoid
 * pulling the pybind module in.
 *
 * STATUS: reference implementation. Numerics (Sim3 retraction, decoder I/O
 * names, descriptor layout) must be validated on the target GPU; the global
 * factor-graph backend and ASMK loop-closure are marked as integration points.
 */
#include "mast3r_slam_core.h"

#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "io.h"
#include "sim3.h"
#include "trt_engine.h"

// ---- reused CUDA kernels (defined in mast3r_slam/backend/src/*.cu) ----------
std::vector<torch::Tensor> iter_proj_cuda(torch::Tensor rays_img_with_grad,
                                          torch::Tensor pts_3d_norm,
                                          torch::Tensor p_init, const int max_iter,
                                          const float lambda_init,
                                          const float cost_thresh);
std::vector<torch::Tensor> refine_matches_cuda(torch::Tensor D11, torch::Tensor D21,
                                               torch::Tensor p1, const int radius,
                                               const int dilation_max);
std::vector<torch::Tensor> gauss_newton_rays_cuda(
    torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs, torch::Tensor ii,
    torch::Tensor jj, torch::Tensor idx_ii2jj, torch::Tensor valid_match,
    torch::Tensor Q, const float sigma_ray, const float sigma_dist,
    const float C_thresh, const float Q_thresh, const int max_iter,
    const float delta_thresh);

namespace mast3r_slam {

namespace F = torch::nn::functional;

// Tracking / matching parameters (defaults from config/base.yaml).
struct Params {
  // matching
  int m_max_iter = 10;
  float m_lambda_init = 1e-8f;
  float m_convergence_thresh = 1e-6f;
  float m_dist_thresh = 1e-1f;
  int m_radius = 3;
  int m_dilation_max = 5;
  // tracking
  float t_C_conf = 0.0f;
  float t_Q_conf = 1.5f;
  int t_max_iters = 50;
  float t_sigma_ray = 0.003f;
  float t_sigma_dist = 1e1f;
  float t_huber = 1.345f;
  float t_rel_error = 1e-3f;
  float t_delta_norm = 1e-3f;
  float t_min_match_frac = 0.05f;
  float t_match_frac_thresh = 0.333f;
};

struct Keyframe {
  uint64_t frame_id = 0;
  double timestamp = 0.0;
  Sim3 T_WC;
  torch::Tensor X_canon;  // (HW,3) cuda
  torch::Tensor C;        // (HW,1) cuda
  torch::Tensor feat;     // (1,N,1024)
  torch::Tensor pos;      // (1,N,2) long
  torch::Tensor gdesc;    // (1024) L2-normalized global descriptor (loop closure)
  int N = 0;
  int H = 0, W = 0;
};

// ---------------------------------------------------------------- Impl
struct Mast3rSlamCore::Impl {
  CoreConfig cfg;
  Params p;
  torch::Device device{torch::kCUDA, 0};
  TrtEngine decoder;
  bool decoder_ok = false;

  int mode = 0;  // 0=INIT, 1=TRACKING, 2=RELOC
  uint64_t frame_count = 0;
  std::vector<Keyframe> keyframes;
  std::vector<TrajSample> traj;  // appended per keyframe at finish
  torch::Tensor idx_f2k;         // previous frame->kf match init

  // --- stereo metric-scale state (DESIGN-STEREO.md) ---
  double scale_ema = 1.0;  // baseline_true / baseline_est, EMA-smoothed
  int scale_updates = 0;

  // --- loop-closure factor graph: two-way edges appended row-by-row ---
  std::vector<int64_t> fg_ii, fg_jj;
  std::vector<torch::Tensor> fg_idx;    // per edge: (HW) long
  std::vector<torch::Tensor> fg_valid;  // per edge: (HW,1) bool
  std::vector<torch::Tensor> fg_Q;      // per edge: (HW,1) float
  // backend params (mirroring config/base.yaml local_opt)
  float bo_sigma_ray = 0.003f, bo_sigma_dist = 1e1f;
  float bo_C_conf = 0.0f, bo_Q_conf = 1.5f;
  float bo_min_match_frac = 0.1f, bo_delta_norm = 1e-8f;
  int bo_max_iters = 10;

  explicit Impl(const CoreConfig &c) : cfg(c), device(torch::kCUDA, c.gpu_id) {}

  torch::TensorOptions f32() {
    return torch::TensorOptions().dtype(torch::kFloat32).device(device);
  }

  // ----- decoder: feat_i,pos_i,feat_j,pos_j -> (Xi,Ci,Di,Qi, Xj,Cj,Dj,Qj) ----
  // Each output is (1,H,W,*). Engine I/O names documented in tools/README.
  struct DecOut {
    torch::Tensor Xi, Ci, Di, Qi, Xj, Cj, Dj, Qj;
  };
  DecOut runDecoder(torch::Tensor feat_i, torch::Tensor pos_i, torch::Tensor feat_j,
                    torch::Tensor pos_j, int H, int W) {
    TORCH_CHECK(decoder_ok, "decoder engine not loaded");
    auto stream = at::cuda::getCurrentCUDAStream(cfg.gpu_id);
    feat_i = feat_i.contiguous();
    feat_j = feat_j.contiguous();
    auto pos_i_i = pos_i.to(torch::kInt32).contiguous();
    auto pos_j_i = pos_j.to(torch::kInt32).contiguous();

    decoder.setInputShape("feat1", {feat_i.size(0), feat_i.size(1), feat_i.size(2)});
    decoder.setInputShape("feat2", {feat_j.size(0), feat_j.size(1), feat_j.size(2)});
    decoder.setInputShape("pos1", {pos_i_i.size(0), pos_i_i.size(1), pos_i_i.size(2)});
    decoder.setInputShape("pos2", {pos_j_i.size(0), pos_j_i.size(1), pos_j_i.size(2)});
    decoder.setTensorAddress("feat1", feat_i.data_ptr());
    decoder.setTensorAddress("feat2", feat_j.data_ptr());
    decoder.setTensorAddress("pos1", pos_i_i.data_ptr());
    decoder.setTensorAddress("pos2", pos_j_i.data_ptr());

    auto mkout = [&](const char *name) {
      auto sh = decoder.tensorShape(name);
      std::vector<int64_t> s(sh.begin(), sh.end());
      auto t = torch::empty(s, f32());
      decoder.setTensorAddress(name, t.data_ptr());
      return t;
    };
    DecOut o;
    o.Xi = mkout("pts3d_1"); o.Ci = mkout("conf_1");
    o.Di = mkout("desc_1");  o.Qi = mkout("desc_conf_1");
    o.Xj = mkout("pts3d_2"); o.Cj = mkout("conf_2");
    o.Dj = mkout("desc_2");  o.Qj = mkout("desc_conf_2");
    TORCH_CHECK(decoder.infer(stream.stream()), "decoder inference failed");
    return o;
  }

  // ----- image gradient (port of mast3r_slam.image.img_gradient) -------------
  std::pair<torch::Tensor, torch::Tensor> imgGradient(torch::Tensor img) {
    int c = img.size(1);
    auto opts = img.options();
    auto gx_k = (1.0 / 32.0) *
                torch::tensor({-3.f, 0.f, 3.f, -10.f, 0.f, 10.f, -3.f, 0.f, 3.f}, opts)
                    .view({1, 1, 3, 3}).repeat({c, 1, 1, 1});
    auto gy_k = (1.0 / 32.0) *
                torch::tensor({-3.f, -10.f, -3.f, 0.f, 0.f, 0.f, 3.f, 10.f, 3.f}, opts)
                    .view({1, 1, 3, 3}).repeat({c, 1, 1, 1});
    auto padded = F::pad(img, F::PadFuncOptions({1, 1, 1, 1}).mode(torch::kReflect));
    auto gx = F::conv2d(padded, gx_k, F::Conv2dFuncOptions().groups(c));
    auto gy = F::conv2d(padded, gy_k, F::Conv2dFuncOptions().groups(c));
    return {gx, gy};
  }

  // ----- matching (port of matching.match_iterative_proj) --------------------
  std::pair<torch::Tensor, torch::Tensor> match(torch::Tensor X11, torch::Tensor X21,
                                                torch::Tensor D11, torch::Tensor D21,
                                                torch::Tensor idx_init) {
    int64_t b = X11.size(0), h = X11.size(1), w = X11.size(2);
    auto rays_img = F::normalize(X11, F::NormalizeFuncOptions().dim(-1)).permute({0, 3, 1, 2});
    auto grads = imgGradient(rays_img);
    auto rays_grad = torch::cat({rays_img, grads.first, grads.second}, 1)
                         .permute({0, 2, 3, 1}).contiguous();
    auto X21_vec = X21.view({b, -1, 3});
    auto pts3d_norm = F::normalize(X21_vec, F::NormalizeFuncOptions().dim(-1)).contiguous();

    if (!idx_init.defined()) {
      idx_init = torch::arange(h * w, torch::TensorOptions().dtype(torch::kLong).device(device))
                     .unsqueeze(0).repeat({b, 1});
    }
    // lin_to_pixel
    auto u = (idx_init % w).unsqueeze(-1);
    auto v = (idx_init / w).unsqueeze(-1);
    auto p_init = torch::cat({u, v}, -1).to(torch::kFloat32).contiguous();

    auto pr = iter_proj_cuda(rays_grad, pts3d_norm, p_init, p.m_max_iter,
                             p.m_lambda_init, p.m_convergence_thresh);
    auto p1 = pr[0].to(torch::kLong);
    auto valid = pr[1];

    // occlusion check by 3D distance
    auto bidx = torch::arange(b, torch::TensorOptions().dtype(torch::kLong).device(device))
                    .unsqueeze(1).repeat({1, h * w});
    auto px = p1.index({"...", 0}).view({b, h * w});
    auto py = p1.index({"...", 1}).view({b, h * w});
    auto gathered = X11.index({bidx, py, px}).view({b, h, w, 3});
    auto dists = torch::linalg_norm(gathered - X21, c10::nullopt, {-1}, false, c10::nullopt);
    auto valid_d = (dists < p.m_dist_thresh).view({b, -1});
    valid = valid & valid_d;

    if (p.m_radius > 0) {
      auto rr = refine_matches_cuda(D11.to(torch::kHalf), D21.view({b, h * w, -1}).to(torch::kHalf),
                                    p1, p.m_radius, p.m_dilation_max);
      p1 = rr[0];
    }
    // pixel_to_lin
    auto idx = p1.index({"...", 0}) + w * p1.index({"...", 1});
    return {idx, valid.unsqueeze(-1)};
  }

  // ----- weighted_pointmap filtering update (frame.py:update_pointmap) -------
  void updatePointmap(torch::Tensor &Xc, torch::Tensor &Cc, int &N,
                      torch::Tensor Xnew, torch::Tensor Cnew) {
    if (N == 0) {
      Xc = Xnew.clone();
      Cc = Cnew.clone();
      N = 1;
      return;
    }
    Xc = ((Cc * Xc) + (Cnew * Xnew)) / (Cc + Cnew);
    Cc = Cc + Cnew;
    N += 1;
  }

  // ----- point_to_ray_dist with optional Jacobian ----------------------------
  void pointToRayDist(torch::Tensor X, torch::Tensor &rd, torch::Tensor *drd_dX) {
    auto d = torch::linalg_norm(X, c10::nullopt, {-1}, true, c10::nullopt);  // (.,1)
    auto d_inv = 1.0 / d;
    auto r = d_inv * X;
    rd = torch::cat({r, d}, -1);
    if (!drd_dX) return;
    auto d_inv2 = d_inv * d_inv;
    auto I = torch::eye(3, X.options()).expand({X.size(0), 3, 3});
    auto XXt = X.unsqueeze(-1).matmul(X.unsqueeze(-2));  // (.,3,3)
    auto dr_dX = d_inv.unsqueeze(-1) * (I - d_inv2.unsqueeze(-1) * XXt);
    auto dd_dX = r.unsqueeze(-2);  // (.,1,3)
    *drd_dX = torch::cat({dr_dX, dd_dX}, -2);  // (.,4,3)
  }

  static torch::Tensor huberWeight(torch::Tensor r, float k) {
    auto r_abs = torch::abs(r);
    return torch::where(r_abs < k, torch::ones_like(r), k / r_abs);
  }

  torch::Tensor RtoTorch(const Sim3 &T) {
    auto R = torch::empty({3, 3}, f32());
    auto Rc = R.to(torch::kCPU);
    auto acc = Rc.accessor<float, 2>();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) acc[i][j] = (float)T.R()(i, j);
    return Rc.to(device);
  }
  torch::Tensor tToTorch(const Sim3 &T) {
    auto t = torch::tensor({(float)T.t().x(), (float)T.t().y(), (float)T.t().z()}, f32());
    return t;
  }

  // pW = s*(X @ R^T) + t ; returns pW and J=(.,3,7) = [I, -skew(pW), pW]
  torch::Tensor actSim3(const Sim3 &T, torch::Tensor X, torch::Tensor *J) {
    auto R = RtoTorch(T);
    auto t = tToTorch(T);
    auto pW = (X.matmul(R.t())) * (float)T.s() + t;  // (HW,3)
    if (J) {
      int64_t n = X.size(0);
      auto I = torch::eye(3, f32()).expand({n, 3, 3});
      auto px = pW.index({"...", 0}), py = pW.index({"...", 1}), pz = pW.index({"...", 2});
      auto zero = torch::zeros_like(px);
      // -skew(pW)
      auto sk = torch::stack({zero, pz, -py, -pz, zero, px, py, -px, zero}, -1).view({n, 3, 3});
      auto col_s = pW.view({n, 3, 1});
      *J = torch::cat({I, sk, col_s}, -1);  // (HW,3,7)
    }
    return pW;
  }

  // GN solve: returns tau(7) and cost. r:(N,4) J:(N,4,7) sqrt_info:(N,4)
  std::pair<torch::Tensor, double> solve(torch::Tensor sqrt_info, torch::Tensor r,
                                         torch::Tensor J) {
    auto whitened = sqrt_info * r;
    auto robust = sqrt_info * torch::sqrt(huberWeight(whitened, p.t_huber));
    auto A = (robust.unsqueeze(-1) * J).view({-1, 7});
    auto b = (robust * r).view({-1, 1});
    auto H = A.t().matmul(A);
    auto g = -A.t().matmul(b);
    double cost = 0.5 * (b.t().matmul(b)).item<double>();
    auto L = torch::linalg_cholesky(H);
    auto tau = torch::cholesky_solve(g, L).view({7});
    return {tau, cost};
  }

  // opt_pose_ray_dist_sim3 (single relative pose) -> updates T_WCf, T_CkCf
  void optPoseRay(torch::Tensor Xf, torch::Tensor Xk, Sim3 &T_WCf, const Sim3 &T_WCk,
                  torch::Tensor Qk, torch::Tensor valid, Sim3 &T_CkCf_out) {
    auto sqrt_Q = torch::sqrt(Qk);
    auto sir = (1.0f / p.t_sigma_ray) * valid * sqrt_Q;
    auto sid = (1.0f / p.t_sigma_dist) * valid * sqrt_Q;
    auto sqrt_info = torch::cat({sir.repeat({1, 3}), sid}, 1);  // (HW,4)

    Sim3 T_CkCf = T_WCk.inverse() * T_WCf;
    torch::Tensor rd_k;
    pointToRayDist(Xk, rd_k, nullptr);

    double old_cost = std::numeric_limits<double>::infinity();
    for (int step = 0; step < p.t_max_iters; ++step) {
      torch::Tensor J_act;
      auto Xf_Ck = actSim3(T_CkCf, Xf, &J_act);  // (HW,3),(HW,3,7)
      torch::Tensor rd_f, drd_dX;
      pointToRayDist(Xf_Ck, rd_f, &drd_dX);  // (HW,4),(HW,4,3)
      auto r = rd_k - rd_f;
      auto J = -drd_dX.matmul(J_act);  // (HW,4,7)
      auto [tau, cost] = solve(sqrt_info, r, J);

      Eigen::Matrix<double, 7, 1> tau_e;
      auto tau_c = tau.to(torch::kCPU).to(torch::kDouble);
      auto acc = tau_c.accessor<double, 1>();
      for (int i = 0; i < 7; ++i) tau_e(i) = acc[i];
      T_CkCf = T_CkCf.retr(tau_e);

      double rel = std::fabs((old_cost - cost) / old_cost);
      double dn = std::sqrt(tau.pow(2).sum().item<double>());
      if (rel < p.t_rel_error || dn < p.t_delta_norm) break;
      old_cost = cost;
    }
    T_WCf = T_WCk * T_CkCf;
    T_CkCf_out = T_CkCf;
  }

  // ------------------------------------------------ stereo scale (hybrid C)
  // Weighted Umeyama similarity alignment X -> Y (both (N,3) cuda float,
  // weights (N,1)). Only the translation norm is needed for the baseline, but
  // the full closed form is computed for robustness. Runs on CPU in double.
  bool umeyamaSim(torch::Tensor Xt, torch::Tensor Yt, torch::Tensor wt,
                  double &scale_out, Eigen::Vector3d &t_out) {
    auto X = Xt.to(torch::kCPU).to(torch::kDouble).contiguous();
    auto Y = Yt.to(torch::kCPU).to(torch::kDouble).contiguous();
    auto w = wt.to(torch::kCPU).to(torch::kDouble).contiguous();
    int64_t n = X.size(0);
    if (n < 100) return false;
    auto xa = X.accessor<double, 2>();
    auto ya = Y.accessor<double, 2>();
    auto wa = w.accessor<double, 2>();

    double wsum = 0.0;
    Eigen::Vector3d mx = Eigen::Vector3d::Zero(), my = Eigen::Vector3d::Zero();
    for (int64_t i = 0; i < n; ++i) {
      double wi = wa[i][0];
      wsum += wi;
      mx += wi * Eigen::Vector3d(xa[i][0], xa[i][1], xa[i][2]);
      my += wi * Eigen::Vector3d(ya[i][0], ya[i][1], ya[i][2]);
    }
    if (wsum <= 1e-9) return false;
    mx /= wsum;
    my /= wsum;

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    double var_x = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      double wi = wa[i][0] / wsum;
      Eigen::Vector3d xc = Eigen::Vector3d(xa[i][0], xa[i][1], xa[i][2]) - mx;
      Eigen::Vector3d yc = Eigen::Vector3d(ya[i][0], ya[i][1], ya[i][2]) - my;
      cov += wi * (yc * xc.transpose());
      var_x += wi * xc.squaredNorm();
    }
    if (var_x <= 1e-12) return false;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d D = Eigen::Vector3d::Ones();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0) D(2) = -1;
    Eigen::Matrix3d R =
        svd.matrixU() * D.asDiagonal() * svd.matrixV().transpose();
    double c = svd.singularValues().dot(D) / var_x;
    Eigen::Vector3d t = my - c * (R * mx);

    scale_out = c;
    t_out = t;
    return std::isfinite(c) && t.allFinite();
  }

  // Estimate the metric scale s = baseline_m / ||t_LR|| from a stereo pair.
  // Two decoder passes give the right view's points in both frames:
  //   decoder(L,R): Xj = X_R in LEFT frame;  decoder(R,L): Xi = X_R in RIGHT frame.
  // Also returns the L,R decode (o_lr) so the caller can reuse the stereo
  // pointmap of the left view.
  bool estimateStereoScale(torch::Tensor featL, torch::Tensor posL,
                           torch::Tensor featR, torch::Tensor posR, int H, int W,
                           DecOut &o_lr_out, double &s_out) {
    auto o_lr = runDecoder(featL, posL, featR, posR, H, W);
    auto o_rl = runDecoder(featR, posR, featL, posL, H, W);

    auto Xrr = o_rl.Xi.view({-1, 3});  // right pts in right frame
    auto Xrl = o_lr.Xj.view({-1, 3});  // right pts in left frame
    auto wgt = torch::sqrt(o_rl.Ci.view({-1, 1}) * o_lr.Cj.view({-1, 1}));

    // keep confident points, subsample to <= 8192 for the CPU solve
    auto mask = (wgt > 1.5f).view({-1});
    auto sel = mask.nonzero().view({-1});
    if (sel.numel() < 500) return false;
    int64_t stride = std::max<int64_t>(1, sel.numel() / 8192);
    sel = sel.index({torch::indexing::Slice(0, torch::indexing::None, stride)});

    double c = 1.0;
    Eigen::Vector3d t;
    if (!umeyamaSim(Xrr.index({sel}), Xrl.index({sel}), wgt.index({sel}), c, t))
      return false;
    double b_est = t.norm();
    if (b_est < 1e-4) return false;

    s_out = cfg.baseline_m / b_est;
    o_lr_out = o_lr;
    return std::isfinite(s_out) && s_out > 0.0;
  }

  void updateScaleEMA(double s_new) {
    if (scale_updates == 0) {
      scale_ema = s_new;
    } else {
      double r = s_new / scale_ema;
      if (r < 0.5 || r > 2.0) {
        std::cerr << "[nvdsmast3rslam] stereo scale outlier rejected: " << s_new
                  << " (ema " << scale_ema << ")\n";
        return;
      }
      scale_ema = 0.7 * scale_ema + 0.3 * s_new;
    }
    scale_updates++;
  }

  // --------------------------------------------- loop closure + backend GN
  torch::Tensor globalDesc(torch::Tensor feat) {
    auto d = feat.mean(1).view({-1});  // (1024)
    return (d / d.norm()).contiguous();
  }

  // Symmetric decode+match of keyframes (i, j); appends both edge directions.
  // Consecutive edges are always kept; loop candidates are gated by the
  // min-match-fraction (geometric verification). Port of FactorGraph.add_factors.
  bool addEdge(int i, int j, bool consecutive) {
    auto &ki = keyframes[i];
    auto &kj = keyframes[j];
    auto o_ij = runDecoder(ki.feat, ki.pos, kj.feat, kj.pos, ki.H, ki.W);
    auto o_ji = runDecoder(kj.feat, kj.pos, ki.feat, ki.pos, kj.H, kj.W);

    auto m_ij = match(o_ij.Xi, o_ij.Xj, o_ij.Di, o_ij.Dj, torch::Tensor());
    auto m_ji = match(o_ji.Xi, o_ji.Xj, o_ji.Di, o_ji.Dj, torch::Tensor());
    auto idx_i2j = m_ij.first.view({-1});           // (HW)
    auto valid_j = m_ij.second.view({-1, 1});       // (HW,1)
    auto idx_j2i = m_ji.first.view({-1});
    auto valid_i = m_ji.second.view({-1, 1});

    auto Qii = o_ij.Qi.view({-1, 1});
    auto Qji = o_ij.Qj.view({-1, 1});
    auto Qjj = o_ji.Qi.view({-1, 1});
    auto Qij = o_ji.Qj.view({-1, 1});
    auto Qj = torch::sqrt(Qii.index({idx_i2j}) * Qji);  // conf of edge i->j
    auto Qi = torch::sqrt(Qjj.index({idx_j2i}) * Qij);  // conf of edge j->i

    auto vj = valid_j & (Qj > bo_Q_conf);
    auto vi = valid_i & (Qi > bo_Q_conf);
    double frac_j = vj.sum().item<double>() / (double)vj.numel();
    double frac_i = vi.sum().item<double>() / (double)vi.numel();
    if (!consecutive && std::min(frac_j, frac_i) < bo_min_match_frac) {
      return false;  // geometric verification failed -> reject candidate
    }

    fg_ii.push_back(i); fg_jj.push_back(j);
    fg_idx.push_back(idx_i2j.contiguous());
    fg_valid.push_back(valid_j.contiguous());
    fg_Q.push_back(Qj.contiguous());
    fg_ii.push_back(j); fg_jj.push_back(i);
    fg_idx.push_back(idx_j2i.contiguous());
    fg_valid.push_back(valid_i.contiguous());
    fg_Q.push_back(Qi.contiguous());
    return true;
  }

  // Global Gauss-Newton over all keyframes (reused CUDA kernel). The first
  // keyframe is pinned; the kernel updates the pose tensor in place.
  void solveBackend() {
    int n = (int)keyframes.size();
    if (n < 2 || fg_ii.empty()) return;

    std::vector<torch::Tensor> xs, cs;
    xs.reserve(n); cs.reserve(n);
    for (auto &kf : keyframes) {
      xs.push_back(kf.X_canon);
      cs.push_back(kf.C / std::max(kf.N, 1));
    }
    auto Xs = torch::stack(xs).contiguous();  // (n,HW,3)
    auto Cs = torch::stack(cs).contiguous();  // (n,HW,1)

    auto Twc_cpu = torch::empty({n, 8}, torch::kFloat32);
    {
      auto acc = Twc_cpu.accessor<float, 2>();
      double d[8];
      for (int k = 0; k < n; ++k) {
        keyframes[k].T_WC.toData(d);
        for (int c = 0; c < 8; ++c) acc[k][c] = (float)d[c];
      }
    }
    auto Twc = Twc_cpu.to(device).contiguous();

    auto lopts = torch::TensorOptions().dtype(torch::kLong);
    auto ii = torch::from_blob(fg_ii.data(), {(int64_t)fg_ii.size()}, lopts)
                  .clone().to(device);
    auto jj = torch::from_blob(fg_jj.data(), {(int64_t)fg_jj.size()}, lopts)
                  .clone().to(device);
    auto idx = torch::stack(fg_idx).contiguous();      // (E,HW)
    auto valid = torch::stack(fg_valid).contiguous();  // (E,HW,1)
    auto Q = torch::stack(fg_Q).contiguous();          // (E,HW,1)

    gauss_newton_rays_cuda(Twc, Xs, Cs, ii, jj, idx, valid, Q, bo_sigma_ray,
                           bo_sigma_dist, bo_C_conf, bo_Q_conf, bo_max_iters,
                           bo_delta_norm);

    // write optimized poses back (keyframe 0 stays pinned)
    auto out = Twc.to(torch::kCPU).to(torch::kDouble).contiguous();
    auto oacc = out.accessor<double, 2>();
    for (int k = 1; k < n; ++k) {
      double d[8];
      for (int c = 0; c < 8; ++c) d[c] = oacc[k][c];
      keyframes[k].T_WC = Sim3::fromData(d);
    }
  }

  // Called after a keyframe is appended: descriptor, edges, global opt.
  void onNewKeyframe() {
    int idx = (int)keyframes.size() - 1;
    keyframes[idx].gdesc = globalDesc(keyframes[idx].feat);

    bool graph_changed = false;
    if (idx >= 1) graph_changed |= addEdge(idx - 1, idx, /*consecutive=*/true);

    if (cfg.loop_closure && idx > cfg.loop_min_gap) {
      int m = idx - cfg.loop_min_gap;  // candidates: keyframes [0, m)
      std::vector<torch::Tensor> ds;
      ds.reserve(m);
      for (int k = 0; k < m; ++k) ds.push_back(keyframes[k].gdesc);
      auto sims = torch::stack(ds).matmul(keyframes[idx].gdesc);  // (m)
      auto best = sims.argmax().item<int64_t>();
      double best_sim = sims[best].item<double>();
      if (best_sim > cfg.loop_sim_thresh) {
        bool ok = addEdge((int)best, idx, /*consecutive=*/false);
        graph_changed |= ok;
        std::cout << "[nvdsmast3rslam] loop candidate kf " << best << " <-> "
                  << idx << " sim=" << best_sim
                  << (ok ? " ACCEPTED" : " rejected (geometry)") << "\n";
      }
    }
    if (graph_changed) solveBackend();
  }

  // -------------------------------------------------------------- per frame
  PoseResult process(const FrameInput &in) { return processImpl(in, nullptr); }
  PoseResult processStereo(const FrameInput &left, const FrameInput &right) {
    return processImpl(left, &right);
  }

  PoseResult processImpl(const FrameInput &in, const FrameInput *right) {
    torch::NoGradGuard ng;
    int H = in.model_h, W = in.model_w;
    // wrap encoder tensors (device memory owned by nvinfer meta)
    auto feat = torch::from_blob((void *)in.feat_dev, {1, in.feat_n, in.feat_dim}, f32());
    auto pos = torch::from_blob((void *)in.pos_dev, {1, in.pos_n, 2},
                                torch::TensorOptions().dtype(torch::kInt32).device(device))
                   .to(torch::kLong);

    // stereo-hybrid: wrap the right frame's encoder tensors too (valid only
    // within this call, so all stereo work happens synchronously below)
    torch::Tensor featR, posR;
    if (right) {
      featR = torch::from_blob((void *)right->feat_dev,
                               {1, right->feat_n, right->feat_dim}, f32());
      posR = torch::from_blob((void *)right->pos_dev, {1, right->pos_n, 2},
                              torch::TensorOptions().dtype(torch::kInt32)
                                  .device(device))
                 .to(torch::kLong);
    }

    PoseResult res;
    res.frame_id = frame_count;
    res.timestamp = in.timestamp;

    if (mode == 0) {  // INIT
      torch::Tensor X, C;
      if (right) {
        // stereo init: decoder(L,R) pointmap is better conditioned than the
        // mono self-pair, and the pair fixes the metric scale from frame one
        DecOut o_lr;
        double s = 1.0;
        if (estimateStereoScale(feat, pos, featR, posR, H, W, o_lr, s)) {
          updateScaleEMA(s);
          X = (o_lr.Xi.view({-1, 3}) * (float)scale_ema).contiguous();
          C = o_lr.Ci.view({-1, 1}).contiguous();
        }
      }
      if (!X.defined()) {  // mono init (or stereo scale estimation failed)
        auto o = runDecoder(feat, pos, feat, pos, H, W);
        X = (o.Xi.view({-1, 3}) * (float)scale_ema).contiguous();
        C = o.Ci.view({-1, 1}).contiguous();
      }
      Keyframe kf;
      kf.frame_id = frame_count;
      kf.timestamp = in.timestamp;
      kf.T_WC = Sim3::Identity();
      kf.feat = feat.clone();
      kf.pos = pos.clone();
      kf.H = H; kf.W = W;
      updatePointmap(kf.X_canon, kf.C, kf.N, X, C);
      keyframes.push_back(std::move(kf));
      onNewKeyframe();
      mode = 1;
      fillPose(res, keyframes.back().T_WC, /*kf=*/true);
      frame_count++;
      return res;
    }

    // TRACKING
    Keyframe &kf = keyframes.back();
    // Asymmetric pair (frame, keyframe): decoder(feat_frame,pos_frame,
    // feat_kf,pos_kf). Output view i = frame canonical, view j = kf-in-frame.
    auto o = runDecoder(feat, pos, kf.feat, kf.pos, H, W);

    auto Xff = o.Xi.view({-1, 3}).contiguous();   // frame canonical pts (HW,3)
    auto Cff = o.Ci.view({-1, 1}).contiguous();
    auto Qff = o.Qi.view({-1, 1}).contiguous();
    auto Xkf = o.Xj.view({-1, 3}).contiguous();   // kf pts in frame prediction
    auto Ckf = o.Cj.view({-1, 1}).contiguous();
    auto Qkf = o.Qj.view({-1, 1}).contiguous();

    auto idx_pair = match(o.Xi, o.Xj, o.Di, o.Dj, idx_f2k);  // {idx(1,HW), valid(1,HW,1)}
    auto idx_f2k_new = idx_pair.first;
    auto valid_k = idx_pair.second;
    idx_f2k = idx_f2k_new.clone();
    auto idx0 = idx_f2k_new.view({-1});            // (HW)
    auto valid0 = valid_k.view({-1, 1});           // (HW,1)

    auto Qk = torch::sqrt(Qff.index({idx0}) * Qkf);  // (HW,1)

    // Metric correction (stereo-hybrid): bring the frame's decoder outputs into
    // the same metric units as the keyframe maps, so the relative Sim3 scale in
    // tracking stays ~1 and does not compound across keyframes. Matching above
    // ran on the raw outputs (both views share the raw scale), which keeps the
    // 3D distance threshold semantics identical to the reference.
    auto Xf_canon = (scale_updates > 0)
                        ? (Xff * (float)scale_ema).contiguous()
                        : Xff;  // frame.update_pointmap with N=0 sets directly
    auto Xkf_m = (scale_updates > 0)
                     ? (Xkf * (float)scale_ema).contiguous()
                     : Xkf;
    auto Cf = Cff;  // average conf with N=1 == C

    // Points/poses (ray mode, no calib)
    auto Xf = Xf_canon.index({idx0});  // (HW,3)
    auto Xk = kf.X_canon;              // (HW,3)
    auto Cf_sel = Cf.index({idx0});
    auto Ck = kf.C / std::max(kf.N, 1);

    auto valid_Cf = Cf_sel > p.t_C_conf;
    auto valid_Ck = Ck > p.t_C_conf;
    auto valid_Q = Qk > p.t_Q_conf;
    auto valid_opt = valid0 & valid_Cf & valid_Ck & valid_Q;

    double match_frac = valid_opt.sum().item<double>() / (double)valid_opt.numel();
    if (match_frac < p.t_min_match_frac) {
      // tracking lost for this frame
      res.mode = 2;  // RELOC
      fillPose(res, kf.T_WC, false);
      frame_count++;
      return res;
    }

    Sim3 T_WCf = kf.T_WC;  // prior = last pose
    Sim3 T_CkCf;
    optPoseRay(Xf, Xk, T_WCf, kf.T_WC, Qk, valid_opt.to(torch::kFloat32), T_CkCf);

    // update keyframe pointmap with aligned current observation (metric units)
    auto Xkk = actSim3(T_CkCf, Xkf_m, nullptr);
    updatePointmap(kf.X_canon, kf.C, kf.N, Xkk, Ckf);

    // keyframe selection
    auto valid_kf = valid0 & valid_Q;
    double match_frac_k = valid_kf.sum().item<double>() / (double)valid_kf.numel();
    auto uniq = std::get<0>(torch::_unique(idx0.index({valid0.view({-1})}))).numel();
    double unique_frac_f = (double)uniq / (double)valid_kf.numel();
    bool new_kf = std::min(match_frac_k, unique_frac_f) < p.t_match_frac_thresh;

    fillPose(res, T_WCf, new_kf);

    if (new_kf) {
      idx_f2k = torch::Tensor();  // reset

      // stereo-hybrid: refresh the metric scale on every keyframe and keep the
      // stereo pointmap as an extra observation for the new keyframe's map
      DecOut o_lr;
      bool have_stereo = false;
      if (right) {
        double s = 1.0;
        if (estimateStereoScale(feat, pos, featR, posR, H, W, o_lr, s)) {
          updateScaleEMA(s);
          have_stereo = true;
        }
      }

      Keyframe nkf;
      nkf.frame_id = frame_count;
      nkf.timestamp = in.timestamp;
      nkf.T_WC = T_WCf;
      nkf.feat = feat.clone();
      nkf.pos = pos.clone();
      nkf.H = H; nkf.W = W;
      int n0 = 0;
      // seed with the frame's canonical pointmap (already metric-corrected)...
      updatePointmap(nkf.X_canon, nkf.C, n0, Xf_canon, Cf);
      // ...and blend in the independent stereo depth observation
      if (have_stereo) {
        updatePointmap(nkf.X_canon, nkf.C, n0,
                       (o_lr.Xi.view({-1, 3}) * (float)scale_ema).contiguous(),
                       o_lr.Ci.view({-1, 1}).contiguous());
      }
      nkf.N = n0;
      keyframes.push_back(std::move(nkf));

      // loop closure + global Gauss-Newton over all keyframes
      onNewKeyframe();
      // the backend may have moved the newest keyframe -> report its pose
      fillPose(res, keyframes.back().T_WC, true);
    }
    frame_count++;
    return res;
  }

  void fillPose(PoseResult &res, const Sim3 &T, bool is_kf) {
    double d[8];
    T.toData(d);
    res.valid = true;
    res.t[0] = d[0]; res.t[1] = d[1]; res.t[2] = d[2];
    res.q[0] = d[3]; res.q[1] = d[4]; res.q[2] = d[5]; res.q[3] = d[6];
    res.scale = d[7];
    res.num_keyframes = (int)keyframes.size();
    res.is_keyframe = is_kf ? 1 : 0;
    if (res.mode == 0) res.mode = mode;
  }

  // -------------------------------------------------------------- finish
  void finish() {
    if (!cfg.save_results || keyframes.empty()) return;
    // trajectory
    traj.clear();
    for (auto &kf : keyframes) traj.push_back({kf.timestamp, kf.T_WC});
    std::string base = cfg.save_dir + "/" + cfg.sequence_name;
    saveTrajectoryTUM(base + ".txt", traj);

    // point cloud (no per-point color extraction yet -> neutral gray)
    std::vector<float> pts;
    std::vector<uint8_t> cols;
    for (auto &kf : keyframes) {
      auto X = kf.X_canon.to(torch::kCPU).contiguous();
      auto Cavg = (kf.C / std::max(kf.N, 1)).to(torch::kCPU).contiguous();
      auto acc = X.accessor<float, 2>();
      auto cacc = Cavg.accessor<float, 2>();
      for (int64_t i = 0; i < X.size(0); ++i) {
        if (cacc[i][0] <= (float)cfg.conf_threshold) continue;
        Eigen::Vector3d pc(acc[i][0], acc[i][1], acc[i][2]);
        Eigen::Vector3d pw = kf.T_WC.act(pc);
        pts.push_back((float)pw.x());
        pts.push_back((float)pw.y());
        pts.push_back((float)pw.z());
        cols.push_back(128); cols.push_back(128); cols.push_back(128);
      }
    }
    savePly(base + ".ply", pts, cols);
    std::cout << "[nvdsmast3rslam] saved " << base << ".{txt,ply}\n";
  }
};

// ---------------------------------------------------------------- facade
Mast3rSlamCore::Mast3rSlamCore(const CoreConfig &cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}
Mast3rSlamCore::~Mast3rSlamCore() = default;

bool Mast3rSlamCore::start() {
  if (!torch::cuda::is_available()) {
    std::cerr << "[nvdsmast3rslam] CUDA not available\n";
    return false;
  }
  if (impl_->cfg.decoder_engine.empty() ||
      !impl_->decoder.load(impl_->cfg.decoder_engine)) {
    std::cerr << "[nvdsmast3rslam] failed to load decoder engine: "
              << impl_->cfg.decoder_engine << "\n";
    return false;
  }
  impl_->decoder_ok = true;
  return true;
}

PoseResult Mast3rSlamCore::process(const FrameInput &in) { return impl_->process(in); }
PoseResult Mast3rSlamCore::processStereo(const FrameInput &left,
                                         const FrameInput &right) {
  return impl_->processStereo(left, right);
}

bool Mast3rSlamCore::copyLatestKeyframeCloud(int max_points,
                                             std::vector<float> &xyz,
                                             uint64_t &kf_id) {
  if (impl_->keyframes.empty()) return false;
  auto &kf = impl_->keyframes.back();
  kf_id = kf.frame_id;

  auto X = kf.X_canon.to(torch::kCPU).contiguous();
  auto Cavg = (kf.C / std::max(kf.N, 1)).to(torch::kCPU).contiguous();
  auto acc = X.accessor<float, 2>();
  auto cacc = Cavg.accessor<float, 2>();
  int64_t n = X.size(0);

  // count survivors first to derive the stride for the max_points cap
  int64_t valid = 0;
  for (int64_t i = 0; i < n; ++i)
    if (cacc[i][0] > (float)impl_->cfg.conf_threshold) valid++;
  if (valid == 0) return false;
  int64_t stride = std::max<int64_t>(1, valid / std::max(1, max_points));

  xyz.clear();
  xyz.reserve((valid / stride + 1) * 3);
  int64_t seen = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (cacc[i][0] <= (float)impl_->cfg.conf_threshold) continue;
    if ((seen++ % stride) != 0) continue;
    Eigen::Vector3d pw =
        kf.T_WC.act(Eigen::Vector3d(acc[i][0], acc[i][1], acc[i][2]));
    xyz.push_back((float)pw.x());
    xyz.push_back((float)pw.y());
    xyz.push_back((float)pw.z());
  }
  return !xyz.empty();
}
void Mast3rSlamCore::finish() { impl_->finish(); }

}  // namespace mast3r_slam
