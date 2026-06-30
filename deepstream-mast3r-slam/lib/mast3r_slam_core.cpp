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

  // -------------------------------------------------------------- per frame
  PoseResult process(const FrameInput &in) {
    torch::NoGradGuard ng;
    int H = in.model_h, W = in.model_w;
    // wrap encoder tensors (device memory owned by nvinfer meta)
    auto feat = torch::from_blob((void *)in.feat_dev, {1, in.feat_n, in.feat_dim}, f32());
    auto pos = torch::from_blob((void *)in.pos_dev, {1, in.pos_n, 2},
                                torch::TensorOptions().dtype(torch::kInt32).device(device))
                   .to(torch::kLong);

    PoseResult res;
    res.frame_id = frame_count;
    res.timestamp = in.timestamp;

    if (mode == 0) {  // INIT — mono inference
      auto o = runDecoder(feat, pos, feat, pos, H, W);
      auto X = o.Xi.view({-1, 3}).contiguous();
      auto C = o.Ci.view({-1, 1}).contiguous();
      Keyframe kf;
      kf.frame_id = frame_count;
      kf.timestamp = in.timestamp;
      kf.T_WC = Sim3::Identity();
      kf.feat = feat.clone();
      kf.pos = pos.clone();
      kf.H = H; kf.W = W;
      updatePointmap(kf.X_canon, kf.C, kf.N, X, C);
      keyframes.push_back(std::move(kf));
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

    // frame canonical pointmap (single update -> just set)
    auto Xf_canon = Xff;  // frame.update_pointmap with N=0 sets directly
    auto Cf = Cff;        // average conf with N=1 == C

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

    // update keyframe pointmap with aligned current observation
    auto Xkk = actSim3(T_CkCf, Xkf, nullptr);
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
      Keyframe nkf;
      nkf.frame_id = frame_count;
      nkf.timestamp = in.timestamp;
      nkf.T_WC = T_WCf;
      nkf.feat = feat.clone();
      nkf.pos = pos.clone();
      nkf.H = H; nkf.W = W;
      int n0 = 0;
      updatePointmap(nkf.X_canon, nkf.C, n0, Xf_canon, Cf);
      nkf.N = n0;
      keyframes.push_back(std::move(nkf));
      // INTEGRATION POINT: enqueue global factor-graph optimisation here
      // (gauss_newton_rays_cuda over retrieval + consecutive edges). See DESIGN.
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
void Mast3rSlamCore::finish() { impl_->finish(); }

}  // namespace mast3r_slam
