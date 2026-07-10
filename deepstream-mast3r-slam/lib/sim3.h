/*
 * sim3.h — minimal Sim(3) Lie group for the camera pose, mirroring the
 * lietorch.Sim3 semantics used by mast3r_slam (tracker.py / lietorch_utils.py).
 *
 * Data layout matches lietorch: [tx, ty, tz, qx, qy, qz, qw, s] (8 floats).
 * The tangent (retraction) order matches geometry.act_Sim3's Jacobian columns:
 * [rho(3) translation, phi(3) rotation, sigma(1) log-scale], applied as a LEFT
 * increment  T <- Exp(tau) * T  (as in T_CkCf.retr(tau)).
 *
 * NOTE: validate numerically against lietorch on the target before production —
 * the closed-form Sim3 exp below follows the standard Sophus derivation but the
 * SLAM is sensitive to the exact retraction convention.
 */
#ifndef MAST3R_SLAM_SIM3_H
#define MAST3R_SLAM_SIM3_H

#include <Eigen/Dense>
#include <cmath>

namespace mast3r_slam {

class Sim3 {
 public:
  Sim3() : R_(Eigen::Matrix3d::Identity()), t_(Eigen::Vector3d::Zero()), s_(1.0) {}
  Sim3(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, double s)
      : R_(R), t_(t), s_(s) {}

  static Sim3 Identity() { return Sim3(); }

  // From lietorch data [t(3), q(xyzw), s].
  static Sim3 fromData(const double d[8]) {
    Eigen::Quaterniond q(d[6], d[3], d[4], d[5]);  // (w, x, y, z)
    q.normalize();
    Eigen::Vector3d t(d[0], d[1], d[2]);
    return Sim3(q.toRotationMatrix(), t, d[7]);
  }

  void toData(double d[8]) const {
    Eigen::Quaterniond q(R_);
    q.normalize();
    d[0] = t_.x(); d[1] = t_.y(); d[2] = t_.z();
    d[3] = q.x(); d[4] = q.y(); d[5] = q.z(); d[6] = q.w();
    d[7] = s_;
  }

  const Eigen::Matrix3d &R() const { return R_; }
  const Eigen::Vector3d &t() const { return t_; }
  double s() const { return s_; }

  // Acts on a 3D point: pW = s * R * pC + t
  Eigen::Vector3d act(const Eigen::Vector3d &p) const { return s_ * (R_ * p) + t_; }

  Sim3 inverse() const {
    double s_inv = 1.0 / s_;
    Eigen::Matrix3d R_inv = R_.transpose();
    Eigen::Vector3d t_inv = -s_inv * (R_inv * t_);
    return Sim3(R_inv, t_inv, s_inv);
  }

  // this * other
  Sim3 operator*(const Sim3 &o) const {
    Eigen::Matrix3d R = R_ * o.R_;
    Eigen::Vector3d t = s_ * (R_ * o.t_) + t_;
    double s = s_ * o.s_;
    return Sim3(R, t, s);
  }

  // Left retraction: T <- Exp(tau) * T,  tau = [rho(3), phi(3), sigma(1)].
  Sim3 retr(const Eigen::Matrix<double, 7, 1> &tau) const {
    return Exp(tau) * (*this);
  }

  // Sim(3) exponential map.
  static Sim3 Exp(const Eigen::Matrix<double, 7, 1> &tau) {
    Eigen::Vector3d rho = tau.segment<3>(0);
    Eigen::Vector3d phi = tau.segment<3>(3);
    double sigma = tau(6);

    double s = std::exp(sigma);
    Eigen::Matrix3d R = expSO3(phi);
    Eigen::Matrix3d W = sim3V(phi, sigma);
    Eigen::Vector3d t = W * rho;
    return Sim3(R, t, s);
  }

 private:
  static Eigen::Matrix3d hat(const Eigen::Vector3d &w) {
    Eigen::Matrix3d S;
    S << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
    return S;
  }

  static Eigen::Matrix3d expSO3(const Eigen::Vector3d &phi) {
    double theta = phi.norm();
    Eigen::Matrix3d Phi = hat(phi);
    if (theta < 1e-8) return Eigen::Matrix3d::Identity() + Phi;
    double a = std::sin(theta) / theta;
    double b = (1.0 - std::cos(theta)) / (theta * theta);
    return Eigen::Matrix3d::Identity() + a * Phi + b * (Phi * Phi);
  }

  // The Sim(3) left-Jacobian "W" matrix coupling rotation and scale into the
  // translation term (Sophus RxSO3/Sim3 derivation).
  static Eigen::Matrix3d sim3V(const Eigen::Vector3d &phi, double sigma) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d Phi = hat(phi);
    const Eigen::Matrix3d Phi2 = Phi * Phi;
    double theta = phi.norm();
    double theta2 = theta * theta;

    double A, B, C;
    const double eps = 1e-8;
    if (std::abs(sigma) < eps) {
      // scale ~ 1: reduce to SO3 left-Jacobian.
      C = 1.0;
      if (theta < eps) {
        A = 0.5;
        B = 1.0 / 6.0;
      } else {
        A = (1.0 - std::cos(theta)) / theta2;
        B = (theta - std::sin(theta)) / (theta2 * theta);
      }
    } else {
      double s = std::exp(sigma);
      C = (s - 1.0) / sigma;
      double sigma2 = sigma * sigma;
      if (theta < eps) {
        A = ((sigma - 1.0) * s + 1.0) / sigma2;
        B = ((0.5 * sigma2 - sigma + 1.0) * s - 1.0) / (sigma2 * sigma);
      } else {
        double sin_t = std::sin(theta);
        double cos_t = std::cos(theta);
        double denom_a = sigma2 + theta2;
        A = (s * sin_t * sigma + (1.0 - s * cos_t) * theta) / (theta * denom_a);
        B = (C - ((s * cos_t - 1.0) * sigma + s * sin_t * theta) / denom_a) / theta2;
      }
    }
    return C * I + A * Phi + B * Phi2;
  }

  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;
  double s_;
};

}  // namespace mast3r_slam

#endif  // MAST3R_SLAM_SIM3_H
