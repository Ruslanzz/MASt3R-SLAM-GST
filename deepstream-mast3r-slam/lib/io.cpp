#include "io.h"

#include <cstdio>
#include <fstream>

namespace mast3r_slam {

void saveTrajectoryTUM(const std::string &path,
                       const std::vector<TrajSample> &samples) {
  std::ofstream f(path, std::ios::out | std::ios::trunc);
  if (!f.good()) return;
  f.setf(std::ios::fixed);
  for (const auto &s : samples) {
    double d[8];
    s.T_WC.toData(d);  // [tx,ty,tz, qx,qy,qz,qw, scale]
    // TUM line: t tx ty tz qx qy qz qw  (scale dropped, matching as_SE3)
    char line[256];
    std::snprintf(line, sizeof(line),
                  "%.6f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", s.timestamp,
                  d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
    f << line;
  }
}

void savePly(const std::string &path, const std::vector<float> &points,
             const std::vector<uint8_t> &colors) {
  const size_t n = points.size() / 3;
  std::ofstream f(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!f.good()) return;

  f << "ply\n";
  f << "format binary_little_endian 1.0\n";
  f << "element vertex " << n << "\n";
  f << "property float x\nproperty float y\nproperty float z\n";
  f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  f << "end_header\n";

  for (size_t i = 0; i < n; ++i) {
    f.write(reinterpret_cast<const char *>(&points[i * 3]), sizeof(float) * 3);
    uint8_t rgb[3] = {colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]};
    f.write(reinterpret_cast<const char *>(rgb), 3);
  }
}

}  // namespace mast3r_slam
