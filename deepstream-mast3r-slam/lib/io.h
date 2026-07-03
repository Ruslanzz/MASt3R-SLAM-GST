/*
 * io.h — output writers mirroring mast3r_slam.evaluate:
 *   - TUM trajectory:  "<t> <tx> <ty> <tz> <qx> <qy> <qz> <qw>"
 *   - colored point cloud .ply (binary little-endian)
 * so the persisted output matches the reference repository.
 */
#ifndef MAST3R_SLAM_IO_H
#define MAST3R_SLAM_IO_H

#include <cstdint>
#include <string>
#include <vector>

#include "sim3.h"

namespace mast3r_slam {

struct TrajSample {
  double timestamp;
  Sim3 T_WC;  // world-from-camera (SE3 part used; scale dropped, as in as_SE3)
};

// Append/overwrite a TUM-format trajectory file.
void saveTrajectoryTUM(const std::string &path,
                       const std::vector<TrajSample> &samples);

// Write a binary little-endian PLY with XYZ + RGB.
// points: flattened [N*3] (x,y,z); colors: flattened [N*3] (r,g,b in 0..255).
void savePly(const std::string &path, const std::vector<float> &points,
             const std::vector<uint8_t> &colors);

}  // namespace mast3r_slam

#endif  // MAST3R_SLAM_IO_H
