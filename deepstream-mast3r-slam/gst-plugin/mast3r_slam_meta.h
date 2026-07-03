/*
 * Custom DeepStream user-meta carrying the MASt3R-SLAM camera pose for a frame.
 *
 * Attached by the nvdsmast3rslam element as NvDsUserMeta on the frame's
 * NvDsFrameMeta. Downstream elements / pad probes read it to obtain the live
 * trajectory (the online output channel; persistent .txt/.ply are written on
 * EOS, see lib/io.h).
 */
#ifndef MAST3R_SLAM_META_H
#define MAST3R_SLAM_META_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* Pick a value in the user/custom meta range to avoid clashing with nvds meta. */
#define NVDS_MAST3R_SLAM_POSE_META (nvds_get_user_meta_type((char *)"NVIDIA.MAST3R_SLAM.POSE"))

typedef enum {
  MAST3R_SLAM_MODE_INIT = 0,
  MAST3R_SLAM_MODE_TRACKING = 1,
  MAST3R_SLAM_MODE_RELOC = 2,
  MAST3R_SLAM_MODE_TERMINATED = 3
} Mast3rSlamMode;

/*
 * World-from-camera Sim3 pose, decomposed for convenience.
 * Quaternion order matches lietorch / mast3r_slam.evaluate: (qx, qy, qz, qw).
 */
typedef struct _NvDsMast3rSlamPoseMeta {
  uint64_t frame_id;     /* SLAM frame index (also the TUM trajectory row)      */
  double timestamp;      /* seconds (from buffer PTS)                            */
  double t[3];           /* translation tx, ty, tz                               */
  double q[4];           /* rotation quaternion qx, qy, qz, qw                   */
  double scale;          /* Sim3 similarity scale                                */
  int num_keyframes;     /* current keyframe count                               */
  int is_keyframe;       /* 1 if this frame became a keyframe                    */
  Mast3rSlamMode mode;   /* tracker mode for this frame                          */
} NvDsMast3rSlamPoseMeta;

#ifdef __cplusplus
}
#endif

#endif /* MAST3R_SLAM_META_H */
