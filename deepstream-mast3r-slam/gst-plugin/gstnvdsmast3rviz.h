/*
 * gstnvdsmast3rviz.h — visualization / ROS 2 bridge for MASt3R-SLAM output.
 *
 * Sits downstream of nvdsmast3rslam. Passthrough on video; consumes the pose
 * (NvDsMast3rSlamPoseMeta) and map (NvDsMast3rSlamCloudMeta) user metas and:
 *   - overlay: attaches NvDsDisplayMeta (HUD text + top-down trajectory
 *     minimap) rendered by a downstream nvdsosd — view locally or stream the
 *     encoded video to a laptop (works on headless hosts);
 *   - ROS 2 (compile-time optional, -DWITH_ROS2): publishes nav_msgs/Odometry,
 *     nav_msgs/Path, TF and sensor_msgs/PointCloud2 for RViz.
 */
#ifndef GST_NVDSMAST3RVIZ_H
#define GST_NVDSMAST3RVIZ_H

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

#include <vector>

G_BEGIN_DECLS

#define GST_TYPE_NVDSMAST3RVIZ (gst_nvdsmast3rviz_get_type())
#define GST_NVDSMAST3RVIZ(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDSMAST3RVIZ, GstNvDsMast3rViz))
#define GST_IS_NVDSMAST3RVIZ(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDSMAST3RVIZ))

typedef struct _GstNvDsMast3rViz GstNvDsMast3rViz;
typedef struct _GstNvDsMast3rVizClass GstNvDsMast3rVizClass;

struct _GstNvDsMast3rViz {
  GstBaseTransform parent;

  /* properties */
  gboolean overlay;          /* draw HUD + trajectory minimap (via nvdsosd)  */
  gboolean ros_enable;       /* publish Odometry/Path/TF/PointCloud2         */
  gchar *frame_id;           /* ROS fixed frame (default "map")              */
  gchar *child_frame_id;     /* ROS moving frame (default "base_link")       */
  gchar *topic_prefix;       /* ROS topic prefix (default "/mast3r")         */

  /* state */
  std::vector<double> *traj; /* flattened x,y,z per pose (minimap history)   */
  void *ros_bridge;          /* opaque Mast3rRosBridge* (WITH_ROS2 only)     */
};

struct _GstNvDsMast3rVizClass {
  GstBaseTransformClass parent_class;
};

GType gst_nvdsmast3rviz_get_type(void);

/* Called from the shared plugin_init in gstnvdsmast3rslam.cpp. */
gboolean nvdsmast3rviz_register(GstPlugin *plugin);

G_END_DECLS

#endif /* GST_NVDSMAST3RVIZ_H */
