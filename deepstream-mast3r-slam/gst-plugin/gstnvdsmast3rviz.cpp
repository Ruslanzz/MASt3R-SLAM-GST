/*
 * gstnvdsmast3rviz.cpp — visualization / ROS 2 bridge element (see header).
 *
 * Overlay drawing uses NvDsDisplayMeta only, so the element never touches
 * pixels itself — a downstream nvdsosd renders the HUD and minimap. The ROS 2
 * bridge is compiled in only with -DWITH_ROS2 (CMake option); without it the
 * ros-enable property logs a warning and does nothing.
 */
#include "gstnvdsmast3rviz.h"

#include <cmath>
#include <cstring>
#include <string>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#include "mast3r_slam_meta.h"

GST_DEBUG_CATEGORY_STATIC(gst_nvdsmast3rviz_debug);
#define GST_CAT_DEFAULT gst_nvdsmast3rviz_debug

/* ------------------------------------------------------------- ROS 2 bridge */
#ifdef WITH_ROS2
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_ros/transform_broadcaster.h>

struct Mast3rRosBridge {
  rclcpp::Node::SharedPtr node;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf;
  nav_msgs::msg::Path path;
  std::string frame_id, child_frame_id;
  bool we_initialized = false;

  Mast3rRosBridge(const std::string &prefix, const std::string &frame,
                  const std::string &child)
      : frame_id(frame), child_frame_id(child) {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
      we_initialized = true;
    }
    node = std::make_shared<rclcpp::Node>("nvdsmast3rviz");
    pub_odom = node->create_publisher<nav_msgs::msg::Odometry>(
        prefix + "/odom", rclcpp::QoS(50));
    pub_path = node->create_publisher<nav_msgs::msg::Path>(
        prefix + "/path", rclcpp::QoS(5));
    pub_cloud = node->create_publisher<sensor_msgs::msg::PointCloud2>(
        prefix + "/map", rclcpp::QoS(5));
    tf = std::make_unique<tf2_ros::TransformBroadcaster>(node);
    path.header.frame_id = frame_id;
  }
  ~Mast3rRosBridge() {
    tf.reset();
    pub_odom.reset();
    pub_path.reset();
    pub_cloud.reset();
    node.reset();
    if (we_initialized && rclcpp::ok()) rclcpp::shutdown();
  }

  rclcpp::Time stamp(double ts) {
    return rclcpp::Time((int64_t)(ts * 1e9), RCL_ROS_TIME);
  }

  void publishPose(const NvDsMast3rSlamPoseMeta *pm) {
    auto st = stamp(pm->timestamp);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = st;
    odom.header.frame_id = frame_id;
    odom.child_frame_id = child_frame_id;
    odom.pose.pose.position.x = pm->t[0];
    odom.pose.pose.position.y = pm->t[1];
    odom.pose.pose.position.z = pm->t[2];
    odom.pose.pose.orientation.x = pm->q[0];
    odom.pose.pose.orientation.y = pm->q[1];
    odom.pose.pose.orientation.z = pm->q[2];
    odom.pose.pose.orientation.w = pm->q[3];
    pub_odom->publish(odom);

    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = st;
    t.header.frame_id = frame_id;
    t.child_frame_id = child_frame_id;
    t.transform.translation.x = pm->t[0];
    t.transform.translation.y = pm->t[1];
    t.transform.translation.z = pm->t[2];
    t.transform.rotation = odom.pose.pose.orientation;
    tf->sendTransform(t);

    geometry_msgs::msg::PoseStamped ps;
    ps.header = odom.header;
    ps.pose = odom.pose.pose;
    path.poses.push_back(ps);
    if (path.poses.size() > 20000)
      path.poses.erase(path.poses.begin(),
                       path.poses.begin() + path.poses.size() / 2);
    if (pm->is_keyframe) {  // path is heavy -> refresh on keyframes only
      path.header.stamp = st;
      pub_path->publish(path);
    }
  }

  void publishCloud(const NvDsMast3rSlamCloudMeta *cm, double ts) {
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = stamp(ts);
    msg.header.frame_id = frame_id;
    msg.height = 1;
    msg.width = cm->num_points;
    msg.is_bigendian = false;
    msg.is_dense = true;
    msg.point_step = 12;
    msg.row_step = msg.point_step * cm->num_points;
    msg.fields.resize(3);
    const char *names[3] = {"x", "y", "z"};
    for (int i = 0; i < 3; ++i) {
      msg.fields[i].name = names[i];
      msg.fields[i].offset = 4 * i;
      msg.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
      msg.fields[i].count = 1;
    }
    msg.data.resize((size_t)msg.row_step);
    std::memcpy(msg.data.data(), cm->points, msg.data.size());
    pub_cloud->publish(msg);
  }
};
#endif  // WITH_ROS2

/* --------------------------------------------------------------- element */
enum {
  PROP_0,
  PROP_OVERLAY,
  PROP_ROS_ENABLE,
  PROP_FRAME_ID,
  PROP_CHILD_FRAME_ID,
  PROP_TOPIC_PREFIX,
};

#define VIZ_CAPS \
  "video/x-raw(memory:NVMM), format=(string){ NV12, RGBA }; " \
  "video/x-raw, format=(string){ NV12, RGBA }"

static GstStaticPadTemplate viz_sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(VIZ_CAPS));
static GstStaticPadTemplate viz_src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(VIZ_CAPS));

G_DEFINE_TYPE(GstNvDsMast3rViz, gst_nvdsmast3rviz, GST_TYPE_BASE_TRANSFORM);

static void viz_set_property(GObject *object, guint prop_id, const GValue *value,
                             GParamSpec *pspec) {
  GstNvDsMast3rViz *self = GST_NVDSMAST3RVIZ(object);
  switch (prop_id) {
    case PROP_OVERLAY: self->overlay = g_value_get_boolean(value); break;
    case PROP_ROS_ENABLE: self->ros_enable = g_value_get_boolean(value); break;
    case PROP_FRAME_ID:
      g_free(self->frame_id); self->frame_id = g_value_dup_string(value); break;
    case PROP_CHILD_FRAME_ID:
      g_free(self->child_frame_id);
      self->child_frame_id = g_value_dup_string(value); break;
    case PROP_TOPIC_PREFIX:
      g_free(self->topic_prefix);
      self->topic_prefix = g_value_dup_string(value); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
  }
}

static void viz_get_property(GObject *object, guint prop_id, GValue *value,
                             GParamSpec *pspec) {
  GstNvDsMast3rViz *self = GST_NVDSMAST3RVIZ(object);
  switch (prop_id) {
    case PROP_OVERLAY: g_value_set_boolean(value, self->overlay); break;
    case PROP_ROS_ENABLE: g_value_set_boolean(value, self->ros_enable); break;
    case PROP_FRAME_ID: g_value_set_string(value, self->frame_id); break;
    case PROP_CHILD_FRAME_ID:
      g_value_set_string(value, self->child_frame_id); break;
    case PROP_TOPIC_PREFIX: g_value_set_string(value, self->topic_prefix); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
  }
}

static gboolean viz_start(GstBaseTransform *trans) {
  GstNvDsMast3rViz *self = GST_NVDSMAST3RVIZ(trans);
  self->traj = new std::vector<double>();
  if (self->ros_enable) {
#ifdef WITH_ROS2
    try {
      self->ros_bridge = new Mast3rRosBridge(
          self->topic_prefix ? self->topic_prefix : "/mast3r",
          self->frame_id ? self->frame_id : "map",
          self->child_frame_id ? self->child_frame_id : "base_link");
      GST_INFO_OBJECT(self, "ROS 2 bridge up (topics under %s)",
                      self->topic_prefix);
    } catch (const std::exception &e) {
      GST_ERROR_OBJECT(self, "ROS 2 bridge init failed: %s", e.what());
      self->ros_bridge = nullptr;
    }
#else
    GST_WARNING_OBJECT(self,
                       "ros-enable=true but the plugin was built without "
                       "WITH_ROS2; rebuild with -DWITH_ROS2=ON");
#endif
  }
  return TRUE;
}

static gboolean viz_stop(GstBaseTransform *trans) {
  GstNvDsMast3rViz *self = GST_NVDSMAST3RVIZ(trans);
  delete self->traj;
  self->traj = nullptr;
#ifdef WITH_ROS2
  delete (Mast3rRosBridge *)self->ros_bridge;
#endif
  self->ros_bridge = nullptr;
  return TRUE;
}

static void viz_finalize(GObject *object) {
  GstNvDsMast3rViz *self = GST_NVDSMAST3RVIZ(object);
  g_free(self->frame_id);
  g_free(self->child_frame_id);
  g_free(self->topic_prefix);
  G_OBJECT_CLASS(gst_nvdsmast3rviz_parent_class)->finalize(object);
}

static const char *mode_str(Mast3rSlamMode m) {
  switch (m) {
    case MAST3R_SLAM_MODE_INIT: return "INIT";
    case MAST3R_SLAM_MODE_TRACKING: return "TRACK";
    case MAST3R_SLAM_MODE_RELOC: return "RELOC";
    default: return "?";
  }
}

/* HUD text + top-down trajectory minimap (world X/Z), all via display meta. */
static void draw_overlay(GstNvDsMast3rViz *self, NvDsBatchMeta *bm,
                         NvDsFrameMeta *fm, const NvDsMast3rSlamPoseMeta *pm) {
  int W = fm->source_frame_width > 0 ? (int)fm->source_frame_width : 1280;
  int H = fm->source_frame_height > 0 ? (int)fm->source_frame_height : 720;

  NvDsDisplayMeta *dm = nvds_acquire_display_meta_from_pool(bm);
  if (!dm) return;

  /* HUD line */
  dm->num_labels = 1;
  NvOSD_TextParams *tp = &dm->text_params[0];
  tp->display_text = g_strdup_printf(
      "MASt3R-SLAM  t=(%.2f, %.2f, %.2f) m  kf=%d  %s%s",
      pm->t[0], pm->t[1], pm->t[2], pm->num_keyframes, mode_str(pm->mode),
      pm->is_keyframe ? "  [KF]" : "");
  tp->x_offset = 12;
  tp->y_offset = 12;
  tp->font_params.font_name = (gchar *)"Serif";
  tp->font_params.font_size = 13;
  tp->font_params.font_color = (NvOSD_ColorParams){1.0, 1.0, 1.0, 1.0};
  tp->set_bg_clr = 1;
  tp->text_bg_clr = (NvOSD_ColorParams){0.0, 0.0, 0.0, 0.6};

  /* minimap box (top-right corner) */
  int box_w = W / 4, box_h = H / 4, margin = 12;
  int bx = W - box_w - margin, by = margin;
  dm->num_rects = 1;
  NvOSD_RectParams *rp = &dm->rect_params[0];
  rp->left = bx; rp->top = by; rp->width = box_w; rp->height = box_h;
  rp->border_width = 2;
  rp->border_color = (NvOSD_ColorParams){1.0, 1.0, 1.0, 0.8};
  rp->has_bg_color = 1;
  rp->bg_color = (NvOSD_ColorParams){0.0, 0.0, 0.0, 0.4};
  nvds_add_display_meta_to_frame(fm, dm);

  /* trajectory polyline: world X (right) / Z (forward), auto-fit into the box */
  auto &tr = *self->traj;
  size_t n = tr.size() / 3;
  if (n < 2) return;
  double minx = 1e30, maxx = -1e30, minz = 1e30, maxz = -1e30;
  for (size_t i = 0; i < n; ++i) {
    minx = std::min(minx, tr[3 * i]); maxx = std::max(maxx, tr[3 * i]);
    minz = std::min(minz, tr[3 * i + 2]); maxz = std::max(maxz, tr[3 * i + 2]);
  }
  double span = std::max({maxx - minx, maxz - minz, 1e-3});
  double cx = 0.5 * (minx + maxx), cz = 0.5 * (minz + maxz);
  double sc = 0.85 * std::min(box_w, box_h) / span;
  auto px = [&](double x) { return (int)(bx + box_w / 2 + (x - cx) * sc); };
  auto pz = [&](double z) { return (int)(by + box_h / 2 - (z - cz) * sc); };

  const size_t MAX_SEG = 48;  /* 3 display metas x 16 lines */
  size_t step = std::max<size_t>(1, (n - 1) / MAX_SEG);
  NvDsDisplayMeta *ld = nullptr;
  for (size_t i = step; i < n; i += step) {
    if (!ld || ld->num_lines >= MAX_ELEMENTS_IN_DISPLAY_META) {
      ld = nvds_acquire_display_meta_from_pool(bm);
      if (!ld) break;
      nvds_add_display_meta_to_frame(fm, ld);
    }
    NvOSD_LineParams *lp = &ld->line_params[ld->num_lines++];
    lp->x1 = px(tr[3 * (i - step)]); lp->y1 = pz(tr[3 * (i - step) + 2]);
    lp->x2 = px(tr[3 * i]);          lp->y2 = pz(tr[3 * i + 2]);
    lp->line_width = 2;
    lp->line_color = (NvOSD_ColorParams){0.2, 1.0, 0.2, 0.9};
  }
}

static GstFlowReturn viz_transform_ip(GstBaseTransform *trans, GstBuffer *buf) {
  GstNvDsMast3rViz *self = GST_NVDSMAST3RVIZ(trans);
  NvDsBatchMeta *bm = gst_buffer_get_nvds_batch_meta(buf);
  if (!bm) return GST_FLOW_OK;

  for (NvDsMetaList *l = bm->frame_meta_list; l != nullptr; l = l->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *)l->data;
    const NvDsMast3rSlamPoseMeta *pm = nullptr;
    const NvDsMast3rSlamCloudMeta *cm = nullptr;
    for (NvDsMetaList *u = fm->frame_user_meta_list; u != nullptr; u = u->next) {
      NvDsUserMeta *um = (NvDsUserMeta *)u->data;
      if (um->base_meta.meta_type == (NvDsMetaType)NVDS_MAST3R_SLAM_POSE_META)
        pm = (const NvDsMast3rSlamPoseMeta *)um->user_meta_data;
      else if (um->base_meta.meta_type ==
               (NvDsMetaType)NVDS_MAST3R_SLAM_CLOUD_META)
        cm = (const NvDsMast3rSlamCloudMeta *)um->user_meta_data;
    }
    if (!pm) continue;

    self->traj->push_back(pm->t[0]);
    self->traj->push_back(pm->t[1]);
    self->traj->push_back(pm->t[2]);
    if (self->traj->size() > 3 * 20000) {  /* decimate history 2x */
      auto &tr = *self->traj;
      size_t w = 0;
      for (size_t i = 0; i < tr.size() / 3; i += 2, ++w) {
        tr[3 * w] = tr[3 * i]; tr[3 * w + 1] = tr[3 * i + 1];
        tr[3 * w + 2] = tr[3 * i + 2];
      }
      tr.resize(3 * w);
    }

    if (self->overlay) draw_overlay(self, bm, fm, pm);

#ifdef WITH_ROS2
    if (self->ros_bridge) {
      auto *rb = (Mast3rRosBridge *)self->ros_bridge;
      rb->publishPose(pm);
      if (cm && cm->num_points > 0) rb->publishCloud(cm, pm->timestamp);
    }
#else
    (void)cm;
#endif
  }
  return GST_FLOW_OK;
}

static void gst_nvdsmast3rviz_class_init(GstNvDsMast3rVizClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass *bt = GST_BASE_TRANSFORM_CLASS(klass);

  gobject_class->set_property = viz_set_property;
  gobject_class->get_property = viz_get_property;
  gobject_class->finalize = viz_finalize;
  bt->start = GST_DEBUG_FUNCPTR(viz_start);
  bt->stop = GST_DEBUG_FUNCPTR(viz_stop);
  bt->transform_ip = GST_DEBUG_FUNCPTR(viz_transform_ip);

  g_object_class_install_property(
      gobject_class, PROP_OVERLAY,
      g_param_spec_boolean("overlay", "overlay",
                           "Draw HUD + trajectory minimap (rendered by a "
                           "downstream nvdsosd)",
                           TRUE, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_ROS_ENABLE,
      g_param_spec_boolean("ros-enable", "ros-enable",
                           "Publish Odometry/Path/TF/PointCloud2 to ROS 2 "
                           "(requires build with -DWITH_ROS2=ON)",
                           FALSE, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_FRAME_ID,
      g_param_spec_string("frame-id", "frame-id", "ROS fixed frame", "map",
                          (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_CHILD_FRAME_ID,
      g_param_spec_string("child-frame-id", "child-frame-id",
                          "ROS moving frame", "base_link",
                          (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_TOPIC_PREFIX,
      g_param_spec_string("topic-prefix", "topic-prefix", "ROS topic prefix",
                          "/mast3r", (GParamFlags)G_PARAM_READWRITE));

  gst_element_class_set_static_metadata(
      element_class, "MASt3R-SLAM visualization / ROS 2 bridge",
      "Filter/Analyzer/Video",
      "Overlays the SLAM trajectory HUD (via nvdsosd display meta) and/or "
      "publishes Odometry, Path, TF and PointCloud2 to ROS 2",
      "MASt3R-SLAM DeepStream integration");
  gst_element_class_add_pad_template(
      element_class, gst_static_pad_template_get(&viz_sink_template));
  gst_element_class_add_pad_template(
      element_class, gst_static_pad_template_get(&viz_src_template));
}

static void gst_nvdsmast3rviz_init(GstNvDsMast3rViz *self) {
  self->overlay = TRUE;
  self->ros_enable = FALSE;
  self->frame_id = g_strdup("map");
  self->child_frame_id = g_strdup("base_link");
  self->topic_prefix = g_strdup("/mast3r");
  self->traj = nullptr;
  self->ros_bridge = nullptr;
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

gboolean nvdsmast3rviz_register(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_nvdsmast3rviz_debug, "nvdsmast3rviz", 0,
                          "MASt3R-SLAM visualization element");
  return gst_element_register(plugin, "nvdsmast3rviz", GST_RANK_NONE,
                              GST_TYPE_NVDSMAST3RVIZ);
}
