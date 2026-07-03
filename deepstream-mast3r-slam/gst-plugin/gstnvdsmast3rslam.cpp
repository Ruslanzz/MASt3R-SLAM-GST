/*
 * gstnvdsmast3rslam.cpp
 *
 * DeepStream element that consumes the MASt3R ViT-encoder tensors produced by an
 * upstream gst-nvinfer (output-tensor-meta=1) and runs MASt3R-SLAM on the GPU:
 * decoder + matching + Sim3 tracking + factor-graph backend (lib/). It attaches
 * the per-frame camera pose as NvDsUserMeta and, on EOS, writes the same outputs
 * as the reference repo (TUM .txt, .ply, keyframe PNGs).
 *
 * Passthrough-in-place: the video buffer is never modified, so downstream NVIDIA
 * elements (nvdsosd, encoders, sinks) keep working.
 */
#include "gstnvdsmast3rslam.h"

#include <cstring>
#include <vector>

#include "gstnvdsinfer.h"   // NvDsInferTensorMeta + NVDSINFER_TENSOR_OUTPUT_META
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
// Older DeepStream shipped a separate nvdsinfer_tensor_meta.h; on DS 7.x the
// tensor meta lives in gstnvdsinfer.h. Include it only if present.
#if defined(__has_include)
#  if __has_include("nvdsinfer_tensor_meta.h")
#    include "nvdsinfer_tensor_meta.h"
#  endif
#endif

#include "mast3r_slam_core.h"
#include "mast3r_slam_meta.h"

GST_DEBUG_CATEGORY_STATIC(gst_nvdsmast3rslam_debug);
#define GST_CAT_DEFAULT gst_nvdsmast3rslam_debug

enum {
  PROP_0,
  PROP_INFER_GIE_ID,
  PROP_CONFIG,
  PROP_CALIB,
  PROP_DECODER_ENGINE,
  PROP_SAVE_DIR,
  PROP_SEQUENCE_NAME,
  PROP_SAVE_RESULTS,
  PROP_CONF_THRESHOLD,
  PROP_GPU_ID,
  PROP_STEREO_MODE,
  PROP_BASELINE,
  PROP_LEFT_SOURCE_ID,
  PROP_RIGHT_SOURCE_ID,
  PROP_LOOP_CLOSURE,
  PROP_LOOP_SIM_THRESH,
  PROP_EMIT_CLOUD,
  PROP_CLOUD_MAX_POINTS,
};

/* gst-nvinfer feeds us NVMM video; we negotiate RGBA (set by nvvideoconvert) but
 * never touch the pixels — only the attached tensor meta. */
#define SUPPORTED_CAPS \
  "video/x-raw(memory:NVMM), format=(string){ NV12, RGBA }; " \
  "video/x-raw, format=(string){ NV12, RGBA }"

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(SUPPORTED_CAPS));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(SUPPORTED_CAPS));

#define gst_nvdsmast3rslam_parent_class parent_class
G_DEFINE_TYPE(GstNvDsMast3rSlam, gst_nvdsmast3rslam, GST_TYPE_BASE_TRANSFORM);

#define GST_TYPE_NVDSMAST3RSLAM_STEREO_MODE \
  (gst_nvdsmast3rslam_stereo_mode_get_type())
static GType gst_nvdsmast3rslam_stereo_mode_get_type(void) {
  static GType mode_type = 0;
  static const GEnumValue modes[] = {
      {GST_NVDSMAST3RSLAM_MODE_AUTO,
       "Per buffer: left+right in the batch -> stereo, otherwise mono", "auto"},
      {GST_NVDSMAST3RSLAM_MODE_MONO,
       "Force monocular SLAM on every batch frame", "mono"},
      {GST_NVDSMAST3RSLAM_MODE_STEREO,
       "Force left/right pairing by source-id (metric scale from baseline)",
       "stereo"},
      {0, nullptr, nullptr},
  };
  if (g_once_init_enter(&mode_type)) {
    GType t = g_enum_register_static("GstNvDsMast3rSlamStereoMode", modes);
    g_once_init_leave(&mode_type, t);
  }
  return mode_type;
}

static void gst_nvdsmast3rslam_set_property(GObject *object, guint prop_id,
                                            const GValue *value,
                                            GParamSpec *pspec);
static void gst_nvdsmast3rslam_get_property(GObject *object, guint prop_id,
                                            GValue *value, GParamSpec *pspec);
static void gst_nvdsmast3rslam_finalize(GObject *object);
static gboolean gst_nvdsmast3rslam_start(GstBaseTransform *trans);
static gboolean gst_nvdsmast3rslam_stop(GstBaseTransform *trans);
static gboolean gst_nvdsmast3rslam_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps, GstCaps *outcaps);
static gboolean gst_nvdsmast3rslam_sink_event(GstBaseTransform *trans,
                                              GstEvent *event);
static GstFlowReturn gst_nvdsmast3rslam_transform_ip(GstBaseTransform *trans,
                                                     GstBuffer *buf);

/* ------------------------------------------------------------------ class */
static void gst_nvdsmast3rslam_class_init(GstNvDsMast3rSlamClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  gobject_class->set_property = gst_nvdsmast3rslam_set_property;
  gobject_class->get_property = gst_nvdsmast3rslam_get_property;
  gobject_class->finalize = gst_nvdsmast3rslam_finalize;

  base_transform_class->start = GST_DEBUG_FUNCPTR(gst_nvdsmast3rslam_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_nvdsmast3rslam_stop);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_nvdsmast3rslam_set_caps);
  base_transform_class->sink_event =
      GST_DEBUG_FUNCPTR(gst_nvdsmast3rslam_sink_event);
  base_transform_class->transform_ip =
      GST_DEBUG_FUNCPTR(gst_nvdsmast3rslam_transform_ip);

  g_object_class_install_property(
      gobject_class, PROP_INFER_GIE_ID,
      g_param_spec_uint("infer-gie-id", "infer-gie-id",
                        "gie-unique-id of the upstream nvinfer (encoder)", 0,
                        G_MAXUINT, 1, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_CONFIG,
      g_param_spec_string("config", "config", "Path to SLAM config YAML",
                          "config/base.yaml", (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_CALIB,
      g_param_spec_string("calib", "calib", "Path to intrinsics YAML (optional)",
                          "", (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_DECODER_ENGINE,
      g_param_spec_string("decoder-engine", "decoder-engine",
                          "TensorRT engine for the MASt3R decoder+heads", "",
                          (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_SAVE_DIR,
      g_param_spec_string("save-dir", "save-dir", "Output directory", "logs",
                          (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_SEQUENCE_NAME,
      g_param_spec_string("sequence-name", "sequence-name",
                          "Base name of output files", "mast3rslam",
                          (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_SAVE_RESULTS,
      g_param_spec_boolean("save-results", "save-results",
                           "Write trajectory/point-cloud/keyframes on EOS", TRUE,
                           (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_CONF_THRESHOLD,
      g_param_spec_double("conf-threshold", "conf-threshold",
                          "Confidence threshold for the exported .ply", 0.0,
                          1e9, 1.5, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_GPU_ID,
      g_param_spec_int("gpu-id", "gpu-id", "CUDA device id", 0, G_MAXINT, 0,
                       (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_STEREO_MODE,
      g_param_spec_enum("stereo-mode", "stereo-mode",
                        "Camera setup: auto (default) detects per buffer — a "
                        "batch carrying both left and right source-ids runs "
                        "stereo (metric scale from the baseline), a single "
                        "frame runs mono; mono/stereo force the mode",
                        GST_TYPE_NVDSMAST3RSLAM_STEREO_MODE,
                        GST_NVDSMAST3RSLAM_MODE_AUTO,
                        (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_BASELINE,
      g_param_spec_double("baseline", "baseline",
                          "Stereo baseline in meters", 1e-4, 10.0, 0.12,
                          (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_LEFT_SOURCE_ID,
      g_param_spec_int("left-source-id", "left-source-id",
                       "nvstreammux source-id of the left camera", 0, G_MAXINT,
                       0, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_RIGHT_SOURCE_ID,
      g_param_spec_int("right-source-id", "right-source-id",
                       "nvstreammux source-id of the right camera", 0, G_MAXINT,
                       1, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_LOOP_CLOSURE,
      g_param_spec_boolean("loop-closure", "loop-closure",
                           "Enable retrieval + factor-graph global optimisation",
                           TRUE, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_LOOP_SIM_THRESH,
      g_param_spec_double("loop-sim-thresh", "loop-sim-thresh",
                          "Cosine similarity threshold for loop candidates",
                          0.0, 1.0, 0.90, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_EMIT_CLOUD,
      g_param_spec_boolean("emit-cloud", "emit-cloud",
                           "Attach the keyframe map as NvDsMast3rSlamCloudMeta "
                           "on keyframe frames (for nvdsmast3rviz / ROS)",
                           TRUE, (GParamFlags)G_PARAM_READWRITE));
  g_object_class_install_property(
      gobject_class, PROP_CLOUD_MAX_POINTS,
      g_param_spec_int("cloud-max-points", "cloud-max-points",
                       "Stride the emitted keyframe cloud down to this many "
                       "points", 100, 500000, 50000,
                       (GParamFlags)G_PARAM_READWRITE));

  gst_element_class_set_static_metadata(
      element_class, "MASt3R-SLAM", "Filter/Analyzer/Video",
      "Dense SLAM driven by gst-nvinfer encoder tensors; works with a mono or "
      "a stereo camera (auto-detected per buffer), emits poses as NvDsUserMeta "
      "and saves trajectory/point-cloud on EOS",
      "MASt3R-SLAM DeepStream integration");

  gst_element_class_add_pad_template(
      element_class, gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(
      element_class, gst_static_pad_template_get(&src_template));
}

static void gst_nvdsmast3rslam_init(GstNvDsMast3rSlam *self) {
  self->infer_gie_id = 1;
  self->config_path = g_strdup("config/base.yaml");
  self->calib_path = g_strdup("");
  self->decoder_engine = g_strdup("");
  self->save_dir = g_strdup("logs");
  self->sequence_name = g_strdup("mast3rslam");
  self->save_results = TRUE;
  self->conf_threshold = 1.5;
  self->gpu_id = 0;
  self->stereo_mode = GST_NVDSMAST3RSLAM_MODE_AUTO;
  self->baseline = 0.12;
  self->left_source_id = 0;
  self->right_source_id = 1;
  self->loop_closure = TRUE;
  self->loop_sim_thresh = 0.90;
  self->emit_cloud = TRUE;
  self->cloud_max_points = 50000;
  self->core = nullptr;
  self->frame_num = 0;
  self->video_info_valid = FALSE;
  self->stereo_seen = FALSE;

  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

static void gst_nvdsmast3rslam_finalize(GObject *object) {
  GstNvDsMast3rSlam *self = GST_NVDSMAST3RSLAM(object);
  g_free(self->config_path);
  g_free(self->calib_path);
  g_free(self->decoder_engine);
  g_free(self->save_dir);
  g_free(self->sequence_name);
  if (self->core) {
    delete self->core;
    self->core = nullptr;
  }
  G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* --------------------------------------------------------------- props */
static void gst_nvdsmast3rslam_set_property(GObject *object, guint prop_id,
                                            const GValue *value,
                                            GParamSpec *pspec) {
  GstNvDsMast3rSlam *self = GST_NVDSMAST3RSLAM(object);
  switch (prop_id) {
    case PROP_INFER_GIE_ID: self->infer_gie_id = g_value_get_uint(value); break;
    case PROP_CONFIG:
      g_free(self->config_path); self->config_path = g_value_dup_string(value); break;
    case PROP_CALIB:
      g_free(self->calib_path); self->calib_path = g_value_dup_string(value); break;
    case PROP_DECODER_ENGINE:
      g_free(self->decoder_engine); self->decoder_engine = g_value_dup_string(value); break;
    case PROP_SAVE_DIR:
      g_free(self->save_dir); self->save_dir = g_value_dup_string(value); break;
    case PROP_SEQUENCE_NAME:
      g_free(self->sequence_name); self->sequence_name = g_value_dup_string(value); break;
    case PROP_SAVE_RESULTS: self->save_results = g_value_get_boolean(value); break;
    case PROP_CONF_THRESHOLD: self->conf_threshold = g_value_get_double(value); break;
    case PROP_GPU_ID: self->gpu_id = g_value_get_int(value); break;
    case PROP_STEREO_MODE:
      self->stereo_mode =
          (GstNvDsMast3rSlamStereoMode)g_value_get_enum(value);
      break;
    case PROP_BASELINE: self->baseline = g_value_get_double(value); break;
    case PROP_LEFT_SOURCE_ID: self->left_source_id = g_value_get_int(value); break;
    case PROP_RIGHT_SOURCE_ID: self->right_source_id = g_value_get_int(value); break;
    case PROP_LOOP_CLOSURE: self->loop_closure = g_value_get_boolean(value); break;
    case PROP_LOOP_SIM_THRESH: self->loop_sim_thresh = g_value_get_double(value); break;
    case PROP_EMIT_CLOUD: self->emit_cloud = g_value_get_boolean(value); break;
    case PROP_CLOUD_MAX_POINTS: self->cloud_max_points = g_value_get_int(value); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
  }
}

static void gst_nvdsmast3rslam_get_property(GObject *object, guint prop_id,
                                            GValue *value, GParamSpec *pspec) {
  GstNvDsMast3rSlam *self = GST_NVDSMAST3RSLAM(object);
  switch (prop_id) {
    case PROP_INFER_GIE_ID: g_value_set_uint(value, self->infer_gie_id); break;
    case PROP_CONFIG: g_value_set_string(value, self->config_path); break;
    case PROP_CALIB: g_value_set_string(value, self->calib_path); break;
    case PROP_DECODER_ENGINE: g_value_set_string(value, self->decoder_engine); break;
    case PROP_SAVE_DIR: g_value_set_string(value, self->save_dir); break;
    case PROP_SEQUENCE_NAME: g_value_set_string(value, self->sequence_name); break;
    case PROP_SAVE_RESULTS: g_value_set_boolean(value, self->save_results); break;
    case PROP_CONF_THRESHOLD: g_value_set_double(value, self->conf_threshold); break;
    case PROP_GPU_ID: g_value_set_int(value, self->gpu_id); break;
    case PROP_STEREO_MODE: g_value_set_enum(value, self->stereo_mode); break;
    case PROP_BASELINE: g_value_set_double(value, self->baseline); break;
    case PROP_LEFT_SOURCE_ID: g_value_set_int(value, self->left_source_id); break;
    case PROP_RIGHT_SOURCE_ID: g_value_set_int(value, self->right_source_id); break;
    case PROP_LOOP_CLOSURE: g_value_set_boolean(value, self->loop_closure); break;
    case PROP_LOOP_SIM_THRESH: g_value_set_double(value, self->loop_sim_thresh); break;
    case PROP_EMIT_CLOUD: g_value_set_boolean(value, self->emit_cloud); break;
    case PROP_CLOUD_MAX_POINTS: g_value_set_int(value, self->cloud_max_points); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec); break;
  }
}

/* --------------------------------------------------------------- lifecycle */
static gboolean gst_nvdsmast3rslam_start(GstBaseTransform *trans) {
  GstNvDsMast3rSlam *self = GST_NVDSMAST3RSLAM(trans);
  mast3r_slam::CoreConfig cfg;
  cfg.config_path = self->config_path ? self->config_path : "";
  cfg.calib_path = self->calib_path ? self->calib_path : "";
  cfg.decoder_engine = self->decoder_engine ? self->decoder_engine : "";
  cfg.save_dir = self->save_dir ? self->save_dir : "logs";
  cfg.sequence_name = self->sequence_name ? self->sequence_name : "mast3rslam";
  cfg.save_results = self->save_results;
  cfg.conf_threshold = self->conf_threshold;
  cfg.gpu_id = self->gpu_id;
  cfg.baseline_m = self->baseline;
  cfg.loop_closure = self->loop_closure;
  cfg.loop_sim_thresh = self->loop_sim_thresh;
  try {
    self->core = new mast3r_slam::Mast3rSlamCore(cfg);
    if (!self->core->start()) {
      GST_ERROR_OBJECT(self, "SLAM core failed to start");
      return FALSE;
    }
  } catch (const std::exception &e) {
    GST_ERROR_OBJECT(self, "SLAM core init error: %s", e.what());
    return FALSE;
  }
  self->frame_num = 0;
  self->stereo_seen = FALSE;
  return TRUE;
}

static gboolean gst_nvdsmast3rslam_stop(GstBaseTransform *trans) {
  GstNvDsMast3rSlam *self = GST_NVDSMAST3RSLAM(trans);
  if (self->core) self->core->finish();
  return TRUE;
}

static gboolean gst_nvdsmast3rslam_set_caps(GstBaseTransform *trans,
                                            GstCaps *incaps, GstCaps *outcaps) {
  GstNvDsMast3rSlam *self = GST_NVDSMAST3RSLAM(trans);
  self->video_info_valid = gst_video_info_from_caps(&self->video_info, incaps);
  return self->video_info_valid;
}

static gboolean gst_nvdsmast3rslam_sink_event(GstBaseTransform *trans,
                                              GstEvent *event) {
  GstNvDsMast3rSlam *self = GST_NVDSMAST3RSLAM(trans);
  if (GST_EVENT_TYPE(event) == GST_EVENT_EOS && self->core) {
    self->core->finish();
  }
  return GST_BASE_TRANSFORM_CLASS(parent_class)->sink_event(trans, event);
}

/* Find an output layer of the tensor meta by name; returns device pointer. */
static const void *find_layer(NvDsInferTensorMeta *tm, const char *name,
                              NvDsInferDims *dims_out) {
  for (unsigned int i = 0; i < tm->num_output_layers; ++i) {
    NvDsInferLayerInfo *li = &tm->output_layers_info[i];
    if (li->layerName && g_strcmp0(li->layerName, name) == 0) {
      if (dims_out) *dims_out = li->inferDims;
      return tm->out_buf_ptrs_dev[i];
    }
  }
  return nullptr;
}

static void pose_release_func(gpointer data, gpointer /*user_data*/) {
  if (data) {
    NvDsUserMeta *um = (NvDsUserMeta *)data;
    g_free(um->user_meta_data);
    um->user_meta_data = nullptr;
  }
}

static gpointer pose_copy_func(gpointer data, gpointer /*user_data*/) {
  NvDsUserMeta *um = (NvDsUserMeta *)data;
  NvDsMast3rSlamPoseMeta *src = (NvDsMast3rSlamPoseMeta *)um->user_meta_data;
  NvDsMast3rSlamPoseMeta *dst =
      (NvDsMast3rSlamPoseMeta *)g_malloc0(sizeof(NvDsMast3rSlamPoseMeta));
  *dst = *src;
  return dst;
}

static void attach_pose_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                             const mast3r_slam::PoseResult &pose) {
  NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_meta) return;
  NvDsMast3rSlamPoseMeta *pm =
      (NvDsMast3rSlamPoseMeta *)g_malloc0(sizeof(NvDsMast3rSlamPoseMeta));
  pm->frame_id = pose.frame_id;
  pm->timestamp = pose.timestamp;
  for (int k = 0; k < 3; ++k) pm->t[k] = pose.t[k];
  for (int k = 0; k < 4; ++k) pm->q[k] = pose.q[k];
  pm->scale = pose.scale;
  pm->num_keyframes = pose.num_keyframes;
  pm->is_keyframe = pose.is_keyframe;
  pm->mode = (Mast3rSlamMode)pose.mode;

  user_meta->user_meta_data = pm;
  user_meta->base_meta.meta_type = (NvDsMetaType)NVDS_MAST3R_SLAM_POSE_META;
  user_meta->base_meta.copy_func = pose_copy_func;
  user_meta->base_meta.release_func = pose_release_func;
  nvds_add_user_meta_to_frame(frame_meta, user_meta);
}

static void cloud_release_func(gpointer data, gpointer /*user_data*/) {
  if (!data) return;
  NvDsUserMeta *um = (NvDsUserMeta *)data;
  NvDsMast3rSlamCloudMeta *cm = (NvDsMast3rSlamCloudMeta *)um->user_meta_data;
  if (cm) {
    g_free(cm->points);
    g_free(cm);
  }
  um->user_meta_data = nullptr;
}

static gpointer cloud_copy_func(gpointer data, gpointer /*user_data*/) {
  NvDsUserMeta *um = (NvDsUserMeta *)data;
  NvDsMast3rSlamCloudMeta *src = (NvDsMast3rSlamCloudMeta *)um->user_meta_data;
  NvDsMast3rSlamCloudMeta *dst =
      (NvDsMast3rSlamCloudMeta *)g_malloc0(sizeof(NvDsMast3rSlamCloudMeta));
  dst->keyframe_id = src->keyframe_id;
  dst->num_points = src->num_points;
  size_t bytes = (size_t)src->num_points * 3 * sizeof(float);
  dst->points = (float *)g_malloc(bytes);
  memcpy(dst->points, src->points, bytes);
  return dst;
}

/* Attach the newest keyframe's world-frame cloud (for nvdsmast3rviz / ROS). */
static void attach_cloud_meta(GstNvDsMast3rSlam *self, NvDsBatchMeta *batch_meta,
                              NvDsFrameMeta *frame_meta) {
  std::vector<float> xyz;
  uint64_t kf_id = 0;
  if (!self->core->copyLatestKeyframeCloud(self->cloud_max_points, xyz, kf_id))
    return;
  NvDsUserMeta *um = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!um) return;
  NvDsMast3rSlamCloudMeta *cm =
      (NvDsMast3rSlamCloudMeta *)g_malloc0(sizeof(NvDsMast3rSlamCloudMeta));
  cm->keyframe_id = kf_id;
  cm->num_points = (uint32_t)(xyz.size() / 3);
  cm->points = (float *)g_malloc(xyz.size() * sizeof(float));
  memcpy(cm->points, xyz.data(), xyz.size() * sizeof(float));
  um->user_meta_data = cm;
  um->base_meta.meta_type = (NvDsMetaType)NVDS_MAST3R_SLAM_CLOUD_META;
  um->base_meta.copy_func = cloud_copy_func;
  um->base_meta.release_func = cloud_release_func;
  nvds_add_user_meta_to_frame(frame_meta, um);
}

/* Build a FrameInput from a frame's encoder tensor meta. Returns FALSE if the
 * tensor meta (or feat/pos layers) is missing. */
static gboolean build_frame_input(GstNvDsMast3rSlam *self, GstBuffer *buf,
                                  NvDsFrameMeta *frame_meta, double ts,
                                  mast3r_slam::FrameInput *fin) {
  NvDsInferTensorMeta *tensor_meta = nullptr;
  for (NvDsMetaList *u = frame_meta->frame_user_meta_list; u != nullptr;
       u = u->next) {
    NvDsUserMeta *um = (NvDsUserMeta *)u->data;
    if (um->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
      NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *)um->user_meta_data;
      if (tm->unique_id == self->infer_gie_id) {
        tensor_meta = tm;
        break;
      }
    }
  }
  if (!tensor_meta) return FALSE;

  NvDsInferDims feat_dims{}, pos_dims{};
  const void *feat_dev = find_layer(tensor_meta, "feat", &feat_dims);
  const void *pos_dev = find_layer(tensor_meta, "pos", &pos_dims);
  if (!feat_dev || !pos_dev) {
    GST_WARNING_OBJECT(self, "encoder tensor meta missing feat/pos layers");
    return FALSE;
  }

  fin->timestamp = ts;
  fin->feat_dev = (const float *)feat_dev;
  fin->pos_dev = pos_dev; /* int32 in the engine; core casts to long */
  /* feat dims: (N, 1024); pos dims: (N, 2). nvinfer drops the batch dim. */
  fin->feat_n = (feat_dims.numDims >= 2) ? feat_dims.d[feat_dims.numDims - 2] : 0;
  fin->feat_dim = (feat_dims.numDims >= 1) ? feat_dims.d[feat_dims.numDims - 1] : 0;
  fin->pos_n = (pos_dims.numDims >= 2) ? pos_dims.d[pos_dims.numDims - 2] : 0;
  fin->model_w = tensor_meta->network_info.width;
  fin->model_h = tensor_meta->network_info.height;
  fin->gst_buffer = buf; /* for optional color extraction */
  fin->frame_meta = frame_meta;
  return TRUE;
}

/* --------------------------------------------------------------- per buffer */
static GstFlowReturn gst_nvdsmast3rslam_transform_ip(GstBaseTransform *trans,
                                                     GstBuffer *buf) {
  GstNvDsMast3rSlam *self = GST_NVDSMAST3RSLAM(trans);
  if (!self->core) return GST_FLOW_OK;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) {
    GST_WARNING_OBJECT(self, "no NvDsBatchMeta on buffer (need nvstreammux)");
    return GST_FLOW_OK;
  }

  double ts = (GST_BUFFER_PTS_IS_VALID(buf))
                  ? (double)GST_BUFFER_PTS(buf) / (double)GST_SECOND
                  : (double)self->frame_num / 30.0;

  /* Explicit MONO: every batch frame feeds the (single) SLAM track. */
  if (self->stereo_mode == GST_NVDSMAST3RSLAM_MODE_MONO) {
    for (NvDsMetaList *l = batch_meta->frame_meta_list; l != nullptr;
         l = l->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l->data;
      mast3r_slam::FrameInput fin{};
      if (!build_frame_input(self, buf, frame_meta, ts, &fin)) continue;

      mast3r_slam::PoseResult pose;
      try {
        pose = self->core->process(fin);
      } catch (const std::exception &e) {
        GST_WARNING_OBJECT(self, "frame processing error: %s", e.what());
        continue;
      }
      if (pose.valid) {
        attach_pose_meta(batch_meta, frame_meta, pose);
        if (pose.is_keyframe && self->emit_cloud)
          attach_cloud_meta(self, batch_meta, frame_meta);
      }
    }
    self->frame_num++;
    return GST_FLOW_OK;
  }

  /* AUTO / STEREO: pair the batch frames by source-id. The left frame drives
   * SLAM, the right one (if present) anchors the metric scale from the
   * baseline (DESIGN-STEREO.md); a batch without the right frame degrades to
   * mono, so one code path serves both camera setups. */
  mast3r_slam::FrameInput left{}, rightf{};
  gboolean have_l = FALSE, have_r = FALSE;
  NvDsFrameMeta *left_fm = nullptr;
  NvDsFrameMeta *sole_fm = nullptr;
  guint n_frames = 0;
  for (NvDsMetaList *l = batch_meta->frame_meta_list; l != nullptr;
       l = l->next) {
    NvDsFrameMeta *fm = (NvDsFrameMeta *)l->data;
    ++n_frames;
    sole_fm = fm;
    if ((gint)fm->source_id == self->left_source_id) {
      have_l = build_frame_input(self, buf, fm, ts, &left);
      left_fm = fm;
    } else if ((gint)fm->source_id == self->right_source_id) {
      have_r = build_frame_input(self, buf, fm, ts, &rightf);
    }
  }

  /* AUTO with a single-camera pipeline whose source-id is not the configured
   * left id (e.g. the camera sits on nvstreammux sink_1): before any stereo
   * evidence a single-frame batch IS the mono camera — process it as such.
   * Once a batch has carried both ids, a lone right frame means the left
   * frame of a stereo batch was dropped and must not join the left track. */
  if (self->stereo_mode == GST_NVDSMAST3RSLAM_MODE_AUTO && !self->stereo_seen &&
      n_frames == 1 && !have_l) {
    have_l = build_frame_input(self, buf, sole_fm, ts, &left);
    left_fm = sole_fm;
    have_r = FALSE;
  }

  if (have_l) {
    if (have_r) self->stereo_seen = TRUE;
    mast3r_slam::PoseResult pose;
    try {
      pose = have_r ? self->core->processStereo(left, rightf)
                    : self->core->process(left);
    } catch (const std::exception &e) {
      GST_WARNING_OBJECT(self, "frame processing error: %s", e.what());
      self->frame_num++;
      return GST_FLOW_OK;
    }
    if (!have_r && self->stereo_seen)
      GST_LOG_OBJECT(self, "right frame missing; processed mono");
    if (pose.valid) {
      attach_pose_meta(batch_meta, left_fm, pose);
      if (pose.is_keyframe && self->emit_cloud)
        attach_cloud_meta(self, batch_meta, left_fm);
    }
  } else if (have_r) {
    GST_LOG_OBJECT(self, "left frame missing; skipped right-only batch");
  } else if (n_frames > 0) {
    GST_WARNING_OBJECT(self,
                       "batch has %u frame(s) but none matches left-source-id="
                       "%d / right-source-id=%d; nothing processed",
                       n_frames, self->left_source_id, self->right_source_id);
  }

  self->frame_num++;
  return GST_FLOW_OK;
}

/* --------------------------------------------------------------- plugin */
#include "gstnvdsmast3rviz.h"

static gboolean plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_nvdsmast3rslam_debug, "nvdsmast3rslam", 0,
                          "MASt3R-SLAM DeepStream element");
  gboolean ok = gst_element_register(plugin, "nvdsmast3rslam", GST_RANK_PRIMARY,
                                     GST_TYPE_NVDSMAST3RSLAM);
  ok &= nvdsmast3rviz_register(plugin);
  return ok;
}

#ifndef PACKAGE
#define PACKAGE "nvdsmast3rslam"
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsmast3rslam,
                  "MASt3R-SLAM DeepStream plugin",
                  plugin_init, "1.0", "Proprietary", "mast3r-slam",
                  "https://github.com/rmurai0610/MASt3R-SLAM")
