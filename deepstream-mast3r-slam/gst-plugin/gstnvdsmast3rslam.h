/*
 * gstnvdsmast3rslam.h — DeepStream GStreamer element that turns the MASt3R
 * encoder tensors produced by gst-nvinfer into a SLAM trajectory + map.
 *
 * Structured after the open-source gst-nvinfer / gst-dsexample skeletons that
 * ship with the DeepStream SDK (sources/gst-plugins/). It is a GstBaseTransform
 * in passthrough-in-place mode: it never alters the video, it only consumes the
 * upstream NvDsInferTensorMeta and attaches NvDsMast3rSlamPoseMeta.
 */
#ifndef GST_NVDSMAST3RSLAM_H
#define GST_NVDSMAST3RSLAM_H

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <memory>

/* Forward declaration of the C++ SLAM core (PIMPL keeps libtorch out of the
 * GObject header). */
namespace mast3r_slam {
class Mast3rSlamCore;
}

G_BEGIN_DECLS

#define GST_TYPE_NVDSMAST3RSLAM (gst_nvdsmast3rslam_get_type())
#define GST_NVDSMAST3RSLAM(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NVDSMAST3RSLAM, GstNvDsMast3rSlam))
#define GST_NVDSMAST3RSLAM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_NVDSMAST3RSLAM, GstNvDsMast3rSlamClass))
#define GST_IS_NVDSMAST3RSLAM(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_NVDSMAST3RSLAM))
#define GST_IS_NVDSMAST3RSLAM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_NVDSMAST3RSLAM))

typedef struct _GstNvDsMast3rSlam GstNvDsMast3rSlam;
typedef struct _GstNvDsMast3rSlamClass GstNvDsMast3rSlamClass;

struct _GstNvDsMast3rSlam {
  GstBaseTransform parent;

  /* properties */
  guint infer_gie_id;        /* gie-unique-id of the upstream nvinfer (encoder) */
  gchar *config_path;        /* SLAM yaml config                                */
  gchar *calib_path;         /* intrinsics yaml (optional)                      */
  gchar *decoder_engine;     /* TensorRT engine for the decoder+heads           */
  gchar *save_dir;           /* output dir for .txt/.ply/keyframes              */
  gchar *sequence_name;      /* base output file name                           */
  gboolean save_results;     /* write outputs on EOS                            */
  gdouble conf_threshold;    /* confidence filter for the .ply                  */
  gint gpu_id;               /* CUDA device                                     */

  /* stereo-hybrid mode (DESIGN-STEREO.md) */
  gboolean stereo_mode;      /* pair batch frames; metric scale from baseline   */
  gdouble baseline;          /* stereo baseline, meters                          */
  gint left_source_id;       /* nvstreammux source-id of the left camera         */
  gint right_source_id;      /* nvstreammux source-id of the right camera        */
  gboolean loop_closure;     /* retrieval + factor graph + global GN             */
  gdouble loop_sim_thresh;   /* cosine threshold for loop candidates             */

  /* negotiated video info */
  GstVideoInfo video_info;
  gboolean video_info_valid;

  /* C++ SLAM core (PIMPL) */
  mast3r_slam::Mast3rSlamCore *core;
  guint64 frame_num;
};

struct _GstNvDsMast3rSlamClass {
  GstBaseTransformClass parent_class;
};

GType gst_nvdsmast3rslam_get_type(void);

G_END_DECLS

#endif /* GST_NVDSMAST3RSLAM_H */
