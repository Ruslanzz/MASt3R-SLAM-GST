# Визуализация и ROS 2: элемент `nvdsmast3rviz`

Второй элемент плагина (регистрируется тем же `libnvdsmast3rslam.so`). Ставится
**после** `nvdsmast3rslam`, видео пропускает без изменений, читает меты позы
(`NvDsMast3rSlamPoseMeta`) и карты (`NvDsMast3rSlamCloudMeta`) и выдаёт их двумя
каналами:

```
nvdsmast3rslam ─▶ nvdsmast3rviz ─▶ nvmultistreamtiler ─▶ nvdsosd ─▶ h264 ─▶ udpsink
                     │  (display meta: HUD + мини-карта траектории)
                     └─▶ ROS 2: /mast3r/odom, /mast3r/path, /mast3r/map, TF
```

## 1. Смотреть глазами (работает на headless-хосте)

Оверлей (HUD: поза в метрах, режим, счётчик ключевых кадров; мини-карта
траектории X/Z в углу) рисуется через `NvDsDisplayMeta` → отрисовывает штатный
`nvdsosd`, а видео кодируется и стримится по UDP:

```bash
VIEW_HOST=<IP ноутбука> bash deepstream-mast3r-slam/pipelines/run_stereo_viz.sh \
    /dev/video0 /dev/video1 0.12
# на ноутбуке:
gst-launch-1.0 udpsrc port=5600 \
  caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! \
  rtpjitterbuffer ! rtph264depay ! avdec_h264 ! autovideosink sync=false
```

(порт 5600 совместим с QGroundControl: Settings → Video → UDP h.264, 5600.)

## 2. Стрим в ROS 2 (полный 3D в RViz)

Мост компилируется опционально: соберите образ с `--build-arg WITH_ROS2=1`
(ставится ROS 2 Humble — родной для Ubuntu 22.04 базового образа DS 7.1) и
включите свойством `ros-enable=true` (в скрипте — `ROS=true`).

Публикуется:

| Топик / канал | Тип | Когда |
|---|---|---|
| `/mast3r/odom` | `nav_msgs/Odometry` | каждый кадр |
| TF `map → base_link` | `tf2` | каждый кадр |
| `/mast3r/path` | `nav_msgs/Path` | на ключевых кадрах |
| `/mast3r/map` | `sensor_msgs/PointCloud2` | облако свежего ключевого кадра (мир, метры) |

RViz на другой машине в той же сети (`--network host` уже проброшен): Fixed
Frame = `map`, добавьте Path, PointCloud2, TF. Один и тот же `ROS_DOMAIN_ID` с
обеих сторон.

Свойства `nvdsmast3rviz`: `overlay` (true), `ros-enable` (false), `frame-id`
("map"), `child-frame-id` ("base_link"), `topic-prefix` ("/mast3r").
У `nvdsmast3rslam` добавлены `emit-cloud` (true) и `cloud-max-points` (50000) —
выдача карты ключевого кадра метой для viz/ROS.

## Замечания

* Координаты — «сырые» камерные (x вправо, y вниз, z вперёд) в системе первого
  ключевого кадра; преобразование в ROS-конвенцию (ENU/FLU) при необходимости —
  static_transform_publisher поверх `map`.
* Без `WITH_ROS2` элемент собирается и работает (overlay); `ros-enable=true`
  лишь пишет предупреждение.
* PointCloud2 шлёт облако **последнего** ключевого кадра (не всю карту целиком)
  — в RViz поставьте Decay Time побольше (например, 3600 с), чтобы карта
  накапливалась.
