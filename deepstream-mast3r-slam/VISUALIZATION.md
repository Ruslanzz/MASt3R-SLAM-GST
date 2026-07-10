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
# универсальный скрипт (моно И стерео): VIZ=udp включает оверлей + UDP-стрим
VIZ=udp VIEW_HOST=<IP ноутбука> bash deepstream-mast3r-slam/pipelines/run_slam.sh \
    /dev/video0 /dev/video1 0.12          # или один источник для моно
# либо специализированный стерео-скрипт:
VIEW_HOST=<IP ноутбука> bash deepstream-mast3r-slam/pipelines/run_stereo_viz.sh \
    /dev/video0 /dev/video1 0.12
# на ноутбуке:
gst-launch-1.0 udpsrc port=5600 \
  caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! \
  rtpjitterbuffer ! rtph264depay ! avdec_h264 ! autovideosink sync=false
```

(порт 5600 совместим с QGroundControl: Settings → Video → UDP h.264, 5600.)

## 1b. Окно на самой машине (sink пайплайна)

Отдельный sink-элемент не нужен: оверлей уже врисован в кадры (`nvdsmast3rviz`
→ display-meta → `nvdsosd`), достаточно завершить пайплайн оконным синком.
Готовые скрипты:

```bash
# на хосте, один раз — разрешить X11 из контейнера:
xhost +local:

# контейнер запускать с пробросом X11 (добавить к обычной команде docker run):
#   -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix

# универсальный скрипт: VIZ=window, моно или стерео по числу источников
VIZ=window bash deepstream-mast3r-slam/pipelines/run_slam.sh /path/video.mp4
VIZ=window bash deepstream-mast3r-slam/pipelines/run_slam.sh /dev/video0 /dev/video1 0.12

# либо специализированные скрипты — видеофайл в окно:
bash deepstream-mast3r-slam/pipelines/run_file_display.sh /path/video.mp4
# стерео в окно (лево|право плиткой):
bash deepstream-mast3r-slam/pipelines/run_stereo_display.sh /dev/video0 /dev/video1 0.12
```

Выбор синка — переменной `SINK`:

| `SINK` | Элемент | Когда использовать |
|---|---|---|
| (не задан) | `autovideosink` → xv/ximagesink | **дефолт**: чистый X11, работает на Optimus-ноутбуках и хостах без модуля `nvidia_modeset` (наш случай); кадр копируется в системную память — при 512×384 и 1–2 FPS незаметно |
| `egl` | `nveglglessink` | полный дисплейный стек NVIDIA: загружен `nvidia_modeset`, контейнер с `NVIDIA_DRIVER_CAPABILITIES=...,graphics,display` |
| `xv` | `xvimagesink` | явный XVideo |

Типичные ошибки: `Could not initialise EGL` / `nvidia-modeset` → используйте
дефолтный X11-синк; `cannot open display` → не проброшен `DISPLAY`/X11-сокет
или не выполнен `xhost +local:`.

## 2. Стрим в ROS 2 → RViz (полный 3D: карта + траектория)

Единственный способ смотреть **плотную 3D-карту и трек живьём** (в окне
пайплайна — только видео с HUD и 2D-мини-картой). Публикуется:

| Топик / канал | Тип | Когда |
|---|---|---|
| `/mast3r/odom` | `nav_msgs/Odometry` | каждый кадр |
| TF `map → base_link` | `tf2` | каждый кадр |
| `/mast3r/path` | `nav_msgs/Path` | на ключевых кадрах |
| `/mast3r/map` | `sensor_msgs/PointCloud2` (xyz, метры) | облако свежего ключевого кадра |

### 2.1. Собрать образ с ROS-мостом

Мост компилируется опционально (в образ ставится ROS 2 Humble — родной для
Ubuntu 22.04 базового образа DS 7.1 — **вместе с rviz2**, так что смотреть
карту можно прямо из этого же контейнера, см. §2.4):

```bash
WITH_ROS2=1 bash deepstream-mast3r-slam/docker/build.sh
# то же самое вручную: docker build ... --build-arg WITH_ROS2=1 ...
```

### 2.2. Запустить пайплайн с публикацией

Контейнер должен быть запущен с `--network host` (он уже есть в канонической
команде из BUILD.md §2 — DDS-обнаружению нужна сеть хоста). При желании
задайте домен: `-e ROS_DOMAIN_ID=0`.

```bash
# внутри контейнера: ROS=true включает ros-enable=true у nvdsmast3rviz
ROS=true bash deepstream-mast3r-slam/pipelines/run_slam.sh /dev/video0 /dev/video1 0.12
ROS=true bash deepstream-mast3r-slam/pipelines/run_slam.sh /path/video.mp4   # моно
# ROS можно совмещать с картинкой: VIZ=udp ROS=true ... / VIZ=window ROS=true ...
# а VIZ=rviz (§2.4) сразу поднимает RViz рядом с пайплайном в этом контейнере
```

### 2.3. RViz на ноутбуке / другой машине в той же сети

На машине с Ubuntu 22.04:

```bash
sudo apt install ros-humble-desktop        # либо минимум: ros-humble-rviz2
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0                     # тот же, что у контейнера

ros2 topic list                            # должны появиться /mast3r/odom|path|map
ros2 topic hz /mast3r/odom                 # проверка, что данные идут

rviz2 -d deepstream-mast3r-slam/configs/mast3r_slam.rviz
```

Готовый конфиг `configs/mast3r_slam.rviz` уже включает: Grid, TF
(`map → base_link`), **Trajectory** (Path `/mast3r/path`, бирюзовая линия),
**Odometry** (`/mast3r/odom`, текущая поза осями XYZ), **Map** (PointCloud2
`/mast3r/map`, раскраска по высоте (ось Y), **Decay Time = 3600 с** — облака
ключевых кадров накапливаются в полную карту). Fixed Frame = `map`.

Если настраиваете RViz вручную: Fixed Frame = `map`, затем Add → By topic →
`/mast3r/path` (Path), `/mast3r/map` (PointCloud2: Color Transformer =
**AxisColor** — в облаке только xyz, Decay Time = 3600), `/mast3r/odom`
(Odometry: Keep = 1, Shape = Axes), TF.

### 2.4. RViz прямо из контейнера MASt3R (`VIZ=rviz`)

Самый короткий путь, если смотреть надо на той же машине: образ собран с
`WITH_ROS2=1` (rviz2 уже внутри), контейнер запущен с пробросом X11 (§1b), и
один запуск поднимает и пайплайн, и окно RViz с готовым конфигом.

**Одна камера — одной командой** (готовый скрипт, RViz открывается сам):

```bash
bash deepstream-mast3r-slam/pipelines/run_mono_rviz.sh /dev/video0
# [seq] — необязательное имя выходных файлов; по умолчанию /dev/video0
```

Скрипт заранее проверяет, что `/dev/videoN` — именно video-capture, а не
metadata-узел (частая ловушка UVC-камер, когда вторая нода отдаёт только
метаданные), и подсказывает `v4l2-ctl --list-devices`, если выбран не тот узел.

Общий вариант (моно/стерео, любой источник) — тот же режим через `VIZ=rviz`:

```bash
VIZ=rviz bash deepstream-mast3r-slam/pipelines/run_slam.sh /dev/video0            # моно
VIZ=rviz bash deepstream-mast3r-slam/pipelines/run_slam.sh /dev/video0 /dev/video2 0.12  # стерео
VIZ=rviz bash deepstream-mast3r-slam/pipelines/run_slam.sh /path/video.mp4        # файл
```

`VIZ=rviz` сам включает `ros-enable=true`; rviz2 закрывается вместе со
скриптом. Свой конфиг — `RVIZ_CFG=/path/my.rviz`. Рендер в контейнере
программный (Mesa, без NVIDIA `graphics`-capability) — на частотах SLAM
(1–3 FPS) это незаметно; если окно RViz падает по OpenGL, добавьте
`LIBGL_ALWAYS_SOFTWARE=1`.

### 2.5. RViz без установки ROS на машину (docker)

```bash
xhost +local:
docker run --rm -it --network host \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e ROS_DOMAIN_ID=0 \
    -v "$PWD/deepstream-mast3r-slam/configs:/cfg:ro" \
    osrf/ros:humble-desktop \
    rviz2 -d /cfg/mast3r_slam.rviz
```

(официальный образ сам подхватывает окружение ROS через entrypoint; работает и
на той же машине, где крутится SLAM-контейнер, — DDS находит узлы через
`--network host`.)

### 2.6. Если топиков не видно

* `ROS_DOMAIN_ID` должен совпадать с обеих сторон (не задан = 0);
* `ROS_LOCALHOST_ONLY` не должен быть `1` ни с одной стороны;
* SLAM-контейнер запущен с `--network host`;
* обе машины в одной подсети, мультикаст не зарезан (VPN и «изоляция клиентов»
  на Wi-Fi-точках ломают DDS-discovery — проще всего проводная сеть/одна точка);
* если мост молчит уже в контейнере (нет строки про ROS в логе элемента) —
  образ собран без `WITH_ROS2=1` либо не включён `ros-enable=true`.

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
