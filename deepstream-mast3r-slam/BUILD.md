# Сборка и запуск C/C++ плагина `nvdsmast3rslam` — GTX 1660 Ti / DeepStream 7.1

Пошаговая инструкция по сборке нативного DeepStream-плагина (ветка
`claude/gstreamer-nvinfer-cpp-gtx1660ti-g29qxg`) и запуску MASt3R-SLAM на
видеокарте **NVIDIA GeForce GTX 1660 Ti (6 ГБ)**.

> **Почему DeepStream 7.1.** GTX 1660 Ti — это архитектура **Turing (compute
> capability 7.5)**, без тензорных ядер. Самая новая DeepStream, которая ещё
> поддерживает Turing на x86 dGPU, — **DeepStream 7.1** (Ubuntu 22.04, GStreamer
> 1.20, CUDA 12.6, TensorRT 10.3, драйвер ≥ 560). Более новые релизы (8.0+) могут
> не заявлять поддержку Turing, поэтому 7.1 — безопасный максимум для этой карты.
> Если в матрице поддержки более новой DeepStream Turing/sm_75 всё же присутствует,
> просто поднимите `BASE_IMAGE` — архитектура CUDA остаётся 75.

> Плагин — нативный C/C++, собирается **только** в окружении с DeepStream 7.1,
> TensorRT, CUDA и libtorch на машине с GPU NVIDIA. Ниже — путь через Docker
> (рекомендуется) и альтернатива «вручную в готовом контейнере».

---

## 0. Требования

* **NVIDIA GeForce GTX 1660 Ti** (Turing, sm_75, 6 ГБ) — или другой Turing-GPU.
* Драйвер NVIDIA **≥ 560** (требование DeepStream 7.1).
* Docker + **nvidia-container-toolkit** (`--gpus all` / `--runtime nvidia`).
* Доступ к базовому образу DeepStream 7.1 в NGC
  (`nvcr.io/nvidia/deepstream:7.1-triton-multiarch`).
* ~25–30 ГБ места (образ + libtorch + движки).

---

## 1. Получить код и веса

```bash
git clone https://github.com/Ruslanzz/MASt3R-SLAM-GST.git
cd MASt3R-SLAM-GST
git checkout claude/gstreamer-nvinfer-cpp-gtx1660ti-g29qxg
git submodule update --init --recursive      # eigen + mast3r/dust3r нужны для сборки/экспорта

# веса MASt3R (нужен как минимум metric-чекпойнт для экспорта ONNX)
mkdir -p checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
# (для будущего loop-closure также:)
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

---

## 2. Собрать Docker-образ (компилирует плагин)

Из корня репозитория:

```bash
bash deepstream-mast3r-slam/docker/build.sh
```

Что делает образ (`deepstream-mast3r-slam/docker/Dockerfile`):
* базовый образ — **DeepStream 7.1** (GStreamer 1.20, CUDA 12.6, TensorRT 10.3);
* ставит `cmake/ninja/pkg-config`, dev-пакеты GStreamer, Eigen, libtorch
  (pip-колесо `torch==2.5.1` под CUDA, с ядрами Turing sm_75);
* через CMake собирает `libnvdsmast3rslam.so` под **sm_75**
  (`-DCMAKE_CUDA_ARCHITECTURES=75`), **переиспользуя CUDA-ядра репозитория**
  (`mast3r_slam/backend/src/*.cu`);
* кладёт `.so` в путь плагинов GStreamer и проверяет `gst-inspect-1.0
  nvdsmast3rslam`.

Параметры (необязательно):

```bash
# другой базовый образ / arch (по умолчанию DS 7.1, sm_75)
BASE_IMAGE=nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
CUDA_ARCH=75 \
IMAGE=nvdsmast3rslam:ds7.1-gtx1660ti \
bash deepstream-mast3r-slam/docker/build.sh
```

Запустить контейнер с GPU, камерами и смонтированным репозиторием:

```bash
docker run --rm -it --gpus all --runtime nvidia \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video --network host \
    $(for d in /dev/video*; do echo --device=$d; done) \
    -v "$PWD:/opt/MASt3R-SLAM-GST" -w /opt/MASt3R-SLAM-GST \
    nvdsmast3rslam:ds7.1-gtx1660ti bash
```

Если планируете смотреть результат в окне на этой же машине
(`pipelines/*_display.sh`, см. `VISUALIZATION.md` §1b), добавьте проброс X11:

```bash
xhost +local:        # один раз на хосте
docker run ... \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    nvdsmast3rslam:ds7.1-gtx1660ti bash
```

Проверка внутри контейнера:

```bash
gst-inspect-1.0 nvdsmast3rslam      # должен показать свойства элемента
```

### 2b. Пересборка после правок (внутри контейнера, без пересборки образа)

```bash
bash deepstream-mast3r-slam/build_local.sh
# GStreamer plugin dir (DeepStream 7.1 uses the system dir, not lib/gstreamer-1.0):
cp deepstream-mast3r-slam/build/libnvdsmast3rslam.so \
   "$(pkg-config --variable=pluginsdir gstreamer-1.0)/"
gst-inspect-1.0 nvdsmast3rslam
```

---

## 3. Подготовить TensorRT-движки

Нужны два движка: **энкодер** (его гоняет `gst-nvinfer`) и **декодер** (его
гоняет наш элемент). Все шаги — внутри контейнера.

### 3.1. Python-зависимости MASt3R — уже в образе

Начиная с текущей версии `docker/Dockerfile`, обе установки выполняются **при
сборке образа** (сборка CUDA-кода без GPU включается через `FORCE_CUDA=1`,
который поддержан в `setup.py`):

```dockerfile
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e thirdparty/mast3r && \
    FORCE_CUDA=1 python3 -m pip install --no-cache-dir --no-build-isolation -e .
```

Вручную в контейнере ничего ставить не нужно. (Если вы монтируете репозиторий
поверх `/opt/MASt3R-SLAM-GST` через `-v`, editable-установки продолжают
работать: пакеты ссылаются на тот же путь.)

> Примечание: `--no-build-isolation` обязателен — curope (CUDA-RoPE из CroCo),
> lietorch и backend репозитория импортируют torch в `setup.py`. Если curope не
> соберётся — не блокер: CroCo откатится на PyTorch-реализацию RoPE, что для
> ONNX-экспорта даже предпочтительнее.

### 3.2–3.3. Экспорт ONNX и сборка движков — один скрипт

Движки **нельзя** подготовить при сборке образа: `trtexec` требует живой GPU, а
готовый `.engine` привязан к конкретной карте и версии TensorRT. Поэтому этот
шаг выполняется один раз **внутри контейнера на машине с 1660 Ti**:

```bash
cd /opt/MASt3R-SLAM-GST
bash deepstream-mast3r-slam/tools/build_engines.sh          # fp16, 384x512 (дефолт)
# варианты: build_engines.sh fp32   |   build_engines.sh fp16 384 512
```

Скрипт делает по порядку:
1. проверяет наличие чекпойнта `checkpoints/MASt3R_..._metric.pth` (если нет —
   печатает команду `wget`);
2. запускает `tools/export_onnx.py` → `checkpoints/mast3r_encoder.onnx` и
   `checkpoints/mast3r_decoder.onnx` (пропускает, если файлы уже есть);
3. находит `trtexec` (`/usr/src/tensorrt/bin`, `/opt/tensorrt/bin` или PATH;
   можно указать явно: `TRTEXEC=/path/to/trtexec bash ...`);
4. собирает оба движка: `--fp16 --memPoolSize=workspace:2048` (FP16 и лимит
   workspace 2 ГБ — чтобы уложиться в 6 ГБ VRAM). Каждый движок строится
   несколько минут — это нормально;
5. в конце показывает `ls -lh` готовых `.engine`.

Результат (используется на шаге 4):
* `checkpoints/mast3r_encoder.engine` — путь прописан в
  `configs/config_infer_mast3r_encoder.txt` (`model-engine-file=...`);
* `checkpoints/mast3r_decoder.engine` — дефолт `DEC_ENGINE` в `pipelines/*.sh`.

`--height/--width` (дефолт 384×512) должны совпадать с
`infer-dims=3;384;512` в конфиге nvinfer и `MUX_W/MUX_H` в пайплайнах.
Повторять шаг нужно только при смене GPU, версии TensorRT или разрешения
(перед пересборкой удалите старые `.onnx`/`.engine`).

> FP32 ближе к эталону, но требует заметно больше VRAM — на 6 ГБ может не
> влезть. `network-mode=2` (FP16) в конфиге nvinfer уже согласован с
> FP16-движком энкодера.

Важно: имена входов/выходов ONNX **должны совпадать** с конфигом и кодом:
* энкодер: выходы `feat`, `pos` (см. `output-blob-names=feat;pos`);
* декодер: входы `feat1,pos1,feat2,pos2`; выходы
  `pts3d_1,conf_1,desc_1,desc_conf_1, pts3d_2,conf_2,desc_2,desc_conf_2`
  (см. `lib/mast3r_slam_core.cpp::runDecoder`).

---

## 4. Запуск

Из корня репозитория, внутри контейнера. Энкодер-движок берётся из конфига
nvinfer (`configs/config_infer_mast3r_encoder.txt`), декодер — из свойства
элемента (`DEC_ENGINE`, по умолчанию `checkpoints/mast3r_decoder.engine`).

**Универсальный скрипт** `run_slam.sh` — один вход для МОНО и СТЕРЕО (элемент
работает в `stereo-mode=auto` и сам определяет режим по батчу); источник —
файл, `/dev/videoN`, `rtsp://` или `udp://:порт`; визуализация — переменной
`VIZ` (`none`|`window`|`udp`|`rviz`), ROS 2 — `ROS=true` (пошаговая инструкция
по RViz — образ с `WITH_ROS2=1`, запуск, готовый конфиг
`configs/mast3r_slam.rviz` — в `VISUALIZATION.md` §2; `VIZ=rviz` поднимает RViz
прямо из этого контейнера рядом с пайплайном):

```bash
cd /opt/MASt3R-SLAM-GST

# МОНО: файл / камера / сеть
bash deepstream-mast3r-slam/pipelines/run_slam.sh /path/to/video.mp4 myseq
bash deepstream-mast3r-slam/pipelines/run_slam.sh /dev/video0
bash deepstream-mast3r-slam/pipelines/run_slam.sh rtsp://host/stream
bash deepstream-mast3r-slam/pipelines/run_slam.sh udp://:5000

# СТЕРЕО (метрическая траектория + loop closure, см. DESIGN-STEREO.md):
bash deepstream-mast3r-slam/pipelines/run_slam.sh /dev/video0 /dev/video1 0.12
bash deepstream-mast3r-slam/pipelines/run_slam.sh left.mp4 right.mp4 0.12 offroad

# с визуализацией: окно (нужен проброс X11, шаг 2) или UDP-стрим оверлея
VIZ=window bash deepstream-mast3r-slam/pipelines/run_slam.sh /dev/video0 /dev/video1 0.12
VIZ=udp VIEW_HOST=<IP ноутбука> bash deepstream-mast3r-slam/pipelines/run_slam.sh /path/video.mp4
# 3D-карта + трек в RViz из этого же контейнера (образ с WITH_ROS2=1 + X11):
VIZ=rviz bash deepstream-mast3r-slam/pipelines/run_slam.sh /dev/video0 /dev/video2 0.12
# ОДНА КАМЕРА + RViz одной командой (RViz открывается автоматически):
bash deepstream-mast3r-slam/pipelines/run_mono_rviz.sh /dev/video0
```

Специализированные скрипты (эквивалентные конфигурации, для справки):

```bash
# видеофайл
bash deepstream-mast3r-slam/pipelines/run_file.sh /path/to/video.mp4 myseq

# v4l2-камера
bash deepstream-mast3r-slam/pipelines/run_v4l2.sh /dev/video0

# сеть: RTP/H264 поверх UDP (порт 5000) или RTSP
bash deepstream-mast3r-slam/pipelines/run_udp.sh udp 5000
bash deepstream-mast3r-slam/pipelines/run_udp.sh rtsp rtsp://host/stream

# стерео-гибрид (явный stereo-mode=stereo)
bash deepstream-mast3r-slam/pipelines/run_stereo_v4l2.sh /dev/video0 /dev/video1 0.12

# визуализация в окно (нужен проброс X11 в docker run, см. шаг 2 и VISUALIZATION.md §1b):
bash deepstream-mast3r-slam/pipelines/run_file_display.sh /path/to/video.mp4
bash deepstream-mast3r-slam/pipelines/run_stereo_display.sh /dev/video0 /dev/video1 0.12
# стрим оверлея по UDP/RTP на другую машину (headless-хост):
VIEW_HOST=<IP ноутбука> bash deepstream-mast3r-slam/pipelines/run_stereo_viz.sh /dev/video0 /dev/video1 0.12
```

Переопределение параметров пайплайна:

```bash
MUX_W=512 MUX_H=384 \
DEC_ENGINE=/opt/MASt3R-SLAM-GST/checkpoints/mast3r_decoder.engine \
bash deepstream-mast3r-slam/pipelines/run_file.sh /data/clip.mp4 clip
```

Эквивалентный «голый» конвейер (что собирают скрипты):

```
source → nvvideoconvert → 'video/x-raw(memory:NVMM),format=RGBA'
       → nvstreammux(batch-size=1,width,height)
       → nvinfer config-file-path=.../config_infer_mast3r_encoder.txt
       → nvdsmast3rslam infer-gie-id=1 decoder-engine=.../mast3r_decoder.engine
                        save-dir=logs sequence-name=<seq>
       → nvvideoconvert → fakesink
```

---

## 5. Результаты

По завершении (EOS — для файла; Ctrl-C с `-e` — для камеры/сети) в `logs/`:

* `logs/<seq>.txt` — траектория в формате TUM: `t tx ty tz qx qy qz qw`;
* `logs/<seq>.ply` — облако точек (цвет точек — TODO, пока нейтрально-серый;
  траектория заполнена полностью).

**Онлайн-поза** каждого кадра кладётся в `NvDsUserMeta` типа
`NVDS_MAST3R_SLAM_POSE_META` (структура `NvDsMast3rSlamPoseMeta`,
`gst-plugin/mast3r_slam_meta.h`): `frame_id, timestamp, t[3], q[4], scale,
num_keyframes, is_keyframe, mode`. Читается pad-probe'ом на src-паде элемента.

---

## 5b. Память на 6 ГБ (GTX 1660 Ti)

6 ГБ — жёсткое ограничение. Что помогает уложиться:

* **FP16-движки** энкодера и декодера (см. шаг 3.3) + `network-mode=2` в конфиге —
  вдвое меньше памяти под веса и активации.
* **Разрешение входа 384×512** (дефолт). Не увеличивайте без запаса по VRAM.
* **Длина последовательности.** Ключевые кадры копятся в памяти (на каждый —
  тензоры `feat` ~3 МБ, `X_canon`/`C` ~2–3 МБ). На длинных прогонах это растёт до
  нескольких ГБ. Рекомендации:
  * прореживайте вход (обрабатывайте не каждый кадр);
  * для длинных сцен снижайте `MUX_W/MUX_H` или дробите запись на отрезки;
  * закрывайте прочие процессы, занимающие VRAM (`nvidia-smi`).
* Мониторьте: `watch -n1 nvidia-smi`. При `out of memory` — уменьшите
  `--memPoolSize=workspace` при сборке движка и/или разрешение.

---

## 6. Свойства элемента `nvdsmast3rslam`

| Свойство | По умолчанию | Назначение |
|----------|--------------|-----------|
| `infer-gie-id` | `1` | `gie-unique-id` апстрим-`nvinfer` (энкодера) |
| `decoder-engine` | `""` | TensorRT-движок декодера+голов |
| `config` | `config/base.yaml` | конфиг SLAM (зарезервирован; дефолты вшиты) |
| `calib` | `""` | YAML интринсик (калиброванный режим) |
| `save-dir` / `sequence-name` | `logs` / `mast3rslam` | путь/имя выходных файлов |
| `save-results` | `true` | сохранять `.txt`/`.ply` по EOS |
| `conf-threshold` | `1.5` | порог уверенности для `.ply` |
| `gpu-id` | `0` | устройство CUDA |
| `stereo-mode` | `auto` | `auto` — режим определяется по батчу (один кадр → моно, лево+право → стерео); `mono`/`stereo` — принудительно (DESIGN-STEREO.md §4) |
| `baseline` | `0.12` | база стереопары, **метры** (задаёт метрический масштаб) |
| `left-source-id` / `right-source-id` | `0` / `1` | source-id левой/правой камеры в `nvstreammux` |
| `loop-closure` | `true` | ретривер + факторный граф + глобальная GN |
| `loop-sim-thresh` | `0.90` | порог косинусной близости для loop-кандидатов |
| `emit-cloud` | `true` | прикладывать облако ключевого кадра метой (для `nvdsmast3rviz`/ROS) |
| `cloud-max-points` | `50000` | прореживание выдаваемого облака |

---

## 7. Диагностика

| Симптом | Причина / решение |
|---------|-------------------|
| `gst-inspect-1.0 nvdsmast3rslam` пусто | `.so` не в пути плагинов — скопируйте в `$(pkg-config --variable=pluginsdir gstreamer-1.0)` (в DS 7.1 каталога `lib/gstreamer-1.0` нет). Если `ldd libnvdsmast3rslam.so` показывает ненайденный libtorch — добавьте `.../torch/lib` в `LD_LIBRARY_PATH` или в `/etc/ld.so.conf.d` + `ldconfig` |
| `no NvDsBatchMeta on buffer` | перед элементом обязателен `nvstreammux` |
| `encoder tensor meta missing feat/pos` | имена выходов ONNX ≠ `feat;pos`, либо `infer-gie-id` ≠ `gie-unique-id` в конфиге nvinfer |
| `failed to load decoder engine` | не задан `decoder-engine` или путь неверный; пересоберите движок под текущую версию TensorRT |
| CMake не находит Torch | задайте `-DTorch_DIR=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake","Torch"))')` |
| Линковка: нет `nvinfer`/`nvds_*` | поправьте `-DDEEPSTREAM_DIR=` и `-DTENSORRT_DIR=` в CMake под вашу установку |
| `mount error: stat failed: /dev/nvidia-modeset` при `docker run` | capability `display` (входит в `NVIDIA_DRIVER_CAPABILITIES=all`) требует модуль ядра `nvidia_modeset`, которого нет на headless-хостах. Используйте `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` (дисплей в контейнере не нужен) или `sudo modprobe nvidia_modeset` на хосте |
| `no element "nvvideoconvert"` / `"nvinfer"` в рантайме | реестр GStreamer с занесёнными в blacklist NVIDIA-плагинами (какой-то разбор плагинов прошёл без доступа к GPU — сборка образа, `docker run` без `--gpus all`, или гонка с драйвером). **В образах с entrypoint это лечится само**: при старте контейнера `mast3r-entrypoint.sh` проверяет `nvvideoconvert` и, если тот не грузится, чистит `~/.cache/gstreamer-1.0` и пересканирует уже с GPU. Если образ старый (без entrypoint) — вручную `rm -rf ~/.cache/gstreamer-1.0` и повторите. Если и после этого не грузится — контейнер запущен без `--gpus all`/`--runtime nvidia` |
| `out of memory` / OOM на 1660 Ti | стройте движки в FP16 (шаг 3.3), уменьшите `--memPoolSize=workspace`, снизьте `MUX_W/MUX_H`, прореживайте кадры (см. §5b) |
| `no kernel image is available` / незапуск ядер | движок/расширение собраны не под sm_75; пересоберите с `-DCMAKE_CUDA_ARCHITECTURES=75` и `TORCH_CUDA_ARCH_LIST=7.5` |
| Сборка образа падает на `apt-get update`: `repository 'https://librealsense.intel.com/... ' is not signed` (`NO_PUBKEY`) | базовый образ DeepStream несёт apt-источник Intel RealSense с протухшим GPG-ключом. RealSense нам не нужен. В слое `WITH_ROS2` источник удаляется автоматически. Если то же падает на **базовом** `apt` (полностью чистая сборка без кэша) — удалите источник вручную перед сборкой или добавьте ту же очистку в первый `RUN apt` Dockerfile: `grep -rlE 'librealsense' /etc/apt/sources.list.d/ \| xargs -r rm -f` |

---

## 8. Замечания по точности (паритет с эталоном)

* FP16/INT8-движки меняют численность → траектория дрейфует от референса.
  Для близости к репозиторию нужен **FP32**, но на 6 ГБ 1660 Ti FP32 может не
  влезть — практический дефолт здесь **FP16** (компромисс «работает vs. паритет»).
  Для точной оценки метрик прогоняйте FP32 на GPU с бо́льшим объёмом памяти.
* На Turing (TU116) **нет тензорных ядер**, поэтому FP16 даёт в основном экономию
  памяти, а не кратный прирост скорости. Ожидайте невысокий FPS на ViT-Large —
  прореживайте вход при необходимости.
* Ресайз `gst-nvinfer` не повторяет «длинная сторона 512 + центр-кроп» из
  MASt3R. Для фиксированной камеры задайте `infer-dims`/`MUX_W/H` под уже
  отресайзенный вход (или предварительно кропайте).
* Перед боевым использованием провалидируйте на целевом GPU: Sim3-ретракцию
  (`lib/sim3.h`), имена входов/выходов декодера, раскладку дескрипторов.
  Глобальный backend (factor-graph) и loop-closure (ASMK) помечены как точки
  интеграции в `lib/mast3r_slam_core.cpp`.
