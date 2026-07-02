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
    -e NVIDIA_DRIVER_CAPABILITIES=all --network host \
    $(for d in /dev/video*; do echo --device=$d; done) \
    -v "$PWD:/opt/MASt3R-SLAM-GST" -w /opt/MASt3R-SLAM-GST \
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

### 3.1. Поставить Python-зависимости MASt3R (для экспорта ONNX)

Экспорт импортирует `mast3r_slam.mast3r_utils`, которому нужны submodule MASt3R и
скомпилированный backend репозитория:

```bash
cd /opt/MASt3R-SLAM-GST
pip install -e thirdparty/mast3r
pip install --no-build-isolation -e .          # собирает CUDA-расширение (нужен nvcc)
```

### 3.2. Экспорт ONNX (энкодер + декодер)

```bash
python deepstream-mast3r-slam/tools/export_onnx.py \
    --checkpoint checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --height 384 --width 512 --which both
# -> checkpoints/mast3r_encoder.onnx, checkpoints/mast3r_decoder.onnx
```

`--height/--width` должны совпадать с разрешением входа модели после ресайза
(для 4:3 это обычно 384×512). Те же значения стоят в
`configs/config_infer_mast3r_encoder.txt` (`infer-dims=3;384;512`) и в
`pipelines/*.sh` (`MUX_W/MUX_H`).

### 3.3. Сборка движков (trtexec)

```bash
TRTEXEC=/usr/src/tensorrt/bin/trtexec
# На 6 ГБ 1660 Ti собираем в FP16 (вдвое меньше памяти движка/активаций) и
# ограничиваем рабочую область, чтобы сборка уложилась в VRAM.
$TRTEXEC --onnx=checkpoints/mast3r_encoder.onnx \
         --saveEngine=checkpoints/mast3r_encoder.engine \
         --fp16 --memPoolSize=workspace:2048
$TRTEXEC --onnx=checkpoints/mast3r_decoder.onnx \
         --saveEngine=checkpoints/mast3r_decoder.engine \
         --fp16 --memPoolSize=workspace:2048
# (FP32 — ближе к эталону, но требует заметно больше VRAM; на 6 ГБ может не влезть.)
```

> `network-mode` в `configs/config_infer_mast3r_encoder.txt` уже стоит `2` (FP16),
> чтобы совпадать с FP16-движком энкодера.

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

```bash
cd /opt/MASt3R-SLAM-GST

# видеофайл
bash deepstream-mast3r-slam/pipelines/run_file.sh /path/to/video.mp4 myseq

# v4l2-камера
bash deepstream-mast3r-slam/pipelines/run_v4l2.sh /dev/video0

# сеть: RTP/H264 поверх UDP (порт 5000) или RTSP
bash deepstream-mast3r-slam/pipelines/run_udp.sh udp 5000
bash deepstream-mast3r-slam/pipelines/run_udp.sh rtsp rtsp://host/stream
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
| `out of memory` / OOM на 1660 Ti | стройте движки в FP16 (шаг 3.3), уменьшите `--memPoolSize=workspace`, снизьте `MUX_W/MUX_H`, прореживайте кадры (см. §5b) |
| `no kernel image is available` / незапуск ядер | движок/расширение собраны не под sm_75; пересоберите с `-DCMAKE_CUDA_ARCHITECTURES=75` и `TORCH_CUDA_ARCH_LIST=7.5` |

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
