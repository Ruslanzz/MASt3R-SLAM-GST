# DeepStream MASt3R-SLAM — плагин на C/C++ на основе gst-nvinfer

Этот вариант реализует MASt3R-SLAM как **нативный C/C++ плагин DeepStream**,
построенный вокруг официального `gst-nvinfer`. В отличие от Python-варианта
(ветка `claude/gstreamer-cuda-tensorrt-plugin-g29qxg`), здесь нейросеть гоняется
через TensorRT силами `gst-nvinfer`, а вся обвязка — на C++.

---

## 1. Почему gst-nvinfer + отдельный C++ элемент

`gst-nvinfer` — это штатный DeepStream-элемент для инференса через TensorRT. Он
умеет: загружать/строить `.engine`, препроцессить кадр на GPU, гонять сеть и
**отдавать сырые выходные тензоры** как `NvDsInferTensorMeta`
(`output-tensor-meta=1`, `network-type=100 // other`). Это ровно то, что нужно
для «прогнать модель внутри GStreamer-плагина с GPU».

Но MASt3R — **двухвидовая** сеть, а `gst-nvinfer` обрабатывает **один** кадр за
раз. Поэтому сеть разрезается на две части:

```
                 nvstreammux(batch=1)
                        │  NV12/RGBA (NVMM)
                        ▼
   ┌───────────────────────────────────────────────┐
   │ gst-nvinfer  (MASt3R ViT-энкодер, TensorRT)     │  ← официальный плагин
   │   in : image 1×3×H×W                            │
   │   out: feat 1×N×1024, pos 1×N×2  (tensor-meta)  │  output-tensor-meta=1
   └───────────────────────┬─────────────────────────┘
                           │  NvDsInferTensorMeta (feat, pos)
                           ▼
   ┌───────────────────────────────────────────────┐
   │ nvdsmast3rslam  (наш C++ элемент)               │  ← разрабатывается здесь
   │  1. decoder+DPT-головы (TensorRT, 2 прогона)    │   feat_i,feat_j → X,C,D,Q
   │  2. matching  (CUDA-ядра репозитория)           │
   │  3. Sim3-трекинг (Gauss-Newton, CUDA/libtorch)  │
   │  4. factor-graph backend (CUDA-ядра)            │
   │  → NvDsUserMeta(pose) + по EOS .txt/.ply/keyfr. │
   └───────────────────────┬─────────────────────────┘
                           ▼  кадр без изменений
                 nvosd / encoder / sink (официальные плагины)
```

Почему именно так:

* **Энкодер** — самый тяжёлый и самый «статичный» блок (1 прогон/кадр, фикс.
  разрешение 512, фикс. число токенов). Идеально ложится в `gst-nvinfer`: один
  вход → сырые тензоры. Это и есть «на основе gst-nvinfer».
* **Декодер** асимметричный и прогоняется **дважды** на пару (в двух порядках) —
  один `gst-nvinfer`-инстанс так не настроить, поэтому декодер выполняется внутри
  нашего элемента отдельным TRT-движком (класс `TrtEngine`).
* **Геометрия SLAM** (матчинг, Sim3-трекинг, глобальная оптимизация факторного
  графа, ретривер) — это собственные CUDA-ядра репозитория
  (`gn_kernels.cu`, `matching_kernels.cu`) и `lietorch`. Они **компилируются в
  состав нашего C++ элемента** (через libtorch C++ API) и переиспользуются как
  есть — переписывать их на TensorRT невозможно (это не сеть, а оптимизация).

> То же разделение «энкодер на TRT, остальное на CUDA» обосновано в Python-ветке
> (раздел CUDA vs TensorRT). Здесь оно реализовано нативно и с использованием
> штатного `gst-nvinfer` для инференса.

## 2. Почему C/C++ (а не Python)

* `gst-nvinfer`, `nvstreammux`, `nvds*`-метаданные — это C/C++ API DeepStream.
  Нативный элемент интегрируется с ними напрямую (читает `NvDsInferTensorMeta`,
  пишет `NvDsUserMeta`), без Python-загрузчика и без GIL.
* Нулевое копирование: тензоры энкодера уже на GPU (в meta как device-указатели);
  декодер и SLAM работают прямо на GPU, без перегонки в системную память.
* Производительность и совместимость с финальным DeepStream-конвейером.

## 3. Соответствие выходу репозитория

* **Онлайн**: поза текущего кадра кладётся в `NvDsUserMeta` фрейма (тип
  `NVDS_MAST3R_SLAM_POSE_META`, см. `mast3r_slam_meta.h`): `frame_id`,
  `timestamp`, `t (3)`, `q (4)`, `scale`, флаг ключевого кадра, режим. Это можно
  читать pad-probe'ом downstream или штатными DeepStream-средствами.
* **Файлы по EOS**: `<seq>.txt` (TUM `t x y z qx qy qz qw`), `<seq>.ply` (цветное
  облако), PNG ключевых кадров — теми же формулами, что и `mast3r_slam.evaluate`
  (порт в `lib/io.cpp`).

## 4. Поток данных и метаданные

1. `gst-nvinfer` (gie-unique-id = 1) кладёт `NvDsInferTensorMeta` на каждый
   `NvDsFrameMeta` (т.к. работаем на полном кадре, `process-mode=1`,
   `network-type=100`, `output-tensor-meta=1`).
2. `nvdsmast3rslam` в `transform_ip`:
   * находит `NvDsInferTensorMeta` своего gie-id, достаёт `feat`, `pos`
     (`NvDsInferLayerInfo.buffer`, device-память);
   * выполняет декодер (TRT) для пары (текущий кадр ↔ опорный ключевой кадр);
   * матчинг + Sim3-трекинг + (в фоне) backend;
   * добавляет `NvDsUserMeta(pose)`; кадр уходит дальше без изменений.

## 5. Вход

Монокулярный поток (как и в Python-ветке: MASt3R-SLAM монокулярный, «пару»
строит сам алгоритм). Источник любой: v4l2 / файл / UDP-RTSP — через
`nvurisrcbin`/`uridecodebin3`/`v4l2src` → `nvstreammux(batch-size=1)`.

## 6. Сборка (эта ветка — GTX 1660 Ti / DeepStream 7.1)

Эта ветка нацелена на **NVIDIA GeForce GTX 1660 Ti** (Turing, sm_75, 6 ГБ).
Базовый образ — **DeepStream 7.1** (GStreamer 1.20, CUDA 12.6, TensorRT 10.3):
это самая новая DeepStream, ещё поддерживающая Turing на x86 dGPU. `gst-nvinfer`
входит в DeepStream, пересобирать его не нужно. Собираем только наш элемент +
SLAM-ядро (CMake, `-DCMAKE_CUDA_ARCHITECTURES=75`, линкуясь с GStreamer,
DeepStream `nvds_meta`/`nvdsinfer`, TensorRT, CUDA и libtorch). Всё — в Docker
на базе DeepStream 7.1 (см. `docker/Dockerfile`, `BUILD.md`).

## 7. Ограничения / статус

* Это **нативный DeepStream-проект**: его нельзя собрать/проверить без SDK
  DeepStream 7.1, TensorRT, CUDA и libtorch на целевой машине с GPU. Код написан
  по конвенциям gst-nvinfer/`dsexample` и снабжён комментариями о точках сборки.
* На 6 ГБ 1660 Ti практический дефолт — **FP16**-движки (FP32 может не влезть).
  Turing (TU116) без тензорных ядер → FP16 экономит память, но не даёт кратного
  ускорения; на ViT-Large ждите скромный FPS.
* Бит-в-бит совпадение с эталоном недостижимо при FP16/INT8 TRT-движках
  (численность меняется). Для максимальной близости стройте движки в FP32.
* SLAM-ядро переиспользует CUDA-ядра репозитория через libtorch C++ API;
  оркестрация (matching/tracker/global_opt) портирована в `lib/`. Перед боевым
  использованием требуется сборка и валидация на целевом GPU.
