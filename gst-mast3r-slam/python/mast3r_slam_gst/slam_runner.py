"""In-process MASt3R-SLAM driver for the GStreamer element.

This is a faithful port of the reference ``main.py`` event loop into a class so
it can be driven one frame at a time from a GStreamer streaming thread. It reuses
the unchanged ``mast3r_slam`` modules (tracker, factor graph, CUDA kernels,
retrieval) and the unchanged ``mast3r_slam.evaluate`` saver, so the persisted
trajectory / point cloud / keyframes are identical to the repository.

Differences from ``main.py`` (none of which alter the SLAM numerics):

* the multiprocessing ``Manager`` is replaced by a lightweight in-process shim
  (``_ThreadManager``) since everything runs in a single process;
* the back-end global optimisation runs in a ``threading.Thread`` instead of a
  separate process (it shares the same CUDA tensors directly);
* there is no 3D viewer; the per-frame pose is returned to the element which
  posts it on the GStreamer bus.
"""

from __future__ import annotations

import threading
import time

import numpy as np


class _MgrValue:
    """Stand-in for ``multiprocessing.Manager().Value`` used in a single process."""

    def __init__(self, value):
        self.value = value


class _ThreadManager:
    """Provides the subset of the ``Manager`` API that the shared structures use.

    ``SharedStates`` / ``SharedKeyframes`` only need ``RLock``, ``Value`` and
    ``list``; in a single process plain threading primitives are sufficient and
    avoid spawning a manager server process inside the GStreamer plugin.
    """

    def RLock(self):
        return threading.RLock()

    def Value(self, typecode, value):
        return _MgrValue(value)

    def list(self):
        return []


class Mast3rSlamRunner:
    def __init__(
        self,
        config_path="config/base.yaml",
        calib_path="",
        checkpoint="",
        retrieval_checkpoint="",
        backend="torch",
        trt_encoder_engine="",
        device="cuda:0",
        save_dir="logs",
        sequence_name="mast3rslam",
        save_results=True,
        conf_threshold=1.5,
        backend_thread=True,
    ):
        self.config_path = config_path
        self.calib_path = calib_path
        self.checkpoint = checkpoint
        self.retrieval_checkpoint = retrieval_checkpoint
        self.backend_kind = backend
        self.trt_encoder_engine = trt_encoder_engine
        self.device = device
        self.save_dir = save_dir
        self.sequence_name = sequence_name
        self.save_results = save_results
        self.conf_threshold = conf_threshold
        self.use_backend_thread = backend_thread

        self.model = None
        self.keyframes = None
        self.states = None
        self.tracker = None
        self.intrinsics = None
        self.K = None
        self.img_size = 512
        self.frame_count = 0
        self.timestamps = []
        self._initialized = False
        self._backend = None
        self._backend_started = False
        self._terminate = False

        # filled in start()
        self._Mode = None
        self._lietorch = None
        self._torch = None

    # ------------------------------------------------------------------ setup
    def start(self):
        """Load config + model. Heavy imports happen here, not at plugin scan."""
        import torch
        import lietorch
        import yaml

        from mast3r_slam.config import load_config, config, set_global_config
        from mast3r_slam.dataloader import Intrinsics
        from mast3r_slam.frame import Mode

        self._torch = torch
        self._lietorch = lietorch
        self._Mode = Mode

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)

        load_config(self.config_path)
        self._config = config

        # Calibration (optional). Must set use_calib before building Intrinsics.
        if self.calib_path:
            with open(self.calib_path, "r") as f:
                calib = yaml.load(f, Loader=yaml.SafeLoader)
            config["use_calib"] = True
            self.intrinsics = Intrinsics.from_calib(
                self.img_size,
                calib["width"],
                calib["height"],
                calib["calibration"],
            )

        # Build the model through the selected backend (CUDA by default).
        from mast3r_slam_gst.backends import make_backend

        backend = make_backend(self.backend_kind)
        if self.backend_kind.lower().startswith(("tensorrt", "trt")):
            backend.configure(self.trt_encoder_engine)
        self.model = backend.load_model(self.checkpoint, self.device)

    def _lazy_init_from_frame(self, img_rgb):
        """Allocate shared structures once we know the (resized) image size."""
        import torch
        from mast3r_slam.config import config
        from mast3r_slam.frame import SharedKeyframes, SharedStates
        from mast3r_slam.mast3r_utils import resize_img
        from mast3r_slam.tracker import FrameTracker

        resized = resize_img(img_rgb, self.img_size)
        h, w = resized["img"][0].shape[1:]

        self.keyframes = SharedKeyframes(_ThreadManager(), h, w, device=self.device)
        self.states = SharedStates(_ThreadManager(), h, w, device=self.device)

        if config["use_calib"] and self.intrinsics is not None:
            self.K = torch.from_numpy(self.intrinsics.K_frame).to(
                self.device, dtype=torch.float32
            )
            self.keyframes.set_intrinsics(self.K)

        self.tracker = FrameTracker(self.model, self.keyframes, self.device)

        if self.use_backend_thread:
            self._backend = threading.Thread(
                target=self._backend_loop, name="mast3r-slam-backend", daemon=True
            )
            self._backend.start()
            self._backend_started = True

        self._initialized = True

    # ----------------------------------------------------------- back-end loop
    def _backend_loop(self):
        """Port of ``main.py:run_backend`` running in a thread."""
        from mast3r_slam.config import config
        from mast3r_slam.frame import Mode
        from mast3r_slam.global_opt import FactorGraph
        from mast3r_slam.mast3r_utils import load_retriever

        factor_graph = FactorGraph(self.model, self.keyframes, self.K, self.device)
        retrieval_database = load_retriever(
            self.model, retriever_path=self.retrieval_checkpoint or None
        )
        self._retrieval_database = retrieval_database

        mode = self.states.get_mode()
        while mode is not Mode.TERMINATED:
            mode = self.states.get_mode()
            if mode == Mode.INIT or self.states.is_paused():
                time.sleep(0.01)
                continue
            if mode == Mode.RELOC:
                frame = self.states.get_frame()
                success = self._relocalization(
                    frame, factor_graph, retrieval_database
                )
                if success:
                    self.states.set_mode(Mode.TRACKING)
                self.states.dequeue_reloc()
                continue
            idx = -1
            with self.states.lock:
                if len(self.states.global_optimizer_tasks) > 0:
                    idx = self.states.global_optimizer_tasks[0]
            if idx == -1:
                time.sleep(0.01)
                continue

            # Graph construction (consecutive + retrieval edges)
            kf_idx = []
            n_consec = 1
            for j in range(min(n_consec, idx)):
                kf_idx.append(idx - 1 - j)
            frame = self.keyframes[idx]
            retrieval_inds = retrieval_database.update(
                frame,
                add_after_query=True,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds

            lc_inds = set(retrieval_inds)
            lc_inds.discard(idx - 1)
            if len(lc_inds) > 0:
                print("Database retrieval", idx, ": ", lc_inds)

            kf_idx = set(kf_idx)
            kf_idx.discard(idx)
            kf_idx = list(kf_idx)
            frame_idx = [idx] * len(kf_idx)
            if kf_idx:
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )

            with self.states.lock:
                self.states.edges_ii[:] = factor_graph.ii.cpu().tolist()
                self.states.edges_jj[:] = factor_graph.jj.cpu().tolist()

            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()

            with self.states.lock:
                if len(self.states.global_optimizer_tasks) > 0:
                    self.states.global_optimizer_tasks.pop(0)

    def _relocalization(self, frame, factor_graph, retrieval_database):
        """Port of ``main.py:relocalization``."""
        from mast3r_slam.config import config

        keyframes = self.keyframes
        with keyframes.lock:
            kf_idx = []
            retrieval_inds = retrieval_database.update(
                frame,
                add_after_query=False,
                k=config["retrieval"]["k"],
                min_thresh=config["retrieval"]["min_thresh"],
            )
            kf_idx += retrieval_inds
            successful_loop_closure = False
            if kf_idx:
                keyframes.append(frame)
                n_kf = len(keyframes)
                kf_idx = list(kf_idx)
                frame_idx = [n_kf - 1] * len(kf_idx)
                print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
                if factor_graph.add_factors(
                    frame_idx,
                    kf_idx,
                    config["reloc"]["min_match_frac"],
                    is_reloc=config["reloc"]["strict"],
                ):
                    retrieval_database.update(
                        frame,
                        add_after_query=True,
                        k=config["retrieval"]["k"],
                        min_thresh=config["retrieval"]["min_thresh"],
                    )
                    print("Success! Relocalized")
                    successful_loop_closure = True
                    keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
                else:
                    keyframes.pop_last()
                    print("Failed to relocalize")

            if successful_loop_closure:
                if config["use_calib"]:
                    factor_graph.solve_GN_calib()
                else:
                    factor_graph.solve_GN_rays()
            return successful_loop_closure

    # --------------------------------------------------------------- per frame
    def process(self, timestamp_s, img_rgb):
        """Feed one RGB float image (``HxWx3`` in ``[0,1]``). Returns pose info.

        Returns ``dict`` with keys ``frame_id, timestamp, T_WC, mode,
        num_keyframes, is_keyframe`` or ``None`` if the frame was skipped before
        a pose exists.
        """
        import lietorch
        from mast3r_slam.config import config
        from mast3r_slam.frame import Mode, create_frame
        from mast3r_slam.mast3r_utils import mast3r_inference_mono

        if not self._initialized:
            self._lazy_init_from_frame(img_rgb)

        # Apply calibration undistort the same way the dataloader does.
        if config["use_calib"] and self.intrinsics is not None:
            img_rgb = self.intrinsics.remap(img_rgb)

        i = self.frame_count
        self.timestamps.append(float(timestamp_s))

        mode = self.states.get_mode()

        T_WC = (
            lietorch.Sim3.Identity(1, device=self.device)
            if i == 0
            else self.states.get_frame().T_WC
        )
        frame = create_frame(
            i, img_rgb, T_WC, img_size=self.img_size, device=self.device
        )

        is_keyframe = False

        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X_init, C_init)
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            self.states.set_mode(Mode.TRACKING)
            self.states.set_frame(frame)
            is_keyframe = True
            self.frame_count += 1
            return self._pose_info(frame, Mode.TRACKING, is_keyframe)

        if mode == Mode.TRACKING:
            add_new_kf, _match_info, try_reloc = self.tracker.track(frame)
            if try_reloc:
                self.states.set_mode(Mode.RELOC)
            self.states.set_frame(frame)
        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            self.states.set_frame(frame)
            self.states.queue_reloc()
            add_new_kf = False
            while config["single_thread"]:
                with self.states.lock:
                    if self.states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)
        else:
            self.frame_count += 1
            return None

        if add_new_kf:
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            is_keyframe = True
            while config["single_thread"]:
                with self.states.lock:
                    if len(self.states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        cur_mode = self.states.get_mode()
        self.frame_count += 1
        return self._pose_info(frame, cur_mode, is_keyframe)

    def _pose_info(self, frame, mode, is_keyframe):
        return {
            "frame_id": frame.frame_id,
            "timestamp": self.timestamps[frame.frame_id],
            "T_WC": frame.T_WC,
            "mode": mode.name if hasattr(mode, "name") else str(mode),
            "num_keyframes": len(self.keyframes),
            "is_keyframe": is_keyframe,
        }

    # ------------------------------------------------------------------ finish
    def finish(self):
        """Terminate the back-end and persist the same outputs as the repo."""
        if self._terminate:
            return
        self._terminate = True

        if self._Mode is not None and self.states is not None:
            self.states.set_mode(self._Mode.TERMINATED)
        if self._backend_started and self._backend is not None:
            self._backend.join(timeout=30.0)

        if not (self.save_results and self.keyframes is not None):
            return
        if len(self.keyframes) == 0:
            return

        import pathlib
        import mast3r_slam.evaluate as eval_mod

        save_dir = pathlib.Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        seq = self.sequence_name
        traj_file = save_dir / f"{seq}.txt"
        recon_file = save_dir / f"{seq}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

        eval_mod.save_traj(save_dir, f"{seq}.txt", self.timestamps, self.keyframes)
        eval_mod.save_reconstruction(
            save_dir, f"{seq}.ply", self.keyframes, self.conf_threshold
        )
        eval_mod.save_keyframes(
            save_dir / "keyframes" / seq, self.timestamps, self.keyframes
        )
        print(f"[mast3rslam] saved trajectory + reconstruction to {save_dir}/{seq}.*")
