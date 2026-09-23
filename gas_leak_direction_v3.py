import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import scipy.fft as fft
import threading
from collections import deque


class GasPlumeAmplifierApp:
    """
    Gas plume analysis from ordinary RGB video.

    Pipeline
    --------
    1) Temporal chroma amplification / band-pass filtering
    2) Adaptive plume mask inside a Detection ROI
    3) Optional binary CNN leak classifier from ONNX
    4) Dense optical flow inside the plume mask
    5) 8-direction output + direction confidence
    6) Leak detection support from plume coverage

    IMPORTANT
    ---------
        - The displayed heatmap is a temporal chroma heatmap, NOT Grad-CAM.
      True Grad-CAM requires a specific trained CNN and access to its
      convolutional feature maps/gradients.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Gas Leak Detection - Direction Analysis")
        self.root.geometry("1180x840")

        # ------------------------------------------------------------
        # VIDEO
        # ------------------------------------------------------------
        self.video_path = None
        self.cap = None
        self.fps = 30.0
        self.num_cores = 8
        self.frame_width = 0
        self.frame_height = 0
        self.current_frame = None

        # ------------------------------------------------------------
        # TEMPORAL AMPLIFICATION
        # ------------------------------------------------------------
        self.levels = 4
        self.alpha = 50
        self.low_freq = 0.1
        self.high_freq = 1.0

        # ------------------------------------------------------------
        # ROI
        # detection_roi: where plume detection/direction analysis is allowed
        # reference_roi: optional area used for automatic frequency estimate
        # ------------------------------------------------------------
        self.detection_roi = None
        self.reference_roi = None
        self.roi_mode = None
        self.roi_start = None

        # ------------------------------------------------------------
        # MASK / PLUME PARAMETERS
        # ------------------------------------------------------------
        self.adaptive_mask = True
        self.fixed_mask_threshold = 40
        self.min_component_area = 25
        self.min_gas_pixels = 40
        self.min_coverage_for_gas = 0.15  # percent of Detection ROI

        # ------------------------------------------------------------
        # OPTICAL FLOW / DIRECTION
        # ------------------------------------------------------------
        self.flow_history = deque(maxlen=15)
        self.flow_min_magnitude = 0.12
        self.flow_max_magnitude = 15.0
        self.flow_alignment_cos = 0.50  # within ~60 degrees of dominant flow
        self.direction_axis_ratio = 1.50  # widen cardinal sectors

        self.last_valid_direction = "WAITING"
        self.last_direction_confidence = 0.0
        self.prev_gray = None
        self.prev_motion_mask = None
        self.no_gas_frames = 0
        self.direction_ttl_frames = 20

        # ------------------------------------------------------------
        # OPTIONAL CNN (ONNX)
        # Expected: binary classifier. For two outputs, class index 1 = leak.
        # For one output, interpreted as leak probability/logit.
        # ------------------------------------------------------------
        self.cnn_net = None
        self.cnn_model_path = None
        self.cnn_input_size = (224, 224)
        self.cnn_positive_class = 1
        self.cnn_threshold = 0.50
        self.cnn_conf_history = deque(maxlen=7)

        # ------------------------------------------------------------
        # PLAYBACK / RESULTS
        # ------------------------------------------------------------
        self.is_playing = False
        self.processed_frames = []
        self.frame_index = 0

        self.current_direction = "WAITING"
        self.current_direction_confidence = 0.0
        self.current_flow_speed = 0.0
        self.current_centroid = None
        self.current_leak_status = "UNKNOWN"
        self.current_leak_confidence = 0.0
        self.current_leak_source = "CV"

        self.setup_ui()

    # ================================================================
    # UI
    # ================================================================
    def setup_ui(self):
        control_frame = tk.Frame(self.root, pady=8)
        control_frame.pack(fill=tk.X)

        tk.Button(
            control_frame, text="Load Video", command=self.load_video
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            control_frame,
            text="Select Detection ROI",
            command=lambda: self.enable_roi_selection("detection")
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            control_frame,
            text="Select Reference ROI",
            command=lambda: self.enable_roi_selection("reference")
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            control_frame,
            text="Reset ROIs",
            command=self.reset_rois
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            control_frame,
            text="Load CNN (ONNX)",
            command=self.load_cnn_model
        ).pack(side=tk.LEFT, padx=4)

        self.btn_process = tk.Button(
            control_frame,
            text="Process Gas Plume",
            command=self.start_processing
        )
        self.btn_process.pack(side=tk.LEFT, padx=4)

        self.btn_play = tk.Button(
            control_frame,
            text="Play/Pause",
            command=self.toggle_play,
            state=tk.DISABLED
        )
        self.btn_play.pack(side=tk.LEFT, padx=4)

        # Parameter row 1
        param_frame = tk.Frame(self.root, pady=3)
        param_frame.pack(fill=tk.X)

        self.alpha_var = tk.IntVar(value=self.alpha)
        self.low_var = tk.DoubleVar(value=self.low_freq)
        self.high_var = tk.DoubleVar(value=self.high_freq)
        self.display_fps_var = tk.DoubleVar(value=30.0)
        self.adaptive_mask_var = tk.BooleanVar(value=self.adaptive_mask)

        tk.Label(param_frame, text="Alpha:").pack(side=tk.LEFT)
        tk.Entry(param_frame, textvariable=self.alpha_var, width=5).pack(
            side=tk.LEFT, padx=(4, 1)
        )
        tk.Scale(
            param_frame, from_=1, to=300, orient=tk.HORIZONTAL,
            variable=self.alpha_var, showvalue=False, length=115
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(param_frame, text="Low Hz:").pack(side=tk.LEFT)
        tk.Entry(param_frame, textvariable=self.low_var, width=5).pack(
            side=tk.LEFT, padx=(4, 1)
        )
        tk.Scale(
            param_frame, from_=0.1, to=15.0, resolution=0.1,
            orient=tk.HORIZONTAL, variable=self.low_var,
            showvalue=False, length=115
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(param_frame, text="High Hz:").pack(side=tk.LEFT)
        tk.Entry(param_frame, textvariable=self.high_var, width=5).pack(
            side=tk.LEFT, padx=(4, 1)
        )
        tk.Scale(
            param_frame, from_=0.2, to=30.0, resolution=0.1,
            orient=tk.HORIZONTAL, variable=self.high_var,
            showvalue=False, length=115
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(param_frame, text="Playback FPS:").pack(side=tk.LEFT)
        tk.Entry(param_frame, textvariable=self.display_fps_var, width=5).pack(
            side=tk.LEFT, padx=(4, 1)
        )
        tk.Scale(
            param_frame, from_=1, to=120, resolution=1,
            orient=tk.HORIZONTAL, variable=self.display_fps_var,
            showvalue=False, length=115
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Checkbutton(
            param_frame,
            text="Adaptive Mask",
            variable=self.adaptive_mask_var
        ).pack(side=tk.LEFT, padx=5)

        # Grid
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.canvas_orig = tk.Canvas(
            self.grid_frame, bg="gray20", width=500, height=320
        )
        self.canvas_orig.grid(row=0, column=0, padx=5, pady=5)
        self.canvas_orig.bind("<ButtonPress-1>", self.on_roi_press)
        self.canvas_orig.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas_orig.bind("<ButtonRelease-1>", self.on_roi_release)

        self.canvas_overlay = tk.Canvas(
            self.grid_frame, bg="gray20", width=500, height=320
        )
        self.canvas_overlay.grid(row=0, column=1, padx=5, pady=5)

        self.canvas_mask = tk.Canvas(
            self.grid_frame, bg="black", width=500, height=320
        )
        self.canvas_mask.grid(row=1, column=0, padx=5, pady=5)

        info_frame = tk.Frame(
            self.grid_frame, bg="white", width=500, height=320
        )
        info_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        info_frame.grid_propagate(False)

        self.lbl_info = tk.Label(
            info_frame,
            text="Load a video to begin.",
            font=("Arial", 14),
            bg="white",
            justify=tk.LEFT,
            anchor="w"
        )
        self.lbl_info.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    # ================================================================
    # VIDEO / MODEL LOADING
    # ================================================================
    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return

        if self.cap is not None:
            self.cap.release()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open the selected video.")
            return

        self.video_path = path
        self.cap = cap
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0

        self.display_fps_var.set(round(self.fps, 1))
        self.direction_ttl_frames = max(3, int(round(self.fps * 0.75)))

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "The video contains no readable frame.")
            return

        h, w = frame.shape[:2]
        max_w, max_h = 500, 320
        scale = min(max_w / w, max_h / h, 1.0)

        self.frame_width = max(1, int(round(w * scale)))
        self.frame_height = max(1, int(round(h * scale)))

        self.current_frame = cv2.resize(
            frame, (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_AREA
        )

        self.detection_roi = (0, 0, self.frame_width, self.frame_height)
        self.reference_roi = None
        self.processed_frames = []
        self.frame_index = 0
        self.is_playing = False
        self.btn_play.config(state=tk.DISABLED)

        self.display_frame_with_rois(self.current_frame)
        self.update_info()

    def load_cnn_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("ONNX model", "*.onnx")]
        )
        if not path:
            return

        try:
            net = cv2.dnn.readNetFromONNX(path)
            self.cnn_net = net
            self.cnn_model_path = path
            self.cnn_conf_history.clear()
            messagebox.showinfo(
                "CNN loaded",
                "ONNX model loaded.\n\n"
                "Expected binary output:\n"
                "- 2 outputs: [no_leak, leak]\n"
                "- 1 output: leak probability/logit\n\n"
                "Input preprocessing uses 224x224 RGB + ImageNet normalization."
            )
        except Exception as e:
            self.cnn_net = None
            self.cnn_model_path = None
            messagebox.showerror("CNN load error", str(e))

    # ================================================================
    # ROI
    # ================================================================
    def enable_roi_selection(self, mode):
        if self.current_frame is None:
            messagebox.showinfo("ROI", "Load a video first.")
            return
        self.roi_mode = mode
        self.roi_start = None
        self.display_frame_with_rois(self.current_frame)

    def reset_rois(self):
        if self.frame_width <= 0 or self.frame_height <= 0:
            return
        self.detection_roi = (0, 0, self.frame_width, self.frame_height)
        self.reference_roi = None
        self.roi_mode = None
        if self.current_frame is not None:
            self.display_frame_with_rois(self.current_frame)
        self.update_info()

    def _clamp_canvas_point(self, x, y):
        x = int(np.clip(x, 0, max(0, self.frame_width - 1)))
        y = int(np.clip(y, 0, max(0, self.frame_height - 1)))
        return x, y

    def on_roi_press(self, event):
        if self.roi_mode is None:
            return
        self.roi_start = self._clamp_canvas_point(event.x, event.y)

    def on_roi_drag(self, event):
        if self.roi_mode is None or self.roi_start is None:
            return

        x2, y2 = self._clamp_canvas_point(event.x, event.y)
        x1, y1 = self.roi_start

        self.canvas_orig.delete("selection_rect")
        color = "red" if self.roi_mode == "detection" else "yellow"
        self.canvas_orig.create_rectangle(
            x1, y1, x2, y2,
            outline=color, width=2, tag="selection_rect"
        )

    def on_roi_release(self, event):
        if self.roi_mode is None or self.roi_start is None:
            return

        x1, y1 = self.roi_start
        x2, y2 = self._clamp_canvas_point(event.x, event.y)

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        if w >= 10 and h >= 10:
            roi = (x, y, w, h)
            if self.roi_mode == "detection":
                self.detection_roi = roi
            else:
                self.reference_roi = roi

        self.roi_mode = None
        self.roi_start = None
        self.display_frame_with_rois(self.current_frame)
        self.update_info()

    def display_frame_with_rois(self, frame, centroid=None):
        if frame is None:
            return
        disp = frame.copy()

        if self.detection_roi is not None:
            x, y, w, h = self.detection_roi
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                disp, "DETECTION ROI", (x + 4, max(18, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA
            )

        if self.reference_roi is not None:
            x, y, w, h = self.reference_roi
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                disp, "REFERENCE ROI", (x + 4, max(18, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA
            )

        if centroid is not None:
            cv2.circle(disp, centroid, 6, (0, 255, 0), -1)

        self.display_image(disp, self.canvas_orig)

    def roi_to_mask(self, shape_hw, roi):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        if roi is None:
            mask[:, :] = 255
            return mask

        x, y, rw, rh = roi
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        x2 = int(np.clip(x + rw, 0, w))
        y2 = int(np.clip(y + rh, 0, h))
        if x2 > x and y2 > y:
            mask[y:y2, x:x2] = 255
        return mask

    # ================================================================
    # SIGNAL / MASK
    # ================================================================
    def build_gaussian_pyramid(self, frame, levels):
        pyramid = [frame]
        for _ in range(max(0, levels - 1)):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def estimate_roi_frequencies(self, frames):
        """Estimate a useful temporal band from Reference ROI if present,
        otherwise Detection ROI. Returns (low, high) or None.
        """
        roi = self.reference_roi or self.detection_roi
        if roi is None or len(frames) < 12:
            return None

        x, y, w, h = roi
        signal = []
        for frame in frames:
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            signal.append(float(np.mean(gray)))

        if len(signal) < 12:
            return None

        signal = np.asarray(signal, dtype=np.float32)
        signal -= np.mean(signal)

        spec = np.abs(fft.fft(signal))
        freqs = fft.fftfreq(len(signal), d=1.0 / self.fps)

        valid = (freqs > 0.05) & (freqs <= self.fps / 2.0)
        freqs = freqs[valid]
        spec = spec[valid]
        if spec.size == 0 or np.max(spec) <= 1e-9:
            return None

        # Ignore the very lowest bins if DC leakage dominates.
        peak_idx = int(np.argmax(spec))
        peak_amp = float(spec[peak_idx])
        sig_idx = np.where(spec >= peak_amp * 0.50)[0]
        if sig_idx.size == 0:
            return None

        low = max(0.1, float(freqs[sig_idx[0]]) * 0.8)
        high = min(self.fps / 2.0, float(freqs[sig_idx[-1]]) * 1.2)

        low = round(low, 2)
        high = round(high, 2)
        if high <= low:
            high = min(self.fps / 2.0, low + 0.5)
        return low, high

    def adaptive_threshold_value(self, norm_diff, roi_mask):
        vals = norm_diff[roi_mask > 0]
        if vals.size < 20:
            return self.fixed_mask_threshold

        vals_u8 = vals.astype(np.uint8).reshape(-1, 1)
        otsu_thr, _ = cv2.threshold(
            vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Bound Otsu so a single unusual frame cannot become too permissive
        # or too strict after per-frame normalization.
        return int(np.clip(otsu_thr, 28, 110))

    def clean_motion_mask(self, mask, roi_mask):
        mask = cv2.bitwise_and(mask, roi_mask)

        close_kernel = np.ones((5, 5), np.uint8)
        open_kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        # Remove tiny isolated components.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self.min_component_area:
                cleaned[labels == label] = 255

        return cleaned

    def calculate_centroid(self, mask):
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy

    def calculate_plume_coverage(self, mask, roi_mask):
        """Return visible plume-mask coverage (%) inside the Detection ROI.

        This value is used only as an internal detection cue. It is not a
        measurement of gas amount or concentration.
        """
        gas_pixels = int(cv2.countNonZero(mask))
        total_pixels = int(cv2.countNonZero(roi_mask))
        if total_pixels <= 0:
            return 0.0

        return (gas_pixels / total_pixels) * 100.0

    # ================================================================
    # OPTIONAL CNN
    # ================================================================
    @staticmethod
    def _softmax(x):
        x = np.asarray(x, dtype=np.float32)
        x = x - np.max(x)
        ex = np.exp(x)
        den = np.sum(ex)
        if den <= 0:
            return np.zeros_like(ex)
        return ex / den

    @staticmethod
    def _sigmoid(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    def preprocess_cnn(self, bgr_image):
        resized = cv2.resize(
            bgr_image, self.cnn_input_size, interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        blob = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        return blob

    def cnn_leak_probability(self, frame):
        if self.cnn_net is None:
            return None

        # Classify only the Detection ROI so irrelevant background is reduced.
        if self.detection_roi is not None:
            x, y, w, h = self.detection_roi
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                crop = frame
        else:
            crop = frame

        try:
            blob = self.preprocess_cnn(crop)
            self.cnn_net.setInput(blob)
            out = np.asarray(self.cnn_net.forward()).reshape(-1)

            if out.size == 0:
                return None

            if out.size == 1:
                val = float(out[0])
                if 0.0 <= val <= 1.0:
                    prob = val
                else:
                    prob = self._sigmoid(val)
            else:
                probs = self._softmax(out)
                idx = int(np.clip(self.cnn_positive_class, 0, len(probs) - 1))
                prob = float(probs[idx])

            prob = float(np.clip(prob, 0.0, 1.0))
            self.cnn_conf_history.append(prob)
            return float(np.median(np.asarray(self.cnn_conf_history)))
        except Exception:
            # Do not stop the entire video pipeline if model dimensions/output
            # do not match the generic assumptions above.
            return None

    # ================================================================
    # DIRECTION
    # ================================================================
    def classify_8_direction(self, dx, dy):
        ax = abs(dx)
        ay = abs(dy)

        if ax < 1e-9 and ay < 1e-9:
            return None

        # Wider cardinal sectors than the standard 22.5-degree split.
        # This prevents a mostly-right plume from flickering RIGHT/UP-RIGHT.
        if ax >= self.direction_axis_ratio * ay:
            return "RIGHT" if dx > 0 else "LEFT"

        if ay >= self.direction_axis_ratio * ax:
            return "DOWN" if dy > 0 else "UP"

        if dx > 0 and dy > 0:
            return "DOWN-RIGHT"
        if dx > 0 and dy < 0:
            return "UP-RIGHT"
        if dx < 0 and dy > 0:
            return "DOWN-LEFT"
        return "UP-LEFT"

    def _hold_or_wait_direction(self, gas_present):
        if gas_present:
            self.no_gas_frames = 0
            if self.last_valid_direction not in ("WAITING", "NO GAS"):
                self.last_direction_confidence *= 0.94
                return (
                    self.last_valid_direction,
                    float(np.clip(self.last_direction_confidence, 0.0, 1.0)),
                    0.0
                )
            return "WAITING", 0.0, 0.0

        self.no_gas_frames += 1
        if (
            self.no_gas_frames <= self.direction_ttl_frames
            and self.last_valid_direction not in ("WAITING", "NO GAS")
        ):
            self.last_direction_confidence *= 0.90
            return (
                self.last_valid_direction,
                float(np.clip(self.last_direction_confidence, 0.0, 1.0)),
                0.0
            )

        self.last_valid_direction = "NO GAS"
        self.last_direction_confidence = 0.0
        self.flow_history.clear()
        return "NO GAS", 0.0, 0.0

    def calculate_flow_direction(
        self, prev_gray, curr_gray, prev_mask, curr_mask, roi_mask
    ):
        gas_pixels = int(cv2.countNonZero(curr_mask))
        gas_present = gas_pixels >= self.min_gas_pixels

        if (
            prev_gray is None or curr_gray is None
            or prev_mask is None or curr_mask is None
        ):
            return self._hold_or_wait_direction(gas_present)

        if not gas_present:
            return self._hold_or_wait_direction(False)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            0.5,   # pyr_scale
            3,     # levels
            19,    # winsize (slightly larger = more stable turbulent flow)
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0
        )

        gas_mask = cv2.bitwise_or(prev_mask, curr_mask)
        gas_mask = cv2.bitwise_and(gas_mask, roi_mask)
        gas_mask = cv2.dilate(gas_mask, np.ones((7, 7), np.uint8), iterations=1)

        valid = gas_mask > 0
        if np.count_nonzero(valid) < self.min_gas_pixels:
            return self._hold_or_wait_direction(True)

        fx = flow[:, :, 0][valid]
        fy = flow[:, :, 1][valid]
        if fx.size < 30:
            return self._hold_or_wait_direction(True)

        mag = np.hypot(fx, fy)
        good = (
            (mag >= self.flow_min_magnitude)
            & (mag <= self.flow_max_magnitude)
            & np.isfinite(mag)
            & np.isfinite(fx)
            & np.isfinite(fy)
        )

        if np.count_nonzero(good) < 30:
            return self._hold_or_wait_direction(True)

        fx = fx[good]
        fy = fy[good]
        mag = mag[good]

        # Initial robust dominant vector.
        seed_dx = float(np.median(fx))
        seed_dy = float(np.median(fy))
        seed_speed = float(np.hypot(seed_dx, seed_dy))
        if seed_speed < self.flow_min_magnitude:
            return self._hold_or_wait_direction(True)

        # Reject local curls that point far away from the dominant plume flow.
        ux = fx / np.maximum(mag, 1e-6)
        uy = fy / np.maximum(mag, 1e-6)
        dux = seed_dx / seed_speed
        duy = seed_dy / seed_speed
        alignment = ux * dux + uy * duy
        aligned = alignment >= self.flow_alignment_cos

        if np.count_nonzero(aligned) >= 20:
            fx2 = fx[aligned]
            fy2 = fy[aligned]
            mag2 = mag[aligned]
        else:
            fx2, fy2, mag2 = fx, fy, mag

        frame_dx = float(np.median(fx2))
        frame_dy = float(np.median(fy2))

        # Directional coherence: 1 = vectors agree, 0 = cancel each other.
        unit_x = fx2 / np.maximum(mag2, 1e-6)
        unit_y = fy2 / np.maximum(mag2, 1e-6)
        coherence = float(
            np.hypot(np.mean(unit_x), np.mean(unit_y))
        )
        coherence = float(np.clip(coherence, 0.0, 1.0))

        self.flow_history.append((frame_dx, frame_dy, coherence))
        self.no_gas_frames = 0

        if len(self.flow_history) < 4:
            return self._hold_or_wait_direction(True)

        hist = np.asarray(self.flow_history, dtype=np.float32)
        dx = float(np.median(hist[:, 0]))
        dy = float(np.median(hist[:, 1]))
        speed = float(np.hypot(dx, dy))
        conf = float(np.median(hist[:, 2]))

        if speed < self.flow_min_magnitude:
            return self._hold_or_wait_direction(True)

        direction = self.classify_8_direction(dx, dy)
        if direction is None:
            return self._hold_or_wait_direction(True)

        # Mild speed support without making slow gas automatically low-confidence.
        speed_support = float(np.clip(speed / 0.40, 0.45, 1.0))
        conf = float(np.clip(conf * speed_support, 0.0, 1.0))

        self.last_valid_direction = direction
        self.last_direction_confidence = conf
        return direction, conf, speed

    # ================================================================
    # PROCESSING
    # ================================================================
    def start_processing(self):
        if self.cap is None or self.video_path is None:
            messagebox.showinfo("Process", "Load a video first.")
            return

        try:
            alpha = int(self.alpha_var.get())
            low = float(self.low_var.get())
            high = float(self.high_var.get())
            adaptive = bool(self.adaptive_mask_var.get())
        except Exception:
            messagebox.showerror("Parameters", "Invalid parameter value.")
            return

        nyquist = max(0.1, self.fps / 2.0)
        low = max(0.01, min(low, nyquist))
        high = max(0.01, min(high, nyquist))
        if high <= low:
            messagebox.showerror(
                "Frequency", "High Hz must be greater than Low Hz."
            )
            return

        params = {
            "alpha": alpha,
            "low": low,
            "high": high,
            "adaptive": adaptive,
        }

        self.is_playing = False
        self.btn_process.config(state=tk.DISABLED)
        self.btn_play.config(state=tk.DISABLED)
        self.lbl_info.config(text="Processing video...\nPlease wait.")

        thread = threading.Thread(
            target=self.process_video_thread,
            args=(params,),
            daemon=True
        )
        thread.start()

    def read_all_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(
                frame,
                (self.frame_width, self.frame_height),
                interpolation=cv2.INTER_AREA
            )
            frames.append(frame)
        cap.release()
        return frames

    def process_video_thread(self, params):
        cv2.setNumThreads(self.num_cores)

        frames = self.read_all_frames()
        if not frames:
            self.root.after(0, self._processing_failed, "No frames found.")
            return

        self.flow_history.clear()
        self.last_valid_direction = "WAITING"
        self.last_direction_confidence = 0.0
        self.prev_gray = None
        self.prev_motion_mask = None
        self.no_gas_frames = 0
        self.cnn_conf_history.clear()

        # Frequency auto-estimate is intentionally conservative:
        # use it only if current values are obviously invalid for this FPS.
        low = params["low"]
        high = params["high"]
        auto_band = self.estimate_roi_frequencies(frames)
        if auto_band is not None:
            est_low, est_high = auto_band
            # Blend user setting with estimate instead of completely replacing it.
            # This keeps UI control predictable while still adapting to the video.
            low = max(0.05, min(low, est_low))
            high = min(self.fps / 2.0, max(high, est_high))
            if high <= low:
                high = min(self.fps / 2.0, low + 0.5)

        alpha = params["alpha"]
        adaptive = params["adaptive"]

        h, w = self.frame_height, self.frame_width
        roi_mask = self.roi_to_mask((h, w), self.detection_roi)

        # ------------------------------------------------------------
        # 1) Build low-resolution chroma representation, then upsample.
        # ------------------------------------------------------------
        tensor = np.zeros((len(frames), h, w, 2), dtype=np.float32)

        for i, frame in enumerate(frames):
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            chroma = ycrcb[:, :, 1:].astype(np.float32)
            pyr = self.build_gaussian_pyramid(chroma, self.levels)
            tensor[i] = cv2.resize(
                pyr[-1], (w, h), interpolation=cv2.INTER_LINEAR
            )

        # ------------------------------------------------------------
        # 2) Correct symmetric temporal band-pass using |frequency|.
        # ------------------------------------------------------------
        fft_data = fft.fft(tensor, axis=0, workers=self.num_cores)
        frequencies = fft.fftfreq(len(frames), d=1.0 / self.fps)
        keep = (np.abs(frequencies) >= low) & (np.abs(frequencies) <= high)
        fft_data[~keep, ...] = 0

        filtered = np.real(
            fft.ifft(fft_data, axis=0, workers=self.num_cores)
        ).astype(np.float32)
        filtered *= float(alpha)

        self.processed_frames = []
        total = len(frames)

        for i, orig_frame in enumerate(frames):
            # --------------------------------------------------------
            # 3) Temporal chroma magnitude -> heatmap -> plume mask
            # --------------------------------------------------------
            chroma_diff = np.hypot(
                filtered[i, :, :, 0],
                filtered[i, :, :, 1]
            )
            chroma_diff = cv2.GaussianBlur(chroma_diff, (7, 7), 0)

            norm_diff = cv2.normalize(
                chroma_diff, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            if adaptive:
                threshold_value = self.adaptive_threshold_value(
                    norm_diff, roi_mask
                )
            else:
                threshold_value = self.fixed_mask_threshold

            _, motion_mask = cv2.threshold(
                norm_diff, threshold_value, 255, cv2.THRESH_BINARY
            )
            motion_mask = self.clean_motion_mask(motion_mask, roi_mask)

            coverage = self.calculate_plume_coverage(
                motion_mask, roi_mask
            )
            centroid = self.calculate_centroid(motion_mask)

            # --------------------------------------------------------
            # 4) Optional CNN leak / no-leak classification
            # --------------------------------------------------------
            cnn_prob = self.cnn_leak_probability(orig_frame)
            if cnn_prob is not None:
                leak_conf = cnn_prob
                leak_status = "DETECTED" if cnn_prob >= self.cnn_threshold else "NO LEAK"
                leak_source = "CNN"
            else:
                # Fallback is NOT a CNN score. It is just a mask-based confidence
                # so the program remains useful before a model is trained.
                leak_conf = float(np.clip(coverage / 10.0, 0.0, 1.0))
                leak_status = (
                    "DETECTED"
                    if coverage >= self.min_coverage_for_gas
                    else "NO GAS"
                )
                leak_source = "CV"

            # --------------------------------------------------------
            # 5) Direction from optical flow inside plume mask
            # --------------------------------------------------------
            curr_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)

            direction, dir_conf, flow_speed = self.calculate_flow_direction(
                self.prev_gray,
                curr_gray,
                self.prev_motion_mask,
                motion_mask,
                roi_mask
            )

            self.prev_gray = curr_gray
            self.prev_motion_mask = motion_mask.copy()

            # --------------------------------------------------------
            # 6) Visualization
            # --------------------------------------------------------
            color_map = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
            color_mask_3ch = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            colored_gas = cv2.bitwise_and(color_map, color_mask_3ch)

            overlay = cv2.addWeighted(
                orig_frame, 0.72, colored_gas, 0.78, 0
            )

            # Detection ROI on overlay
            if self.detection_roi is not None:
                x, y, rw, rh = self.detection_roi
                cv2.rectangle(
                    overlay, (x, y), (x + rw, y + rh), (0, 0, 255), 1
                )

            if centroid is not None:
                cx, cy = centroid
                cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), -1)
                cv2.putText(
                    overlay,
                    f"{direction}  {dir_conf * 100:.0f}%",
                    (max(5, cx - 90), max(24, cy - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # Small status line on overlay
            cv2.putText(
                overlay,
                f"Leak:{leak_status} ({leak_source})",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            mask_frame_color = color_mask_3ch

            result = {
                "orig": orig_frame,
                "overlay": overlay,
                "mask": mask_frame_color,
                "direction": direction,
                "direction_confidence": dir_conf,
                "flow_speed": flow_speed,
                "centroid": centroid,
                "leak_status": leak_status,
                "leak_confidence": leak_conf,
                "leak_source": leak_source,
                "threshold": threshold_value,
            }
            self.processed_frames.append(result)

            self.current_direction = direction
            self.current_direction_confidence = dir_conf
            self.current_flow_speed = flow_speed
            self.current_centroid = centroid
            self.current_leak_status = leak_status
            self.current_leak_confidence = leak_conf
            self.current_leak_source = leak_source

            # Throttle UI refreshes while processing; results still store every frame.
            if i % max(1, int(self.fps // 10)) == 0 or i == total - 1:
                self.root.after(
                    0,
                    self.update_live_view,
                    result,
                    i + 1,
                    total
                )

        self.root.after(0, self.finish_processing, low, high)

    def _processing_failed(self, msg):
        self.btn_process.config(state=tk.NORMAL)
        self.btn_play.config(state=tk.DISABLED)
        self.lbl_info.config(text=f"Processing failed:\n{msg}")

    def finish_processing(self, used_low, used_high):
        self.btn_process.config(state=tk.NORMAL)
        self.btn_play.config(state=tk.NORMAL)

        # Show actual frequency band used for this run.
        self.low_var.set(round(used_low, 2))
        self.high_var.set(round(used_high, 2))

        self.is_playing = True
        self.frame_index = 0
        self.play_loop()

    # ================================================================
    # DISPLAY / PLAYBACK
    # ================================================================
    def build_info_text(self, data=None, current=None, total=None):
        if data is None:
            direction = self.current_direction
            dir_conf = self.current_direction_confidence
            flow_speed = self.current_flow_speed
            centroid = self.current_centroid
            leak_status = self.current_leak_status
            leak_conf = self.current_leak_confidence
            leak_source = self.current_leak_source
            threshold = None
        else:
            direction = data["direction"]
            dir_conf = data["direction_confidence"]
            flow_speed = data["flow_speed"]
            centroid = data["centroid"]
            leak_status = data["leak_status"]
            leak_conf = data["leak_confidence"]
            leak_source = data["leak_source"]
            threshold = data["threshold"]

        lines = []
        if current is not None and total is not None:
            lines.append(f"Frame : {current}/{total}")
            lines.append("")

        lines += [
            "GAS LEAK / PLUME",
            "========================",
            f"Leak       : {leak_status}",
            f"Leak source: {leak_source}",
            f"Confidence : {leak_conf * 100:.1f}%",
            "",
            f"Direction  : {direction}",
            f"Dir. conf. : {dir_conf * 100:.1f}%",
            f"Flow speed : {flow_speed:.3f} px/frame",
            "",
        ]

        if centroid is not None:
            lines.append(f"Centroid   : {centroid}")
        else:
            lines.append("Centroid   : NONE")

        if threshold is not None:
            lines.append(f"Mask thr.  : {threshold}")

        return "\n".join(lines)

    def update_info(self):
        self.lbl_info.config(text=self.build_info_text())

    def update_live_view(self, result, current, total):
        self.current_frame = result["orig"]
        self.display_frame_with_rois(result["orig"], result["centroid"])
        self.display_image(result["overlay"], self.canvas_overlay)
        self.display_image(result["mask"], self.canvas_mask)
        self.lbl_info.config(
            text=self.build_info_text(result, current, total)
        )

    def display_image(self, cv_img, canvas):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_tk = ImageTk.PhotoImage(image=im_pil)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=im_tk)
        canvas.image = im_tk

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_loop()

    def play_loop(self):
        if not self.is_playing or not self.processed_frames:
            return

        data = self.processed_frames[self.frame_index]
        self.current_frame = data["orig"]

        self.display_frame_with_rois(data["orig"], data["centroid"])
        self.display_image(data["overlay"], self.canvas_overlay)
        self.display_image(data["mask"], self.canvas_mask)
        self.lbl_info.config(text=self.build_info_text(data))

        self.frame_index = (self.frame_index + 1) % len(self.processed_frames)

        try:
            current_fps = max(1.0, float(self.display_fps_var.get()))
        except Exception:
            current_fps = max(1.0, self.fps)

        delay = max(1, int(round(1000.0 / current_fps)))
        self.root.after(delay, self.play_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = GasPlumeAmplifierApp(root)
    root.mainloop()
