import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import scipy.fft as fft           
import threading
import multiprocessing            

class GasPlumeAmplifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gas Leak Detection (Color/Plume Amplification)")
        self.root.geometry("1100x800") 

        # --- Parameters ---
        self.video_path = None
        self.cap = None
        self.fps = 30
        self.num_cores = 8          
        self.frame_width = 0
        self.frame_height = 0
        
        self.levels = 4 
        self.alpha = 50 
        self.low_freq = 5.0    # Hz
        self.high_freq = 15.0  # Hz
        
        self.roi = None
        self.selecting_roi = False
        self.roi_start = None

        # Playback control
        self.is_playing = False
        self.current_frame = None
        self.processed_frames = []
        self.frame_index = 0

        self.setup_ui()

    def setup_ui(self):
        # Top Control Frame
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(fill=tk.X)

        btn_load = tk.Button(control_frame, text="Load Video", command=self.load_video)
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_roi = tk.Button(control_frame, text="Select ROI (Reference)", command=self.enable_roi_selection)
        btn_roi.pack(side=tk.LEFT, padx=5)
        
        self.btn_process = tk.Button(control_frame, text="Process Gas Plume", command=self.start_processing)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        self.btn_play = tk.Button(control_frame, text="Play/Pause", command=self.toggle_play, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        # Parameter Sliders & Entries
        param_frame = tk.Frame(self.root, pady=5)
        param_frame.pack(fill=tk.X)
        
        self.alpha_var = tk.IntVar(value=self.alpha)
        self.low_var = tk.DoubleVar(value=self.low_freq)
        self.high_var = tk.DoubleVar(value=self.high_freq)
        self.display_fps_var = tk.DoubleVar(value=30.0)
        
        # Alpha
        tk.Label(param_frame, text="Alpha:").pack(side=tk.LEFT)
        self.alpha_entry = tk.Entry(param_frame, textvariable=self.alpha_var, width=5)
        self.alpha_entry.pack(side=tk.LEFT, padx=(5, 2))
        self.alpha_slider = tk.Scale(param_frame, from_=1, to=300, orient=tk.HORIZONTAL, variable=self.alpha_var, showvalue=False)
        self.alpha_slider.pack(side=tk.LEFT, padx=(0, 10))
        
        # Low Hz
        tk.Label(param_frame, text="Low Hz:").pack(side=tk.LEFT)
        self.low_entry = tk.Entry(param_frame, textvariable=self.low_var, width=5)
        self.low_entry.pack(side=tk.LEFT, padx=(5, 2))
        self.low_slider = tk.Scale(param_frame, from_=0.1, to=15, resolution=0.1, orient=tk.HORIZONTAL, variable=self.low_var, showvalue=False)
        self.low_slider.pack(side=tk.LEFT, padx=(0, 10))
        
        # High Hz
        tk.Label(param_frame, text="High Hz:").pack(side=tk.LEFT)
        self.high_entry = tk.Entry(param_frame, textvariable=self.high_var, width=5)
        self.high_entry.pack(side=tk.LEFT, padx=(5, 2))
        self.high_slider = tk.Scale(param_frame, from_=1, to=30, resolution=0.1, orient=tk.HORIZONTAL, variable=self.high_var, showvalue=False)
        self.high_slider.pack(side=tk.LEFT, padx=(0, 10))
        
        # Display FPS
        tk.Label(param_frame, text="| Playback FPS:").pack(side=tk.LEFT, padx=(10, 0))
        self.fps_entry = tk.Entry(param_frame, textvariable=self.display_fps_var, width=5)
        self.fps_entry.pack(side=tk.LEFT, padx=(5, 2))
        self.fps_slider = tk.Scale(param_frame, from_=1, to=120, resolution=1, orient=tk.HORIZONTAL, variable=self.display_fps_var, showvalue=False)
        self.fps_slider.pack(side=tk.LEFT, padx=(0, 10))

        # 2x2 Grid Layout
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top-Left: Original
        self.canvas_orig = tk.Canvas(self.grid_frame, bg="gray20", width=400, height=300)
        self.canvas_orig.grid(row=0, column=0, padx=5, pady=5)
        self.canvas_orig.bind("<ButtonPress-1>", self.on_roi_press)
        self.canvas_orig.bind("<B1-Motion>", self.on_roi_drag)
        self.canvas_orig.bind("<ButtonRelease-1>", self.on_roi_release)

        # Top-Right: Color Amplified 
        self.canvas_overlay = tk.Canvas(self.grid_frame, bg="gray20", width=400, height=300)
        self.canvas_overlay.grid(row=0, column=1, padx=5, pady=5)

        # Bottom-Left: Plume Mask
        self.canvas_mask = tk.Canvas(self.grid_frame, bg="black", width=400, height=300)
        self.canvas_mask.grid(row=1, column=0, padx=5, pady=5)

        # Bottom-Right: Info Text
        info_frame = tk.Frame(self.grid_frame, bg="white", width=400, height=300)
        info_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        info_frame.grid_propagate(False)
        self.lbl_info = tk.Label(info_frame, text="ข้อมูลต่างๆ\nรอโหลดวิดีโอ...", font=("Arial", 12), bg="white", justify=tk.LEFT)
        self.lbl_info.pack(expand=True)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not self.video_path: return
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if np.isnan(self.fps) or self.fps == 0: self.fps = 30
        
        self.display_fps_var.set(round(self.fps, 1))
        
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            max_dim = 400
            scale = min(max_dim/w, max_dim/h)
            self.frame_width = int(w * scale)
            self.frame_height = int(h * scale)
            
            self.current_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            self.display_image(self.current_frame, self.canvas_orig)
            
            # ค่าเริ่มต้น ROI คือเต็มจอ
            self.roi = (0, 0, self.frame_width, self.frame_height)
            self.update_info()

    def update_info(self):
        info_text = f"Video Data:\n"
        info_text += f"Video FPS: {self.fps:.2f}\n"
        info_text += f"Display FPS: {self.display_fps_var.get():.1f}\n"
        info_text += f"Frame Size: {self.frame_width}x{self.frame_height}\n"
        info_text += f"Current Settings (Color/Plume Mode):\n"
        info_text += f"Alpha: {self.alpha_var.get()}\n"
        info_text += f"Filter: {self.low_var.get():.1f} - {self.high_var.get():.1f} Hz\n"
        info_text += f"CPU Cores Fixed: {self.num_cores}"
        self.lbl_info.config(text=info_text)

    def enable_roi_selection(self):
        self.selecting_roi = True
        self.canvas_orig.delete("roi_rect")

    def on_roi_press(self, event):
        if not self.selecting_roi: return
        self.roi_start = (event.x, event.y)

    def on_roi_drag(self, event):
        if not self.selecting_roi or not self.roi_start: return
        self.canvas_orig.delete("roi_rect")
        self.canvas_orig.create_rectangle(self.roi_start[0], self.roi_start[1], event.x, event.y, 
                                          outline="red", width=2, tag="roi_rect")

    def on_roi_release(self, event):
        if not self.selecting_roi: return
        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        
        if w > 10 and h > 10:
            self.roi = (x, y, w, h)
        else:
            self.roi = (0, 0, self.frame_width, self.frame_height) 
            
        self.selecting_roi = False
        self.update_info()

    def build_gaussian_pyramid(self, frame, levels):
        pyramid = [frame]
        for i in range(levels - 1):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    # --- ฟังก์ชั่นคำนวณช่วงความถี่ (Hz) จาก ROI อัตโนมัติ ---
    def auto_detect_roi_frequencies(self, frames):
        if not self.roi:
            return

        x, y, w, h = self.roi
        # ดึงเฉพาะพื้นที่ ROI จากทุกเฟรม
        roi_signals = []
        for f in frames:
            crop = f[y:y+h, x:x+w]
            if crop.size > 0:
                # หาค่าสัญญาณสีเฉลี่ยใน ROI (ใช้ Grayscale)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                roi_signals.append(np.mean(gray))
        
        if len(roi_signals) < 10:
            return

        # คำนวณ FFT ของสัญญาณใน ROI
        roi_signals = np.array(roi_signals) - np.mean(roi_signals) # ตัดค่า DC offset ออก
        fft_spectrum = np.abs(fft.fft(roi_signals))
        freqs = fft.fftfreq(len(roi_signals), d=1.0/self.fps)

        # พิจารณาเฉพาะความถี่ที่เป็นบวก
        pos_mask = freqs > 0.1
        freqs = freqs[pos_mask]
        fft_spectrum = fft_spectrum[pos_mask]

        if len(fft_spectrum) == 0:
            return

        # หาค่า Peak สูงสุด และคำนวณช่วง Bandwidth รอบๆ Peak
        max_idx = np.argmax(fft_spectrum)
        peak_freq = freqs[max_idx]

        # กำหนดช่วงความถี่ตอบสนองรอบๆ Peak (เช่น +- 30% หรือครอบคลุมพลังงาน 50%)
        half_power = fft_spectrum[max_idx] * 0.5
        significant_indices = np.where(fft_spectrum >= half_power)[0]

        if len(significant_indices) > 0:
            calc_low = freqs[significant_indices[0]]
            calc_high = freqs[significant_indices[-1]]
            
            # กำหนดขอบเขตให้เหมาะสม (Padding เล็กน้อยเพื่อให้ครอบคลุม)
            calc_low = max(0.1, round(calc_low * 0.8, 1))
            calc_high = min(self.fps / 2.0, round(calc_high * 1.2, 1))
            
            if calc_high <= calc_low:
                calc_high = calc_low + 2.0

            # อัปเดตค่าไปยัง UI Variables
            self.low_var.set(calc_low)
            self.high_var.set(calc_high)

    def start_processing(self):
        if not self.cap: return
        
        self.is_playing = False
        self.btn_process.config(state=tk.DISABLED)
        self.btn_play.config(state=tk.DISABLED)
        self.lbl_info.config(text="สถานะ:\nกำลังวิเคราะห์ ROI และสกัดสัญญาณความถี่...")
        
        process_thread = threading.Thread(target=self.process_video_thread)
        process_thread.daemon = True
        process_thread.start()

    def update_live_view(self, orig, overlay, mask, current, total):
        orig_disp = orig.copy()
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(orig_disp, (x, y), (x+w, y+h), (0, 0, 255), 2)

        self.display_image(orig_disp, self.canvas_orig)
        self.display_image(overlay, self.canvas_overlay)
        self.display_image(mask, self.canvas_mask)
        self.lbl_info.config(text=f"สถานะ:\nกำลังขยายกลุ่มก๊าซ (Live Preview)...\nเฟรมที่ {current} / {total}")

    def process_video_thread(self):
        cv2.setNumThreads(self.num_cores)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret: 
                break
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            frames.append(frame)
            
        if not frames: 
            self.root.after(0, lambda: self.btn_process.config(state=tk.NORMAL))
            return

        # -------------------------------------------------------------
        # 1. วิเคราะห์เพื่ออัปเดต Low Hz และ High Hz จาก ROI โดยอัตโนมัติ
        # -------------------------------------------------------------
        self.auto_detect_roi_frequencies(frames)
        
        # ดึงค่า Hz ใหม่ที่ได้จากการวิเคราะห์ ROI มาใช้ประมวลผล
        self.low_freq = self.low_var.get()
        self.high_freq = self.high_var.get()
        self.alpha = self.alpha_var.get()
        
        # อัปเดต UI ให้แสดงค่าความถี่ใหม่
        self.root.after(0, self.update_info)

        # 2. เตรียมข้อมูล Chroma Tensor (YCrCb)
        tensor = np.zeros((len(frames), self.frame_height, self.frame_width, 2), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            chroma = ycrcb[:, :, 1:].astype(np.float32)
            
            gauss_pyr = self.build_gaussian_pyramid(chroma, self.levels)
            extracted_chroma = cv2.resize(gauss_pyr[-1], (self.frame_width, self.frame_height))
            tensor[i] = extracted_chroma
            
        # 3. กรองความถี่ตาม Low Hz - High Hz ที่คำนวณมาใหม่
        fft_data = fft.fft(tensor, axis=0, workers=self.num_cores)
        frequencies = fft.fftfreq(len(frames), d=1.0/self.fps)
        
        bound_low = (np.abs(frequencies - self.low_freq)).argmin()
        bound_high = (np.abs(frequencies - self.high_freq)).argmin()
        
        fft_data[:bound_low] = 0
        fft_data[bound_high:-bound_high] = 0
        fft_data[-bound_low:] = 0
        
        filtered_tensor = np.real(fft.ifft(fft_data, axis=0, workers=self.num_cores))
        
        # 4. Amplify สัญญาณ
        filtered_tensor *= self.alpha
        
        self.processed_frames = []
        total_frames = len(frames)
        
        for i in range(total_frames):
            orig_frame = frames[i].copy()
            
            chroma_diff = np.sqrt(filtered_tensor[i, :, :, 0]**2 + filtered_tensor[i, :, :, 1]**2)
            chroma_diff = cv2.GaussianBlur(chroma_diff, (7, 7), 0)
            
            norm_diff = cv2.normalize(chroma_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            color_map = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
            _, motion_mask = cv2.threshold(norm_diff, 40, 255, cv2.THRESH_BINARY)
            
            color_mask_3ch = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            colored_gas = cv2.bitwise_and(color_map, color_mask_3ch)
            
            amplified_frame = cv2.addWeighted(orig_frame, 0.7, colored_gas, 0.8, 0)
            mask_frame_color = color_mask_3ch.copy()
            
            self.processed_frames.append({
                "orig": orig_frame,
                "overlay": amplified_frame,   
                "mask": mask_frame_color      
            })
            
            self.root.after(0, self.update_live_view, orig_frame, amplified_frame, mask_frame_color, i+1, total_frames)
            
        def finish_processing():
            self.update_info() 
            self.lbl_info.config(text=self.lbl_info.cget("text") + "\n\nประมวลผลเสร็จสิ้น!\nกด Play เพื่อเล่นวนลูป")
            self.btn_process.config(state=tk.NORMAL)
            self.btn_play.config(state=tk.NORMAL)
            self.is_playing = True
            self.frame_index = 0
            self.play_loop()
            
        self.root.after(0, finish_processing)

    def display_image(self, cv_img, canvas):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_tk = ImageTk.PhotoImage(image=im_pil)
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
        
        orig_disp = data["orig"].copy()
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(orig_disp, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
        self.display_image(orig_disp, self.canvas_orig)
        self.display_image(data["overlay"], self.canvas_overlay)
        self.display_image(data["mask"], self.canvas_mask)
        
        self.frame_index = (self.frame_index + 1) % len(self.processed_frames)
        
        current_fps = max(1.0, self.display_fps_var.get())
        delay = int(1000 / current_fps)
        
        self.root.after(delay, self.play_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = GasPlumeAmplifierApp(root)
    root.mainloop()