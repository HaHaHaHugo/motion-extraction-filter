import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
import json
import os
from scipy.signal import butter, lfilter_zi, lfilter
from scipy.ndimage import gaussian_filter
from PIL import ImageFont, ImageDraw, Image

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
CAMERA_INDEX = 0
FPS = 30.0
PYRAMID_LEVELS = 3
SMOOTH_SIGMA = 2.0
CONFIG_FILE = "evm_config.json" # ไฟล์สำหรับบันทึกการตั้งค่า

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_config():
    """ โหลดการตั้งค่าจากไฟล์ ถ้าไม่มีไฟล์ให้ใช้ค่าเริ่มต้น """
    defaults = {'LOW_FREQ': 0.2, 'HIGH_FREQ': 2.5, 'ALPHA': 20.0}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("โหลดไฟล์ config ไม่สำเร็จ ใช้ค่าเริ่มต้นแทน:", e)
    return defaults

def save_config(config):
    """ บันทึกการตั้งค่าลงไฟล์ """
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print("บันทึกการตั้งค่าเรียบร้อยแล้ว!")
    except Exception as e:
        print("บันทึกไฟล์ config ไม่สำเร็จ:", e)

def put_thai_text(img, text, position, font_size=24, color=(255, 255, 255)):
    """ ฟังก์ชันสำหรับพิมพ์ภาษาไทยลงบนภาพ OpenCV """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def setup_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)        
        cap.set(cv2.CAP_PROP_GAIN, 0)             
    except Exception as e:
        print("ไม่สามารถล็อคค่า Exposure/Gain ได้:", e)
    return cap

def stabilize_frame(prev_gray, curr_gray, curr_frame):
    if prev_gray is None: return curr_frame
    shift, _ = cv2.phaseCorrelate(prev_gray, curr_gray)
    dx, dy = shift
    if abs(dx) < 0.2 and abs(dy) < 0.2: return curr_frame
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(curr_frame, M, (curr_frame.shape[1], curr_frame.shape[0]))

def riesz_transform(img):
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f_norm = np.sqrt(fx**2 + fy**2)
    f_norm[0, 0] = 1.0
    H_x, H_y = -1j * (fx / f_norm), -1j * (fy / f_norm)
    H_x[0, 0], H_y[0, 0] = 0, 0
    F = np.fft.fft2(img)
    Rx, Ry = np.fft.ifft2(F * H_x), np.fft.ifft2(F * H_y)
    amplitude = np.sqrt(img**2 + np.abs(Rx)**2 + np.abs(Ry)**2 + 1e-6)
    phase = np.arctan2(np.abs(Rx) + np.abs(Ry), img + 1e-6)
    return amplitude, phase

def amplitude_weighted_blur(phase_diff, amplitude, sigma):
    numerator = gaussian_filter(phase_diff * amplitude, sigma=sigma)
    denominator = gaussian_filter(amplitude, sigma=sigma) + 1e-6
    return numerator / denominator

def build_butterworth_filter(low, high, fps):
    nyq = 0.5 * fps
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return b, a

def on_trackbar(val):
    """ Dummy function สำหรับ Trackbar """
    pass

# ==============================================================================
# MAIN REAL-TIME PROCESSING LOOP
# ==============================================================================

def main():
    cap = setup_camera()
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้อง Webcam ได้")
        return

    # 1. โหลดการตั้งค่าล่าสุด
    config = load_config()
    current_lf = config['LOW_FREQ']
    current_hf = config['HIGH_FREQ']
    current_alpha = config['ALPHA']

    print("เริ่มต้นระบบ Eulerian Video Magnification...")
    print("กด 'q' เพื่อเลิกทำงานและบันทึกการตั้งค่า")

    # 2. สร้างหน้าต่างแสดงผลและ Slider
    window_name = "Eulerian Video Magnification - Gas Leak Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 560) # ปรับขนาดหน้าต่างให้กว้างพอ

    # สร้าง Trackbar (OpenCV Slider รับค่าทศนิยมไม่ได้ จึงต้องคูณ 10)
    # เช่น 0.2 Hz = 2, 14.9 Hz = 149
    cv2.createTrackbar("Low Freq (x10)", window_name, int(current_lf * 10), 140, on_trackbar)
    cv2.createTrackbar("High Freq (x10)", window_name, int(current_hf * 10), 149, on_trackbar)
    cv2.createTrackbar("Alpha (Zoom)", window_name, int(current_alpha), 100, on_trackbar)

    # 3. เตรียมตัวแปรสำหรับ Filter
    b, a = build_butterworth_filter(current_lf, current_hf, FPS)
    filter_states = [None] * PYRAMID_LEVELS
    prev_gray = None
    reference_phases = [None] * PYRAMID_LEVELS

    while True:
        ret, frame = cap.read()
        if not ret: break

        # ---- ดึงค่าจาก Slider อัปเดตแบบ Real-time ----
        slider_lf = cv2.getTrackbarPos("Low Freq (x10)", window_name)
        slider_hf = cv2.getTrackbarPos("High Freq (x10)", window_name)
        slider_alpha = cv2.getTrackbarPos("Alpha (Zoom)", window_name)

        # ป้องกันค่าติดลบหรือความถี่ต่ำสูงกว่าความถี่สูง (จำกัดขอบเขต)
        new_lf = max(1, slider_lf) / 10.0
        new_hf = max(slider_lf + 1, slider_hf) / 10.0 # ห้าม High <= Low
        new_alpha = float(max(1, slider_alpha))

        # หากมีการปรับเปลี่ยน Slider ความถี่ ต้องคำนวณ Filter ใหม่
        if new_lf != current_lf or new_hf != current_hf:
            current_lf, current_hf = new_lf, new_hf
            b, a = build_butterworth_filter(current_lf, current_hf, FPS)
            # รีเซ็ต State ของ Filter ป้องกันภาพกระตุก
            filter_states = [None] * PYRAMID_LEVELS
            reference_phases = [None] * PYRAMID_LEVELS
        
        current_alpha = new_alpha
        # -----------------------------------------------

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_stab = stabilize_frame(prev_gray, gray, gray)
        prev_gray = gray_stab.copy()

        current_img = gray_stab.copy()
        output_pyramid_levels = []

        for level in range(PYRAMID_LEVELS):
            amplitude, phase = riesz_transform(current_img)

            if reference_phases[level] is None:
                reference_phases[level] = phase.copy()
                zi = lfilter_zi(b, a)
                filter_states[level] = np.zeros((len(b) - 1, *phase.shape))

            phase_diff = phase - reference_phases[level]
            filtered_phase_diff, filter_states[level] = lfilter(
                b, a, phase_diff[None, ...], axis=0, zi=filter_states[level]
            )
            filtered_phase_diff = filtered_phase_diff[0]

            smoothed_phase_diff = amplitude_weighted_blur(
                filtered_phase_diff, amplitude, sigma=SMOOTH_SIGMA
            )

            # ใช้ current_alpha จาก Slider
            magnified_phase = phase + (current_alpha * smoothed_phase_diff)
            
            reconstructed_level = amplitude * np.cos(magnified_phase)
            output_pyramid_levels.append(reconstructed_level)

            if level < PYRAMID_LEVELS - 1:
                current_img = cv2.pyrDown(current_img)

        final_output = output_pyramid_levels[0]
        for level in range(1, PYRAMID_LEVELS):
            up = final_output
            for _ in range(level): up = cv2.pyrUp(up)
            up = cv2.resize(up, (final_output.shape[1], final_output.shape[0]))
            final_output += up

        final_output = np.clip(final_output, 0, 1)
        display_frame = (final_output * 255).astype(np.uint8)

        # จัดการแสดงผลและตัวหนังสือภาษาไทย
        original_color = cv2.cvtColor((gray_stab * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        display_color = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        combined_display = np.hstack((original_color, display_color))

        text_left = f"ต้นฉบับ (Stabilized)"
        text_right = f"ขยายสัญญาณ ({current_alpha}x) | ความถี่: {current_lf}-{current_hf} Hz"
        
        combined_display = put_thai_text(combined_display, text_left, (10, 10), font_size=24, color=(0, 255, 0))
        combined_display = put_thai_text(combined_display, text_right, (650, 10), font_size=24, color=(0, 255, 255))

        cv2.imshow(window_name, combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # บันทึกการตั้งค่าเมื่อกด 'q' ก่อนออกจากลูป
            config_to_save = {
                'LOW_FREQ': current_lf,
                'HIGH_FREQ': current_hf,
                'ALPHA': current_alpha
            }
            save_config(config_to_save)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()