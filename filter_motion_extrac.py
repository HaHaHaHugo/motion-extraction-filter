import cv2
import numpy as np
from collections import deque

def nothing(x):
    pass

def main():
    # 1. เปิดกล้อง Webcam
    cap = cv2.VideoCapture(0)
    
    # สร้างหน้าต่างสำหรับแสดงผล
    cv2.namedWindow('Motion Extraction Filter')
    
    # 2. สร้างแถบเลื่อน (Trackbars)
    # Time Shift (เลื่อนเวลา/เฟรม) - เพื่อให้ได้ผลลัพธ์ Motion Extraction แบบในคลิป
    cv2.createTrackbar('Time Shift (Frames)', 'Motion Extraction Filter', 1, 60, nothing)
    
    # Pixel Shift X และ Y (เลื่อนพิกเซล) - เพื่อสร้างเอฟเฟกต์นูน (Emboss) ตามที่คุณต้องการ
    cv2.createTrackbar('Pixel Shift X', 'Motion Extraction Filter', 0, 50, nothing)
    cv2.createTrackbar('Pixel Shift Y', 'Motion Extraction Filter', 0, 50, nothing)
    
    # Opacity (ความโปร่งใสของเลเยอร์ที่ Invert) - ค่า 0 ถึง 100 (ค่า Default ที่ 50%)
    cv2.createTrackbar('Opacity (%)', 'Motion Extraction Filter', 50, 100, nothing)
    
    # สร้าง Buffer สำหรับเก็บเฟรมย้อนหลัง (เก็บสูงสุด 60 เฟรม หรือประมาณ 2 วินาที)
    frame_buffer = deque(maxlen=60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # ลดขนาดภาพลงเล็กน้อยเพื่อให้ประมวลผลได้ลื่นไหลขึ้น
        frame = cv2.resize(frame, (640, 480))
        
        # เก็บเฟรมปัจจุบันลงใน Buffer
        frame_buffer.append(frame)
        
        # อ่านค่าปัจจุบันจากแถบเลื่อน
        time_shift = cv2.getTrackbarPos('Time Shift (Frames)', 'Motion Extraction Filter')
        pixel_shift_x = cv2.getTrackbarPos('Pixel Shift X', 'Motion Extraction Filter')
        pixel_shift_y = cv2.getTrackbarPos('Pixel Shift Y', 'Motion Extraction Filter')
        opacity = cv2.getTrackbarPos('Opacity (%)', 'Motion Extraction Filter') / 100.0
        
        # ตรวจสอบว่ามีเฟรมใน Buffer พอสำหรับค่า time_shift หรือไม่
        if time_shift >= len(frame_buffer):
            time_shift = len(frame_buffer) - 1
            
        # ดึงเฟรมในอดีต (Delayed Frame) ตามค่า Time Shift ที่เลือก
        delayed_frame = frame_buffer[-(time_shift + 1)]
        
        # 3. ถ้ามีการปรับ Pixel Shift ให้ทำการเลื่อนตำแหน่งพิกเซล (Translation)
        if pixel_shift_x > 0 or pixel_shift_y > 0:
            M = np.float32([[1, 0, pixel_shift_x], [0, 1, pixel_shift_y]])
            delayed_frame = cv2.warpAffine(delayed_frame, M, (delayed_frame.shape[1], delayed_frame.shape[0]))
            
        # 4. ทำการ Invert สีของเฟรมที่คัดลอกมา (สลับสี)
        inverted_frame = cv2.bitwise_not(delayed_frame)
        
        # 5. ผสมเฟรมปัจจุบัน กับ เฟรมที่ Invert ตามค่า Opacity
        # สูตรทางคณิตศาสตร์: (Current Frame * (1 - Opacity)) + (Inverted Frame * Opacity)
        # ถ้า Opacity = 50% ส่วนที่ไม่มีการเปลี่ยนแปลงจะกลายเป็นสีเทากลาง (127.5) พอดี
        result = cv2.addWeighted(frame, 1.0 - opacity, inverted_frame, opacity, 0)
        
        # แสดงผลลัพธ์
        cv2.imshow('Motion Extraction Filter', result)
        
        # กดปุ่ม 'q' บนคีย์บอร์ดเพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()