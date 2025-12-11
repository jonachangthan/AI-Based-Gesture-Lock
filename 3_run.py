import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
from utils import extract_features 

def run():
    # 1. 檢查模型是否存在
    if not os.path.exists('gesture_password_model.h5'):
        print("錯誤：找不到模型檔案，請先執行 2_train.py")
        return

    print("=== 啟動手勢解密系統 ===")
    print("載入 AI 模型中...")
    
    # 2. 載入模型與設定
    try:
        model = tf.keras.models.load_model('gesture_password_model.h5')
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return

    # ==========================================
    # [設定區] 請在此定義您的「正確密碼」
    # 對應您在步驟 1 收集的: 0=拳頭, 1=手掌, 2=OK
    # ==========================================
    PASSWORD_SEQUENCE = [0, 1, 2] 
    GESTURE_NAMES = {
        0: "Fist", 
        1: "Palm", 
        2: "OK", 
        3: "Unknown"
    }
    # ==========================================

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.8, 
        min_tracking_confidence=0.8
    )
    
    # 系統變數初始化
    input_sequence = []       # 存放使用者目前的輸入
    last_prediction = -1      # 上一幀的預測結果
    stability_counter = 0     # 穩定度計數器
    STABILITY_THRESHOLD = 10  # 需連續 10 幀相同才算數 (防手抖)
    last_input_time = time.time()
    
    # UI 狀態
    system_state = "LOCKED"   # LOCKED / UNLOCKED
    feedback_color = (0, 0, 255) # 紅色代表鎖定

    cap = cv2.VideoCapture(0)
    
    print("系統啟動完成！請對鏡頭輸入手勢密碼。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 畫面翻轉與色彩轉換
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        current_gesture = -1
        display_msg = "Scan your hand..."

        # --- A. 手勢偵測與推論 ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 繪製骨架
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 提取特徵並預測
                feats = extract_features(hand_landmarks.landmark)
                pred = model.predict(np.array([feats]), verbose=0)
                confidence = np.max(pred)
                
                # 若信心度夠高 (>0.85)，才視為有效手勢
                if confidence > 0.85:
                    current_gesture = np.argmax(pred)
                    gesture_name = GESTURE_NAMES.get(current_gesture, str(current_gesture))
                    display_msg = f"Detecting: {gesture_name}"

        # --- B. 邏輯核心 (狀態機) ---
        current_time = time.time()
        
        # 1. 超時重置：如果超過 5 秒沒動作，清空輸入
        if current_time - last_input_time > 5.0 and len(input_sequence) > 0:
            input_sequence = []
            print("--- 輸入超時，已重置 ---")
        
        # 2. 防抖動確認輸入
        if current_gesture != -1:
            if current_gesture == last_prediction:
                stability_counter += 1
                if stability_counter == STABILITY_THRESHOLD:
                    # 只有當「輸入序列為空」或者「新輸入與上一個不同」時才加入
                    # (避免比著拳頭不放，系統一直重複輸入 0 0 0 0)
                    if not input_sequence or input_sequence[-1] != current_gesture:
                        input_sequence.append(current_gesture)
                        last_input_time = current_time
                        print(f"輸入確認: {GESTURE_NAMES.get(current_gesture)}")
            else:
                stability_counter = 0
                last_prediction = current_gesture
        else:
            stability_counter = 0

        # --- C. 密碼驗證 ---
        if input_sequence == PASSWORD_SEQUENCE:
            system_state = "UNLOCKED"
            feedback_color = (0, 255, 0) # 轉綠色
            # 這裡可以加入實際功能，例如打開一個檔案或發送訊號
            
        elif len(input_sequence) >= len(PASSWORD_SEQUENCE) and input_sequence != PASSWORD_SEQUENCE:
            # 輸入長度已滿但密碼錯誤
            system_state = "WRONG PWD"
            input_sequence = [] # 錯誤重置
            print("!!! 密碼錯誤 !!!")

        # --- D. 繪製精美 UI ---
        # 1. 頂部黑色背景條
        cv2.rectangle(frame, (0, 0), (640, 80), (20, 20, 20), -1)
        
        # 2. 狀態顯示 (LOCKED / UNLOCKED)
        cv2.putText(frame, f"STATUS: {system_state}", (400, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)

        # 3. 顯示目前輸入序列 (將數字轉為手勢名稱)
        seq_names = " > ".join([str(x) for x in input_sequence])
        cv2.putText(frame, f"Input: {seq_names}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 4. 顯示目前偵測到的手勢文字 (在手部附近或左下角)
        cv2.putText(frame, display_msg, (20, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # 5. 解鎖特效
        if system_state == "UNLOCKED":
            cv2.putText(frame, "ACCESS GRANTED", (130, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.imshow('Gesture Security', frame)
            
            # 暫停 2 秒展示成功畫面，然後重置系統
            if cv2.waitKey(1): # 保持畫面刷新
                time.sleep(2)
                system_state = "LOCKED"
                feedback_color = (0, 0, 255)
                input_sequence = []
                print("--- 系統重新鎖定 ---")

        cv2.imshow('Gesture Security', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()