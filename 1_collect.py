# 1_collect.py
import cv2
import mediapipe as mp
import csv
import os
from utils import extract_features  # 引用 utils.py

def run():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    
    # 若檔案不存在則建立
    if not os.path.exists('gesture_data.csv'):
        with open('gesture_data.csv', 'w', newline='') as f:
            pass 

    print("=== [步驟 1] 資料收集模式 ===")
    print("請面對鏡頭，做出手勢並長按數字鍵存檔：")
    print("  按 '0': 存為手勢 0 (拳頭)")
    print("  按 '1': 存為手勢 1 (手掌)")
    print("  按 '2': 存為手勢 2 (OK)")
    print("  按 'q': 離開")

    with open('gesture_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            status_text = "Press 0, 1, or 2"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    key = cv2.waitKey(1)
                    if key >= 48 and key <= 57: # 0-9
                        label = key - 48
                        features = extract_features(hand_landmarks.landmark)
                        writer.writerow([label] + features)
                        status_text = f"Saved Label: {label}"
                        print(f"已儲存 -> Label: {label}")
            
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Step 1: Collection', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()