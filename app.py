import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import json 
from utils import extract_features 

# ==========================================
# 1. 系統與路徑設定
# ==========================================
st.set_page_config(page_title="AI Gesture Security System", page_icon="🔒", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASSWORD_FILE = os.path.join(BASE_DIR, "password_config.json")
MODEL_PATH = os.path.join(BASE_DIR, 'gesture_password_model.h5')

# --- 檔案存取函式 ---
def save_password_to_file(sequence):
    """將密碼寫入 JSON 檔案"""
    try:
        # 確保轉為標準 int
        clean_sequence = [int(x) for x in sequence]
        
        with open(PASSWORD_FILE, "w", encoding='utf-8') as f:
            json.dump(clean_sequence, f)
            f.flush()
            os.fsync(f.fileno()) 
        return True, PASSWORD_FILE
    except Exception as e:
        return False, str(e)

def load_password_from_file():
    """讀取密碼"""
    if not os.path.exists(PASSWORD_FILE):
        default_pwd = [0, 1, 2] 
        save_password_to_file(default_pwd)
        return default_pwd
    
    try:
        with open(PASSWORD_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    except:
        return [0, 1, 2]

# --- AI 模型載入 ---
@st.cache_resource
def load_ai_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    model = tf.keras.models.load_model(MODEL_PATH)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1, 
        min_detection_confidence=0.8, 
        min_tracking_confidence=0.8
    )
    return model, hands, mp_hands

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# ==========================================
# 2. 登入邏輯 (維持 AI 辨識)
# ==========================================

def run_login_process(model, hands, mp_hands):
    GESTURE_NAMES = {0: "拳頭", 1: "手掌", 2: "OK"}
    SAVED_PASSWORD = load_password_from_file()
    
    input_sequence = []
    last_prediction = -1
    stability_counter = 0
    STABILITY_THRESHOLD = 8
    last_input_time = time.time()
    gesture_triggered = False 

    st.info(f"🟢 請輸入手勢密碼進行解鎖 (密碼長度: {len(SAVED_PASSWORD)})")
    st.info("💡 提示：若要輸入連續相同手勢（如拳頭、拳頭），請在兩次之間將手放下。")
    stop_btn = st.button("停止/返回")
    
    image_placeholder = st.empty()
    sequence_display = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        current_gesture = -1
        display_msg = "Waiting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                feats = extract_features(hand_landmarks.landmark)
                pred = model.predict(np.array([feats]), verbose=0)
                if np.max(pred) > 0.85:
                    current_gesture = np.argmax(pred)
                    display_msg = f"Detected: {GESTURE_NAMES.get(current_gesture)}"

        # 狀態機
        current_time = time.time()
        if current_time - last_input_time > 5.0 and input_sequence:
            input_sequence = []
            st.toast("⏳ 超時重置", icon="⚠️")
            gesture_triggered = False

        if current_gesture != -1:
            if current_gesture == last_prediction:
                stability_counter += 1
                if stability_counter == STABILITY_THRESHOLD and not gesture_triggered:
                    input_sequence.append(int(current_gesture))
                    last_input_time = current_time
                    gesture_triggered = True 
                    st.toast(f"輸入：{GESTURE_NAMES.get(current_gesture)}", icon="✅")
            else:
                stability_counter = 0
                last_prediction = current_gesture
                gesture_triggered = False
        else:
            stability_counter = 0
            gesture_triggered = False

        # 驗證
        if input_sequence == SAVED_PASSWORD:
            st.success("✅ 密碼正確！正在登入...")
            st.session_state['logged_in'] = True
            cap.release()
            time.sleep(1)
            st.rerun()
            break
        elif len(input_sequence) >= len(SAVED_PASSWORD) and input_sequence != SAVED_PASSWORD:
            input_sequence = []
            st.toast("❌ 密碼錯誤", icon="🚫")

        # 畫面更新
        seq_str = " -> ".join([str(x) for x in input_sequence]) if input_sequence else "..."
        sequence_display.metric("目前輸入", seq_str)
        cv2.putText(frame, display_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        image_placeholder.image(frame, channels="RGB")
        time.sleep(0.01)

    cap.release()

# ==========================================
# 3. 註冊介面 (改為數字輸入版)
# ==========================================
def register_tab_ui():
    st.markdown("### 📝 設定新密碼")
    
    current_pwd = load_password_from_file()
    
    # 顯示對照表
    st.info("""
    **手勢代碼對照表：**
    * `0` : ✊ 拳頭 (Fist)
    * `1` : ✋ 手掌 (Palm)
    * `2` : 👌 OK手勢 (OK)
    """)
    
    st.divider()
    
    # 顯示目前密碼
    st.write(f"目前密碼序列: `{current_pwd}`")
    
    # --- 文字輸入框 ---
    user_input = st.text_input("請輸入新的密碼序列 (請用逗號分隔，例如: 0, 0, 1, 2)")
    
    if st.button("💾 更新密碼", type="primary"):
        if not user_input.strip():
            st.warning("密碼不能為空！")
        else:
            try:
                # 1. 解析字串轉為數字列表
                # 例如 "0, 1, 2" -> [0, 1, 2]
                new_sequence = []
                parts = user_input.split(',')
                
                valid = True
                for p in parts:
                    num = int(p.strip())
                    if num not in [0, 1, 2]: # 檢查是否為有效手勢 ID
                        valid = False
                        st.error(f"錯誤：數字 '{num}' 無效！只能輸入 0, 1 或 2。")
                        break
                    new_sequence.append(num)
                
                # 2. 儲存
                if valid:
                    success, msg = save_password_to_file(new_sequence)
                    if success:
                        st.success(f"密碼更新成功！新序列為: {new_sequence}")
                        time.sleep(1.5)
                        st.rerun() # 重新整理頁面
                    else:
                        st.error(f"儲存失敗: {msg}")
                        
            except ValueError:
                st.error("格式錯誤！請只輸入數字和逗號 (例如: 0, 1, 2)")

# ==========================================
# 4. 主程式入口
# ==========================================

def auth_portal():
    st.title("🔒 AI Gesture Security System")
    
    model, hands, mp_hands = load_ai_resources()
    if model is None:
        st.error("錯誤：找不到模型檔案 gesture_password_model.h5")
        return

    tab1, tab2 = st.tabs(["🔑 登入系統", "📝 設定密碼 (數字輸入)"])

    with tab1:
        if st.button("啟動登入辨識", key="start_login", type="primary"):
            run_login_process(model, hands, mp_hands)
    
    with tab2:
        # 這裡不需要傳入 model 了，因為只剩下純文字操作
        register_tab_ui()

def main_dashboard():
    st.balloons()
    st.title("👋 Welcome Admin!")
    st.success("Identity Verified: Access Granted.")
    st.divider()
    
    st.write("### 安全控制台")
    st.json({
        "User": "Administrator",
        "Access Level": "Root",
        "System Time": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

if __name__ == "__main__":
    if st.session_state['logged_in']:
        main_dashboard()
    else:

        auth_portal()
