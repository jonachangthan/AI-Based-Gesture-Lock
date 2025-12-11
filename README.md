# 🔒 AI-Based-Gesture-Lock

[](https://www.python.org/)
[](https://tensorflow.org/)
[](https://mediapipe.dev/)
[](https://streamlit.io/)

這是一個基於電腦視覺與深度學習的生物辨識安全系統。透過 Google MediaPipe 追蹤手部骨架，並利用 TensorFlow/Keras 進行手勢分類，使用者可以透過特定的「手勢序列」（例如：拳頭 -\> 手掌 -\> OK）來解鎖系統。

本專案包含完整的 **資料收集**、**模型訓練**、**即時偵測 (CLI)** 以及 **Web 登入介面 (Streamlit)**。

-----

## ✨ 功能特色

  * **即時手勢辨識**：利用 MediaPipe 進行毫秒級的手部骨架追蹤。
  * **自定義密碼**：支援設定任意長度的手勢序列作為密碼（預設：✊ -\> ✋ -\> 👌）。
  * **狀態機邏輯**：內建防抖動 (Debounce) 與超時重置機制，防止誤觸。
  * **雙重介面**：
      * 🛠 **Debug 模式**：透過 OpenCV 視窗查看即時預測數據。
      * 🌐 **Web 登入系統**：現代化的 Streamlit 介面，模擬真實登入情境。

-----

## 📂 檔案結構

```text
Project/
├── 1_collect.py              # [步驟1] 資料收集工具
├── 2_train.py                # [步驟2] AI 模型訓練腳本
├── 3_run.py                  # [步驟3] CLI 即時解密/除錯工具
├── app.py                    # [步驟4] Streamlit Web 登入介面
├── utils.py                  # 共用工具 (特徵提取函式)
├── gesture_data.csv          # 收集到的手勢數據 (自動生成)
├── gesture_password_model.h5 # 訓練好的 AI 模型 (自動生成)
├── requirements.txt          # 專案依賴套件清單
└── README.md                 # 說明文件
```

-----

## 🚀 快速安裝

### 1\. 環境準備

建議使用 Python 3.8 或以上版本。

### 2\. 安裝依賴套件

在終端機 (Terminal) 執行以下指令：

```bash
pip install -r requirements.txt
```

> **注意**：若遇到 `protobuf` 相關錯誤，請執行 `pip install "protobuf<3.20.x"` 進行降版。

-----

## 📖 使用教學 (Step-by-Step)

請依照順序執行以下腳本：

### Step 1: 收集手勢資料

建立屬於您的手勢資料庫。

```bash
python 1_collect.py
```

  * **操作方式**：
      * 對著鏡頭比出 **拳頭**，長按鍵盤 **`0`** 儲存數據。
      * 對著鏡頭比出 **手掌**，長按鍵盤 **`1`** 儲存數據。
      * 對著鏡頭比出 **OK**，長按鍵盤 **`2`** 儲存數據。
      * 按 **`q`** 離開。
  * *建議每個手勢至少收集 300 筆數據。*

### Step 2: 訓練 AI 模型

讓電腦學習您剛剛收集的手勢。

```bash
python 2_train.py
```

  * 程式會讀取 `gesture_data.csv` 並進行訓練。
  * 訓練完成後會產生 `gesture_password_model.h5`。

### Step 3: 測試與除錯 (CLI)

在終端機查看辨識信心度與密碼邏輯。

```bash
python 3_run.py
```

  * **解鎖密碼**：依序比出 `0 (拳頭)` -\> `1 (手掌)` -\> `2 (OK)`。
  * 觀察終端機輸出的 Confidence (信心度) 數值。

### Step 4: 啟動 Web 登入系統

漂亮的圖形化介面體驗。

```bash
streamlit run app.py
```

  * 瀏覽器會自動開啟 (預設網址 `http://localhost:8501`)。
  * 點擊 **「啟動攝影機辨識」** 開始解鎖。
  * 解鎖成功後會自動跳轉至後台管理頁面。

-----

## ⚙️ 進階設定

### 修改密碼順序
修改密碼兩種方式 : 
1. 執行app.py時可再登入頁面進行修改密碼

2. 若要更改密碼，請開啟 `3_run.py` 或 `app.py`，找到以下段落進行修改：
```python
# 例如改成: OK -> 拳頭 -> OK
PASSWORD_SEQUENCE = [2, 0, 2] 
```

### 修改手勢名稱

若您收集了不同的手勢 (例如把 OK 改成 讚)，請修改字典：

```python
GESTURE_NAMES = {
    0: "Fist (拳頭)", 
    1: "Palm (手掌)", 
    2: "ThumbsUp (讚)"
}
```

-----

## 🛠 常見問題排解 (Troubleshooting)

**Q1: 執行時閃退，出現 `SymbolDatabase.GetPrototype() is deprecated` 錯誤？**

  * **原因**：`protobuf` 版本與 `mediapipe` 不相容。
  * **解法**：執行 `pip install "protobuf<3.20.x"`。

**Q2: 找不到攝影機 (Camera index out of range)？**

  * **解法**：請開啟 `1_collect.py` 或 `app.py`，將 `cv2.VideoCapture(0)` 改為 `(1)` 或 `(2)` 試試看。

**Q3: AI 辨識很不準確？**

  * **解法**：
    1.  請重新執行 `Step 1`，收集更多數據。
    2.  收集時請稍微旋轉手腕、改變遠近，增加資料的多樣性 (Data Augmentation)。
    3.  確保背景不要太雜亂，光線充足。

-----
## DEMO影片
https://github.com/user-attachments/assets/eb3d7c3e-0ff1-40a0-899c-25e64cf3be6b

