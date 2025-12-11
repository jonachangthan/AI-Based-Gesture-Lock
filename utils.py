# utils.py
def extract_features(landmarks):
    """
    將 21 個關節點轉換為相對於手腕的座標 (42維向量)。
    這是為了確保手勢在畫面任何位置都能被辨識。
    """
    base_x, base_y = landmarks[0].x, landmarks[0].y
    features = []
    for lm in landmarks:
        features.append(lm.x - base_x)
        features.append(lm.y - base_y)
    return features