import cv2
import os
import glob
import random

# --- 設定 ---
IMAGE_DIR = 'output'
LABEL_DIR = 'merged_output'
CLASSES_FILE = 'classes.txt'
WINDOW_NAME = "YOLO Annotation Verification"

# ★ 画面に収まる最大サイズを指定してください
MAX_WIDTH = 1280
MAX_HEIGHT = 720

CLASS_COLORS = {}

def get_unique_color(class_name):
    if class_name not in CLASS_COLORS:
        CLASS_COLORS[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return CLASS_COLORS[class_name]

def load_classes(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]
        return {i: name for i, name in enumerate(class_list)}
    except FileNotFoundError:
        print(f"❌ エラー: '{file_path}' が見つかりません。")
        return {}

def get_yolo_annotations(label_path, img_w, img_h, class_map):
    annotations = []
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            class_id = int(parts[0])
            x_c, y_c, w_norm, h_norm = [float(p) for p in parts[1:]]
            x_min = int((x_c - w_norm / 2) * img_w)
            y_min = int((y_c - h_norm / 2) * img_h)
            x_max = int((x_c + w_norm / 2) * img_w)
            y_max = int((y_c + h_norm / 2) * img_h)
            annotations.append({'bbox': [x_min, y_min, x_max, y_max], 'label': class_map.get(class_id, f"ID:{class_id}")})
    return annotations

def visualize_yolo():
    class_map = load_classes(CLASSES_FILE)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_paths:
        print(f"❌ エラー: 画像が {IMAGE_DIR} に見つかりません。")
        return

    print("--- 検証開始 (Q:終了 / S:次へ) ---")

    for img_path in sorted(image_paths):
        img_filename = os.path.basename(img_path)
        img_id = os.path.splitext(img_filename)[0]
        label_path = os.path.join(LABEL_DIR, f"{img_id}.txt")

        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape

        annos = get_yolo_annotations(label_path, w, h, class_map)
        temp_img = img.copy()

        # 描画
        cv2.putText(temp_img, f"File: {img_filename}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        for ann in annos:
            x1, y1, x2, y2 = ann['bbox']
            label = ann['label']
            color = get_unique_color(label)
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 > 25 else y1 + 25
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(temp_img, (x1, text_y - th - 5), (x1 + tw, text_y + 5), color, -1)
            cv2.putText(temp_img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # --- リサイズ処理 ---
        # 縦横どちらかが制限を超えている場合のみ縮小
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
            new_w, new_h = int(w * scale), int(h * scale)
            display_img = cv2.resize(temp_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_img = temp_img

        cv2.imshow(WINDOW_NAME, display_img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_yolo()