import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import glob
import urllib.request

# --- 設定 ---
CLASS_NAMES = {
    0: "Mesh collapse",
    1: "Parts missing",
    2: "Face not displayed",
    3: "Texture discoloration"
}

IMAGE_DIR = r"C:\workspace\sotsuron\reindexed_images"
LABEL_DIR = r"C:\workspace\sotsuron\reindexed_labels"
OUTPUT_DIR = "mediapipe_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- モデルファイルのダウンロード ---
DETECTOR_MODEL = "blaze_face_short_range.tflite"
LANDMARKER_MODEL = "face_landmarker.task"

def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")

download_model(
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    DETECTOR_MODEL
)
download_model(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    LANDMARKER_MODEL
)

# --- IoU計算 ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

# --- 描画ユーティリティ ---
# FaceMesh の主要輪郭接続を手動定義 (mp.solutions依存を排除)
FACE_MESH_CONNECTIONS = [
    # 顔の輪郭
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),(356,454),
    (454,323),(323,361),(361,288),(288,397),(397,365),(365,379),(379,378),(378,400),
    (400,377),(377,152),(152,148),(148,176),(176,149),(149,150),(150,136),(136,172),
    (172,58),(58,132),(132,93),(93,234),(234,127),(127,162),(162,21),(21,54),
    (54,103),(103,67),(67,109),(109,10),
    # 左目
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
    (133,173),(173,157),(157,158),(158,159),(159,160),(160,161),(161,246),(246,33),
    # 右目
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),(249,263),
    (263,466),(466,388),(388,387),(387,386),(386,385),(385,384),(384,398),(398,362),
    # 口
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
    (321,375),(375,291),(291,409),(409,270),(270,269),(269,267),(267,0),(0,37),
    (37,39),(39,40),(40,185),(185,61),
    # 鼻
    (1,2),(2,98),(98,97),(2,326),(326,327),
]

def draw_landmarks_on_image(image, detection_result):
    """face_landmarker の結果をimage上に描画する"""
    h, w = image.shape[:2]
    for face_landmarks in detection_result.face_landmarks:
        # ランドマーク点を描画
        for lm in face_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)
        # メッシュの接続線を描画
        for conn in FACE_MESH_CONNECTIONS:
            a, b = conn
            x0 = int(face_landmarks[a].x * w)
            y0 = int(face_landmarks[a].y * h)
            x1 = int(face_landmarks[b].x * w)
            y1 = int(face_landmarks[b].y * h)
            cv2.line(image, (x0, y0), (x1, y1), (0, 200, 0), 1)

def evaluate_glitch_detection():
    stats = {i: {"total": 0, "detected": 0, "landmarked": 0} for i in CLASS_NAMES.keys()}

    # --- Tasks API オプション設定 ---
    detector_options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=DETECTOR_MODEL),
        min_detection_confidence=0.5
    )
    landmarker_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL),
        num_faces=20,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )

    with mp_vision.FaceDetector.create_from_options(detector_options) as detector, \
         mp_vision.FaceLandmarker.create_from_options(landmarker_options) as landmarker:

        image_files = glob.glob(os.path.join(IMAGE_DIR, "*"))

        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
            if not os.path.exists(label_path):
                continue

            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            h, w = image_bgr.shape[:2]

            # MediaPipe Image に変換 (RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            )

            vis_img = image_bgr.copy()

            # 1. FaceDetector 実行
            det_result = detector.detect(mp_image)
            mp_det_boxes = []
            for det in det_result.detections:
                bb = det.bounding_box
                # bounding_box は pixel 座標なので正規化する
                box = [
                    bb.origin_x / w,
                    bb.origin_y / h,
                    (bb.origin_x + bb.width) / w,
                    (bb.origin_y + bb.height) / h
                ]
                mp_det_boxes.append(box)
                # 検出枠を描画 (青)
                cv2.rectangle(
                    vis_img,
                    (bb.origin_x, bb.origin_y),
                    (bb.origin_x + bb.width, bb.origin_y + bb.height),
                    (255, 0, 0), 2
                )

            # 2. FaceLandmarker 実行
            lm_result = landmarker.detect(mp_image)
            mp_mesh_boxes = []
            for face_landmarks in lm_result.face_landmarks:
                xs = [lm.x for lm in face_landmarks]
                ys = [lm.y for lm in face_landmarks]
                mp_mesh_boxes.append([min(xs), min(ys), max(xs), max(ys)])
            draw_landmarks_on_image(vis_img, lm_result)

            # 3. 集計ロジック
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    cls_id = int(float(parts[0]))
                    if cls_id not in stats:
                        continue

                    x_c, y_c, bw, bh = map(float, parts[1:])
                    gt_box = [x_c - bw/2, y_c - bh/2, x_c + bw/2, y_c + bh/2]
                    stats[cls_id]["total"] += 1

                    if any(calculate_iou(gt_box, mb) >= 0.5 for mb in mp_det_boxes):
                        stats[cls_id]["detected"] += 1

                    if any(calculate_iou(gt_box, mb) >= 0.5 for mb in mp_mesh_boxes):
                        stats[cls_id]["landmarked"] += 1

            # 結果画像を保存
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"mp_{base_name}.jpg"), vis_img)
            print(f"Processed: {base_name}")

    # --- 最終集計結果の出力 ---
    print(f"\n{'Glitch Category':<25} | {'Count':<6} | {'Detect%':<10} | {'Mesh%':<10}")
    print("-" * 60)
    for cid, name in CLASS_NAMES.items():
        s = stats[cid]
        d_rate = (s['detected'] / s['total'] * 100) if s['total'] > 0 else 0
        l_rate = (s['landmarked'] / s['total'] * 100) if s['total'] > 0 else 0
        print(f"{name:<25} | {s['total']:<6} | {d_rate:>8.1f}% | {l_rate:>8.1f}%")

if __name__ == "__main__":
    evaluate_glitch_detection()