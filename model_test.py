import cv2
import os
import glob
import csv
import gc
import urllib.request

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# --- 設定 ---
CLASS_NAMES = {
    0: "Mesh collapse",
    1: "Parts missing",
    2: "Face not displayed",
    3: "Texture discoloration"
}

IMAGE_DIR = r"C:\workspace\sotsuron\reindexed_images"
LABEL_DIR = r"C:\workspace\sotsuron\reindexed_labels"
OUTPUT_CSV = "model_comparison_results.csv"
OUTPUT_IMG_DIR = "visualized_results"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# --- ファイル自動ダウンロード ---
def download_model_file(url, filename):
    if not os.path.exists(filename):
        print(f"[{filename}] をダウンロード中...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"ダウンロード完了: {filename}")
        except Exception as e:
            print(f"ダウンロード失敗: {filename} ({e})")

YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
BLAZEFACE_MODEL = "blaze_face_short_range.tflite"
BLAZEFACE_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
DNN_PROTOTXT = "deploy.prototxt"
DNN_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

download_model_file(YUNET_URL, YUNET_MODEL)
download_model_file(BLAZEFACE_URL, BLAZEFACE_MODEL)
download_model_file(DNN_PROTOTXT_URL, DNN_PROTOTXT)
download_model_file(DNN_CAFFEMODEL_URL, DNN_CAFFEMODEL)


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

# --- メモリ解放ユーティリティ ---
def clear_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

# =========================================================
# モデル別処理関数群
# =========================================================

def evaluate_mtcnn(image_data):
    from mtcnn import MTCNN
    model = MTCNN()
    results_dict = {}
    for base_name, img_bgr, h, w in image_data:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            detections = model.detect_faces(img_rgb)
            boxes = [[x/w, y/h, (x+bw)/w, (y+bh)/h] for det in detections for x, y, bw, bh in [det['box']]]
        except:
            boxes = []
        results_dict[base_name] = boxes
    del model
    return results_dict

def evaluate_retinaface(image_data):
    from retinaface import RetinaFace
    results_dict = {}
    for base_name, img_bgr, h, w in image_data:
        detections = RetinaFace.detect_faces(img_bgr)
        boxes = []
        if type(detections) == dict:
            for key in detections.keys():
                x1, y1, x2, y2 = detections[key]["facial_area"]
                boxes.append([x1/w, y1/h, x2/w, y2/h])
        results_dict[base_name] = boxes
    return results_dict

def evaluate_blazeface(image_data):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not os.path.exists(BLAZEFACE_MODEL):
        print(f"エラー: {BLAZEFACE_MODEL} がありません。")
        return {}

    base_options = mp_python.BaseOptions(model_asset_path=BLAZEFACE_MODEL)
    options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    
    results_dict = {}
    with mp_vision.FaceDetector.create_from_options(options) as detector:
        for base_name, img_bgr, h, w in image_data:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            detections = detector.detect(mp_image)
            boxes = []
            for det in detections.detections:
                bb = det.bounding_box
                boxes.append([bb.origin_x / w, bb.origin_y / h, (bb.origin_x + bb.width) / w, (bb.origin_y + bb.height) / h])
            results_dict[base_name] = boxes
    return results_dict

def evaluate_yunet(image_data):
    if not os.path.exists(YUNET_MODEL):
        print(f"エラー: {YUNET_MODEL} が見つかりません。")
        return {}
        
    results_dict = {}
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320))
    
    for base_name, img_bgr, h, w in image_data:
        detector.setInputSize((w, h))
        _, results = detector.detect(img_bgr)
        boxes = []
        if results is not None:
            for face in results:
                x, y, bw, bh = face[:4]
                boxes.append([x/w, y/h, (x+bw)/w, (y+bh)/h])
        results_dict[base_name] = boxes
    del detector
    return results_dict

def evaluate_cv2_dnn(image_data):
    if not os.path.exists(DNN_PROTOTXT) or not os.path.exists(DNN_CAFFEMODEL):
        print(f"エラー: OpenCV DNNのモデルファイルが見つかりません。")
        return {}
        
    net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_CAFFEMODEL)
    results_dict = {}
    
    for base_name, img_bgr, h, w in image_data:
        blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = max(0.0, detections[0, 0, i, 3])
                y1 = max(0.0, detections[0, 0, i, 4])
                x2 = min(1.0, detections[0, 0, i, 5])
                y2 = min(1.0, detections[0, 0, i, 6])
                boxes.append([x1, y1, x2, y2])
        results_dict[base_name] = boxes
    del net
    return results_dict

def evaluate_yolo(image_data, model_path='yolov11n-face.pt'):
    from ultralytics import YOLO
    if not os.path.exists(model_path):
        print(f"エラー: {model_path} が見つかりません。")
        return {}
        
    model = YOLO(model_path)
    results_dict = {}
    for base_name, img_bgr, h, w in image_data:
        detections = model(img_bgr, verbose=False)
        boxes = [box.xyxyn[0].tolist() for box in detections[0].boxes]
        results_dict[base_name] = boxes
    del model
    return results_dict

# =========================================================
# メイン処理
# =========================================================

def main():
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.[pj][pn][g]"))
    
    print("画像の事前読み込みと正解ラベルのパースを開始します...")
    image_data = [] # [(base_name, img_bgr, h, w), ...]
    gt_data = {}    # {base_name: [(cls_id, [x1, y1, x2, y2]), ...]}
    
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
        if not os.path.exists(label_path): continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        h, w = img_bgr.shape[:2]
        
        gt_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                cls_id = int(float(parts[0]))
                if cls_id not in CLASS_NAMES: continue
                x_c, y_c, bw, bh = map(float, parts[1:])
                gt_boxes.append((cls_id, [x_c - bw/2, y_c - bh/2, x_c + bw/2, y_c + bh/2]))
        
        if gt_boxes: 
            image_data.append((base_name, img_bgr, h, w))
            gt_data[base_name] = gt_boxes

    print(f"有効な画像ファイル: {len(image_data)} 件")

    models_to_run = {
        "MTCNN": evaluate_mtcnn,
        "RetinaFace": evaluate_retinaface,
        "BlazeFace": evaluate_blazeface,
        "YuNet": evaluate_yunet,
        "OpenCV_DNN": evaluate_cv2_dnn,
        # "YOLOv11": evaluate_yolo,
    }

    all_stats = {
        model_name: {i: {"total": 0, "detected": 0} for i in CLASS_NAMES.keys()}
        for model_name in models_to_run.keys()
    }

    # --- 順番にロードして評価 ---
    for model_name, func in models_to_run.items():
        print(f"\n[{model_name}] の処理を開始します...")
        
        model_out_dir = os.path.join(OUTPUT_IMG_DIR, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        try:
            predictions = func(image_data)
            
            # --- 判定と画像描画ループ ---
            for base_name, img_bgr, h, w in image_data:
                pred_boxes = predictions.get(base_name, [])
                gt_boxes_for_img = gt_data.get(base_name, [])
                
                vis_img = img_bgr.copy()

                # 1. 正解ラベル (GT) のIoU判定と集計 (描画はしない)
                for cls_id, gt_box in gt_boxes_for_img:
                    all_stats[model_name][cls_id]["total"] += 1
                    if any(calculate_iou(gt_box, p_box) >= 0.5 for p_box in pred_boxes):
                        all_stats[model_name][cls_id]["detected"] += 1

                # 2. モデルの検出結果のみを描画 (青色)
                for p_box in pred_boxes:
                    x1, y1 = int(p_box[0] * w), int(p_box[1] * h)
                    x2, y2 = int(p_box[2] * w), int(p_box[3] * h)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(vis_img, model_name, (x1, max(y1-20, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                out_path = os.path.join(model_out_dir, f"{base_name}.jpg")
                cv2.imwrite(out_path, vis_img)

            print(f"[{model_name}] の処理が完了し、画像を出力しました。メモリを解放します。")
        except Exception as e:
            print(f"[{model_name}] の処理中にエラーが発生しました: {e}")
            
        clear_memory()

    # --- CSVファイルへ書き出し ---
    print(f"\n評価完了。結果を {OUTPUT_CSV} に書き出します。")
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Glitch Category", "Total Instances", "Detected", "Detection Rate (%)"])
        
        for model_name, stats in all_stats.items():
            for cls_id, name in CLASS_NAMES.items():
                s = stats[cls_id]
                total = s['total']
                detected = s['detected']
                if total > 0:
                    rate = (detected / total * 100)
                    writer.writerow([model_name, name, total, detected, f"{rate:.1f}"])
                
    print("完了しました！")

if __name__ == "__main__":
    main()