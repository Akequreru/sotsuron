import cv2
import numpy as np
import json
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import glob
import sys
import math

# --- 設定 ---
IMAGE_DIR = 'input_images'
CLASSES_FILE = 'classes.txt'
MASTER_OUTPUT_DIR = 'annotations_output'

OUTPUT_FOLDERS = {
    'yolo': 'yolo_txt',
    'voc': 'voc_xml',
    'coco': 'coco_json'
}

IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']

# --- 新しい設定: 最大表示サイズ (横幅, 高さ) ---
MAX_DISPLAY_SIZE = (1200, 800) # 画面に収まる最大の幅と高さ (ピクセル)

# --- グローバル変数 ---
drawing = False
start_point = (-1, -1)
class_list = []
class_to_id = {}
current_img_file = ""
current_img_copy = None

SCALE_FACTOR = 1.0 
ORIGINAL_IMG = None 
RESIZED_IMG_BASE = None 

EXISTING_ANNOTATIONS = []
NEW_ANNOTATIONS = []

# ==============================
# マウスコールバック
# ==============================
def draw_rectangle(event, x, y, flags, param):
    """
    OpenCVのマウスコールバック関数。
    x, y はリサイズ後の座標なので、元の画像サイズに変換してから保存する。
    """
    global drawing, start_point, NEW_ANNOTATIONS, current_img_copy, SCALE_FACTOR

    # リサイズ後の座標を元の画像座標に変換するヘルパー関数
    def get_original_coords(x_resized, y_resized):
        x_orig = int(round(x_resized / SCALE_FACTOR))
        y_orig = int(round(y_resized / SCALE_FACTOR))
        return x_orig, y_orig

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y) # リサイズ後の座標

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = current_img_copy.copy()
            # 描画はリサイズ後の画像上で行う
            cv2.rectangle(temp_img, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotation Tool", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y) # リサイズ後の座標

        x_min_resized = min(start_point[0], end_point[0])
        y_min_resized = min(start_point[1], end_point[1])
        x_max_resized = max(start_point[0], end_point[0])
        y_max_resized = max(start_point[1], end_point[1])

        # 描画した矩形が十分大きいかチェック (リサイズ後の座標で)
        if (x_max_resized - x_min_resized > 5) and (y_max_resized - y_min_resized > 5):
            
            # 座標を元の画像サイズに戻す
            x_min, y_min = get_original_coords(x_min_resized, y_min_resized)
            x_max, y_max = get_original_coords(x_max_resized, y_max_resized)
            
            print("\n--- ラベル入力 ---")
            print(f"クラスリスト: {class_list}")
            class_name = input("クラス名を入力してください: ").strip()

            if class_name in class_list:
                img_h, img_w, _ = ORIGINAL_IMG.shape # 元の画像のサイズを使用
                new_bbox = {'class_name': class_name, 'bbox_xyxy': [x_min, y_min, x_max, y_max], 'img_w': img_w, 'img_h': img_h}

                # 重複チェック (元の座標で比較)
                all_annotations = EXISTING_ANNOTATIONS + NEW_ANNOTATIONS
                if any(ann['class_name'] == new_bbox['class_name'] and ann['bbox_xyxy'] == new_bbox['bbox_xyxy'] for ann in all_annotations):
                    print("⚠️ 既存のアノテーションと重複しています。保存しませんでした。")
                    return

                NEW_ANNOTATIONS.append(new_bbox)

                # 描画はリサイズ後の画像上で行う
                cv2.rectangle(current_img_copy, (x_min_resized, y_min_resized), (x_max_resized, y_max_resized), (0, 255, 0), 2)
                cv2.putText(current_img_copy, class_name, (x_min_resized, y_min_resized - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Annotation Tool", current_img_copy)
                print(f"[{class_name}] のバウンディングボックスを保存しました (元座標: ({x_min}, {y_min}) - ({x_max}, {y_max}))。")
            else:
                print("⚠️ 無効なクラス名です。保存しませんでした。")

# ==============================
# 既存アノテーション読み込み
# ==============================
def load_existing_annotations(file_name):
    global EXISTING_ANNOTATIONS, class_list, ORIGINAL_IMG
    img_no_ext = os.path.splitext(file_name)[0]
    EXISTING_ANNOTATIONS.clear()
    
    if ORIGINAL_IMG is None: return

    # YOLO
    yolo_file = os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['yolo'], img_no_ext + ".txt")
    if os.path.exists(yolo_file):
        try:
            with open(yolo_file, "r") as f:
                lines = f.readlines()
            img_h, img_w, _ = ORIGINAL_IMG.shape
            for line in lines:
                cls_id, cx, cy, w, h = map(float, line.strip().split(" "))
                x1 = int(round((cx - w / 2) * img_w))
                y1 = int(round((cy - h / 2) * img_h))
                x2 = int(round((cx + w / 2) * img_w))
                y2 = int(round((cy + h / 2) * img_h))
                if 0 <= int(cls_id) < len(class_list):
                     cls_name = class_list[int(cls_id)]
                else:
                     cls_name = "UNKNOWN"
                EXISTING_ANNOTATIONS.append({"class_name": cls_name, "bbox_xyxy": [x1, y1, x2, y2], "img_w": img_w, "img_h": img_h})
            print(f"💡 既存YOLOアノテーション読み込み: {yolo_file}")
            return
        except Exception as e:
            print(f"⚠️ YOLO読み込みエラー: {e}")

    # VOC
    voc_file = os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['voc'], img_no_ext + ".xml")
    if os.path.exists(voc_file):
        try:
            tree = ET.parse(voc_file)
            root = tree.getroot()
            img_w = int(root.find("size/width").text)
            img_h = int(root.find("size/height").text)
            for obj in root.findall("object"):
                cls = obj.find("name").text
                x1 = int(obj.find("bndbox/xmin").text)
                y1 = int(obj.find("bndbox/ymin").text)
                x2 = int(obj.find("bndbox/xmax").text)
                y2 = int(obj.find("bndbox/ymax").text)
                EXISTING_ANNOTATIONS.append({"class_name": cls, "bbox_xyxy": [x1, y1, x2, y2], "img_w": img_w, "img_h": img_h})
            print(f"💡 既存VOCアノテーション読み込み: {voc_file}")
            return
        except Exception as e:
            print(f"⚠️ VOC読み込みエラー: {e}")

    # COCO
    coco_file = os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['coco'], "dataset_coco.json")
    if os.path.exists(coco_file):
        try:
            with open(coco_file, "r", encoding="utf-8") as f:
                coco = json.load(f)
            img_entry = next((img for img in coco["images"] if img["file_name"] == file_name), None)
            if img_entry:
                img_id = img_entry["id"]
                img_w, img_h = img_entry["width"], img_entry["height"]
                for ann in coco["annotations"]:
                    if ann["image_id"] == img_id:
                        x1, y1, w, h = ann["bbox"]
                        x2 = x1 + w
                        y2 = y1 + h
                        cls = coco["categories"][ann["category_id"] - 1]["name"]
                        EXISTING_ANNOTATIONS.append({"class_name": cls, "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)], "img_w": img_w, "img_h": img_h})
                print(f"💡 既存COCOアノテーション読み込み: {coco_file}")
                return
        except Exception as e:
            print(f"⚠️ COCO読み込みエラー: {e}")


# ==============================
# 画像処理ループ (ウィンドウ設定の強化)
# ==============================
def process_single_image(image_path):
    global NEW_ANNOTATIONS, current_img_copy, current_img_file, SCALE_FACTOR, ORIGINAL_IMG, RESIZED_IMG_BASE

    NEW_ANNOTATIONS = []
    current_img_file = os.path.basename(image_path)
    print(f"\n=====================================")
    print(f"➡️ 画像処理: {current_img_file}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 画像を読み込めません: {image_path}")
        return False
    
    # 元の画像を保持
    ORIGINAL_IMG = img.copy()
    original_h, original_w, _ = ORIGINAL_IMG.shape
    
    # --- 画像リサイズとスケーリング係数の計算 ---
    
    w_ratio = MAX_DISPLAY_SIZE[0] / original_w
    h_ratio = MAX_DISPLAY_SIZE[1] / original_h
    
    SCALE_FACTOR = min(w_ratio, h_ratio)
    
    if SCALE_FACTOR < 1.0:
        # 縮小が必要な場合のみリサイズ
        new_w = int(original_w * SCALE_FACTOR)
        new_h = int(original_h * SCALE_FACTOR)
        RESIZED_IMG_BASE = cv2.resize(ORIGINAL_IMG, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"ℹ️ 高解像度のため画像をリサイズ: {original_w}x{original_h} -> {new_w}x{new_h} (Scale: {SCALE_FACTOR:.3f})")
    else:
        # リサイズ不要な場合は元の画像をそのままベースとする
        RESIZED_IMG_BASE = ORIGINAL_IMG.copy()
        SCALE_FACTOR = 1.0
        print(f"ℹ️ 画像サイズ {original_w}x{original_h} は画面内に収まります。")

    # ベース画像から作業用コピーを作成
    current_img_copy = RESIZED_IMG_BASE.copy()

    # 既存アノテーション読み込み
    load_existing_annotations(current_img_file)

    # 既存アノテーション描画 (青)
    for ann in EXISTING_ANNOTATIONS:
        x1, y1, x2, y2 = ann['bbox_xyxy']
        class_name = ann['class_name']
        
        # 描画はリサイズ後の座標に変換して行う
        x1_r = int(x1 * SCALE_FACTOR)
        y1_r = int(y1 * SCALE_FACTOR)
        x2_r = int(x2 * SCALE_FACTOR)
        y2_r = int(y2 * SCALE_FACTOR)
        
        cv2.rectangle(current_img_copy, (x1_r, y1_r), (x2_r, y2_r), (255, 0, 0), 2)
        cv2.putText(current_img_copy, class_name, (x1_r, y1_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # --- ⭐ ウィンドウ設定の強化 ---
    WINDOW_NAME = "Annotation Tool"
    # ウィンドウ作成時にリサイズ可能フラグを設定 (最も重要)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 
    
    # リサイズ後にウィンドウサイズを画像サイズに合わせる
    cv2.resizeWindow(WINDOW_NAME, current_img_copy.shape[1], current_img_copy.shape[0])

    cv2.setMouseCallback(WINDOW_NAME, draw_rectangle)

    while True:
        cv2.setWindowTitle(WINDOW_NAME, f"Annotation Tool - {current_img_file}")
        cv2.imshow(WINDOW_NAME, current_img_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("\n💾 アノテーション保存＆エクスポート...")
            break
        elif key == ord('z'):
            if NEW_ANNOTATIONS:
                removed = NEW_ANNOTATIONS.pop()
                print(f"↩️ Undo: {removed['class_name']} 削除")
                
                # 描画をクリアするためにリサイズ後のベース画像から再作成する
                current_img_copy = RESIZED_IMG_BASE.copy()
                
                # 既存アノテーションと残りの新規アノテーションを再描画
                for ann in EXISTING_ANNOTATIONS + NEW_ANNOTATIONS:
                    x1, y1, x2, y2 = ann['bbox_xyxy']
                    class_name = ann['class_name']
                    
                    # 再描画もリサイズ後の座標に変換
                    x1_r = int(x1 * SCALE_FACTOR)
                    y1_r = int(y1 * SCALE_FACTOR)
                    x2_r = int(x2 * SCALE_FACTOR)
                    y2_r = int(y2 * SCALE_FACTOR)
                    
                    color = (255, 0, 0) if ann in EXISTING_ANNOTATIONS else (0, 255, 0) # 青または緑
                    cv2.rectangle(current_img_copy, (x1_r, y1_r), (x2_r, y2_r), color, 2)
                    cv2.putText(current_img_copy, class_name, (x1_r, y1_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                cv2.imshow(WINDOW_NAME, current_img_copy) # 更新された画像を表示

            else:
                print("⚠️ 削除可能なアノテーションはありません。")
        elif key == ord('q'):
            print("\n🛑 中断します")
            cv2.destroyAllWindows()
            sys.exit()

    export_annotations(current_img_file)
    return True

# ==============================
# エクスポート
# ==============================
# (変更なし)
def export_to_yolo(data, yolo_dir, file_name):
    os.makedirs(yolo_dir, exist_ok=True)
    if not data: return
    img_w, img_h = data[0]['img_w'], data[0]['img_h']
    yolo_lines = []
    for ann in data:
        x1, y1, x2, y2 = ann['bbox_xyxy']
        class_id = class_to_id[ann['class_name']]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        yolo_lines.append(f"{class_id} {x_center/img_w:.6f} {y_center/img_h:.6f} {width/img_w:.6f} {height/img_h:.6f}")
    with open(os.path.join(yolo_dir, os.path.splitext(file_name)[0] + ".txt"), 'w') as f:
        f.write('\n'.join(yolo_lines))
    print(f" - YOLO形式: {file_name} 保存")

def export_to_voc(data, voc_dir, file_name):
    os.makedirs(voc_dir, exist_ok=True)
    
    if not data: return 
    img_w, img_h = data[0]['img_w'], data[0]['img_h']
    root_name = os.path.splitext(file_name)[0]

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = file_name
    ET.SubElement(annotation, 'folder').text = os.path.basename(os.getcwd())

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_w)
    ET.SubElement(size, 'height').text = str(img_h)
    ET.SubElement(size, 'depth').text = '3'

    for ann in data:
        x1, y1, x2, y2 = ann['bbox_xyxy']

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = ann['class_name']
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'


        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(x1) 
        ET.SubElement(bndbox, 'ymin').text = str(y1)
        ET.SubElement(bndbox, 'xmax').text = str(x2)
        ET.SubElement(bndbox, 'ymax').text = str(y2)

    tree = ET.ElementTree(annotation)
    output_file = root_name + '.xml'
    xml_path = os.path.join(voc_dir, output_file)

    try:
        ET.indent(tree, space="  ", level=0) 
    except AttributeError:
        pass
        
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    print(f"   - VOC形式: {os.path.join(voc_dir, output_file)}")

def export_to_coco(data, coco_dir, file_name):
    os.makedirs(coco_dir, exist_ok=True)
    coco_path = os.path.join(coco_dir, "dataset_coco.json")
    if os.path.exists(coco_path):
        with open(coco_path, 'r', encoding='utf-8') as f:
            try:
                coco_json = json.load(f)
            except:
                coco_json = None

    if coco_json is None:
        coco_json = {"info": {"description": "Custom Dataset"}, "licenses": [], "categories": [{"id": i+1, "name": name, "supercategory": "none"} for i, name in enumerate(class_list)], "images": [], "annotations": []}

    img_entry = next((img for img in coco_json["images"] if img["file_name"] == file_name), None)
    if img_entry:
        img_id = img_entry["id"]
        coco_json["images"] = [img for img in coco_json["images"] if img["id"] != img_id]
        coco_json["annotations"] = [ann for ann in coco_json["annotations"] if ann["image_id"] != img_id]
    else:
        max_img_id = max([img['id'] for img in coco_json["images"]] + [0])
        img_id = max_img_id + 1

    if not data: return
    img_w, img_h = data[0]['img_w'], data[0]['img_h']
    coco_json["images"].append({"id": img_id, "width": img_w, "height": img_h, "file_name": file_name})
    ann_start = len(coco_json["annotations"]) + 1
    for i, ann in enumerate(data):
        x1, y1, x2, y2 = ann['bbox_xyxy']
        coco_json["annotations"].append({
            "id": ann_start + i,
            "image_id": img_id,
            "category_id": class_to_id[ann['class_name']] + 1,
            "bbox": [x1, y1, x2-x1, y2-y1],
            "area": (x2-x1)*(y2-y1),
            "iscrowd": 0
        })
    with open(coco_path, 'w', encoding='utf-8') as f:
        json.dump(coco_json, f, ensure_ascii=False, indent=2)
    print(f" - COCO形式: {file_name} 保存")

def export_annotations(file_name):
    all_data = EXISTING_ANNOTATIONS + NEW_ANNOTATIONS
    if not all_data:
        print("💡 アノテーションなし。スキップ")
        return

    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
    export_to_yolo(all_data, os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['yolo']), file_name)
    export_to_voc(all_data, os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['voc']), file_name)
    export_to_coco(all_data, os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['coco']), file_name)
    print(f"✅ {file_name} のエクスポート完了")


# ==============================
# メインループ
# ==============================
def main_annotation_loop():
    global class_list, class_to_id
    try:
        with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]
        class_to_id = {name: i for i, name in enumerate(class_list)}
        print(f"✅ クラスリスト読み込み ({len(class_list)} 件): {class_list}")
    except FileNotFoundError:
        print(f"❌ クラスファイルなし: {CLASSES_FILE}")
        return

    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    if not image_paths:
        print(f"❌ 画像なし: {IMAGE_DIR}")
        return

    print(f"🖼️ 合計 {len(image_paths)} 枚を処理")
    for path in image_paths:
        process_single_image(path)

    print("\n🎉 すべて完了")
    cv2.destroyAllWindows()

# --- 実行 ---
if __name__ == "__main__":
    main_annotation_loop()