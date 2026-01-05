import cv2
import json
import os
import glob
import xml.etree.ElementTree as ET
import random # 色をランダムに生成するために追加

# --- 設定 ---
IMAGE_DIR = 'input_images'
CLASSES_FILE = 'classes.txt'
WINDOW_NAME = "Annotation Verification"

# フォルダ設定はアノテーションツールと一致させる必要があります
OUTPUT_FOLDERS = {
    'yolo': 'annotations_output/yolo_txt',
    'voc': 'annotations_output/voc_xml',
    'coco': 'annotations_output/coco_json'
}

# クラスごとの色を保持する辞書
# この辞書は、スクリプトの実行中に一度だけ生成されます
CLASS_COLORS = {}

def get_unique_color(class_name):
    """クラス名に基づいて一意の色を生成または取得する"""
    if class_name not in CLASS_COLORS:
        # ランダムな色を生成 (BGR形式)
        # 0-255 の範囲でランダムな整数を生成
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        CLASS_COLORS[class_name] = (b, g, r)
    return CLASS_COLORS[class_name]


def load_classes(file_path):
    """classes.txt を読み込み、IDと名前のマップを返す"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            class_list = [line.strip() for line in f if line.strip()]
        
        # ここでCLASS_COLORSを初期化
        for class_name in class_list:
            get_unique_color(class_name) # 各クラスに色を割り当てておく
            
        return {i: name for i, name in enumerate(class_list)} 
    except FileNotFoundError:
        print(f"❌ エラー: クラスファイル '{file_path}' が見つかりません。")
        return {}

def get_annotations(img_filename_no_ext, img_filename, img_w, img_h, format_type):

    """指定された形式のアノテーションを読み込み、共通形式 (xmin, ymin, xmax, ymax, label) に変換する"""
    
    annotations = []
    class_map = load_classes(CLASSES_FILE) # class_map をロードし、同時に CLASS_COLORS も初期化
    
    # --- YOLO 形式 (.txt) の読み込み ---
    if format_type == 'yolo':
        yolo_path = os.path.join(OUTPUT_FOLDERS['yolo'], f"{img_filename_no_ext}.txt")
        if not os.path.exists(yolo_path): return []
        
        with open(yolo_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                class_id = int(parts[0])
                x_c, y_c, w_norm, h_norm = [float(p) for p in parts[1:]]
                
                x = x_c * img_w; y = y_c * img_h
                w = w_norm * img_w; h = h_norm * img_h
                
                x_min = int(x - w / 2); y_min = int(y - h / 2)
                x_max = int(x + w / 2); y_max = int(y + h / 2)
                
                annotations.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'label': class_map.get(class_id, f"ID:{class_id}_Unknown")
                })
        
    # --- PASCAL VOC 形式 (.xml) の読み込み ---
    elif format_type == 'voc':
        voc_path = os.path.join(OUTPUT_FOLDERS['voc'], f"{img_filename_no_ext}.xml")
        if not os.path.exists(voc_path): return []

        try:
            tree = ET.parse(voc_path)
            root = tree.getroot()
        except ET.ParseError:
            print(f"⚠️ XMLパースエラー: {voc_path}")
            return []
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            x_min = int(float(bndbox.find('xmin').text))
            y_min = int(float(bndbox.find('ymin').text))
            x_max = int(float(bndbox.find('xmax').text))
            y_max = int(float(bndbox.find('ymax').text))
            
            annotations.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'label': label
            })
            
    # --- COCO 形式 (.json) の読み込み ---
    elif format_type == 'coco': 
        coco_path = os.path.join(OUTPUT_FOLDERS['coco'], "dataset_coco.json") 
        if not os.path.exists(coco_path): 
            return [] 

        # COCO データセットの全情報をロード 
        with open(coco_path, 'r', encoding='utf-8') as f: 
            coco_data = json.load(f) 

        # file_name で画像を特定 
        image_entry = None 
        for img in coco_data.get('images', []): 
            if img.get("file_name") == img_filename:  # ← 修正ポイント
                image_entry = img 
                break 

        if image_entry is None: 
            return [] 

        image_id = image_entry["id"] 

        # category ID → name の対応表 
        coco_id_to_label = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])} 

        # 対象画像の annotation を抽出 
        for ann in coco_data.get('annotations', []): 
            if ann['image_id'] != image_id: 
                continue 

            x, y, w, h = ann['bbox']   # COCO bbox: [x_min, y_min, width, height] 
            x_min = int(x) 
            y_min = int(y) 
            x_max = int(x + w) 
            y_max = int(y + h) 

            annotations.append({ 
                'bbox': [x_min, y_min, x_max, y_max], 
                'label': coco_id_to_label.get(ann['category_id'], f"ID:{ann['category_id']}_Unknown") 
            }) 

    return annotations



        

def visualize_annotations(format_type):
    """指定された形式の全てのアノテーションを画像上に描画し、検証する"""
    
    print(f"\n--- 検証開始: {format_type.upper()} 形式 ---")
    
    # クラスリストを最初にロードし、CLASS_COLORSを初期化
    # これにより、load_classesが複数回呼ばれても色が再生成されないようにする
    load_classes(CLASSES_FILE) 

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_paths:
        print(f"❌ エラー: 画像ファイルが '{IMAGE_DIR}' に見つかりません。")
        return

    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        img_filename_no_ext, _ = os.path.splitext(img_filename)

        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w, _ = img.shape

        annotations = get_annotations(img_filename_no_ext, img_filename, img_w, img_h, format_type)


        if not annotations:
            print(f"⚠️ アノテーションファイルが存在しないか、空です: {img_filename}")
            continue

        temp_img = img.copy()
        
        # 描画
        for ann in annotations:
            x_min, y_min, x_max, y_max = ann['bbox']
            label = ann['label']
            
            # クラス名に基づいて色を取得
            color = get_unique_color(label) 
            
            cv2.rectangle(temp_img, (x_min, y_min), (x_max, y_max), color, 2)
            # テキストの色もバウンディングボックスの色と合わせる
            cv2.putText(temp_img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.setWindowTitle(WINDOW_NAME, f"検証 ({format_type.upper()}): {img_filename}")
        cv2.imshow(WINDOW_NAME, temp_img)
        
        # キー入力待ち: 's'で次の画像へ, 'q'で終了
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"検証 ({format_type.upper()}): 終了しました。")


if __name__ == "__main__":
    
    # 1. YOLO形式の検証
    visualize_annotations('yolo') 
    
    # 2. PASCAL VOC形式の検証
    visualize_annotations('voc') 
    
    # 3. COCO形式の検証
    visualize_annotations('coco')