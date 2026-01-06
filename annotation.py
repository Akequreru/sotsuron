import cv2
import numpy as np
import json
import os
import xml.etree.ElementTree as ET
import glob
import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk

# --- 設定 ---
IMAGE_DIR = 'output'
CLASSES_FILE = 'classes.txt'
MASTER_OUTPUT_DIR = 'annotations_output'
OUTPUT_FOLDERS = {
    'yolo': 'yolo_txt',
    'voc': 'voc_xml',
    'coco': 'coco_json'
}
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']
MAX_DISPLAY_SIZE = (1100, 750)

class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python Fully GUI Annotation Tool")
        
        # 1. データの初期化
        self.class_list = self.load_classes()
        self.class_to_id = {name: i for i, name in enumerate(self.class_list)}
        self.image_paths = self.get_image_paths()
        self.current_idx = 0
        
        if not self.image_paths:
            messagebox.showerror("Error", f"画像が見つかりません: {IMAGE_DIR}")
            root.destroy()
            return

        # 2. 状態管理変数
        self.current_anns = []
        self.drawing = False
        self.start_x = -1
        self.start_y = -1
        self.scale_factor = 1.0
        self.cv_img_rgb = None
        
        # 3. UIの構築
        self.setup_ui()
        
        # 4. 最初の画像読み込み
        self.load_image()

    def load_classes(self):
        if not os.path.exists(CLASSES_FILE):
            with open(CLASSES_FILE, 'w', encoding='utf-8') as f: f.write("object")
            return ["object"]
        with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def get_image_paths(self):
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        return sorted(paths)

    def setup_ui(self):
        # メインフレーム
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 左側: キャンバス
        self.canvas_frame = tk.Frame(self.main_frame, bg="gray")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 右側: 操作パネル
        self.ctrl_panel = tk.Frame(self.main_frame, width=250, bg="#f0f0f0", padx=10, pady=10)
        self.ctrl_panel.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.ctrl_panel, text="【1. クラス選択】", font=("", 10, "bold")).pack(anchor=tk.W)
        # --- 修正箇所: tk.Listbox を使用 ---
        self.class_box = tk.Listbox(self.ctrl_panel, height=10, exportselection=False)
        for c in self.class_list: self.class_box.insert(tk.END, c)
        self.class_box.select_set(0)
        self.class_box.pack(fill=tk.X, pady=5)

        tk.Label(self.ctrl_panel, text="【2. 操作】", font=("", 10, "bold")).pack(anchor=tk.W, pady=(10, 0))
        tk.Button(self.ctrl_panel, text="Undo (Z)", command=self.undo, bg="#fff").pack(fill=tk.X, pady=2)
        tk.Button(self.ctrl_panel, text="保存して次へ (S)", command=self.save_and_next, bg="#c8e6c9").pack(fill=tk.X, pady=2)
        tk.Button(self.ctrl_panel, text="前の画像へ", command=self.prev_image).pack(fill=tk.X, pady=2)

        self.info_label = tk.Label(self.ctrl_panel, text="", justify=tk.LEFT)
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        path = self.image_paths[self.current_idx]
        file_name = os.path.basename(path)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            self.next_image()
            return
        self.cv_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = self.cv_img_rgb.shape
        w_ratio = MAX_DISPLAY_SIZE[0] / w
        h_ratio = MAX_DISPLAY_SIZE[1] / h
        self.scale_factor = min(w_ratio, h_ratio, 1.0)
        self.display_w = int(w * self.scale_factor)
        self.display_h = int(h * self.scale_factor)
        self.current_anns = self.fetch_existing_annotations(file_name, w, h)
        self.render_image()
        self.update_info()

    def render_image(self):
        resized = cv2.resize(self.cv_img_rgb, (self.display_w, self.display_h))
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.config(width=self.display_w, height=self.display_h)
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        for ann in self.current_anns:
            x1, y1, x2, y2 = [int(c * self.scale_factor) for c in ann['bbox_xyxy']]
            color = "cyan" if ann.get('is_existing') else "green"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.canvas.create_text(x1, y1-2, text=ann['class_name'], fill=color, anchor=tk.SW, font=("", 10, "bold"))

    def on_button_press(self, event):
        self.drawing = True
        self.start_x, self.start_y = event.x, event.y
        self.temp_rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="yellow", width=2)

    def on_move_press(self, event):
        if self.drawing:
            self.canvas.coords(self.temp_rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        self.drawing = False
        self.canvas.delete(self.temp_rect)
        x1_orig = int(min(self.start_x, event.x) / self.scale_factor)
        y1_orig = int(min(self.start_y, event.y) / self.scale_factor)
        x2_orig = int(max(self.start_x, event.x) / self.scale_factor)
        y2_orig = int(max(self.start_y, event.y) / self.scale_factor)
        if (x2_orig - x1_orig) > 5 and (y2_orig - y1_orig) > 5:
            selected_idx = self.class_box.curselection()
            class_name = self.class_list[selected_idx[0]] if selected_idx else self.class_list[0]
            self.current_anns.append({
                'class_name': class_name,
                'bbox_xyxy': [x1_orig, y1_orig, x2_orig, y2_orig],
                'img_w': self.cv_img_rgb.shape[1],
                'img_h': self.cv_img_rgb.shape[0],
                'is_existing': False
            })
            self.redraw()

    def undo(self):
        if self.current_anns:
            self.current_anns.pop()
            self.redraw()

    def save_and_next(self):
        file_name = os.path.basename(self.image_paths[self.current_idx])
        self.export_all_formats(file_name, self.current_anns)
        self.next_image()

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_image()
        else:
            messagebox.showinfo("完了", "最後の画像です。")

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def update_info(self):
        path = self.image_paths[self.current_idx]
        self.info_label.config(text=f"画像: {self.current_idx + 1} / {len(self.image_paths)}\nファイル: {os.path.basename(path)}")

    def fetch_existing_annotations(self, file_name, img_w, img_h):
        base_name = os.path.splitext(file_name)[0]
        anns = []
        yolo_p = os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['yolo'], base_name + ".txt")
        if os.path.exists(yolo_p):
            with open(yolo_p, "r") as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) != 5: continue
                    cid, cx, cy, w, h = map(float, vals)
                    x1 = int((cx - w/2) * img_w)
                    y1 = int((cy - h/2) * img_h)
                    x2 = int((cx + w/2) * img_w)
                    y2 = int((cy + h/2) * img_h)
                    cname = self.class_list[int(cid)] if int(cid) < len(self.class_list) else "unknown"
                    anns.append({'class_name': cname, 'bbox_xyxy': [x1, y1, x2, y2], 'img_w': img_w, 'img_h': img_h, 'is_existing': True})
            return anns
        return []

    def export_all_formats(self, file_name, data):
        if not data: return
        os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)
        img_w, img_h = data[0]['img_w'], data[0]['img_h']
        base_name = os.path.splitext(file_name)[0]

        # 1. YOLO
        yolo_dir = os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['yolo'])
        os.makedirs(yolo_dir, exist_ok=True)
        yolo_lines = []
        for ann in data:
            x1, y1, x2, y2 = ann['bbox_xyxy']
            cid = self.class_to_id.get(ann['class_name'], 0)
            cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
            w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
            yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(os.path.join(yolo_dir, base_name + ".txt"), 'w') as f:
            f.write('\n'.join(yolo_lines))

        # 2. VOC
        voc_dir = os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['voc'])
        os.makedirs(voc_dir, exist_ok=True)
        root_el = ET.Element('annotation')
        ET.SubElement(root_el, 'filename').text = file_name
        size_el = ET.SubElement(root_el, 'size')
        ET.SubElement(size_el, 'width').text = str(img_w)
        ET.SubElement(size_el, 'height').text = str(img_h)
        ET.SubElement(size_el, 'depth').text = '3'
        for ann in data:
            obj = ET.SubElement(root_el, 'object')
            ET.SubElement(obj, 'name').text = ann['class_name']
            bbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = str(ann['bbox_xyxy'][0])
            ET.SubElement(bbox, 'ymin').text = str(ann['bbox_xyxy'][1])
            ET.SubElement(bbox, 'xmax').text = str(ann['bbox_xyxy'][2])
            ET.SubElement(bbox, 'ymax').text = str(ann['bbox_xyxy'][3])
        tree = ET.ElementTree(root_el)
        tree.write(os.path.join(voc_dir, base_name + ".xml"))

        # 3. COCO
        coco_dir = os.path.join(MASTER_OUTPUT_DIR, OUTPUT_FOLDERS['coco'])
        os.makedirs(coco_dir, exist_ok=True)
        coco_path = os.path.join(coco_dir, "dataset_coco.json")
        coco_dict = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(self.class_list)]}
        if os.path.exists(coco_path):
            try:
                with open(coco_path, 'r') as f: coco_dict = json.load(f)
            except: pass
        coco_dict["images"] = [i for i in coco_dict["images"] if i["file_name"] != file_name]
        img_id = len(coco_dict["images"]) + 1
        coco_dict["images"].append({"id": img_id, "file_name": file_name, "width": img_w, "height": img_h})
        for ann in data:
            x1, y1, x2, y2 = ann['bbox_xyxy']
            coco_dict["annotations"].append({
                "id": len(coco_dict["annotations"]) + 1,
                "image_id": img_id,
                "category_id": self.class_to_id.get(ann['class_name'], 0),
                "bbox": [x1, y1, x2-x1, y2-y1],
                "area": (x2-x1)*(y2-y1),
                "iscrowd": 0
            })
        with open(coco_path, 'w') as f:
            json.dump(coco_dict, f, indent=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationApp(root)
    root.bind("<s>", lambda e: app.save_and_next())
    root.bind("<z>", lambda e: app.undo())
    root.mainloop()