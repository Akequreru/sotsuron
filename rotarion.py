import cv2
import os

# ==========================================
# 設定：ここに回転させたいファイルのパスを入力
# ==========================================
target_image_path = "reindexed_images/glitch_image_000221.jpg"
# 対象のテキストファイルパス（画像に対応するもの）
target_txt_path = "reindexed_labels/glitch_image_000221.txt"

# 出力ファイル名
OUT_IMAGE = 'glitch_image_00001_rot90.jpg'
OUT_LABEL = 'glitch_image_00001_rot90.txt'
# ==========================================

def rotate_single_file(img_p, lbl_p, out_img_p, out_lbl_p):
    # --- 1. 画像の回転 ---
    img = cv2.imread(img_p)
    if img is None:
        print(f"❌ 画像が見つかりません: {img_p}")
        return

    # 時計回りに90度回転
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(out_img_p, rotated_img)
    print(f"✅ 画像を保存しました: {out_img_p}")

    # --- 2. ラベルの回転 ---
    if not os.path.exists(lbl_p):
        print(f"⚠️ ラベルファイルが見つかりません。画像のみ回転しました。")
        return

    new_lines = []
    with open(lbl_p, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            cls_id = parts[0]
            x, y, w, h = map(float, parts[1:])

            # 座標変換（時計回り90度）
            new_x = 1.0 - y
            new_y = x
            new_w = h
            new_h = w

            new_lines.append(f"{cls_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")

    with open(out_lbl_p, 'w') as f:
        f.writelines(new_lines)
    print(f"✅ ラベルを保存しました: {out_lbl_p}")

if __name__ == "__main__":
    rotate_single_file(target_image_path, target_txt_path, OUT_IMAGE, OUT_LABEL)