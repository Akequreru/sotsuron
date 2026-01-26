import os
import shutil
from PIL import Image
import re

# --- 設定（モード2用） ---
# 画像が入っているフォルダと、ラベル(txt)が入っているフォルダを指定してください
IMG_SRC = "output"
LBL_SRC = "merged_output"
IMG_DST = "reindexed_images"
LBL_DST = "reindexed_labels"

def reindex_pairs(img_src, lbl_src, img_dst, lbl_dst):
    """画像とテキストのペアを維持したまま欠番を詰める"""
    for d in [img_dst, lbl_dst]:
        if not os.path.exists(d): os.makedirs(d)

    # 対応する画像拡張子
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    # 画像フォルダ内のファイルを取得
    all_files = os.listdir(img_src)
    
    # ペアが存在するものだけを抽出
    pairs = []
    for f in all_files:
        base, ext = os.path.splitext(f)
        if ext.lower() in img_exts:
            lbl_path = os.path.join(lbl_src, f"{base}.txt")
            if os.path.exists(lbl_path):
                pairs.append((f, f"{base}.txt")) # (画像名, テキスト名)

    # 元のファイル名でソート（順番を維持するため）
    # 数字が含まれる場合は数値順になるようにソート
    pairs.sort(key=lambda x: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', x[0])])

    if not pairs:
        print("整合するペアが見つかりませんでした。フォルダ設定を確認してください。")
        return

    print(f"欠番詰めを開始します: {len(pairs)} ペア見つかりました。")

    for i, (img_name, lbl_name) in enumerate(pairs, 1):
        new_base = f"glitch_image_{str(i).zfill(6)}"
        
        # 画像のコピー
        old_img_path = os.path.join(img_src, img_name)
        new_img_ext = os.path.splitext(img_name)[1]
        new_img_path = os.path.join(img_dst, f"{new_base}{new_img_ext}")
        shutil.copy2(old_img_path, new_img_path)

        # ラベルのコピー
        old_lbl_path = os.path.join(lbl_src, lbl_name)
        new_lbl_path = os.path.join(lbl_dst, f"{new_base}.txt")
        shutil.copy2(old_lbl_path, new_lbl_path)

        print(f"[連番] {new_base} <- {img_name} & {lbl_name}")

    print(f"\n完了しました。出力先: '{img_dst}', '{lbl_dst}'")

# --- 既存の関数（0, 1）はそのまま保持 ---
def rename_and_convert_avif_webp_to_jpg(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
        print(f"'{src_dir}' フォルダを作成しました。")
        return
    if not os.path.exists(dst_dir): os.makedirs(dst_dir)
    convert_exts = {".webp", ".avif", ".gif"}
    files = sorted([f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])
    if not files: return
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(src_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        new_name_base = f"glitch_image_{str(i).zfill(6)}"
        try:
            if ext in convert_exts:
                save_path = os.path.join(dst_dir, f"{new_name_base}.jpg")
                with Image.open(file_path) as img:
                    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
                    img.save(save_path, "JPEG", quality=95)
            else:
                new_filename = f"{new_name_base}{ext}"
                shutil.copy2(file_path, os.path.join(dst_dir, new_filename))
        except Exception as e: print(f"Error: {e}")

def rename_zfill(src_dir, dst_dir):
    if not os.path.exists(src_dir): return
    if not os.path.exists(dst_dir): os.makedirs(dst_dir)
    files = sorted([f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))])
    for filename in files:
        numbers = re.findall(r'\d+', filename)
        if not numbers: continue
        new_number = numbers[0].zfill(6)
        ext = os.path.splitext(filename)[1]
        new_filename = f"glitch_image_{new_number}{ext}"
        shutil.copy2(os.path.join(src_dir, filename), os.path.join(dst_dir, new_filename))

def main():
    print("0: 画像をjpg変換+連番 / 1: 既存番号を6桁化 / 2: 画像とtxtの欠番を詰めて連番")
    mode = input("選択 (0, 1, 2): ")
    
    if mode == "0":
        rename_and_convert_avif_webp_to_jpg("input", "output")
    elif mode == "1":
        rename_zfill("input", "output")
    elif mode == "2":
        # モード2は画像とラベルの両方を処理
        reindex_pairs(IMG_SRC, LBL_SRC, IMG_DST, LBL_DST)
    else:
        print("0, 1, 2 のいずれかを入力してください")

if __name__ == "__main__":
    main()