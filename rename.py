import os
import shutil
from PIL import Image
import re

def rename_and_convert_avif_webp_to_jpg(src_dir, dst_dir) :
    # フォルダが存在しない場合は作成
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
        print(f"'{src_dir}' フォルダを作成しました。ここに画像をいれて再実行してください。")
        return

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 変換対象の拡張子
    convert_exts = {".webp", ".avif", ".gif"}
    
    # フォルダ内のファイル一覧を取得してソート
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    files.sort()

    if not files:
        print(f"'{src_dir}' フォルダにファイルが見つかりません。")
        return

    print(f"処理を開始します: {len(files)} 件のファイル")

    for i, filename in enumerate(files, 1):
        file_path = os.path.join(src_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        # 新しいファイル名のベース
        new_name_base = f"glitch_image_{str(i).zfill(6)}"
        
        try:
            if ext in convert_exts:
                # --- WebP/AVIF を JPG に変換して出力 ---
                save_path = os.path.join(dst_dir, f"{new_name_base}.jpg")
                with Image.open(file_path) as img:
                    # 透過情報がある場合は白背景で埋める（JPG対応）
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(save_path, "JPEG", quality=95)
                print(f"[変換] {filename} -> {new_name_base}.jpg")
            
            else:
                # --- それ以外（JPG/PNG等）は拡張子を維持してリネームコピー ---
                new_filename = f"{new_name_base}{ext}"
                save_path = os.path.join(dst_dir, new_filename)
                shutil.copy2(file_path, save_path)
                print(f"[コピー] {filename} -> {new_filename}")

        except Exception as e:
            print(f"[エラー] {filename} の処理に失敗しました: {e}")

    print("\nすべての処理が完了しました。'output' フォルダを確認してください。")

def rename_zfill(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)
        print(f"'{src_dir}' フォルダを作成しました。ここに画像をいれて再実行してください。")
        return

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # フォルダ内のファイル一覧を取得してソート
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    files.sort()

    if not files:
        print(f"'{src_dir}' フォルダにファイルが見つかりません。")
        return

    print(f"処理を開始します: {len(files)} 件のファイル")
    
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(src_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        numbers = re.findall(r'\d+', filename)

        new_number = numbers[0].zfill(6)

        # 新しいファイル名のベース
        new_name_base = f"glitch_image_{new_number}"
        
        try:
                # --- それ以外（JPG/PNG等）は拡張子を維持してリネームコピー ---
                new_filename = f"{new_name_base}{ext}"
                save_path = os.path.join(dst_dir, new_filename)
                shutil.copy2(file_path, save_path)
                print(f"[コピー] {filename} -> {new_filename}")

        except Exception as e:
            print(f"[エラー] {filename} の処理に失敗しました: {e}")

    print("\nすべての処理が完了しました。'output' フォルダを確認してください。")


def main():
    # フォルダの設定
    src_dir = "input"
    dst_dir = "output"
    print("0: 画像をjpgに変換、ファイル名をglitch_image_0xxxxxに変換 / 1: ファイル名をファイル名をglitch_image_0xxxxxに変換")
    mode = input("input 0 or 1:")
    if mode == "0":
        rename_and_convert_avif_webp_to_jpg(src_dir, dst_dir)
    elif mode == "1":
        rename_zfill(src_dir, dst_dir)
    else :
        print("input 0 or 1")

    

if __name__ == "__main__":
    main()