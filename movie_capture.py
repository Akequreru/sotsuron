import cv2
import yt_dlp
import os
import random
import csv
import pickle
from datetime import timedelta
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# --- 設定エリア ---
CLIENT_SECRET_FILE = 'credentials.json' 
SCOPES = ['https://www.googleapis.com/auth/drive.file']
COUNTER_FILE = 'last_index.txt'
CSV_FILE = 'captures_log.csv'
URL_LIST_FILE = 'urls.txt'

# 【重要】ここにGoogleドライブのフォルダIDを入力してください
# 空（None）にするとマイドライブのルートに保存されます
FOLDER_ID = '1qKmIlYTqYuXxwyu4_XzbF0b2exdlcutc'

RESOLUTIONS = [
    (256, 144), (426, 240), (640, 360), (854, 480), (1280, 720), (1920, 1080), (3840, 2160)
]

# --- Google Drive API 関連 ---

def get_drive_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def upload_or_update_to_drive(file_name, mimetype='image/jpeg'):
    service = get_drive_service()
    
    # 指定フォルダ内にある同名ファイルを検索
    query = f"name = '{file_name}' and trashed = false"
    if FOLDER_ID:
        query += f" and '{FOLDER_ID}' in parents"
        
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    media = MediaFileUpload(file_name, mimetype=mimetype, resumable=True)
    
    if items:
        # 既存ファイルの更新
        file_id = items[0]['id']
        service.files().update(fileId=file_id, media_body=media).execute()
        print(f"  -> Drive更新完了: {file_name}")
    else:
        # 新規作成（親フォルダを指定）
        file_metadata = {'name': file_name}
        if FOLDER_ID:
            file_metadata['parents'] = [FOLDER_ID]
            
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"  -> Drive新規保存: {file_name} (ID: {file.get('id')})")

# --- 補助機能 ---

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_next_index():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r') as f:
            try: return int(f.read().strip())
            except: return 1
    return 1

def save_next_index(index):
    with open(COUNTER_FILE, 'w') as f:
        f.write(str(index))

def log_to_csv(title, url, timestamp, res_text):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['題名', 'URL', '時間', '解像度'])
        writer.writerow([title, url, timestamp, res_text])

# --- 個別動画のキャプチャ処理 ---

def process_single_video(youtube_url):
    current_index = get_next_index()
    
    # 1. 4K(2160p)までの映像ストリームを狙う設定
    ydl_opts = {
        'format': 'bestvideo[height<=2160]', 
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(youtube_url, download=False)
            
            stream_url = None
            if 'formats' in info:
                # 【修正ポイント】 heightがNoneの場合を考慮してソートする
                # x.get('height') が None の場合は 0 として扱う
                formats = sorted(
                    info['formats'], 
                    key=lambda x: (x.get('height') if x.get('height') is not None else 0), 
                    reverse=True
                )
                
                for f in formats:
                    u = f.get('url', '')
                    # OpenCVで開ける直接URLであり、かつ映像データがあるもの
                    if u and '.m3u8' not in u and f.get('vcodec') != 'none':
                        stream_url = u
                        break
            
            if not stream_url:
                stream_url = info.get('url')

            if not stream_url:
                raise Exception("ストリームURLの取得に失敗しました。")
                
        except Exception as e:
            # ここで発生していた比較エラーを上記 lambda で回避しています
            print(f"URL解析エラー ({youtube_url}): {e}")
            return current_index

    duration = info.get('duration', 0)
    title = info.get('title', 'Unknown Title')
    source_h = info.get('height', 0)

    print(f"\n🎥 処理中: {title} (解析上の最高画質: {source_h}p)")
    
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("エラー: 動画ストリームを開けませんでした。")
        return current_index

    # --- 最初の位置を1分(60秒)に設定 ---
    start_time = 60 if duration > 60 else duration // 2
    current_time_sec = start_time
    
    def save_and_cleanup(frame_data, time_str, index):
        # OpenCVが実際にデコードした生の解像度
        actual_h, actual_w, _ = frame_data.shape
        
        # 拡大防止ロジック
        valid_res = [res for res in RESOLUTIONS if res[1] <= actual_h]
        
        # 4Kが利用可能なら、50%の確率で4Kを維持、50%でランダムリサイズ
        
        target_res = random.choice(valid_res) if valid_res else (actual_w, actual_h)

        final_frame = cv2.resize(frame_data, (target_res[0], target_res[1]), interpolation=cv2.INTER_AREA)
        
        file_name = f"not_glitch_image_{index:05d}.jpg"
        cv2.imwrite(file_name, final_frame)
        
        log_to_csv(title, youtube_url, time_str, f"{target_res[1]}p")
        upload_or_update_to_drive(file_name)
        
        print(f"  -> 保存完了: {target_res[1]}p (デコード元: {actual_h}p)")
        
        if os.path.exists(file_name):
            os.remove(file_name)

    # --- 以降、キャプチャループ ---
    # (既存の cap.set / cap.read / save_and_cleanup のループを続けてください)

    # --- 最初のキャプチャ（1分後） ---
    cap.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000)
    success, frame = cap.read()
    if success:
        timestamp = format_time(current_time_sec)
        print(f"[{timestamp}] 最初のキャプチャ（開始1分後）を実行中...")
        save_and_cleanup(frame, timestamp, current_index)
        current_index += 1
        save_next_index(current_index)

    # --- その後、2~4分おきにランダムキャプチャ ---
    while current_time_sec < duration:
        interval = random.randint(120, 240) # 2~4分
        current_time_sec += interval
        if current_time_sec >= duration: break

        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000)
        success, frame = cap.read()
        
        if success:
            timestamp = format_time(current_time_sec)
            print(f"[{timestamp}] キャプチャ中...")
            save_and_cleanup(frame, timestamp, current_index)
            current_index += 1
            save_next_index(current_index)
        else:
            print(f"[{format_time(current_time_sec)}] フレーム取得失敗")

    cap.release()
    return current_index

# --- メイン実行部 ---

def main():
    if not os.path.exists(URL_LIST_FILE):
        print(f"エラー: {URL_LIST_FILE} が見つかりません。")
        return

    with open(URL_LIST_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"合計 {len(urls)} 本の動画を処理します。")

    for i, url in enumerate(urls, 1):
        print(f"\n--- 進捗: {i}/{len(urls)} ---")
        try:
            process_single_video(url)
            upload_or_update_to_drive(CSV_FILE, mimetype='text/csv')
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            continue

    print("\n✨ すべての処理が完了しました。指定フォルダを確認してください。")

if __name__ == "__main__":
    main()