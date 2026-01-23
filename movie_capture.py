import cv2
import os
import random
import csv
import pickle
import time
from datetime import timedelta
from pytubefix import YouTube
from pytubefix.cli import on_progress
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
FOLDER_ID = '1qKmIlYTqYuXxwyu4_XzbF0b2exdlcutc'
TEMP_VIDEO_NAME = 'temp_video.mp4'

# 顔認識分類器の定義
frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

RESOLUTIONS = [
    (256, 144), (426, 240), (640, 360), (854, 480), (1280, 720), (1920, 1080), (3840, 2160)
]

# --- Google Drive API / 補助機能 (以前のものを継承) ---

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
            creds = flow.run_local_server(port=9099, host='localhost', open_browser=False)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def upload_or_update_to_drive(file_name, mimetype='image/jpeg'):
    try:
        service = get_drive_service()
        query = f"name = '{file_name}' and trashed = false"
        if FOLDER_ID: query += f" and '{FOLDER_ID}' in parents"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        media = MediaFileUpload(file_name, mimetype=mimetype, resumable=True)
        if items:
            service.files().update(fileId=items[0]['id'], media_body=media).execute()
        else:
            meta = {'name': file_name}
            if FOLDER_ID: meta['parents'] = [FOLDER_ID]
            service.files().create(body=meta, media_body=media).execute()
        print(f"  -> Drive同期: {file_name}")
    except Exception as e:
        print(f"  [Drive Error]: {e}")

def format_time(seconds): return str(timedelta(seconds=int(seconds)))
def get_next_index():
    if not os.path.exists(COUNTER_FILE): return 1
    with open(COUNTER_FILE, 'r') as f:
        try: return int(f.read().strip())
        except: return 1
def save_next_index(index):
    with open(COUNTER_FILE, 'w') as f: f.write(str(index))

def log_to_csv(title, url, timestamp, res_text):
    exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not exists: writer.writerow(['題名', 'URL', '時間', '解像度'])
        writer.writerow([title, url, timestamp, res_text])

# --- 顔認識 (ガバガバ設定) ---
def contains_face(frame_data):
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    f_faces = frontal_face_cascade.detectMultiScale(gray, 1.1, 20, minSize=(50, 50))
    if len(f_faces) > 0: return True
    p_faces = profile_face_cascade.detectMultiScale(gray, 1.1, 20, minSize=(50, 50))
    if len(p_faces) > 0: return True
    gray_flipped = cv2.flip(gray, 1)
    p_flipped = profile_face_cascade.detectMultiScale(gray_flipped, 1.1, 20, minSize=(50, 50))
    return len(p_flipped) > 0

# --- メインロジック ---

def process_single_video(youtube_url):
    current_index = get_next_index()
    
    # 以前の残骸があれば削除
    if os.path.exists(TEMP_VIDEO_NAME): os.remove(TEMP_VIDEO_NAME)

    try:
        # OAuth認証を有効にしてYouTubeオブジェクトを作成
        # 初回実行時、コンソールに「Go to https://www.google.com/device and enter XXX-XXX」と出ます
        yt = YouTube(youtube_url, on_progress_callback=on_progress, use_oauth=True, allow_oauth_cache=True)
        
        # 4K映像のみ(video/mp4 or webm)を検索
        video_stream = yt.streams.filter(res="2160p", only_video=True).first()
        
        # もし4Kがなければ1080p、それもなければ最高画質
        if not video_stream:
            print("  [Info] 4Kが見つかりません。1080pを探します。")
            video_stream = yt.streams.filter(res="1080p", only_video=True).first()
        if not video_stream:
            video_stream = yt.streams.get_highest_resolution()

        print(f"\n📥 ダウンロード開始 ({video_stream.resolution}): {yt.title}")
        video_stream.download(filename=TEMP_VIDEO_NAME)
        
    except Exception as e:
        print(f"❌ Pytubefixエラー: {e}")
        return current_index

    # OpenCVで解析
    cap = cv2.VideoCapture(TEMP_VIDEO_NAME)
    duration = int(yt.length)
    current_time_sec = 30
    
    while current_time_sec < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000)
        success, frame = cap.read()
        
        if success:
            timestamp = format_time(current_time_sec)
            if contains_face(frame):
                h, w, _ = frame.shape
                valid_res = [res for res in RESOLUTIONS if res[1] <= h]
                t_res = (w, h) if (h >= 2160 and random.random() < 0.5) else random.choice(valid_res)
                
                final_frame = cv2.resize(frame, (t_res[0], t_res[1]), interpolation=cv2.INTER_AREA)
                file_name = f"not_glitch_image_{current_index:05d}.jpg"
                cv2.imwrite(file_name, final_frame)
                
                log_to_csv(yt.title, youtube_url, timestamp, f"{t_res[1]}p")
                upload_or_update_to_drive(file_name)
                if os.path.exists(file_name): os.remove(file_name)
                
                print(f"  [{timestamp}] ✅ 顔あり保存: {t_res[1]}p")
                current_index += 1
                save_next_index(current_index)
            else:
                print(f"  [{timestamp}] ⏩ 顔なし")
        
        current_time_sec += random.randint(30, 90)

    cap.release()
    if os.path.exists(TEMP_VIDEO_NAME): os.remove(TEMP_VIDEO_NAME)
    return current_index

def main():
    if not os.path.exists(URL_LIST_FILE):
        print(f"エラー: {URL_LIST_FILE} が必要です。")
        return
    with open(URL_LIST_FILE, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"🚀 Pytubefixによる解析を開始します（計 {len(urls)} 本）")
    for i, url in enumerate(urls, 1):
        print(f"\n--- Progress: {i}/{len(urls)} ---")
        try:
            process_single_video(url)
            upload_or_update_to_drive(CSV_FILE, mimetype='text/csv')
        except Exception as e:
            print(f"⚠️ スキップ: {e}")
            continue
    print("\n✨ 完了しました。")

if __name__ == "__main__":
    main()