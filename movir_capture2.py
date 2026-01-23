import cv2

import yt_dlp

import os

import random

import csv

import pickle

import time

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

COOKIE_FILE = 'youtube_cookies.txt' # Macから転送したクッキーファイル

FOLDER_ID = '1qKmIlYTqYuXxwyu4_XzbF0b2exdlcutc'



# 顔認識分類器の定義

frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')



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

            # サーバー環境向けに port=9099, open_browser=False に設定

            creds = flow.run_local_server(port=9099, host='localhost', open_browser=False)

        with open('token.pickle', 'wb') as token:

            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)



def upload_or_update_to_drive(file_name, mimetype='image/jpeg'):

    try:

        service = get_drive_service()

        query = f"name = '{file_name}' and trashed = false"

        if FOLDER_ID:

            query += f" and '{FOLDER_ID}' in parents"

        results = service.files().list(q=query, fields="files(id, name)").execute()

        items = results.get('files', [])

        media = MediaFileUpload(file_name, mimetype=mimetype, resumable=True)

        if items:

            file_id = items[0]['id']

            service.files().update(fileId=file_id, media_body=media).execute()

            print(f"  -> Drive更新完了: {file_name}")

        else:

            file_metadata = {'name': file_name}

            if FOLDER_ID:

                file_metadata['parents'] = [FOLDER_ID]

            service.files().create(body=file_metadata, media_body=media, fields='id').execute()

            print(f"  -> Drive新規保存: {file_name}")

    except Exception as e:

        print(f"  [Drive Error]: {e}")



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

   

    ydl_opts = {

        # 【重要】AV1(av01)を除外し、VP9等の互換性のある4K(2160p)を取得

        'format': 'bestvideo[height<=2160][vcodec!^=av01]+bestaudio/best[height<=2160]',

        'cookiefile': COOKIE_FILE,

        'quiet': True,

        'no_warnings': True,

    }

   

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        try:

            info = ydl.extract_info(youtube_url, download=False)

            stream_url = None

            formats = info.get('formats', [])

            # 逆順（高画質順）で最適なURLを探す

            for f in reversed(formats):

                u = f.get('url', '')

                if u and all(x not in u for x in ['.m3u8', 'manifest', '.mpd']):

                    vcodec = f.get('vcodec', 'none')

                    # AV1以外の4Kを探す

                    if vcodec != 'none' and not vcodec.startswith('av01'):

                        if (f.get('height') or 0) <= 2160:

                            stream_url = u

                            break

           

            if not stream_url:

                stream_url = info.get('url')

            if not stream_url:

                raise Exception("ストリームURLの取得に失敗しました。")

        except Exception as e:

            print(f"URL解析エラー ({youtube_url}): {e}")

            return current_index



    duration = info.get('duration', 0)

    title = info.get('title', 'Unknown Title')

    source_h = info.get('height', 0)



    print(f"\n🎥 処理中: {title} (ターゲット画質: {source_h}p)")

   

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():

        print("エラー: 動画ストリームを開けませんでした。")

        return current_index



    current_time_sec = 30 if duration > 30 else duration // 2

   

    def contains_face(frame_data):

        gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)

        # 4Kの場合、顔が相対的に小さくなるため、minNeighborsとminSizeを微調整

        frontal_faces = frontal_face_cascade.detectMultiScale(gray, 1.1, 15, minSize=(50, 50))

        if len(frontal_faces) > 0: return True

        profile_faces = profile_face_cascade.detectMultiScale(gray, 1.1, 15, minSize=(50, 50))

        if len(profile_faces) > 0: return True

        gray_flipped = cv2.flip(gray, 1)

        profile_flipped = profile_face_cascade.detectMultiScale(gray_flipped, 1.1, 15, minSize=(50, 50))

        return len(profile_flipped) > 0



    def save_and_cleanup(frame_data, time_str, index):

        if not contains_face(frame_data):

            print(f"  [Skip] 顔なし ({time_str})")

            return index



        actual_h, actual_w, _ = frame_data.shape

        valid_res = [res for res in RESOLUTIONS if res[1] <= actual_h]

       

        if actual_h >= 2160:

            target_res = (actual_w, actual_h) if random.random() < 0.5 else random.choice(valid_res)

        else:

            target_res = random.choice(valid_res) if valid_res else (actual_w, actual_h)



        final_frame = cv2.resize(frame_data, target_res, interpolation=cv2.INTER_AREA)

        file_name = f"not_glitch_image_{index:05d}.jpg"

        cv2.imwrite(file_name, final_frame)

       

        log_to_csv(title, youtube_url, time_str, f"{target_res[1]}p")

        upload_or_update_to_drive(file_name)

       

        print(f"  -> 保存完了: {target_res[1]}p (入力: {actual_h}p)")

        if os.path.exists(file_name): os.remove(file_name)

        return index + 1



    while current_time_sec < duration:

        cap.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000)

        cap.grab() # シーク安定化

        success, frame = cap.read()

       

        if success:

            timestamp = format_time(current_time_sec)

            print(f"[{timestamp}] スキャン中...")

            current_index = save_and_cleanup(frame, timestamp, current_index)

            save_next_index(current_index)

        else:

            print(f"[{format_time(current_time_sec)}] フレーム取得失敗。再接続します。")

            cap.release()

            time.sleep(5)

            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)



        current_time_sec += random.randint(30, 75)



    cap.release()

    return current_index



def main():

    if not os.path.exists(URL_LIST_FILE):

        print(f"エラー: {URL_LIST_FILE} が見つかりません。")

        return

    with open(URL_LIST_FILE, 'r') as f:

        urls = [line.strip() for line in f if line.strip()]



    print(f"合計 {len(urls)} 本（計約450時間）を4K解析します。")

    for i, url in enumerate(urls, 1):

        print(f"\n--- Progress: {i}/{len(urls)} ---")

        try:

            process_single_video(url)

            upload_or_update_to_drive(CSV_FILE, mimetype='text/csv')

        except Exception as e:

            print(f"致命的エラー: {e}")

            continue

    print("\n✨ すべての処理が完了しました。")



if __name__ == "__main__":

    main()