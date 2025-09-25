import os
import cv2
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import cvzone

# =============================
# 1️⃣ Ekstraksi Keypoints dari Video
# =============================
def extract_keypoints(video_root='dataset_videos', output_csv='all_keypoints_dataset.csv', frame_per_video=100):
    """
    video_root: folder yang berisi subfolder Suspicious/Normal
    output_csv: CSV untuk menyimpan keypoints
    frame_per_video: jumlah frame per video yang diekstrak
    """
    model = YOLO("yolo11s-pose.pt")
    all_data = []

    categories = ['Suspicious', 'Normal']

    for category in categories:
        folder_path = os.path.join(video_root, category)
        for video_name in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            i = 0

            while cap.isOpened() and i < frame_per_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * (frame_count // frame_per_video))
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                for r in results:
                    boxes = r.boxes.xyxy
                    conf = r.boxes.conf.tolist()
                    keypoints = r.keypoints.xyn.tolist()

                    for idx, box in enumerate(boxes):
                        if conf[idx] > 0.5:  # threshold confidence
                            data = {}
                            for j in range(len(keypoints[idx])):
                                data[f'x{j}'] = keypoints[idx][j][0]
                                data[f'y{j}'] = keypoints[idx][j][1]

                            data['label'] = category
                            all_data.append(data)

                i += 1

            cap.release()
            print(f"Processed video: {video_name}")

    # Simpan ke CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved: {output_csv}")

# =============================
# 2️⃣ Training XGBoost dari Dataset Gabungan
# =============================
def train_xgboost(dataset_csv='all_keypoints_dataset.csv', model_output='trained_model_multi_video.json'):
    df = pd.read_csv(dataset_csv)
    X = df.drop('label', axis=1)
    y = df['label'].map({'Suspicious':0, 'Normal':1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        eta=0.1,
        eval_metric='logloss',
        objective='binary:logistic',
        tree_method='hist'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    model.save_model(model_output)
    print(f"Trained model saved: {model_output}")
    return model

# =============================
# 3️⃣ Prediksi Video Baru
# =============================
def predict_video(video_path, model_path='trained_model_multi_video.json'):
    model_yolo = YOLO("yolo11s-pose.pt")

    xgb_model = xgb.Booster()
    xgb_model.load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1018, 600))
        results = model_yolo(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            boxes = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for idx, box in enumerate(boxes):
                if conf[idx] > 0.55:
                    x1, y1, x2, y2 = box.tolist()
                    data = {}
                    for j in range(len(keypoints[idx])):
                        data[f'x{j}'] = keypoints[idx][j][0]
                        data[f'y{j}'] = keypoints[idx][j][1]

                    df_kp = pd.DataFrame(data, index=[0])
                    dmatrix = xgb.DMatrix(df_kp)
                    sus = xgb_model.predict(dmatrix)
                    pred = (sus > 0.3).astype(int)

                    if pred == 0:  # Suspicious
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                        cvzone.putTextRect(annotated_frame, "Suspicious", (int(x1), int(y1)+50), 1,1)
                    else:
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        cvzone.putTextRect(annotated_frame, "Normal", (int(x1), int(y1)+50), 1,1)

        cv2.imshow("Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================
# 4️⃣ Contoh penggunaan
# =============================
if __name__ == "__main__":
    # 1. Ekstrak keypoints dari semua video
    extract_keypoints(video_root='dataset_videos', output_csv='all_keypoints_dataset.csv', frame_per_video=100)

    # 2. Train model XGBoost
    train_xgboost(dataset_csv='all_keypoints_dataset.csv', model_output='trained_model_multi_video.json')

    # 3. Prediksi video baru
    predict_video(video_path='new_video.mp4', model_path='trained_model_multi_video.json')
