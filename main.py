# ================== SYSTEM SETUP ==================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import re
import numpy as np
import pandas as pd
from datetime import datetime, date, time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# ================== CONFIG ==================
SIM_THRESHOLD = 0.35
FRAME_SKIP = 3
EXIT_DELAY = 10
MIN_CONFIRM_FRAMES = 3
MIN_REQUIRED_MINUTES = 45

# ================== LECTURE SCHEDULE ==================
LECTURES = [
    ("09:00", "10:00"),
    ("10:00", "11:00"),
    ("11:00", "12:00"),
    # BREAK 12–1
    ("13:00", "14:00"),
    ("14:00", "15:00"),
    ("15:00", "16:00"),
    ("16:00", "17:00"),
]

# ================== UTILS ==================
def get_current_lecture(now):
    for start, end in LECTURES:
        s = datetime.combine(date.today(), time.fromisoformat(start))
        e = datetime.combine(date.today(), time.fromisoformat(end))
        if s <= now < e:
            return f"{start}-{end}", s, e
    return None, None, None

def log_event(text):
    with open("attendance_log.txt", "a") as f:
        f.write(f"{datetime.now()} - {text}\n")

# ================== LOAD STUDENTS ==================
students = pd.read_csv("student_data.csv")
id_to_name = dict(zip(students.RollNo, students.Name))

# ================== MODELS ==================
detector = MTCNN()
embedder = FaceNet()

# ================== LOAD DATASET ==================
known_embeddings = {}
for file in os.listdir("dataset"):
    if not file.lower().endswith((".jpg", ".png")):
        continue
    match = re.search(r"_(\d+)", file)
    if not match:
        continue

    roll = int(match.group(1))
    img = cv2.imread(os.path.join("dataset", file))
    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if not faces:
        continue

    x, y, w, h = faces[0]["box"]
    x, y = max(0, x), max(0, y)
    face = cv2.resize(rgb[y:y+h, x:x+w], (160,160))
    known_embeddings[roll] = embedder.embeddings([face])[0]

print("Registered students:", len(known_embeddings))

# ================== ATTENDANCE FILE ==================
today = str(date.today())
attendance_file = f"attendance_{today}.csv"

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=[
        "Date", "Lecture", "RollNo", "Name",
        "FirstEntry", "LastExit",
        "TotalMinutes", "Status"
    ]).to_csv(attendance_file, index=False)

attendance_df = pd.read_csv(attendance_file)

# ================== STATE ==================
present = {}
last_seen = {}
total_seconds = {}
confirm_frames = {}
first_entry = {}
last_exit = {}
manual_override = set()

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
frame_count = 0
print("Attendance running | ESC to quit | Q end lecture | M manual present")

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.now()
    lecture, lec_start, lec_end = get_current_lecture(now)

    # ---------- BREAK ----------
    if lecture is None:
        cv2.putText(frame, "BREAK TIME - NO ATTENDANCE",
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # ---------- FRAME SKIP ----------
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    current_seen = set()

    for det in detections:
        x, y, w, h = det["box"]
        x, y = max(0, x), max(0, y)
        face = cv2.resize(rgb[y:y+h, x:x+w], (160,160))
        emb = embedder.embeddings([face])[0]

        best_roll, best_score = None, 1.0
        for roll, ref in known_embeddings.items():
            score = cosine(ref, emb)
            if score < best_score:
                best_roll, best_score = roll, score

        if best_score < SIM_THRESHOLD:
            name = id_to_name.get(best_roll)
            confidence = round((1 - best_score) * 100, 1)
            current_seen.add(best_roll)
            last_seen[best_roll] = now
            confirm_frames[best_roll] = confirm_frames.get(best_roll, 0) + 1

            if confirm_frames[best_roll] >= MIN_CONFIRM_FRAMES:
                if best_roll not in present:
                    present[best_roll] = now
                    total_seconds.setdefault(best_roll, 0)
                    first_entry.setdefault(best_roll, now.strftime("%H:%M:%S"))
                    log_event(f"{name} ENTERED")

            # ---------- DISPLAY ----------
            mins = round(total_seconds.get(best_roll,0)/60,1)
            label = f"{name} | {confidence}% | {mins} min"
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ---------- EXIT ----------
    for roll in list(present.keys()):
        if roll not in current_seen:
            if (now - last_seen.get(roll, now)).seconds >= EXIT_DELAY:
                total_seconds[roll] += (now - present.pop(roll)).seconds
                last_exit[roll] = now.strftime("%H:%M:%S")
                confirm_frames[roll] = 0
                log_event(f"{id_to_name.get(roll)} EXITED")

    # ---------- STATS ----------
    elapsed = int((now - lec_start).seconds / 60)
    cv2.putText(frame, f"Lecture: {lecture}", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
    cv2.putText(frame, f"Elapsed: {elapsed} / 60 min", (30,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
    cv2.putText(frame, f"Present: {len(present)}", (30,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.imshow("Attendance", frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ================== FINAL SAVE ==================
now = datetime.now()
for roll, entry in present.items():
    total_seconds[roll] += (now - entry).seconds
    last_exit[roll] = now.strftime("%H:%M:%S")

for roll, secs in total_seconds.items():
    minutes = round(secs/60,2)
    status = "Present" if minutes >= MIN_REQUIRED_MINUTES else "Absent"

    attendance_df.loc[len(attendance_df)] = [
        today,
        lecture,
        roll,
        id_to_name.get(roll),
        first_entry.get(roll,"-"),
        last_exit.get(roll,"-"),
        minutes,
        status
    ]

attendance_df.to_csv(attendance_file, index=False)
print("Attendance saved successfully.")
[]