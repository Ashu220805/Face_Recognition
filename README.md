# Face_Recognition
Intelligent Time-Based Attendance System
📖 Overview

This project implements an AI-assisted, time-based classroom attendance system using computer vision.
Unlike traditional attendance systems that mark students simply as present or absent, this system calculates actual time spent in class and marks attendance fairly and objectively.

Attendance is decided lecture-wise, automatically handling breaks, late arrivals, and early exits without manual intervention.

🎯 Problem Statement

In real classroom environments, students may:

enter late,

leave early,

look down while writing notes,

or temporarily move out of camera view.

Conventional attendance systems often:

rely on single face detection,

fail during note-taking,

ignore actual duration of presence,

and mark students unfairly absent.

This project solves these issues by using cumulative time-based presence tracking instead of binary detection.

✅ Key Features

⏱ Time-Based Attendance
Attendance is calculated using total minutes spent in class.

🕒 Lecture-Wise Attendance (9 AM – 5 PM)
Automatically detects active lecture slots and ignores breaks (12–1 PM).

⛔ Break Handling
Attendance is paused automatically during break time.

✔ Fair Attendance Rule
Students attending ≥ 45 minutes are marked Present, otherwise Absent.

🎥 Face Recognition-Based Presence Detection
Uses FaceNet embeddings with cosine similarity.

🧠 Multi-Frame Confirmation
Prevents false entry/exit due to momentary detection errors.

🖥 Live Visual Feedback

Student name with confidence score

Real-time minutes attended

Current lecture and elapsed time

Present student count

📄 Lecture-Wise CSV Reports

First entry time

Last exit time

Total minutes

Final attendance status

🧾 Event Logging
Entry and exit events are stored in a log file for transparency.

🔐 Privacy-Friendly

No video recordings are stored

Only attendance data is saved

Technologies Used

Python

OpenCV

MTCNN – Face Detection

FaceNet (Keras-Facenet) – Face Recognition

NumPy, Pandas

Cosine Similarity

🗓 Lecture Schedule Used
Time	Session
09:00 – 10:00	Lecture
10:00 – 11:00	Lecture
11:00 – 12:00	Lecture
12:00 – 13:00	Break
13:00 – 14:00	Lecture
14:00 – 15:00	Lecture
15:00 – 16:00	Lecture
16:00 – 17:00	Lecture

How to Run
1️⃣ Install Dependencies
pip install opencv-python mtcnn keras-facenet pandas numpy scipy tensorflow

2️⃣ Prepare Dataset

Store one image per student in dataset/

File format: Name_RollNo.jpg

3️⃣ Run the System
python main.py

Press ESC or Q to safely stop and save attendance.

**Attendance Logic Summary**

System detects current lecture from timetable

Face detection & recognition confirm student presence

Time is accumulated across multiple entries/exits

Break periods are ignored automatically

Final attendance is decided based on ≥ 45 minutes rule

**Limitations**

Uses a single camera; occlusion may affect detection.

Face recognition accuracy depends on lighting and image quality.

Writing activity, posture, or engagement is not directly detected.

Multiple students overlapping may reduce detection accuracy.

**Future Scope (Not Implemented)**

The following features are intentionally excluded from the current version to maintain fairness and explainability:

Writing detection using YOLO

Emotion recognition

Engagement scoring

Mobile phone punishment logic

These can be explored as independent experimental modules without affecting core attendance logic.

**Academic Use Disclaimer**

This project is designed for educational and research purposes.
Attendance decisions are based purely on time-based presence inference, ensuring transparency and fairness.

**Final Note**

This system focuses on doing one thing correctly:

Marking attendance fairly based on actual classroom presence.
