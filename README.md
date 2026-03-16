# 🎧 See2Sound — Backend

Backend service for **See2Sound**, an AI-powered system that transforms visual media into **contextual audio descriptions** to improve accessibility for visually impaired users.

The backend is responsible for processing videos, running AI modules, generating descriptions, and returning results to the frontend platform.

---

# 🚀 Project Goal

See2Sound aims to make digital media more accessible by automatically generating **audio descriptions for videos** using artificial intelligence.

Instead of manually describing visual scenes, the system analyzes the video and produces contextual narration that explains important visual elements.

---

# 🧠 System Responsibilities

The backend handles the core logic of the system:

- 📥 Receive videos uploaded by the frontend
- 🎞 Process video frames
- 🔊 Analyze audio tracks
- 👁 Run visual recognition models
- 📝 Generate contextual descriptions
- 🎤 Convert descriptions into audio narration
- 📤 Return processed results to the user

---

# ⚙️ Architecture Overview

User Uploads Video  
        ↓  
Frontend  
        ↓  
Backend API  
        ↓  
Video Processing Pipeline  
        ↓  
AI Modules  
   ├── Spectra (visual recognition)  
   ├── Narrative Engine  
   └── Voice Engine  
        ↓  
Audio Description Generation  
        ↓  
Processed Video Returned  

---

# 📦 Project Structure

see2sound-backend/
│
├── app/                # API application
│   ├── api/            # API routes
│   ├── core/           # configurations and settings
│   ├── services/       # business logic
│   ├── models/         # data models
│   └── utils/          # helper utilities
│
├── ai/                 # AI modules
│   ├── spectra/        # visual recognition
│   ├── narrative/      # description generation
│   └── voice/          # text-to-speech
│
├── pipeline/           # video processing pipeline
│   ├── video/
│   ├── audio/
│   └── orchestration/
│
├── tests/              # automated tests
├── docs/               # documentation
│
├── requirements.txt
└── README.md

---

# 🛠 Technologies

Main technologies used in this project:

- 🐍 Python
- ⚡ FastAPI
- 🧠 PyTorch
- 👁 OpenCV
- 🔊 Speech-to-Text
- 🎤 Text-to-Speech
- 🎞 FFmpeg

---

# 🔬 AI Modules

The backend integrates multiple AI components.

### 👁 SPECTRA
Visual recognition module responsible for detecting important visual elements in video frames.

### 📝 Narrative Engine
Transforms visual detections into natural language descriptions.

### 🎤 Voice Engine
Converts generated descriptions into spoken audio narration.

---

# 🧪 Testing

Testing includes:

- Unit tests for individual modules
- Integration tests for the pipeline
- Functional tests for the full video processing workflow

---

# 📚 Development Status

🚧 Project currently under development as part of a technical research project.

---

# 🎯 Future Goals

- Improve scene understanding accuracy
- Support longer videos
- Optimize processing time
- Expand accessibility features

---

# ❤️ Accessibility First

This project is built with a focus on **digital accessibility**, helping visually impaired users better understand visual media.

---

# 👨‍💻 Authors

Developed as part of the **See2Sound Project**.

- 💫​ Clara Alves dos Santos — [GitHub](https://github.com/ClarAlvez)
- 🐶​​ João Victor Cussolim   — [GitHub](https://github.com/JoaoCussolim)
