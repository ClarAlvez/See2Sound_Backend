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

- 📥 Receive videos sent by the application
- 🎞 Process video frames
- 🔊 Analyze audio tracks
- 🧩 Detect speech pauses and useful narration intervals
- 👁 Run visual recognition models
- 📝 Generate contextual descriptions
- 🎤 Convert descriptions into audio narration
- 📤 Return processed results to the application

---

# ⚙️ Architecture Overview

User
↓
See2Sound App
↓
Backend API
↓
Video Processing Pipeline
↓
AI Modules
├── Spectra, visual recognition
├── Narrative Engine
└── Voice Engine
↓
Audio Description Generation
↓
Processed Media Returned  

---

# 📦 Project Structure

```bash
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
│   ├── video/          # video metadata, frames and scene detection
│   ├── audio/          # audio extraction and speech analysis
│   └── orchestration/  # pipeline execution flow
│
├── tests/              # automated tests
├── docs/               # documentation
│
├── requirements.txt
└── README.md
```

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
- 🧪 Automated testing tools

---

# 🔬 AI Modules

The backend integrates multiple AI components responsible for analyzing video, understanding context, generating narration and producing the final audio description.

### 👁 SPECTRA
**SPECTRA** is the visual recognition module responsible for detecting important visual elements in video frames.

It may identify elements such as:
- People
- Objects
- Actions
- Scenarios
- Scene changes
- Visual context

### 📝 Narrative Engine
The **Narrative Engine** transforms visual detections and audio context into natural language descriptions.

The goal is to generate descriptions that are:
- Contextual
- Clear
- Non-repetitive
- Aligned with available speech pauses
- Useful for understanding the visual content

### 🎤 Voice Engine
The **Voice Engine** converts generated descriptions into spoken audio narration.

This module is responsible for transforming text-based descriptions into an audio format that can be inserted into or played alongside the original media.

---

# 🧪 Testing

Testing includes:

- Unit tests for individual modules
- Integration tests for the pipeline
- Functional tests for the full video processing workflow
- Validation of video, audio and AI processing steps

---

# 🚧 Development Status

🚧 Project currently under development as part of a technical research project.

---

# 🎯 Future Goals

- Improve scene understanding accuracy
- Support longer videos
- Optimize processing time
- Improve speech pause detection
- Expand accessibility features
- Integrate more advanced contextual narration
- Improve communication with the See2Sound App

---

# ❤️ Accessibility First

This project is built with a focus on **digital accessibility**, helping visually impaired users better understand visual media.

---

## 🔗 Related Repositories

This project is part of the **See2Sound system**.

- 🌐 Frontend Website → https://github.com/ClarAlvez/See2Sound_Frontend
- 📱 Desktop/Mobile App → https://github.com/cc24136/See2Sound_App

---

# 👨‍💻 Authors

Developed as part of the **See2Sound Project**.

- 💫​ Clara Alves dos Santos — [GitHub](https://github.com/ClarAlvez)
- 🐶​​ João Victor Cussolim   — [GitHub](https://github.com/JoaoCussolim)
