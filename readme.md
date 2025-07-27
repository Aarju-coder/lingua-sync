# Lingua Sync 🎙️🌐

**Lingua Sync** is a modular, containerized pipeline that automates the conversion of video URLs into multilingual narrated audio. It integrates YouTube audio downloading, transcription (STT), optional translation, and TTS voice synthesis into a streamlined API.

## 🚀 Features

- 🎧 **Audio Downloading**: Extracts audio from video URLs using `yt-dlp` and converts it to 16kHz mono WAV.
- 🧠 **Speech-to-Text**: Uses `faster-whisper` with GPU acceleration for transcription or direct translation to English.
- 🌍 **Translation**: Converts transcribed text to the desired language using a pluggable backend.
- 🗣️ **Text-to-Speech**: Reconstructs speech with voice cloning and style control via OpenVoice TTS.
- 🔗 **Pipeline API**: Simple REST endpoint `/pipeline` to handle the entire video-to-voice flow.

## 🧱 Architecture

```mermaid
graph LR
A[Downloader] --> B[STT]
B --> C[Translate]
C --> D[TTS]
A --> D
