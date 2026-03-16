# MLX Qwen3-TTS Stream

<div align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MLX-green" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/Qwen3--TTS-blue" alt="Qwen3-TTS">
  <img src="https://img.shields.io/badge/Python-3.9+-yellow" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-orange" alt="License">
</div>

Real-time streaming voice synthesis with intelligent sentence segmentation and voice cloning support, optimized for Apple Silicon.

## Demo

https://github.com/aa2246740/mlx-qwen3tts-stream/raw/main/demo.mp4

## Features

- **Real-time Streaming**: LLM + TTS dual streaming for ultra-low latency
- **Apple Silicon Optimized**: Built on MLX framework for M-series chips
- **Voice Cloning**: Clone any voice with a reference audio file
- **Intelligent Segmentation (SimpleSegmenter v8)**:
  - Chinese: Character-based counting
  - English: Word-based counting with integrity protection
  - Acronym handling (J.A.R.V.I.S → JARVIS)
- **Anti-Click Processing**: Smooth audio transitions between sentences
  - 5ms smooth fade at audio end
  - 50ms gap between audio segments
  - 150ms pause after sentence-ending punctuation
- **Multi-language Support**: Chinese, English, and mixed content

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/aa2246740/mlx-qwen3tts-stream.git
cd mlx-qwen3tts-stream

# 2. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment (optional, for LLM+TTS mode)
cp .env.example .env
# Edit .env with your API key

# 4. (Optional) Prepare reference audio for voice cloning
# Place your audio file in training/ directory
# Example: training/1st.wav and training/1st.md (corresponding text)

# 5. Start the server
python3 server.py

# Open http://localhost:8004 in your browser
```

## Two Usage Modes

### Mode 1: Pure TTS
直接将文本转换为语音，无需 LLM：
```
Text → TTS → Audio
```

### Mode 2: LLM + TTS Streaming
LLM 实时生成内容，TTS 同步转换：
```
Prompt → LLM (streaming) → Sentence Segmentation → TTS → Audio
```
适合对话、朗读、语音助手等场景。

## Architecture / 架构说明

### System Overview / 系统概述

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Application                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Swift-Speech Server                           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐ │
│  │   POST /tts │    │ POST /stream │    │ GET / (Demo Page)   │ │
│  │  单次合成    │    │  流式合成     │    │    Web 界面         │ │
│  └──────┬──────┘    └──────┬───────┘    └─────────────────────┘ │
│         │                  │                                    │
│         │           ┌──────▼───────┐                            │
│         │           │ SimpleSegmenter │ ← 智能断句              │
│         │           │     v8        │   (中英文分别处理)         │
│         │           └──────┬───────┘                            │
│         │                  │                                    │
│         ▼                  ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Qwen3-TTS Engine                       │   │
│  │              (mlx-community/Qwen3-TTS-8bit)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                    Audio Output (WAV)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Speaker/Player │
                    └─────────────────┘
```

### How LLM + TTS Streaming Works / 流式工作原理

```
用户输入: "介绍一下Python"
         │
         ▼
┌─────────────────┐
│   LLM 开始生成   │  ← 你需要提供 OpenAI 兼容的 LLM API
│  "Python 是..." │
└────────┬────────┘
         │ token by token
         ▼
┌─────────────────┐
│  累积到完整句子   │  ← SimpleSegmenter 自动检测句子边界
│  "Python是一门"  │     支持中英文混合
│  "编程语言，"    │
└────────┬────────┘
         │ 句子完成
         ▼
┌─────────────────┐
│   TTS 立即合成   │  ← 不等 LLM 全部完成，边生成边合成
│   输出音频片段   │     实现超低延迟
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  音频队列播放   │  ← 前端按顺序播放，用户几乎无等待
└─────────────────┘
```

## How to Connect Your LLM / 如何对接你的 LLM

### Step 1: 确认你的 LLM 兼容性

本系统支持 **任何 OpenAI 兼容的 API**，包括：
- OpenAI GPT 系列
- 阿里云通义千问 (DashScope)
- DeepSeek
- 智谱 AI (GLM)
- 本地 Ollama
- 任何自部署的 LLM（如 vLLM, LM Studio）

### Step 2: 配置方式

**方式 A: 环境变量（推荐服务端部署）**
```bash
# .env 文件
LLM_BASE_URL=https://your-llm-api.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-name
```

**方式 B: 请求参数（推荐前端调用）**
```javascript
// 每次请求时传入
fetch('/llm-tts/stream', {
  method: 'POST',
  body: JSON.stringify({
    prompt: '你好',
    llm_base_url: 'https://your-llm-api.com/v1',
    llm_api_key: 'your-api-key',
    llm_model: 'your-model',
  })
})
```

### Step 3: API 响应格式 (SSE)

服务端使用 **Server-Sent Events (SSE)** 返回流式数据：

```
data: {"type": "start"}

data: {"type": "llm_token", "token": "Python"}

data: {"type": "llm_token", "token": "是"}

data: {"type": "sentence", "text": "Python是一门编程语言，"}

data: {"type": "audio", "audio_base64": "UklGRiQAAABXQVZFZm10..."}

data: {"type": "done"}
```

### Step 4: 完整对接示例

```python
import requests
import json
import base64

def stream_llm_tts(prompt: str, llm_config: dict):
    """
    对接你自己的 LLM 进行流式 TTS

    Args:
        prompt: 用户输入
        llm_config: 你的 LLM 配置
            {
                "base_url": "https://api.your-llm.com/v1",
                "api_key": "your-key",
                "model": "your-model"
            }
    """
    url = "http://localhost:8004/llm-tts/stream"

    payload = {
        "prompt": prompt,
        "llm_base_url": llm_config["base_url"],
        "llm_api_key": llm_config["api_key"],
        "llm_model": llm_config["model"],
        # 可选：声音克隆
        # "ref_audio_base64": "...",
        # "ref_text": "..."
    }

    response = requests.post(url, json=payload, stream=True)

    for line in response.iter_lines():
        if not line or not line.startswith(b'data: '):
            continue

        data = json.loads(line[6:])  # 去掉 "data: " 前缀

        if data["type"] == "llm_token":
            # LLM 生成的文本 token
            print(data["token"], end="", flush=True)

        elif data["type"] == "audio":
            # TTS 生成的音频（base64 编码的 WAV）
            audio_bytes = base64.b64decode(data["audio_base64"])
            yield audio_bytes  # 返回音频数据

        elif data["type"] == "done":
            print("\n[完成]")
            break

# 使用示例
llm_config = {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-your-key",
    "model": "qwen3.5-flash"
}

for audio_chunk in stream_llm_tts("介绍一下Python", llm_config):
    # 在这里处理音频：播放、保存、发送给客户端等
    with open("output.wav", "ab") as f:
        f.write(audio_chunk)
```

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | https://dashscope.aliyuncs.com/compatible-mode/v1 | LLM API base URL (OpenAI-compatible) |
| `LLM_API_KEY` | - | Your LLM API key |
| `LLM_MODEL` | qwen3.5-flash | LLM model name |
| `SERVER_PORT` | 8004 | Server port |

### LLM Provider Configuration

This project supports any **OpenAI-compatible API**:

**1. 阿里云 DashScope (推荐中文用户)**
```env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=sk-your-dashscope-key
LLM_MODEL=qwen3.5-flash
```

**2. OpenAI**
```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-openai-key
LLM_MODEL=gpt-3.5-turbo
```

**3. 本地 Ollama (无需 API Key)**
```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3
```

**4. 其他兼容服务 (DeepSeek, 智谱, etc.)**
```env
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=deepseek-chat
```

### Demo Page Configuration

The demo page (`http://localhost:8004`) supports dynamic configuration:

- **LLM Base URL**: Any OpenAI-compatible API endpoint
- **LLM API Key**: Your API key
- **LLM Model**: Model name (e.g., gpt-3.5-turbo, qwen3.5-flash)
- **Reference Audio**: Upload a WAV file for voice cloning
- **Reference Text**: The text corresponding to the reference audio

## Usage Examples

### 1. Pure TTS (without LLM)

Convert text directly to speech:

```python
import requests

response = requests.post('http://localhost:8004/tts', json={
    'text': '你好，这是一个测试。',
    'temperature': 0.4,
})

# Save audio
with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### 2. LLM + TTS Streaming

Stream LLM response with real-time TTS:

```python
import requests
import base64

response = requests.post('http://localhost:8004/llm-tts/stream', json={
    'prompt': '介绍一下Python编程语言',
    'llm_base_url': 'https://api.openai.com/v1',
    'llm_api_key': 'sk-your-key',
    'llm_model': 'gpt-3.5-turbo',
}, stream=True)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])

        if data.get('type') == 'llm_token':
            print(data['token'], end='', flush=True)

        elif data.get('type') == 'audio':
            # Play or save audio
            audio_data = base64.b64decode(data['audio_base64'])
            # ... handle audio
```

### 3. Voice Cloning

Clone a voice with reference audio:

```python
import base64

# Read reference audio
with open('reference.wav', 'rb') as f:
    ref_audio_base64 = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:8004/tts', json={
    'text': '这段话会用克隆的声音朗读',
    'ref_audio_base64': ref_audio_base64,
    'ref_text': '这是参考音频中说话的内容',
})
```

### 4. Connect Your Own Stream Source

You can use this as a TTS backend for any text stream:

```python
import requests
import json

def synthesize_stream(text_iterator):
    """Convert any text stream to audio stream"""
    buffer = ""

    for text_chunk in text_iterator:
        buffer += text_chunk

        # When you have a complete sentence, send to TTS
        if ends_with_punctuation(buffer):
            response = requests.post('http://localhost:8004/tts', json={
                'text': buffer,
            })
            yield response.content  # audio bytes
            buffer = ""
```

### 5. Use with Other Applications

**With OBS/Twitch**: Use virtual audio device to route audio

**With Chat Apps**: Build a bot that reads messages aloud

**With Accessibility Tools**: Screen reader enhancement

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Demo page |
| `/health` | GET | Health check |
| `/tts` | POST | Single TTS synthesis |
| `/llm-tts/stream` | POST | LLM + TTS dual streaming |

### POST /tts

Single TTS synthesis request.

```json
{
  "text": "Hello, world!",
  "temperature": 0.4,
  "top_p": 0.85,
  "seed": 42,
  "ref_audio_base64": "...",
  "ref_text": "..."
}
```

### POST /llm-tts/stream

LLM + TTS dual streaming request (Server-Sent Events).

```json
{
  "prompt": "Tell me about Python",
  "temperature": 0.4,
  "top_p": 0.85,
  "seed": 42,
  "llm_base_url": "https://api.openai.com/v1",
  "llm_api_key": "sk-...",
  "llm_model": "gpt-3.5-turbo",
  "ref_audio_base64": "...",
  "ref_text": "..."
}
```

Response events:
- `start`: Streaming started
- `llm_token`: LLM token received
- `sentence`: Complete sentence segmented
- `audio`: Audio data (base64 encoded WAV)
- `done`: Streaming completed
- `error`: Error occurred

## Technical Details

- **Sample Rate**: 24000 Hz
- **TTS Model**: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`
- **Sentence Segmenter**: SimpleSegmenter v8
  - Chinese: Character-based counting
  - English: Word-based counting
  - English word integrity protection
  - Acronym normalization (J.A.R.V.I.S → JARVIS)

## Project Structure

```
mlx-qwen3tts-stream/
├── server.py              # Main server file
├── requirements.txt       # Python dependencies
├── .env.example           # Environment config template
├── .env                   # Your local config (not in git)
├── static/
│   └── index.html         # Demo page
└── training/              # Reference audio files (local only, not in git)
    ├── *.wav              # Your reference audio
    └── *.md               # Corresponding text
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- ~2GB RAM for model loading

## License

MIT License

## Acknowledgments

- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - MLX audio framework
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-Audio) - TTS model
