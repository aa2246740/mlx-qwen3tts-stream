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
