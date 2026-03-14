#!/usr/bin/env python3
"""
Chatterbox Realtime Streaming Voice SDK Server

基于 MLX Chatterbox 的实时流式语音合成服务

特性:
- Apple Silicon 优化 (MLX 框架)
- 声音克隆 (需要参考音频)
- 低延迟合成
- 支持 LLM + TTS 双流式

模型: mlx-community/chatterbox-4bit
参考音频: training/1st.wav
"""

import os
import re
import json
import base64
import wave
import time
import io
from typing import Generator, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import scipy.io.wavfile as wavfile

# ============== Chatterbox MLX 导入 ==============
CHATTERBOX_AVAILABLE = False
mlx_load = None

try:
    from mlx_audio.tts import load as mlx_load
    CHATTERBOX_AVAILABLE = True
    print("[Chatterbox] mlx_audio.tts.load 导入成功")
except ImportError as e:
    print(f"[Chatterbox] 导入失败: {e}")
    print("[Chatterbox] 请安装: pip install mlx-audio")

# OpenAI 兼容的 LLM 客户端
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[Chatterbox] OpenAI 库未安装，LLM 功能不可用")


# ============== 配置 ==============

# Chatterbox 模型配置 - 使用 4bit 模型更快加载
CHATTERBOX_MODEL = "mlx-community/chatterbox-4bit"
OUTPUT_SAMPLE_RATE = 24000  # Chatterbox 输出 24kHz

# 默认参考音频路径 (用于声音克隆)
# 支持两种模式:
# 1. 使用预设参考音频 (DEFAULT_REF_AUDIO)
# 2. 每次请求动态指定 ref_audio
DEFAULT_REF_AUDIO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "training", "1st.wav"
)

# 默认参数 (根据 Chatterbox 官方推荐)
DEFAULT_TEMPERATURE = 0.8      # 采样温度
DEFAULT_TOP_P = 0.95           # 核采样
DEFAULT_CFG_WEIGHT = 0.5       # CFG 权重
DEFAULT_EXAGGERATION = 0.5     # 情绪夸张度 (0-1)
DEFAULT_REPETITION_PENALTY = 1.2  # 重复惩罚

# LLM 配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")


# ============== 全局模型 ==============

_chatterbox_model = None
_model_warmed = False
_default_conds_prepared = False


def get_chatterbox_model():
    """获取或初始化 Chatterbox 模型"""
    global _chatterbox_model

    if _chatterbox_model is None:
        if not CHATTERBOX_AVAILABLE:
            raise RuntimeError("Chatterbox 未安装。请运行: pip install mlx-audio")

        print(f"[Chatterbox] 加载模型: {CHATTERBOX_MODEL}")
        _chatterbox_model = mlx_load(CHATTERBOX_MODEL)
        print(f"[Chatterbox] 模型加载完成，采样率: {_chatterbox_model.sample_rate}")

    return _chatterbox_model


def synthesize_chatterbox(
    text: str,
    ref_audio: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    cfg_weight: float = DEFAULT_CFG_WEIGHT,
    exaggeration: float = DEFAULT_EXAGGERATION,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    lang_code: str = "en",  # 语言代码，支持 23 种语言
) -> bytes:
    """
    使用 Chatterbox 合成语音

    Args:
        text: 要合成的文本
        ref_audio: 参考音频路径 (用于声音克隆，建议 > 6 秒)
        temperature: 生成温度 (0.6-1.0 推荐)
        top_p: 核采样参数 (0.8-1.0 推荐)
        cfg_weight: CFG 权重 (0.0-1.0, 0.5 效果好)
        exaggeration: 情绪夸张度 (0.0-1.0)
        repetition_penalty: 重复惩罚 (1.0-2.0)
        lang_code: 语言代码 (支持: en, zh, es, fr, de, ja, ko, pt, it, ru, ar, hi 等)

    Returns:
        WAV 格式的音频字节
    """
    model = get_chatterbox_model()

    # 使用默认参考音频或用户指定的
    audio_ref = ref_audio or DEFAULT_REF_AUDIO

    # 验证参考音频存在
    if not os.path.exists(audio_ref):
        raise FileNotFoundError(f"参考音频不存在: {audio_ref}")

    try:
        audio_segments = []
        sample_rate = model.sample_rate

        # Chatterbox API - 使用 ref_audio 进行声音克隆
        results = list(model.generate(
            text=text,
            ref_audio=audio_ref,       # 关键: 参考音频用于声音克隆
            temperature=temperature,
            top_p=top_p,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            repetition_penalty=repetition_penalty,
            lang_code=lang_code,       # 语言代码，支持 zh, en, es, fr, de 等 23 种语言
            verbose=False
        ))

        if not results:
            raise RuntimeError("Chatterbox 生成失败：没有结果")

        for r in results:
            if hasattr(r, 'audio') and r.audio is not None:
                # MLX array 转 numpy
                audio_np = np.array(r.audio)
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()
                audio_segments.append(audio_np)

        if not audio_segments:
            raise RuntimeError("Chatterbox 生成失败：没有音频数据")

        combined = np.concatenate(audio_segments, axis=0)

        # RMS 归一化
        target_rms = 0.15
        current_rms = np.sqrt(np.mean(combined ** 2))
        if current_rms > 0:
            combined = combined * (target_rms / current_rms)
            max_val = np.max(np.abs(combined))
            if max_val > 0.95:
                combined = combined * (0.95 / max_val)

        # 转换为 int16
        audio_int16 = (combined * 32767).astype(np.int16)

        # 写入 WAV 字节
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        return buffer.getvalue()

    except Exception as e:
        print(f"[Chatterbox] 合成错误: {e}")
        raise


def warmup_model():
    """预热模型"""
    global _model_warmed, _default_conds_prepared
    if _model_warmed:
        return

    print("[Chatterbox] 预热模型...")
    model = get_chatterbox_model()

    # 检查默认参考音频
    if os.path.exists(DEFAULT_REF_AUDIO):
        print(f"[Chatterbox] 使用参考音频: {DEFAULT_REF_AUDIO}")

        # 预热合成
        results = list(model.generate(
            text="Ready.",
            ref_audio=DEFAULT_REF_AUDIO,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            cfg_weight=DEFAULT_CFG_WEIGHT,
            exaggeration=DEFAULT_EXAGGERATION,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            verbose=False
        ))
        _default_conds_prepared = True
    else:
        print(f"[Chatterbox] 警告: 默认参考音频不存在: {DEFAULT_REF_AUDIO}")
        print("[Chatterbox] 每次请求必须提供 ref_audio 参数")

    _model_warmed = True
    print("[Chatterbox] 模型预热完成")


# ============== FastAPI 应用 ==============

app = FastAPI(
    title="Chatterbox Realtime Streaming Voice SDK",
    description="基于 MLX Chatterbox 的实时流式语音合成 (支持声音克隆)"
)


class TTSRequest(BaseModel):
    text: str
    ref_audio: Optional[str] = None  # 可选，不提供则使用默认
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    cfg_weight: float = DEFAULT_CFG_WEIGHT
    exaggeration: float = DEFAULT_EXAGGERATION
    lang_code: str = "en"  # 语言代码，支持 23 种语言: en, zh, es, fr, de, ja, ko, pt, it, ru, ar, hi 等


class LLMStreamRequest(BaseModel):
    prompt: str
    ref_audio: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    cfg_weight: float = DEFAULT_CFG_WEIGHT
    exaggeration: float = DEFAULT_EXAGGERATION
    lang_code: str = "en"  # 语言代码


@app.get("/")
async def root():
    """返回测试页面"""
    html_path = os.path.join(os.path.dirname(__file__), "examples", "demo.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Chatterbox SDK</h1><p>访问 /docs 查看 API</p>")


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "chatterbox_available": CHATTERBOX_AVAILABLE,
        "llm_available": LLM_AVAILABLE and bool(LLM_API_KEY),
        "model_warmed": _model_warmed,
        "default_conds_prepared": _default_conds_prepared,
        "model": CHATTERBOX_MODEL,
        "default_ref_audio": DEFAULT_REF_AUDIO,
        "ref_audio_exists": os.path.exists(DEFAULT_REF_AUDIO)
    }


@app.get("/config")
async def get_config():
    """获取当前配置"""
    return {
        "model": CHATTERBOX_MODEL,
        "sample_rate": OUTPUT_SAMPLE_RATE,
        "default_ref_audio": DEFAULT_REF_AUDIO,
        "default_parameters": {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "cfg_weight": DEFAULT_CFG_WEIGHT,
            "exaggeration": DEFAULT_EXAGGERATION,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY
        }
    }


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """单次 TTS 合成 (支持声音克隆和多语言)"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    # 检查参考音频
    ref_audio = req.ref_audio or DEFAULT_REF_AUDIO
    if not os.path.exists(ref_audio):
        raise HTTPException(
            status_code=400,
            detail=f"参考音频不存在: {ref_audio}。请提供有效的 ref_audio 参数或设置默认参考音频。"
        )

    try:
        start_time = time.time()
        audio_bytes = synthesize_chatterbox(
            text=req.text,
            ref_audio=ref_audio,
            temperature=req.temperature,
            top_p=req.top_p,
            cfg_weight=req.cfg_weight,
            exaggeration=req.exaggeration,
            lang_code=req.lang_code,  # 语言代码，支持 zh, en, es, fr, de 等 23 种语言
        )
        latency = (time.time() - start_time) * 1000

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "success": True,
            "audio_base64": audio_base64,
            "latency_ms": latency,
            "text_length": len(req.text),
            "ref_audio": ref_audio
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm-tts/stream")
async def llm_tts_stream(req: LLMStreamRequest):
    """LLM + TTS 双流式接口"""

    async def generate():
        if not LLM_AVAILABLE or not LLM_API_KEY:
            yield f"data: {json.dumps({'type': 'error', 'message': 'LLM 未配置'})}\n\n"
            return

        # 检查参考音频
        ref_audio = req.ref_audio or DEFAULT_REF_AUDIO
        if not os.path.exists(ref_audio):
            yield f"data: {json.dumps({'type': 'error', 'message': f'参考音频不存在: {ref_audio}'})}\n\n"
            return

        try:
            # 预热模型
            warmup_model()

            # 初始化 LLM 客户端
            client = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL
            )

            yield f"data: {json.dumps({'type': 'start', 'model': LLM_MODEL, 'engine': 'Chatterbox', 'ref_audio': ref_audio})}\n\n"

            # 句子缓冲区
            sentence_buffer = ""
            sentence_count = 0
            token_count = 0
            total_audio_duration = 0
            first_audio_latency = None
            start_time = time.time()

            # 调用 LLM 流式 API
            stream = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": req.prompt}
                ],
                stream=True,
                max_tokens=500
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    token_count += 1

                    yield f"data: {json.dumps({'type': 'llm_token', 'content': content, 'index': token_count})}\n\n"

                    sentence_buffer += content

                    # 句子边界检测 (英文优先)
                    is_sentence_end = re.search(r'[.!?]$', sentence_buffer.strip())
                    is_clause_end = (
                        re.search(r'[,;]$', sentence_buffer.strip()) and
                        len(sentence_buffer.strip()) >= 10
                    )

                    if is_sentence_end or is_clause_end:
                        sentence = sentence_buffer.strip()
                        sentence_buffer = ""

                        if len(sentence) > 1:
                            sentence_count += 1

                            yield f"data: {json.dumps({'type': 'sentence', 'sentence': sentence, 'index': sentence_count})}\n\n"

                            try:
                                tts_start = time.time()
                                audio_bytes = synthesize_chatterbox(
                                    text=sentence,
                                    ref_audio=ref_audio,
                                    temperature=req.temperature,
                                    top_p=req.top_p,
                                    cfg_weight=req.cfg_weight,
                                    exaggeration=req.exaggeration,
                                    lang_code=req.lang_code  # 语言代码
                                )
                                tts_latency = (time.time() - tts_start) * 1000

                                audio_duration = len(audio_bytes) / (OUTPUT_SAMPLE_RATE * 2)
                                total_audio_duration += audio_duration

                                if first_audio_latency is None:
                                    first_audio_latency = (time.time() - start_time) * 1000
                                    print(f"[Chatterbox] 首句延迟: {first_audio_latency:.0f}ms")

                                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                                yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_base64, 'duration_ms': audio_duration * 1000, 'tts_latency_ms': tts_latency, 'sentence_index': sentence_count})}\n\n"

                            except Exception as e:
                                yield f"data: {json.dumps({'type': 'tts_error', 'message': str(e), 'sentence_index': sentence_count})}\n\n"

            # 处理剩余缓冲区
            if sentence_buffer.strip():
                sentence = sentence_buffer.strip()
                sentence_count += 1

                yield f"data: {json.dumps({'type': 'sentence', 'sentence': sentence, 'index': sentence_count})}\n\n"

                try:
                    audio_bytes = synthesize_chatterbox(
                        text=sentence,
                        ref_audio=ref_audio,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        cfg_weight=req.cfg_weight,
                        exaggeration=req.exaggeration,
                        lang_code=req.lang_code  # 语言代码
                    )
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_base64, 'duration_ms': len(audio_bytes) / (OUTPUT_SAMPLE_RATE * 2) * 1000, 'sentence_index': sentence_count})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'tts_error', 'message': str(e), 'sentence_index': sentence_count})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'total_tokens': token_count, 'total_sentences': sentence_count, 'total_audio_duration': total_audio_duration, 'first_audio_latency': first_audio_latency})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ============== 主函数 ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Chatterbox Realtime Streaming Voice SDK")
    print("=" * 60)
    print(f"  Chatterbox 可用: {CHATTERBOX_AVAILABLE}")
    print(f"  LLM 可用: {LLM_AVAILABLE and bool(LLM_API_KEY)}")
    print(f"  LLM 模型: {LLM_MODEL}")
    print(f"  TTS 模型: {CHATTERBOX_MODEL}")
    print(f"  默认参考音频: {DEFAULT_REF_AUDIO}")
    print(f"  参考音频存在: {os.path.exists(DEFAULT_REF_AUDIO)}")
    print("=" * 60)

    # 预热模型
    if CHATTERBOX_AVAILABLE:
        print("\n预热 Chatterbox 模型...")
        try:
            warmup_model()
        except Exception as e:
            print(f"Chatterbox 预热失败: {e}")

    print("\n启动服务: http://localhost:8004")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8004)
