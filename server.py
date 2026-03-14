#!/usr/bin/env python3
"""
Swift-Speech - Realtime Streaming Voice SDK

基于 Qwen3-TTS Base (MLX) 的实时流式语音合成服务

特性:
- Apple Silicon 优化 (MLX 框架)
- 支持中文、英文、中英混合
- 支持参考音频声音克隆
- 低延迟流式合成
- LLM + TTS 双流式
- 内存管理（模型卸载）
- 智能断句（上下文预测）

模型: mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit
"""

import os
import re
import json
import base64
import time
import io
import gc

# 加载 .env 文件（本地开发用）
from dotenv import load_dotenv
load_dotenv()
import atexit
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import scipy.io.wavfile as wavfile

# ============== Qwen3-TTS MLX 导入 ==============
QWEN3_TTS_AVAILABLE = False
mlx_load = None
mx = None

try:
    from mlx_audio.tts import load as mlx_load
    import mlx.core as mx
    QWEN3_TTS_AVAILABLE = True
    print("[Swift-Speech] mlx_audio.tts.load 导入成功")
except ImportError as e:
    print(f"[Swift-Speech] 导入失败: {e}")
    print("[Swift-Speech] 请安装: pip install mlx-audio")

# OpenAI 兼容的 LLM 客户端
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("[Swift-Speech] OpenAI 库未安装，LLM 功能不可用")


# ============== 配置 ==============

# Qwen3-TTS Base 模型配置
# 使用 1.7B-8bit 模型（效果更好）
QWEN3_TTS_MODEL = os.getenv("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
OUTPUT_SAMPLE_RATE = 24000

# 参考音频配置 (1st 声音)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", os.path.join(PROJECT_ROOT, "training/1st.wav"))
REF_TEXT_PATH = os.getenv("REF_TEXT_PATH", os.path.join(PROJECT_ROOT, "training/1st.md"))

# 默认参数 (专家建议优化)
DEFAULT_TEMPERATURE = 0.4  # Qwen3 最稳的温度 (0.4 是黄金值)
DEFAULT_TOP_P = 0.85       # 收拢采样范围
DEFAULT_SEED = 42          # 固定随机种子，消除随机性

# 句间停顿配置（解决破音和句子太密集问题）
SENTENCE_PAUSE_MS = 150    # 句间停顿时长（毫秒），只在句号结尾时添加
ANTI_CLICK_SMOOTH_MS = 5   # 抗爆音平滑时长（毫秒），技术性处理

# 句子切分配置已移到 SimpleSegmenter 类内部
# 首句优化目标：15-25 字极速响应
# 预期首句延迟：3-5 秒（vs 之前 18 秒）


# ============== 简单断句器（参考 Fish SDK）==============

class SimpleSegmenter:
    """
    简单断句器 - 低延迟优化版 v8

    核心策略：
    1. 首句极速响应：达到 15 字（中文）或 8 词（英文）就切
    2. 后续句子：逗号 >= 10 字切，或 25 字无标点强制切
    3. 假断句检测：数字列表、英文缩写、省略号
    4. 文本清洗：保留语义，删除格式符号
    5. 中英文分别处理：中文按字符，英文按单词
    6. 英文切割保护：不切断英文单词
    7. 缩写词处理：J.A.R.V.I.S → JARVIS

    延迟优化目标：
    - 首句从 88 字降到 15-25 字
    - 首句延迟从 18s 降到 3-5s
    """

    # 配置常量（中文按字符，英文按单词）
    # 首句配置：目标是快速响应但不要太碎
    FIRST_SENTENCE_MIN_CN = 15     # 首句最小长度（中文，字符数）
    FIRST_SENTENCE_MIN_EN = 12     # 首句最小长度（英文，单词数）- 提高避免过早切
    FIRST_SENTENCE_MAX_CN = 30     # 首句最大长度（中文）- 提高
    FIRST_SENTENCE_MAX_EN = 20     # 首句最大长度（英文，单词数）- 提高

    # 后续句子配置：允许更长更自然的句子
    NORMAL_MIN_CHUNK_CN = 15       # 后续逗号切分最小长度（中文）
    NORMAL_MIN_CHUNK_EN = 10       # 后续逗号切分最小长度（英文，单词数）
    NORMAL_MAX_CHUNK_CN = 35       # 后续最大长度（中文）
    NORMAL_MAX_CHUNK_EN = 25       # 后续最大长度（英文，单词数）
    EXTREME_MAX = 50               # 极端情况最大长度

    def __init__(self):
        self.buffer = ""
        self.is_first_sentence = True

    @staticmethod
    def count_text_units(text: str) -> tuple:
        """
        计算文本的语义单位数
        返回: (中文字符数, 英文单词数)

        中文：按字符数计算（一个汉字 ≈ 一个语义单位）
        英文：按单词数计算（一个单词 ≈ 一个语义单位）
        """
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return chinese_chars, english_words

    @staticmethod
    def get_text_length(text: str) -> int:
        """
        获取文本的"等效长度"
        中文：字符数
        英文：单词数 × 2（因为英文单词通常比中文字符长）
        """
        cn_chars, en_words = SimpleSegmenter.count_text_units(text)
        # 英文单词 × 2 是为了平衡中英文的等效长度
        return cn_chars + en_words * 2

    def add_text(self, text: str) -> Optional[str]:
        """添加文本，返回完整句子（如果有的话）"""
        self.buffer += text
        return self._try_segment()

    def _is_false_break(self, text: str) -> bool:
        """检查是否是假断句"""
        text_lower = text.lower().strip()

        # 英文省略号 ...
        if text_lower.endswith('...'):
            return True

        # 中文省略号 ……
        if text.endswith('……') or text.endswith('…'):
            return True

        # 数字列表符：1. 2. 3. 等（后面必须跟空格或换行，排除小数点）
        # 注意：这个检测放在小数点检测之后，避免误判

        # a.m. / p.m.
        if re.search(r'a\.?\s*m\.?$|p\.?\s*m\.?$', text_lower):
            return True

        # 数字中的小数点 (3.14, 0.5 等) - 完整小数
        if re.search(r'\d+\.\d+\s*$', text_lower):
            return True

        # 小数点开头 (2. 后面可能还有数字) - 数字加点结尾，很可能是小数
        # 例如：身高 2. -> 后面可能是 26
        if re.search(r'\d+\.$', text_lower):
            return True

        # 数字列表符：1. 2. 3. 等（单独一行或后面跟空格+大写字母）
        # 排除小数点情况（上面已经处理了数字.结尾的情况）
        # 这里处理：换行后的 1. 或句子开头的 1.
        if re.search(r'(^|\n)\s*\d+\.$', text_lower):
            return True

        # 常见英文缩写
        abbreviations = [
            r'mr\.?', r'mrs\.?', r'ms\.?', r'dr\.?', r'prof\.?',
            r'vs\.?', r'etc\.?', r'inc\.?', r'ltd\.?', r'co\.?', r'corp\.?',
            r'st\.?', r'ave\.?', r'jr\.?', r'sr\.?', r'ph\.?d\.?',
            r'e\.?g\.?', r'i\.?e\.?', r'approx\.?', r'no\.?', r'vol\.?',
            r'fig\.?', r'p\.?p\.?m\.?', r'ft\.?', r'lb\.?', r'oz\.?'
        ]
        for abbr in abbreviations:
            if re.search(abbr + r'\s*$', text_lower):
                return True

        return False

    def _is_safe_to_cut_english(self, text: str, split_idx: int) -> bool:
        """
        检查切割点是否安全（不会切断英文单词）

        如果切割点左右都是英文字母，说明切在了单词中间，不安全
        """
        if split_idx <= 0 or split_idx >= len(text):
            return True

        left_char = text[split_idx - 1]
        right_char = text[split_idx] if split_idx < len(text) else ''

        # 如果左边是英文字母，右边也是英文字母/数字，说明切在单词中间
        if left_char.isalpha() and left_char.isascii():
            if right_char.isalpha() or right_char.isdigit():
                return False

        return True

    @staticmethod
    def normalize_acronyms(text: str) -> str:
        """
        处理 J.A.R.V.I.S, U.S.A., A.I. 等带点缩写

        原理：把缩写词中的句点去掉，避免被错误切分
        例如：J.A.R.V.I.S → JARVIS
        """
        # 匹配至少包含两个 "字母+句点 " 的组合
        # 例如：J.A.R.V.I.S / U.S.A / A.I.
        acronym_pattern = re.compile(r'\b(?:[a-zA-Z]\.){2,}[a-zA-Z]?\.?\b')

        def replace_acronym(match):
            raw_word = match.group(0)
            # 去掉所有句点
            clean_word = raw_word.replace('.', '')
            # 首尾加空格保护
            return f" {clean_word} "

        return acronym_pattern.sub(replace_acronym, text)

    def _try_segment(self) -> Optional[str]:
        """尝试断句"""
        buffer = self.buffer.strip()
        if not buffer:
            return None

        # 使用等效长度（中英文分别处理）
        buffer_len = self.get_text_length(buffer)
        cn_chars, en_words = self.count_text_units(buffer)

        # 1. 检测标点
        # 强标点：句号、问号、感叹号（句子结束）
        has_strong_punc = re.search(r'[。！？.!?]$', buffer)
        # 弱标点：逗号、分号、冒号、顿号（可以在处切分）
        # 包含：中文 ，；：、 和英文 ,;:
        has_comma = re.search(r'[，,；;：:、]$', buffer)

        # 2. 首句策略：极速响应
        if self.is_first_sentence:
            # 根据文本类型确定最小长度
            if en_words > cn_chars:
                # 英文为主：按单词数
                min_len = self.FIRST_SENTENCE_MIN_EN
                max_len = self.FIRST_SENTENCE_MAX_EN
            else:
                # 中文为主：按字符数
                min_len = self.FIRST_SENTENCE_MIN_CN
                max_len = self.FIRST_SENTENCE_MAX_CN

            # 首句达到最小长度 + 任意标点 → 立即切
            if buffer_len >= min_len and (has_strong_punc or has_comma):
                sentence = buffer
                self.buffer = ""
                self.is_first_sentence = False
                return self._finalize(sentence)

            # 首句达到最大长度 → 强制切（找最后一个合适的分割点）
            if buffer_len >= max_len:
                return self._force_split_first_sentence()

            # 首句还在积累中，继续等待
            return None

        # 3. 后续句子：强标点切分
        if has_strong_punc:
            if self._is_false_break(buffer):
                return None

            sentence = buffer
            self.buffer = ""
            return self._finalize(sentence)

        # 4. 后续句子：逗号切分（区分中英文阈值）
        if has_comma:
            # 根据文本类型确定最小长度
            if en_words > cn_chars:
                # 英文为主：按单词数
                min_chunk = self.NORMAL_MIN_CHUNK_EN * 2  # 等效长度
            else:
                # 中文为主：按字符数
                min_chunk = self.NORMAL_MIN_CHUNK_CN

            if buffer_len >= min_chunk:
                sentence = buffer
                self.buffer = ""
                return self._finalize(sentence)

        # 5. 后续句子：长度超限强制切（区分中英文阈值）
        if en_words > cn_chars:
            max_chunk = self.NORMAL_MAX_CHUNK_EN * 2  # 等效长度
        else:
            max_chunk = self.NORMAL_MAX_CHUNK_CN

        if buffer_len >= max_chunk:
            return self._force_split_normal()

        return None

    def _force_split_first_sentence(self) -> Optional[str]:
        """首句强制切分（优先在标点处切，保护英文单词完整性）"""
        buffer = self.buffer.strip()
        buffer_len = len(buffer)

        # 根据文本类型确定阈值
        cn_chars, en_words = self.count_text_units(buffer)
        if en_words > cn_chars:
            # 英文为主
            min_len = self.FIRST_SENTENCE_MIN_EN
            max_len = self.FIRST_SENTENCE_MAX_EN
        else:
            # 中文为主
            min_len = self.FIRST_SENTENCE_MIN_CN
            max_len = self.FIRST_SENTENCE_MAX_CN

        # 优先找标点符号（包含中英文常见标点）
        # 强标点 + 弱标点
        punctuations = '，,；;：:、。！？.!?'
        for i in range(buffer_len - 1, max(buffer_len - 15, 0), -1):
            if buffer[i] in punctuations:
                # 检查切割点是否安全（不会切断英文单词）
                if self._is_safe_to_cut_english(buffer, i + 1):
                    sentence = buffer[:i+1]
                    self.buffer = buffer[i+1:]
                    self.is_first_sentence = False
                    return self._finalize(sentence)

        # 没有标点，在空格处切（这是英文最安全的切割点）
        last_space = buffer.rfind(' ', 0, max_len)
        if last_space > min_len:
            sentence = buffer[:last_space]
            self.buffer = buffer[last_space:]
            self.is_first_sentence = False
            return self._finalize(sentence)

        # 极端情况：找最后一个空格（即使超出预期长度）
        any_space = buffer.rfind(' ')
        if any_space > 5:  # 至少保留 5 个字符
            sentence = buffer[:any_space]
            self.buffer = buffer[any_space:]
            self.is_first_sentence = False
            return self._finalize(sentence)

        # 实在没有空格，只能直接切（但尽量避免）
        sentence = buffer[:max_len]
        self.buffer = buffer[max_len:]
        self.is_first_sentence = False
        return self._finalize(sentence)

    def _force_split_normal(self) -> Optional[str]:
        """后续句子强制切分（保护英文单词完整性）"""
        buffer = self.buffer.strip()
        buffer_len = len(buffer)

        # 根据文本类型确定阈值
        cn_chars, en_words = self.count_text_units(buffer)
        if en_words > cn_chars:
            # 英文为主
            min_chunk = self.NORMAL_MIN_CHUNK_EN
            max_chunk = self.NORMAL_MAX_CHUNK_EN
        else:
            # 中文为主
            min_chunk = self.NORMAL_MIN_CHUNK_CN
            max_chunk = self.NORMAL_MAX_CHUNK_CN

        # 找标点（包含中英文常见标点）
        punctuations = '，,；;：:、。！？.!?'
        for i in range(buffer_len - 1, max(buffer_len - 15, 0), -1):
            if buffer[i] in punctuations:
                # 检查切割点是否安全
                if self._is_safe_to_cut_english(buffer, i + 1):
                    sentence = buffer[:i+1]
                    self.buffer = buffer[i+1:]
                    return self._finalize(sentence)

        # 找空格（英文最安全的切割点）
        last_space = buffer.rfind(' ', 0, max_chunk)
        if last_space > min_chunk:
            sentence = buffer[:last_space]
            self.buffer = buffer[last_space:]
            return self._finalize(sentence)

        # 找任意空格
        any_space = buffer.rfind(' ')
        if any_space > 5:
            sentence = buffer[:any_space]
            self.buffer = buffer[any_space:]
            return self._finalize(sentence)

        # 直接切（最后手段，用字符数作为 fallback）
        sentence = buffer[:max_chunk]
        self.buffer = buffer[max_chunk:]
        return self._finalize(sentence)

    def _finalize(self, text: str) -> str:
        """文本最终处理：缩写词处理 + 清洗 + 尾部补丁"""
        text = text.strip()
        if not text or len(text) <= 1:
            return None

        # 1. 处理缩写词（J.A.R.V.I.S → JARVIS）
        text = self.normalize_acronyms(text)

        # 2. 清洗文本
        text = clean_text_for_tts(text)

        if not text or len(text) <= 1:
            return None

        # 3. 尾部补丁（让 TTS 更自然）
        text += "  "

        return text

    def flush(self) -> Optional[str]:
        """强制输出剩余缓冲区"""
        if self.buffer.strip():
            sentence = self.buffer
            self.buffer = ""
            self.is_first_sentence = False
            return self._finalize(sentence)
        return None

    def reset(self):
        """重置缓冲区"""
        self.buffer = ""
        self.is_first_sentence = True


# LLM 配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5-flash")

# 服务端口
SERVER_PORT = int(os.getenv("SERVER_PORT", "8004"))


# ============== 文本预处理 ==============

# 预编译正则：Emoji（精确范围，不覆盖中文）
# 参考：https://unicode.org/emoji/charts/full-emoji-list.html
# 重要：中文字符范围是 U+4E00 到 U+9FFF，绝对不能包含！
RE_EMOJI = re.compile(
    '['
    '\U0001F600-\U0001F64F'  # Emoticons 表情 (😀-🙏)
    '\U0001F300-\U0001F5FF'  # Misc Symbols 杂项符号 (🌀-🗿)
    '\U0001F680-\U0001F6FF'  # Transport 交通 (🚀-🛿)
    '\U0001F900-\U0001F9FF'  # Supplemental 补充 (🤀-🧿)
    '\U0001FA70-\U0001FAFF'  # Extended-A 扩展 (🩰-🫿)
    '\U00002600-\U000026FF'  # Misc Symbols 杂项 (☀-⛿)
    '\U00002700-\U000027BF'  # Dingbats 装饰 (✀-➿)
    '\U0001F1E0-\U0001F1FF'  # Flags 旗帜 (🇦-🇿)
    '\U00002300-\U000023FF'  # Misc Technical 技术符号
    '\U000020A0-\U000020CF'  # Currency Symbols 货币符号
    '\U00002190-\U000021FF'  # Arrows 箭头
    '\U000025A0-\U000025FF'  # Geometric Shapes 几何形状
    '\U00002900-\U0000297F'  # Misc Technical 杂项技术
    ']+',
    flags=re.UNICODE
)


def clean_text_for_tts(text: str) -> str:
    """
    清洗 LLM 输出文本，使其适合 TTS 合成

    核心原则：
    1. 只删除格式符号，不删除内容
    2. 需要停顿的地方用逗号替代
    3. 保留原文语义完整性

    处理内容：
    - Markdown 格式符号：**粗体** → 粗体，*斜体* → 斜体
    - Emoji：删除（会导致变声）
    - 代码块：删除（TTS 读不了代码）
    - URL/邮箱：删除
    - 列表符：保留数字，删除 - • 符号
    """
    # 1. 移除代码块（先处理，避免被其他规则干扰）
    text = re.sub(r'```[\s\S]*?```', '', text)

    # 2. 移除行内代码 `code` → code（保留内容）
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # 3. Markdown 链接 [text](url) → text（保留链接文字）
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 4. 移除 URL 和邮箱
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)

    # 5. 移除 Emoji
    text = RE_EMOJI.sub('', text)

    # 6. Markdown 格式符号（只删除符号，保留内容）
    # **粗体** → 粗体，*斜体* → 斜体，__下划线__ → 下划线
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **粗体**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *斜体*
    text = re.sub(r'__([^_]+)__', r'\1', text)     # __下划线__
    text = re.sub(r'_([^_]+)_', r'\1', text)       # _斜体_
    text = re.sub(r'~~([^~]+)~~', r'\1', text)     # ~~删除线~~

    # 7. 标题符号 # ## ### → 删除
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # 8. 列表符号（保留数字，删除 - • * 符号）
    text = re.sub(r'^[-•*]\s*', '', text, flags=re.MULTILINE)

    # 9. 引用符号 > → 删除
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # 10. 分隔线 --- *** → 删除
    text = re.sub(r'^[-*]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 11. 标点符号标准化
    # 顿号 → 逗号（TTS 模型通常不认识顿号，需要转换成逗号才能正确停顿）
    text = text.replace('、', '，')

    # 12. 多余的空白字符
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/Tab → 单个空格
    text = re.sub(r'\n\s*\n', '\n', text)  # 多个空行 → 单个换行
    text = re.sub(r'\n', ' ', text)  # 换行 → 空格
    text = text.strip()

    return text


def load_reference_audio():
    """加载参考音频和文本"""
    ref_audio = None
    ref_text = None

    # 加载参考音频
    if os.path.exists(REF_AUDIO_PATH):
        try:
            import scipy.io.wavfile as wav
            sample_rate, audio_data = wav.read(REF_AUDIO_PATH)
            # 转换为 float32 格式
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            # 转换为 MLX array
            ref_audio = mx.array(audio_data)
            print(f"[Swift-Speech] 加载参考音频: {REF_AUDIO_PATH} ({len(audio_data)/sample_rate:.1f}s)")
        except Exception as e:
            print(f"[Swift-Speech] 加载参考音频失败: {e}")

    # 加载参考文本
    if os.path.exists(REF_TEXT_PATH):
        try:
            with open(REF_TEXT_PATH, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
            print(f"[Swift-Speech] 加载参考文本: {REF_TEXT_PATH} ({len(ref_text)} 字符)")
        except Exception as e:
            print(f"[Swift-Speech] 加载参考文本失败: {e}")

    return ref_audio, ref_text


# ============== 全局模型 ==============

_qwen3_model = None
_model_warmed = False
_ref_audio = None
_ref_text = None


def get_qwen3_model():
    """获取或初始化 Qwen3-TTS 模型"""
    global _qwen3_model, _ref_audio, _ref_text

    if _qwen3_model is None:
        if not QWEN3_TTS_AVAILABLE:
            raise RuntimeError("Qwen3-TTS 未安装。请运行: pip install mlx-audio")

        print(f"[Swift-Speech] 加载模型: {QWEN3_TTS_MODEL}")
        _qwen3_model = mlx_load(QWEN3_TTS_MODEL)
        print(f"[Swift-Speech] 模型加载完成，采样率: {_qwen3_model.sample_rate}")

        # 加载参考音频
        _ref_audio, _ref_text = load_reference_audio()

    return _qwen3_model


def unload_model():
    """卸载模型，释放内存"""
    global _qwen3_model, _model_warmed, _ref_audio, _ref_text

    if _qwen3_model is not None:
        print("[Swift-Speech] 卸载模型，释放内存...")
        del _qwen3_model
        _qwen3_model = None
        _model_warmed = False
        _ref_audio = None
        _ref_text = None

        # 强制垃圾回收
        gc.collect()

        # 清理 MLX 缓存
        if mx is not None:
            mx.clear_cache()

        print("[Swift-Speech] 模型已卸载，内存已释放")


# 注册退出时清理
atexit.register(unload_model)


def synthesize_qwen3_base(
    text: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    add_pause: bool = False,
    ref_audio: Optional[np.ndarray] = None,
    ref_text: Optional[str] = None,
) -> bytes:
    """
    使用 Qwen3-TTS Base 模型合成语音

    Args:
        text: 要合成的文本（支持中英混合）
        temperature: 生成温度 (0.4 推荐)
        top_p: 核采样参数 (0.85 推荐)
        seed: 随机种子 (固定值确保一致性)
        add_pause: 是否在末尾添加停顿（只有强标点结尾的句子才需要）

    Returns:
        WAV 格式的音频字节
    """
    model = get_qwen3_model()

    # 【核心！】锁定随机种子
    mx.random.seed(seed)

    try:
        audio_segments = []
        sample_rate = model.sample_rate

        # 构建生成参数
        gen_params = {
            "text": text,
            "temperature": temperature,
            "top_p": top_p,
            "lang_code": "auto",  # 自动检测语言
            "verbose": False
        }

        # 如果有参考音频，添加声音克隆参数（优先使用传入的参数，否则使用全局变量）
        use_ref_audio = ref_audio if ref_audio is not None else _ref_audio
        use_ref_text = ref_text if ref_text is not None else _ref_text

        if use_ref_audio is not None and use_ref_text is not None:
            gen_params["ref_audio"] = use_ref_audio
            gen_params["ref_text"] = use_ref_text
            print(f"[Swift-Speech] 使用参考音频克隆声音")

        results = list(model.generate(**gen_params))

        if not results:
            raise RuntimeError("Qwen3-TTS 生成失败：没有结果")

        for r in results:
            if hasattr(r, 'audio') and r.audio is not None:
                audio_np = np.array(r.audio)
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()
                audio_segments.append(audio_np)

        if not audio_segments:
            raise RuntimeError("Qwen3-TTS 生成失败：没有音频数据")

        combined = np.concatenate(audio_segments, axis=0)

        # 抗爆音处理：在音频末尾添加极短的平滑过渡（5ms）
        # 这不是说话的淡入淡出，而是消除音频边界不连续导致的爆音
        smooth_samples = int(OUTPUT_SAMPLE_RATE * ANTI_CLICK_SMOOTH_MS / 1000)
        if smooth_samples > 0 and len(combined) > smooth_samples:
            # 创建一个极短的余弦平滑窗口（从 1 平滑降到 0）
            smooth_window = np.cos(np.linspace(0, np.pi/2, smooth_samples))
            # 对末尾应用平滑
            combined[-smooth_samples:] *= smooth_window

        # 只在强标点结尾的句子添加停顿（解决句子太密集问题）
        if add_pause:
            pause_samples = int(OUTPUT_SAMPLE_RATE * SENTENCE_PAUSE_MS / 1000)
            silence = np.zeros(pause_samples, dtype=combined.dtype)
            combined = np.concatenate([combined, silence], axis=0)

        # float32 -> int16 转换
        audio_int16 = (combined * 32767).astype(np.int16)

        # 写入 WAV 字节
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        return buffer.getvalue()

    except Exception as e:
        print(f"[Swift-Speech] 合成错误: {e}")
        raise


def warmup_model():
    """预热模型"""
    global _model_warmed
    if _model_warmed:
        return

    print("[Swift-Speech] 预热模型...")
    model = get_qwen3_model()

    # 锁定随机种子
    mx.random.seed(DEFAULT_SEED)

    # 预热合成
    gen_params = {
        "text": "准备就绪。",
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "lang_code": "auto",
        "verbose": False
    }

    # 预热时也使用参考音频
    if _ref_audio is not None and _ref_text is not None:
        gen_params["ref_audio"] = _ref_audio
        gen_params["ref_text"] = _ref_text

    results = list(model.generate(**gen_params))

    _model_warmed = True
    print("[Swift-Speech] 模型预热完成")


# ============== FastAPI 应用 ==============

app = FastAPI(
    title="Swift-Speech",
    description="基于 Qwen3-TTS (MLX) 的实时流式语音合成 SDK"
)


class TTSRequest(BaseModel):
    text: str
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    seed: int = DEFAULT_SEED
    # 参考音频配置（可选）
    ref_audio_base64: Optional[str] = None  # base64 编码的参考音频
    ref_text: Optional[str] = None          # 参考音频对应的文本


class LLMStreamRequest(BaseModel):
    prompt: str
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    seed: int = DEFAULT_SEED
    # LLM 配置（可选，不提供则使用默认）
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    # 参考音频配置（可选）
    ref_audio_base64: Optional[str] = None
    ref_text: Optional[str] = None


@app.get("/")
async def root():
    """返回测试页面"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Swift-Speech</h1><p>访问 /docs 查看 API</p>")


@app.get("/config")
async def get_config():
    """获取默认配置（供前端使用）"""
    return {
        "llm_base_url": LLM_BASE_URL,
        "llm_api_key": LLM_API_KEY[:8] + "..." if LLM_API_KEY and len(LLM_API_KEY) > 8 else "",
        "llm_api_key_full": LLM_API_KEY,  # 本地开发时返回完整 key
        "llm_model": LLM_MODEL,
        "server_port": SERVER_PORT,
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "engine": "swift-speech",
        "model_type": "base",
        "qwen3_tts_available": QWEN3_TTS_AVAILABLE,
        "llm_available": LLM_AVAILABLE and bool(LLM_API_KEY),
        "model_warmed": _model_warmed,
        "model": QWEN3_TTS_MODEL,
        "ref_audio_loaded": _ref_audio is not None,
        "ref_text_loaded": _ref_text is not None,
        "default_temperature": DEFAULT_TEMPERATURE,
        "default_seed": DEFAULT_SEED,
        "sentence_pause_ms": SENTENCE_PAUSE_MS,
        "segmenter": "SimpleSegmenter v8 (英文保护+缩写处理)"
    }


@app.post("/shutdown")
async def shutdown():
    """卸载模型并准备关闭"""
    unload_model()
    return {"status": "ok", "message": "模型已卸载"}


@app.post("/warmup")
async def warmup():
    """预热模型"""
    global _model_warmed

    if _model_warmed:
        return {"status": "ok", "message": "模型已预热", "already_warmed": True}

    try:
        start_time = time.time()
        warmup_model()
        elapsed = (time.time() - start_time) * 1000

        return {
            "status": "ok",
            "message": "模型预热完成",
            "elapsed_ms": elapsed,
            "already_warmed": False,
            "ref_audio_loaded": _ref_audio is not None,
            "ref_text_loaded": _ref_text is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speakers")
async def list_speakers():
    """列出可用声音"""
    return {
        "type": "voice_cloning",
        "ref_audio": REF_AUDIO_PATH,
        "ref_text": REF_TEXT_PATH,
        "loaded": _ref_audio is not None
    }


@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """单次 TTS 合成"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="文本不能为空")

    # 处理动态参考音频（如果前端提供了）
    dynamic_ref_audio = None
    dynamic_ref_text = None

    if req.ref_audio_base64 and req.ref_text:
        try:
            audio_bytes = base64.b64decode(req.ref_audio_base64)
            sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            dynamic_ref_audio = mx.array(audio_data)
            dynamic_ref_text = req.ref_text
            print(f"[Swift-Speech] /tts 使用动态参考音频 ({len(audio_data)/sample_rate:.1f}s)")
        except Exception as e:
            print(f"[Swift-Speech] 解析动态参考音频失败: {e}")

    # 确定使用哪个参考音频（优先动态，否则全局）
    use_ref_audio = dynamic_ref_audio if dynamic_ref_audio is not None else _ref_audio
    use_ref_text = dynamic_ref_text if dynamic_ref_text is not None else _ref_text

    try:
        start_time = time.time()
        audio_bytes = synthesize_qwen3_base(
            text=req.text,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            ref_audio=use_ref_audio,
            ref_text=use_ref_text,
        )
        latency = (time.time() - start_time) * 1000

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "success": True,
            "engine": "swift-speech",
            "model_type": "base",
            "audio_base64": audio_base64,
            "latency_ms": latency,
            "text_length": len(req.text),
            "voice_cloning": _ref_audio is not None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm-tts/stream")
async def llm_tts_stream(req: LLMStreamRequest):
    """LLM + TTS 双流式接口（使用智能断句器）"""

    async def generate():
        # 获取 LLM 配置（优先使用请求中的配置）
        llm_base_url = req.llm_base_url or LLM_BASE_URL
        llm_api_key = req.llm_api_key or LLM_API_KEY
        llm_model = req.llm_model or LLM_MODEL

        # 处理动态参考音频（如果前端提供了）
        dynamic_ref_audio = _ref_audio  # 默认使用全局加载的参考音频
        dynamic_ref_text = _ref_text

        if req.ref_audio_base64 and req.ref_text:
            try:
                # 解码 base64 音频
                audio_bytes = base64.b64decode(req.ref_audio_base64)
                # 使用 scipy 读取 WAV
                sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))
                # 转换为 float32
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                # 转换为 MLX array
                dynamic_ref_audio = mx.array(audio_data)
                dynamic_ref_text = req.ref_text
                print(f"[Swift-Speech] 使用动态参考音频 ({len(audio_data)/sample_rate:.1f}s)")
            except Exception as e:
                print(f"[Swift-Speech] 解析动态参考音频失败: {e}")

        voice_cloning = dynamic_ref_audio is not None

        if not LLM_AVAILABLE or not llm_api_key:
            yield f"data: {json.dumps({'type': 'error', 'message': 'LLM 未配置，请提供 API Key'})}\n\n"
            return

        try:
            # 预热模型
            warmup_model()

            # 初始化 LLM 客户端（使用动态配置）
            client = OpenAI(
                api_key=llm_api_key,
                base_url=llm_base_url
            )

            yield f"data: {json.dumps({'type': 'start', 'engine': 'swift-speech', 'model_type': 'base', 'model': llm_model, 'voice_cloning': voice_cloning, 'temperature': req.temperature, 'seed': req.seed, 'segmenter': 'SimpleSegmenter v8'})}\n\n"

            # 使用智能断句器
            segmenter = SimpleSegmenter()
            sentence_count = 0
            token_count = 0
            total_audio_duration = 0
            first_audio_latency = None
            start_time = time.time()

            # 调用 LLM 流式 API（使用动态配置的 model）
            stream = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "你是一个友好的AI助手，请用简洁的语言回答问题。"},
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

                    # 使用智能断句器处理文本
                    sentence = segmenter.add_text(content)

                    if sentence:
                        sentence_count += 1

                        yield f"data: {json.dumps({'type': 'sentence', 'sentence': sentence, 'index': sentence_count, 'length': len(sentence)})}\n\n"

                        try:
                            tts_start = time.time()
                            # 判断是否以强标点结尾（只有强标点结尾才添加停顿）
                            is_strong_end = bool(re.search(r'[。！？.!?]$', sentence.strip()))
                            audio_bytes = synthesize_qwen3_base(
                                text=sentence,
                                temperature=req.temperature,
                                top_p=req.top_p,
                                seed=req.seed,
                                add_pause=is_strong_end,
                                ref_audio=dynamic_ref_audio,
                                ref_text=dynamic_ref_text,
                            )
                            tts_latency = (time.time() - tts_start) * 1000

                            # 计算音频时长
                            wav_header_size = 44
                            pcm_bytes = len(audio_bytes) - wav_header_size
                            num_samples = pcm_bytes // 2
                            audio_duration = num_samples / OUTPUT_SAMPLE_RATE
                            total_audio_duration += audio_duration

                            if first_audio_latency is None:
                                first_audio_latency = (time.time() - start_time) * 1000
                                print(f"[Swift-Speech] 首句延迟: {first_audio_latency:.0f}ms")

                            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                            yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_base64, 'duration_ms': audio_duration * 1000, 'tts_latency_ms': tts_latency, 'sentence_index': sentence_count})}\n\n"

                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'tts_error', 'message': str(e), 'sentence_index': sentence_count})}\n\n"

            # 处理剩余缓冲区
            final_sentence = segmenter.flush()
            if final_sentence:
                sentence_count += 1

                yield f"data: {json.dumps({'type': 'sentence', 'sentence': final_sentence, 'index': sentence_count, 'length': len(final_sentence)})}\n\n"

                try:
                    # 最后一句通常以强标点结尾，添加停顿
                    is_strong_end = bool(re.search(r'[。！？.!?]$', final_sentence.strip()))
                    audio_bytes = synthesize_qwen3_base(
                        text=final_sentence,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        seed=req.seed,
                        add_pause=is_strong_end,
                        ref_audio=dynamic_ref_audio,
                        ref_text=dynamic_ref_text,
                    )
                    wav_header_size = 44
                    pcm_bytes = len(audio_bytes) - wav_header_size
                    num_samples = pcm_bytes // 2
                    audio_duration = num_samples / OUTPUT_SAMPLE_RATE

                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    yield f"data: {json.dumps({'type': 'audio', 'audio_base64': audio_base64, 'duration_ms': audio_duration * 1000, 'sentence_index': sentence_count})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'tts_error', 'message': str(e), 'sentence_index': sentence_count})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'total_tokens': token_count, 'total_sentences': sentence_count, 'total_audio_duration': total_audio_duration, 'first_audio_latency': first_audio_latency})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ============== 主函数 ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Swift-Speech - Realtime Streaming Voice SDK")
    print("=" * 60)
    print(f"  Qwen3-TTS 可用: {QWEN3_TTS_AVAILABLE}")
    print(f"  LLM 可用: {LLM_AVAILABLE and bool(LLM_API_KEY)}")
    print(f"  LLM 模型: {LLM_MODEL}")
    print(f"  TTS 模型: {QWEN3_TTS_MODEL}")
    print(f"  参考音频: {REF_AUDIO_PATH}")
    print(f"  参考文本: {REF_TEXT_PATH}")
    print(f"  默认温度: {DEFAULT_TEMPERATURE}")
    print(f"  随机种子: {DEFAULT_SEED}")
    print(f"  句间停顿: {SENTENCE_PAUSE_MS}ms")
    print(f"  断句器: SimpleSegmenter v8 (英文保护+缩写处理)")
    print("=" * 60)

    # 预热模型
    if QWEN3_TTS_AVAILABLE:
        print("\n预热模型...")
        try:
            warmup_model()
        except Exception as e:
            print(f"模型预热失败: {e}")

    print(f"\n启动服务: http://localhost:{SERVER_PORT}")
    print("  POST /shutdown - 卸载模型释放内存")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
