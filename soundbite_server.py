# =============================================================================
# SOUNDBITE — single-file backend.  Colab / Kaggle / local, one cell.
# =============================================================================
# 1. Install once:
#      pip install -q qwen-tts soundfile librosa openai-whisper \
#                     paddlepaddle paddleocr anthropic \
#                     fastapi "uvicorn[standard]" python-multipart nest-asyncio pyngrok
#
# 2. Optional env vars:
#      ANTHROPIC_API_KEY  enables /api/rewrite (the "adlib personality" toggle)
#      NGROK_AUTHTOKEN    needed for a public URL
#      NGROK_DOMAIN       your reserved domain, so index.html never needs editing
#
# 3. Run.  It prints the URL to paste into API_BASE in index.html.
#
# Re-running is cheap: models are cached in a module global, so a second run
# rebinds the routes without re-downloading weights.
# =============================================================================

import io, os, json, re, uuid, tempfile, threading
from typing import List, Optional

import numpy as np
import soundfile as sf
from PIL import Image, ImageOps

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------- 0. paths --
# Kaggle has no /content and Colab has no /kaggle/working — pick whichever
# exists so the same cell runs on both.

def _workdir() -> str:
    for p in ("/kaggle/working", "/content"):
        if os.path.isdir(p):
            return p
    return os.getcwd()

BASE_DIR  = os.path.join(_workdir(), "soundbite")
VOICE_DIR = os.path.join(BASE_DIR, "voices")
os.makedirs(VOICE_DIR, exist_ok=True)

PORT = int(os.environ.get("SOUNDBITE_PORT", "8000"))


# --------------------------------------------------------------- 1. models --
# Cached across cell re-runs.

_M = globals().get("_SOUNDBITE_MODELS", {})
globals()["_SOUNDBITE_MODELS"] = _M

# Each model loads independently and on demand. They used to load together,
# which meant a PaddleOCR install problem — common on Kaggle, where paddleocr
# 3.x drags in paddlex and the CPU paddlepaddle wheel often mismatches — took
# voice cloning down with it. Now an OCR failure only breaks OCR.


def _require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU. Colab: Runtime > Change runtime type > GPU (T4). "
            "Kaggle: Settings > Accelerator > GPU (and Internet: On)."
        )
    if not _M.get("gpu_logged"):
        print("GPU:", torch.cuda.get_device_name(0))
        _M["gpu_logged"] = True
    return torch


def _need_tts():
    if "tts" not in _M:
        torch = _require_gpu()
        from qwen_tts import Qwen3TTSModel
        print("loading Qwen3-TTS…")
        _M["tts"] = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        print("  Qwen3-TTS ready")
    return _M["tts"]


def _need_asr():
    if "asr" not in _M:
        _require_gpu()
        import whisper
        print("loading Whisper…")
        _M["asr"] = whisper.load_model("small")
        print("  Whisper ready")
    return _M["asr"]


def _need_ocr():
    if "ocr" not in _M:
        from paddleocr import PaddleOCR      # ImportError propagates as a 503
        print("loading PaddleOCR…")
        # PaddleOCR's default caps detection input at 960px on the long side,
        # which blurs small type on a 2000px+ scan. 1600 keeps columns legible.
        # Kwarg names moved between 2.x and 3.x, so try newest first.
        last = None
        for kw in (
            dict(use_textline_orientation=True, lang="en",
                 text_det_limit_type="max", text_det_limit_side_len=1600),
            dict(use_textline_orientation=True, lang="en"),
            dict(use_angle_cls=True, lang="en",
                 det_limit_type="max", det_limit_side_len=1600),
            dict(use_angle_cls=True, lang="en"),
        ):
            try:
                _M["ocr"] = PaddleOCR(**kw)
                break
            except (TypeError, ValueError) as e:
                last = e
        else:
            raise last
        print("  PaddleOCR ready")
    return _M["ocr"]


# ------------------------------------------------------------- 2. chunking --
# Referenced by /api/chunks. Splits on paragraphs first, then sentences, so the
# layout-aware OCR's paragraph breaks actually buy something.

_SENT = re.compile(r'(?<=[.!?。！？])\s+')


def _split_chunks(text: str, max_chars: int = 320) -> List[str]:
    text = re.sub(r'[ \t]+', ' ', text or '').strip()
    if not text:
        return []

    chunks: List[str] = []
    for para in re.split(r'\n\s*\n', text):
        para = para.strip()
        if not para:
            continue
        cur = ''
        for s in _SENT.split(para):
            s = s.strip()
            if not s:
                continue
            while len(s) > max_chars:              # one runaway sentence
                cut = s.rfind(' ', 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                if cur:
                    chunks.append(cur)
                    cur = ''
                chunks.append(s[:cut].strip())
                s = s[cut:].strip()
            if not cur:
                cur = s
            elif len(cur) + 1 + len(s) <= max_chars:
                cur += ' ' + s
            else:
                chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------- 3. layout-aware OCR ---
# Recognition is stock PaddleOCR; only the ordering changed. Sorting lines by
# banded y then x interleaves a two-column page (colA-1, colB-1, colA-2, …).
# A recursive XY-cut over the polygons handles columns and block text in one
# code path — block text is simply the zero-gutter case, not a separate mode.

def _poly_bbox(poly):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _ocr_lines(img):
    """-> [(text, score, (x0, y0, x1, y1))] in raw detection order."""
    engine = _need_ocr()
    res = None
    if hasattr(engine, "predict"):
        try:
            res = engine.predict(img)
        except Exception:
            res = None
    if res is None:
        try:
            res = engine.ocr(img, cls=True)
        except TypeError:
            res = engine.ocr(img)

    out = []
    for page in (res or []):
        if page is None:
            continue

        if hasattr(page, "get") and page.get("rec_texts") is not None:   # 3.x
            texts  = page.get("rec_texts") or []
            scores = page.get("rec_scores") or [1.0] * len(texts)
            polys  = page.get("rec_polys")
            if polys is None or len(polys) == 0:
                polys = page.get("dt_polys") or []
            for t, s, p in zip(texts, scores, polys):
                if t and t.strip():
                    out.append((t.strip(), float(s), _poly_bbox(p)))
            continue

        for entry in page:                                                # 2.x
            try:
                box, txt, score = entry[0], entry[1][0], float(entry[1][1])
            except Exception:
                continue
            if txt and txt.strip():
                out.append((txt.strip(), score, _poly_bbox(box)))
    return out


# Every threshold below is a multiple of median line height, so nothing is tied
# to a pixel constant — a 600dpi scan and a phone photo behave the same.

def _median_h(lines):
    hs = [b[3] - b[1] for _, _, b in lines]
    return float(np.median(hs)) if hs else 1.0


def _sorted_y(lines):
    return sorted(lines, key=lambda l: (l[2][1], l[2][0]))


def _hsplit(lines, gap):
    ls = _sorted_y(lines)
    bands, cur, bottom = [], [ls[0]], ls[0][2][3]
    for ln in ls[1:]:
        if ln[2][1] - bottom >= gap:
            bands.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
        bottom = max(bottom, ln[2][3])
    bands.append(cur)
    return bands


def _vsplit(lines, min_gap_frac=0.035):
    """Columns from the union of every line's x-interval.

    In block text the lines all overlap horizontally, so the union collapses to
    a single span and this returns immediately. That's what lets one setting
    cover both layouts.
    """
    xs = [(b[0], b[2]) for _, _, b in lines]
    x0, x1 = min(a for a, _ in xs), max(b for _, b in xs)
    W = x1 - x0
    if W <= 0:
        return [lines]
    gap_min = max(min_gap_frac * W, 0.9 * _median_h(lines))

    iv = sorted(xs)
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if len(merged) == 1:
        return [lines]

    cuts = [(merged[i][1] + merged[i + 1][0]) / 2
            for i in range(len(merged) - 1)
            if merged[i + 1][0] - merged[i][1] >= gap_min]
    if not cuts:
        return [lines]

    bounds = [x0 - 1] + cuts + [x1 + 1]
    cols = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        grp = [l for l in lines if lo <= (l[2][0] + l[2][2]) / 2 < hi]
        if grp:
            cols.append(grp)
    return cols if len(cols) > 1 else [lines]


def _peel_wide(lines, thresh=0.70):
    """A full-width title bridges the gutter and hides it — peel it out."""
    x0 = min(b[0] for _, _, b in lines)
    x1 = max(b[2] for _, _, b in lines)
    W = max(x1 - x0, 1.0)
    wide_idx = {i for i, l in enumerate(lines)
                if (l[2][2] - l[2][0]) / W >= thresh}
    if not wide_idx or len(wide_idx) == len(lines):
        return [lines]

    wides = sorted((lines[i] for i in wide_idx), key=lambda l: l[2][1])
    remaining = [l for i, l in enumerate(lines) if i not in wide_idx]
    parts = []
    for w in wides:
        wy = w[2][1]
        above = [l for l in remaining if (l[2][1] + l[2][3]) / 2 < wy]
        remaining = [l for l in remaining if (l[2][1] + l[2][3]) / 2 >= wy]
        if above:
            parts.append(above)
        parts.append([w])
    if remaining:
        parts.append(remaining)
    return parts


def _leaf_blocks(lines, depth=0):
    if len(lines) <= 1 or depth >= 12:
        return [_sorted_y(lines)] if lines else []

    h = _median_h(lines)

    bands = _hsplit(lines, gap=1.8 * h)
    if len(bands) > 1:
        return [b for band in bands for b in _leaf_blocks(band, depth + 1)]

    cols = _vsplit(lines)
    if len(cols) > 1:
        return [b for c in cols for b in _leaf_blocks(c, depth + 1)]

    # Only accept the peel if it actually reveals a gutter — otherwise a
    # paragraph whose last line is short gets shredded into one block per line.
    parts = _peel_wide(lines)
    if len(parts) > 1 and any(len(_vsplit(p)) > 1 for p in parts):
        return [b for p in parts for b in _leaf_blocks(p, depth + 1)]

    return [_sorted_y(lines)]


_CJK = re.compile('[　-鿿＀-￯]')
_SENT_END = re.compile(r'[.!?。！？"\'\)\]]\s*$')


def _join(a, b):
    if not a:
        return b
    if a.endswith('-') and b[:1].islower():
        return a[:-1] + b                     # de-hyphenate across line breaks
    if _CJK.search(a[-1]) or _CJK.search(b[0]):
        return a + b
    return a + ' ' + b


def _paragraphs(block):
    if not block:
        return []
    h = _median_h(block)
    ls = _sorted_y(block)
    right = max(b[2] for _, _, b in ls)

    paras, cur, prev = [], '', None
    for ln in ls:
        txt, _, bb = ln
        brk = False
        if prev is not None:
            gap = bb[1] - prev[2][3]
            short = prev[2][2] < right - 2.0 * h
            brk = gap > 1.5 * h or (short and _SENT_END.search(prev[0]) is not None)
        if brk and cur:
            paras.append(cur)
            cur = txt
        else:
            cur = _join(cur, txt)
        prev = ln
    if cur:
        paras.append(cur)
    return paras


def ocr_image_to_text(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    lines = _ocr_lines(arr)
    if not lines:
        return "", 0
    paras = []
    for block in _leaf_blocks(lines):
        paras.extend(_paragraphs(block))
    return "\n\n".join(p for p in paras if p.strip()), len(lines)


# ------------------------------------------------------- 4. voice profiling --
# Two passes. First measure what a transcript cannot tell you — pace, pause
# structure, pitch movement — straight off the waveform. Then hand those numbers
# plus the transcript to Claude and get back a structured style card with worked
# examples, which every later rewrite uses as few-shot context.
#
# This runs once per voice, in a background thread, and is persisted to disk.

FILLERS = {"um", "uh", "er", "ah", "like", "yeah", "okay", "so", "right",
           "basically", "actually", "literally", "honestly", "anyway", "mean"}

DISCOURSE = ["so", "but", "and then", "you know", "i mean", "right", "okay",
             "well", "anyway", "basically", "actually", "the thing is",
             "here's the thing", "look", "now"]

CONTRACTIONS = re.compile(r"\b\w+'(s|re|ve|ll|d|t|m)\b", re.I)


def _transcribe(wav_path):
    """Word timestamps when we can get them, a plain transcript when we can't."""
    asr = _need_asr()
    try:
        return asr.transcribe(wav_path, word_timestamps=True)
    except Exception as e:
        print("word_timestamps unavailable, falling back:", e)
        return asr.transcribe(wav_path)


def _words_from_whisper(result):
    out = []
    for seg in result.get("segments") or []:
        for w in seg.get("words") or []:
            tok = (w.get("word") or "").strip()
            if tok:
                out.append((tok, float(w["start"]), float(w["end"])))
    return out


def _delivery_stats(wav_path, result):
    """Prosody. A fast talker who rarely pauses writes long run-on sentences;
    someone with big pitch swings and frequent short pauses writes in punchy
    fragments. These numbers are what turn a transcript into a style."""
    import librosa
    stats = {}

    words = _words_from_whisper(result)
    if words:
        span = max(words[-1][2] - words[0][1], 1e-6)
        stats["words_per_minute"] = round(len(words) / span * 60, 1)

        gaps = [words[i + 1][1] - words[i][2] for i in range(len(words) - 1)]
        gaps = [g for g in gaps if g > 0]
        pauses = [g for g in gaps if g >= 0.25]
        stats["pauses_per_100_words"] = round(len(pauses) / len(words) * 100, 1)
        stats["mean_pause_sec"] = round(float(np.mean(pauses)), 2) if pauses else 0.0
        stats["longest_pause_sec"] = round(max(pauses), 2) if pauses else 0.0

        runs, cur = [], 1                      # words between breaths
        for g in gaps:
            if g >= 0.30:
                runs.append(cur)
                cur = 1
            else:
                cur += 1
        runs.append(cur)
        stats["median_words_per_burst"] = int(np.median(runs))

        low = [w[0].lower().strip(".,!?;:") for w in words]
        stats["filler_rate_pct"] = round(
            sum(1 for w in low if w in FILLERS) / len(low) * 100, 1)

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    try:
        # pyin is the slow step (tens of seconds on 30s of audio) — it's why
        # this whole pass runs in a background thread.
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        voiced = f0[~np.isnan(f0)]
        if voiced.size:
            semis = 12 * np.log2(voiced / np.median(voiced))
            stats["pitch_median_hz"] = round(float(np.median(voiced)), 1)
            stats["pitch_range_semitones"] = round(
                float(np.percentile(semis, 95) - np.percentile(semis, 5)), 1)
    except Exception:
        pass

    rms = librosa.feature.rms(y=y)[0]
    if rms.size:
        stats["loudness_variation"] = round(
            float(np.std(rms) / (np.mean(rms) + 1e-9)), 2)
    return stats


def _repeated_ngrams(words, n_min=3, n_max=5, top=8):
    from collections import Counter
    c = Counter()
    for n in range(n_min, n_max + 1):
        for i in range(len(words) - n + 1):
            c[" ".join(words[i:i + n])] += 1
    hits = sorted(((p, k) for p, k in c.items() if k >= 2),
                  key=lambda x: (-x[1], -len(x[0])))
    return [{"phrase": p, "times": k} for p, k in hits[:top]]


def _text_stats(text):
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    words = re.findall(r"[a-z']+", text.lower())
    st = {}

    if sents:
        lens = [len(re.findall(r"[a-z']+", s.lower())) for s in sents]
        st["mean_sentence_words"] = round(float(np.mean(lens)), 1)
        st["sentence_length_sd"] = round(float(np.std(lens)), 1)
        st["question_rate_pct"] = round(
            sum(1 for s in sents if s.endswith('?')) / len(sents) * 100, 1)

    if words:
        st["vocabulary_ratio"] = round(len(set(words)) / len(words), 2)
        st["contractions_per_100_words"] = round(
            len(CONTRACTIONS.findall(text)) / len(words) * 100, 1)
        st["first_person_pct"] = round(
            sum(1 for w in words if w in ("i", "me", "my", "we", "our")) / len(words) * 100, 1)
        st["second_person_pct"] = round(
            sum(1 for w in words if w in ("you", "your")) / len(words) * 100, 1)

    low = " " + " ".join(words) + " "
    st["discourse_markers"] = {d: low.count(" " + d + " ")
                               for d in DISCOURSE if low.count(" " + d + " ") > 0}
    st["repeated_phrases"] = _repeated_ngrams(words)
    return st


# Structured-output schema. No minItems/maxItems/minLength — the API rejects
# those; counts are stated in the prompt instead.
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "traits": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "trait": {"type": "string"},
                "shows_up_as": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["trait", "shows_up_as", "evidence"],
            "additionalProperties": False}},
        "rhythm": {"type": "string"},
        "reaches_for": {"type": "array", "items": {"type": "string"}},
        "avoids": {"type": "array", "items": {"type": "string"}},
        "tics": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "phrase": {"type": "string"},
                "used_when": {"type": "string"},
                "frequency": {"type": "string",
                              "enum": ["often", "sometimes", "rarely"]},
            },
            "required": ["phrase", "used_when", "frequency"],
            "additionalProperties": False}},
        "situations": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "situation": {"type": "string"},
                "approach": {"type": "string"},
            },
            "required": ["situation", "approach"],
            "additionalProperties": False}},
        "examples": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "situation": {"type": "string"},
                "neutral": {"type": "string"},
                "in_voice": {"type": "string"},
                "source": {"type": "string",
                           "enum": ["from recording", "extrapolated"]},
            },
            "required": ["situation", "neutral", "in_voice", "source"],
            "additionalProperties": False}},
    },
    "required": ["summary", "traits", "rhythm", "reaches_for", "avoids",
                 "tics", "situations", "examples"],
    "additionalProperties": False,
}


EXTRACT_SYSTEM = """You build a style guide for one person, from a recording of \
them speaking, so that later text can be rewritten to sound like them reading it \
aloud.

You are given measured delivery statistics and a verbatim transcript. Ground \
every claim in one or the other. Do not infer demographics, background, mood, or \
anything about them as a person beyond how they talk — you are describing a \
speaking style, not profiling a human being.

Read the numbers as style, not trivia. A high words-per-minute with few pauses \
means long unbroken sentences and comma splices. A short median burst with wide \
pitch range means short punchy sentences and fragments. High contraction and \
second-person rates mean they talk to the reader, not at them.

For `examples`, prefer real ones. Find sentences in the transcript that are \
characteristic of how this person talks. For each, write the flat, neutral, \
encyclopedic version of that same content — what a generic narrator would have \
said — and put that in `neutral`, with their actual words in `in_voice`, and \
"from recording" as the source. A real pair teaches the style far better than an \
invented one. Then add two or three "extrapolated" examples, but only to cover \
reading situations the recording does not show: stating a number or statistic, \
moving from one topic to the next, and flagging something as important. Aim for \
six to eight examples total.

Give four to six traits. Keep `evidence` to a short verbatim quote."""


def _extract_profile(vid):
    """Runs in a background thread. Clone returns long before this finishes."""
    v = VOICES.get(vid)
    if not v:
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        v["profile_status"] = "skipped: no ANTHROPIC_API_KEY"
        _persist(vid)
        return

    v["profile_status"] = "working"
    _persist(vid)
    try:
        # Word timings aren't persisted (they're bulky and derivable), so a
        # rebuild after a restart has to re-transcribe — otherwise every pacing
        # statistic silently comes back empty.
        asr_result = v.get("_asr_result")
        if not asr_result:
            asr_result = _transcribe(v["ref_path"])
        delivery = _delivery_stats(v["ref_path"], asr_result)
        lexical = _text_stats(v["ref_text"])

        hint = ""
        if v.get("tags"):
            hint += "\nThe user tagged this voice: " + ", ".join(v["tags"])
        if v.get("prompt"):
            hint += "\nThe user described it as: " + v["prompt"]
        if v.get("intents"):
            hint += "\nWhat the recordings were meant to be about: " + \
                    "; ".join(v["intents"])

        from anthropic import Anthropic
        msg = Anthropic().messages.create(
            model="claude-opus-5",
            max_tokens=16000,
            output_config={"effort": "high",
                           "format": {"type": "json_schema",
                                      "schema": PROFILE_SCHEMA}},
            system=EXTRACT_SYSTEM,
            messages=[{"role": "user", "content":
                       f"SPEAKER: {v['name']}{hint}\n\n"
                       f"DELIVERY (measured from the audio):\n"
                       f"{json.dumps(delivery, indent=2)}\n\n"
                       f"LANGUAGE (measured from the transcript):\n"
                       f"{json.dumps(lexical, indent=2)}\n\n"
                       f"TRANSCRIPT:\n{v['ref_text'][:12000]}"}],
        )
        if msg.stop_reason == "refusal":
            raise RuntimeError("profile extraction declined")

        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        if not text:
            raise RuntimeError(f"empty response (stop_reason={msg.stop_reason})")
        v["profile"] = json.loads(text)
        v["measured"] = {"delivery": delivery, "language": lexical}
        v["profile_status"] = "ready"
    except Exception as e:
        v["profile_status"] = "failed: " + str(e)[:200]
        print("profile extraction failed for", vid, "-", e)
    finally:
        v.pop("_asr_result", None)
        _persist(vid)


def _render_profile(v) -> str:
    """The style card, as it goes into the rewrite system prompt."""
    p = v.get("profile")
    if not p:
        bits = [f"SPEAKER: {v['name']}"]
        if v.get("tags"):
            bits.append("Described as: " + ", ".join(v["tags"]))
        if v.get("prompt"):
            bits.append("How they should sound: " + v["prompt"])
        if v.get("ref_text"):
            bits.append("Transcript of them speaking:\n" + v["ref_text"][:4000])
        return "\n\n".join(bits)

    L = [f"SPEAKER: {v['name']}", "", p["summary"], "", "TRAITS"]
    for t in p["traits"]:
        L.append(f'- {t["trait"]}: {t["shows_up_as"]}  (heard as: "{t["evidence"]}")')

    L += ["", "RHYTHM: " + p["rhythm"]]
    if p.get("reaches_for"):
        L.append("REACHES FOR: " + ", ".join(p["reaches_for"]))
    if p.get("avoids"):
        L.append("AVOIDS: " + ", ".join(p["avoids"]))

    if p.get("tics"):
        L += ["", "VERBAL HABITS (reproduce at the stated frequency, no more)"]
        for t in p["tics"]:
            L.append(f'- "{t["phrase"]}" — {t["used_when"]} ({t["frequency"]})')

    if p.get("situations"):
        L += ["", "BY SITUATION"]
        for s in p["situations"]:
            L.append(f'- {s["situation"]}: {s["approach"]}')

    if p.get("examples"):
        L += ["", "WORKED EXAMPLES — flat prose on the left, how they say it on the right"]
        for e in p["examples"]:
            mark = "" if e["source"] == "from recording" else "  (extrapolated)"
            L += ["", f'[{e["situation"]}]{mark}',
                  f'  flat: {e["neutral"]}',
                  f'  them: {e["in_voice"]}']

    if v.get("prompt"):
        L += ["", "THE USER ALSO ASKED FOR: " + v["prompt"]]
    return "\n".join(L)


# --- persistence: a voice outlives the process ------------------------------

def _persist(vid):
    v = VOICES.get(vid)
    if not v:
        return
    try:
        safe = {k: val for k, val in v.items() if not k.startswith("_")}
        with open(os.path.join(VOICE_DIR, vid + ".json"), "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2)
    except Exception as e:
        print("could not persist", vid, "-", e)


def _restore():
    for fn in os.listdir(VOICE_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(VOICE_DIR, fn), encoding="utf-8") as f:
                v = json.load(f)
            if os.path.exists(v.get("ref_path", "")):
                VOICES[fn[:-5]] = v
        except Exception:
            pass
    if VOICES:
        print(f"restored {len(VOICES)} voice(s) from {VOICE_DIR}")


# ------------------------------------------------------------------ 5. app --

VOICES = {}

app = FastAPI(title="soundbite")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


def _decode_audio(raw: bytes, filename: str):
    """mp3 / m4a / wav -> (float32 mono, sr) via librosa."""
    import librosa
    suffix = os.path.splitext(filename)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        return librosa.load(tmp.name, sr=None, mono=True)
    finally:
        os.unlink(tmp.name)


@app.post("/api/clone")
async def clone(name: str = Form(...),
                consent: str = Form(...),
                files: List[UploadFile] = File(...),
                intents: str = Form("[]"),
                prompt: str = Form(""),
                tags: str = Form("[]"),
                source: str = Form("learned from recordings")):
    if consent != "true":
        raise HTTPException(403, "Consent required.")
    if not files:
        raise HTTPException(400, "No recording uploaded.")

    _need_asr()          # cloning needs the transcript, not the TTS model

    try:
        intent_list = json.loads(intents) or []
    except Exception:
        intent_list = []
    try:
        tag_list = json.loads(tags) or []
    except Exception:
        tag_list = []

    # Concatenate the uploads into one reference clip, capped at 30s. Cloning
    # quality peaks around 10-20s of clean speech; past that it just costs time.
    REF_SECONDS = 30.0
    segments, target_sr, used = [], None, 0
    for f in files:
        raw = await f.read()
        try:
            data, sr = _decode_audio(raw, f.filename)
        except Exception as e:
            raise HTTPException(400, f"Could not decode {f.filename}: {e}")
        if target_sr is None:
            target_sr = sr
        elif sr != target_sr:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        segments.append(data)
        used += 1
        if sum(len(s) for s in segments) / target_sr >= REF_SECONDS:
            break

    gap = np.zeros(int(0.25 * target_sr), dtype=np.float32)
    ref = np.concatenate(
        [x for s in segments for x in (s.astype(np.float32), gap)][:-1]
    )[: int(REF_SECONDS * target_sr)]

    if ref.size < target_sr * 0.5:
        raise HTTPException(400, "Recording is under half a second of audio — "
                                 "too short to clone from.")

    vid = "qv_" + uuid.uuid4().hex[:8]
    ref_path = os.path.join(VOICE_DIR, vid + ".wav")
    sf.write(ref_path, ref, target_sr)

    # Whisper transcript doubles as the TTS reference text AND as the raw
    # material for the style profile. word_timestamps is what makes the pause
    # and pacing measurements possible — but it's the fragile option, so a
    # failure there falls back to a plain transcribe rather than costing us the
    # transcript altogether (clone quality depends on ref_text).
    asr_result, ref_text = {}, ""
    try:
        asr_result = _transcribe(ref_path)
        ref_text = (asr_result.get("text") or "").strip()
    except Exception as e:
        print("transcription failed:", e)
    if not ref_text:
        ref_text = "Reference audio."

    VOICES[vid] = {
        "name": name,
        "ref_path": ref_path,
        "ref_text": ref_text,
        "intents": [i for i in intent_list if i and i.strip()],
        "prompt": prompt.strip(),
        "tags": tag_list,
        "source": source,
        "profile": None,
        "profile_status": "queued",
        "_asr_result": asr_result,
    }
    _persist(vid)

    # Profiling takes a while (pyin plus a high-effort model call) and nothing
    # needs it until the user hits play, so it runs behind the response.
    threading.Thread(target=_extract_profile, args=(vid,), daemon=True).start()

    return {"voice_id": vid, "files_used": used, "files_received": len(files),
            "profile_status": "queued"}


@app.post("/api/ocr")
async def ocr(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No image uploaded.")
    try:
        _need_ocr()
    except Exception as e:
        # An OCR install problem shouldn't read like a bug in the upload.
        raise HTTPException(503, f"OCR unavailable on this server: {e}")

    out, n_lines = [], 0
    for f in files:
        raw = await f.read()
        try:
            img = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(400, f"Unreadable image {f.filename}: {e}")
        img = ImageOps.exif_transpose(img).convert("RGB")   # phone photos rotate
        if max(img.size) < 1000:                            # small crops lose type
            s = 1000 / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        try:
            text, n = ocr_image_to_text(img)
        except Exception as e:
            raise HTTPException(500, f"OCR error on {f.filename}: {e}")
        out.append(text)
        n_lines += n

    return {"text": "\n\n".join(t for t in out if t.strip()), "lines": n_lines}


@app.post("/api/chunks")
async def chunks(text: str = Form(...)):
    out = _split_chunks(text)
    if not out and (text or "").strip():
        out = [text.strip()[:320]]      # never hand the player an empty queue
    return {"chunks": out}


# --- adlib: rewrite the text in the speaker's style before it gets read ------

REWRITE_SYSTEM = """You rewrite text so it sounds like one specific person \
reading it aloud in their own words.

You will be given samples of how that person actually speaks. Match their \
rhythm, sentence length, vocabulary, and habits of phrasing.

Hard rules:
- Preserve every fact, name, number, and claim. Add nothing that was not in \
the original. Remove nothing substantive.
- Output ONLY the rewritten text. No preamble, no notes, no quotes around it.
- The output is fed straight to a speech synthesizer, so write plain spoken \
prose: no markdown, no headings, no bullet characters, no emoji, no \
parentheticals, no stage directions.
- Keep the length within roughly 20% of the original."""

MAX_REWRITE_CHARS = 40_000
BATCH_CHARS = 6_000


def _batch(text: str, limit: int = BATCH_CHARS) -> List[str]:
    out, cur = [], ""
    for para in re.split(r'\n\s*\n', text):
        if cur and len(cur) + len(para) + 2 > limit:
            out.append(cur)
            cur = para
        else:
            cur = (cur + "\n\n" + para) if cur else para
    if cur:
        out.append(cur)
    return out


@app.post("/api/rewrite")
async def rewrite(text: str = Form(...), voice_id: str = Form(...)):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set — adlib is disabled.")
    v = VOICES.get(voice_id)
    if not v:
        raise HTTPException(404, "Unknown voice_id (session may have reset).")
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "Empty text.")

    truncated = len(text) > MAX_REWRITE_CHARS
    if truncated:
        text = text[:MAX_REWRITE_CHARS]

    # If the profile is still extracting, wait briefly — a rewrite without the
    # style card is markedly worse. But the page shows no progress during this,
    # so a long stall just reads as a hang: give up after 45s and use the
    # raw-transcript fallback. The profile will be ready for the next play.
    import asyncio
    waited = 0
    while v.get("profile_status") in ("queued", "working") and waited < 45:
        await asyncio.sleep(1)
        waited += 1

    from anthropic import AsyncAnthropic
    client = AsyncAnthropic()

    # The style card is identical across every batch of one document, so it is
    # a cacheable prefix; only the chunk in the user turn varies.
    system = [
        {"type": "text", "text": REWRITE_SYSTEM},
        {"type": "text", "text": _render_profile(v),
         "cache_control": {"type": "ephemeral"}},
    ]

    pieces = []
    for part in _batch(text):
        # Thinking stays on at low effort. Disabling it on Opus 5 can leak
        # <thinking> tags into the visible response — which the TTS would
        # then read aloud. Low effort is fast and has no such failure mode.
        async with client.messages.stream(
            model="claude-opus-5",
            max_tokens=16000,
            output_config={"effort": "low"},
            system=system,
            messages=[{"role": "user", "content":
                       "Rewrite the following in that person's voice:\n\n" + part}],
        ) as stream:
            msg = await stream.get_final_message()

        if msg.stop_reason == "refusal":
            raise HTTPException(422, "Rewrite declined for this content.")
        pieces.append("".join(b.text for b in msg.content if b.type == "text").strip())

    return {"text": "\n\n".join(p for p in pieces if p),
            "truncated": truncated,
            "profile_used": v.get("profile_status") == "ready"}


@app.get("/api/profile/{voice_id}")
def get_profile(voice_id: str):
    v = VOICES.get(voice_id)
    if not v:
        raise HTTPException(404, "Unknown voice_id.")
    return {"status": v.get("profile_status", "none"),
            "profile": v.get("profile"),
            "measured": v.get("measured"),
            "rendered": _render_profile(v)}


@app.post("/api/profile/{voice_id}")
async def put_profile(voice_id: str, profile: str = Form(...)):
    """Overwrite an extracted profile with a hand-edited one."""
    v = VOICES.get(voice_id)
    if not v:
        raise HTTPException(404, "Unknown voice_id.")
    try:
        v["profile"] = json.loads(profile)
    except Exception as e:
        raise HTTPException(400, f"Not valid JSON: {e}")
    v["profile_status"] = "ready"
    _persist(voice_id)
    return {"ok": True}


@app.post("/api/profile/{voice_id}/rebuild")
def rebuild_profile(voice_id: str):
    if voice_id not in VOICES:
        raise HTTPException(404, "Unknown voice_id.")
    threading.Thread(target=_extract_profile, args=(voice_id,), daemon=True).start()
    return {"ok": True, "status": "queued"}


@app.post("/api/synthesize")
async def synthesize(chunk: str = Form(...), voice_id: str = Form(...)):
    v = VOICES.get(voice_id)
    if not v:
        raise HTTPException(404, "Unknown voice_id (session may have reset).")
    if not chunk.strip():
        raise HTTPException(400, "Empty chunk.")
    if not os.path.exists(v["ref_path"]):
        raise HTTPException(410, "Reference audio is gone — re-create this voice.")

    tts = _need_tts()
    wavs, out_sr = tts.generate_voice_clone(
        text=chunk,
        ref_audio=v["ref_path"],
        ref_text=v.get("ref_text", "Reference audio."),
    )
    audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), out_sr, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/health")
def health():
    return {
        "ok": True,
        "voices": len(VOICES),
        "loaded": {"tts": "tts" in _M, "asr": "asr" in _M, "ocr": "ocr" in _M},
        "adlib": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "workdir": BASE_DIR,
    }


_restore()


# ---------------------------------------------------------------- 6. serve --

def serve(load_now: bool = True,
          ngrok_token: Optional[str] = None,
          ngrok_domain: Optional[str] = None,
          tunnel: bool = True):
    if load_now:
        # Preload what every request path needs. OCR is optional — if its
        # install is broken we still want cloning and playback to work, so it
        # warns here rather than taking the server down.
        _need_asr()
        _need_tts()
        try:
            _need_ocr()
        except Exception as e:
            print(f"\n  ! OCR unavailable — image uploads will return 503.\n"
                  f"    Everything else still works. Cause: {e}\n")

    import time, urllib.request, uvicorn
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT,
                                   log_level="warning"),
        daemon=True,
    ).start()

    for _ in range(20):                       # wait for the port to answer
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read()
            break
        except Exception:
            pass
    else:
        raise RuntimeError("local server never came up — check for a traceback above")

    print("routes:", sorted(r.path for r in app.routes if hasattr(r, "path")))
    print(f"local:  http://127.0.0.1:{PORT}/health")

    if not tunnel:
        return None

    token = ngrok_token or os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        print("\nNo ngrok token — no public URL. Pass ngrok_token=... or set "
              "NGROK_AUTHTOKEN.")
        return None

    from pyngrok import ngrok
    ngrok.kill()                              # drop tunnels from a prior run
    ngrok.set_auth_token(token)
    domain = ngrok_domain or os.environ.get("NGROK_DOMAIN")
    t = ngrok.connect(PORT, "http", domain=domain) if domain else ngrok.connect(PORT, "http")

    ok = False
    for attempt in range(12):
        time.sleep(5)
        try:
            req = urllib.request.Request(
                t.public_url + "/health",
                headers={"ngrok-skip-browser-warning": "1"})
            print("public URL live:", urllib.request.urlopen(req, timeout=10).read().decode())
            ok = True
            break
        except Exception as e:
            print(f"  not ready yet ({attempt + 1}/12): {e}")

    print("\n" + "=" * 60)
    print("  API_BASE =", t.public_url if ok else "TUNNEL FAILED — rerun the cell")
    print("=" * 60)
    if not domain:
        print("  Tip: reserve a domain and pass ngrok_domain=... to keep this\n"
              "       URL stable, so index.html never needs editing again.")
    return t.public_url if ok else None


if __name__ == "__main__":
    serve()
