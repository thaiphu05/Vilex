import json
import random
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from src.config import cfg_get, load_config
from src.llm_client import make_client, no_thinking_extra_body

# ================================
# Configuration
# ================================
# Defaults are now Qwen (local, free). Override via CLI. The legacy gpt-4.1-mini
# path is still available with --prompt legacy --model_name gpt-4.1-mini.
MODEL_NAME = "Qwen/Qwen3-14B"
BASE_URL = "http://localhost:8008/v1"
API_KEY = "EMPTY"
PROMPT_KIND = "qwen"  # "qwen" (new, engineered) | "legacy" (original gpt prompt)
TARGET_LANGUAGE = "vi"  # "en" | "vi" (default vi for Vilex)

# Set from --input_root/--output_root, which are required: the old defaults
# named a directory no stage writes, so a bare run globbed nothing and exited 0.
INPUT_ROOT = None
OUTPUT_ROOT = None

client = None  # built in main() from the resolved config

# ================================
# Prompt
# ================================
SYSTEM_PROMPT = """
You are generating a LISTENER backchannel to insert into a conversation.

# Backchannel definition:
- A very short listener vocalization showing attention/understanding
- It does NOT take the floor and does NOT change the topic
- The speaker continues immediately after it

# Insertion rule:
- Output what the listener says exactly at [INSERT] while the speaker keeps talking.
- Do not add any other text, quotes, labels, or explanations.

# Output constraints:
- 1–2 words only (3 only if absolutely necessary)
- lowercase
- no emojis, no ellipses, no exclamation marks
- "huh?" may include a question mark; otherwise avoid punctuation

# Decision rubric (choose the mildest that fits):
- ack/continue: mhm, mm-hmm, uh-huh, yeah, right, okay
- mild surprise/interest (new/unexpected info): oh, wow, really
- confusion (ONLY if unclear/contradictory): huh?

# Default:
- If uncertain, output "mhm".

# Allowed outputs (choose exactly one):
mhm | mm-hmm | uh-huh | yeah | right | okay | oh | wow | really | huh?"""

# New, engineered prompt for Qwen (do NOT edit SYSTEM_PROMPT above). Goal: pick the
# backchannel a real listener would actually say HERE, matched to the speaker's
# content and emotional register, while staying a true backchannel (no floor-take,
# no new topic, no question that demands an answer).
SYSTEM_PROMPT_QWEN = """You insert a LISTENER BACKCHANNEL at the [INSERT] point: the tiny
sound the listener makes while the speaker keeps talking. It does NOT take the floor,
NOT change the topic, and NOT ask anything that needs an answer. The speaker continues
right after it.

Pick the ONE backchannel a real person would say HERE. Match it to what the speaker is
saying and their emotional tone — do not default to the same token every time:
- neutral attention / "I'm following": mhm, mm-hmm, uh-huh, right, okay, yeah
- agreement / affirmation: yeah, right, exactly, totally, for sure
- new or surprising info: oh, oh wow, really, no way
- bad/sad news: oh no, aw, oh man
- mild understanding of a problem: i see, gotcha, ah
- genuine confusion (only if the speaker was unclear/contradictory): huh?, wait

Rules:
- 1-3 words, lowercase. No emojis, no ellipses, no exclamation marks. A question mark
  is allowed only for "huh?" / "wait?".
- Output ONLY the backchannel text — no quotes, labels, or explanation.
- Prefer the mildest option that fits; reserve "oh wow / really / no way / oh no" for
  genuinely notable content, not routine statements.
- It must read as natural spoken English in this exact spot."""

# Vietnamese listener backchannel prompt. Output MUST be a short Vietnamese
# backchannel (ưm, à, ừ, vâng, phải, thật không, ồ...), optionally with OmniVoice tags.
SYSTEM_PROMPT_QWEN_VI = """Bạn chèn một BACKCHANNEL của người nghe vào điểm [INSERT]: tiếng
thì thầm ngắn người nghe phát ra trong khi người nói vẫn tiếp tục nói. Nó KHÔNG giành
lượt, KHÔNG đổi chủ đề, và KHÔNG đặt câu hỏi cần trả lời. Người nói sẽ tiếp tục ngay sau đó.

Chọn ĐÚNG MỘT backchannel tiếng Việt một người thật sẽ nói Ở ĐÂY, khớp với nội dung và
sắc thái cảm xúc của người nói — đừng luôn dùng cùng một từ. BẠN CÓ THỂ DÙNG CÁC TAG SAU ĐỂ TĂNG TÍNH CHÂN THỰC:
- chú ý / theo dõi: ưm, à, ừ, ừ [confirmation-en]
- cười nhẹ (khi đối phương nói đùa): [laughter]
- thông tin mới / bất ngờ nhẹ: ồ, ồ [surprise-oh], à [surprise-ah]
- đồng cảm / thấu hiểu (tin buồn, thở dài): [sigh]
- đồng ý / xác nhận: phải, đúng, ra vậy
- thực sự bối rối: hả?, khoan đã?
- tin buồn / xấu: ôi không, trời ơi


Quy tắc:
- 1-3 từ, viết thường. Không emoji, không dấu ba chấm, không dấu chấm cảm.
- Chỉ xuất ra đúng text backchannel — không dấu ngoặc, không nhãn, không giải thích.
- Chọn phương án nhẹ nhàng nhất phù hợp cho nội dung thực sự đáng chú ý, không phải câu nói thông thường."""

FALLBACK_BC_VI = ["ừ", "à", "vâng", "phải", "ồ"]
FALLBACK_BC_EN = ["mhm", "yeah", "right", "okay", "oh"]

# Configurable at runtime from config.yaml (stage4b_backchannel.*).
BC_MAX_TOKENS = 32
BC_TEMPERATURE = 0.7
BC_MAX_RETRIES = 2
VALID_MAX_WORDS = 3
MAX_FILE_ATTEMPTS = 3


def _is_valid_backchannel(text: str) -> bool:
    """Reject empty/garbage/explanation-like LLM outputs."""
    if not text:
        return False
    words = text.split()
    if len(words) < 1 or len(words) > VALID_MAX_WORDS:
        return False
    lowered = text.lower()
    # explanations / labels instead of the actual backchannel
    if lowered.startswith(("backchannel", "listener", "speaker", "the ", "this ")):
        return False
    if "..." in text or "!" in text or "—" in text:
        return False
    return True


def _build_messages(context: str):
    if TARGET_LANGUAGE == "vi":
        sys_p = SYSTEM_PROMPT_QWEN_VI
    else:
        sys_p = SYSTEM_PROMPT_QWEN if PROMPT_KIND == "qwen" else SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_p},
        {
            "role": "user",
            "content": f"# Conversation context:\n{context}\n\nGenerate the most natural listener backchannel for the [INSERT] point above.",
        },
    ]


def generate_backchannel(context: str, max_retries: int = None) -> str:
    if max_retries is None:
        max_retries = BC_MAX_RETRIES
    fallback_pool = FALLBACK_BC_VI if TARGET_LANGUAGE == "vi" else FALLBACK_BC_EN
    fallback = random.choice(fallback_pool)
    kwargs = dict(
        model=MODEL_NAME,
        messages=_build_messages(context),
        temperature=BC_TEMPERATURE,
        max_tokens=BC_MAX_TOKENS,
    )
    # Disable thinking for reasoning models. For Gemini we emit a marker that
    # gemini_client.py consumes as types.ThinkingConfig(thinking_budget=0).
    kwargs["extra_body"] = no_thinking_extra_body(client)
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception:
            if attempt == max_retries:
                return fallback
            continue
        out = (response.choices[0].message.content or "").strip()
        out = out.split("</think>")[-1].strip().strip('"').strip().lower()
        out = " ".join(out.split())
        if _is_valid_backchannel(out):
            return out
    return fallback


# ================================
# Core logic
# ================================
def process_dialogue(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    turns = dialogue.get("history", [])

    for t_idx, turn in enumerate(turns):
        if "history" not in turn:
            continue

        # Align to the raw content before TOKEN_BC/TOKEN_FT were inserted.
        raw_text = (
            turn["history"][0].get("full_content", turn["content"])
            if turn["history"]
            else turn["content"]
        )
        raw_text = raw_text.replace("[BACKCHANNEL]", "").replace("[TAKE_FLOOR]", "").strip()
        words = raw_text.split()
        for item in turn["history"]:
            if item.get("decision") != "backchannel":
                continue

            idx = item["word_index"]
            if idx >= len(words):
                continue

            # Current turn prefix up to the backchannel point
            current_prefix = " ".join(words[: idx + 1])
            current_postfix = " ".join(words[idx + 1 :])

            # Full context passed to LLM
            context_lines = []
            if t_idx > 0:
                context_lines.append("Conversation:")
                context_lines.append("[PREVIOUS]")
                context_lines.append(f"{turns[t_idx-1].get('content','').strip()}")

            context_lines.append("[SPEAKER CURRENT]")
            context_lines.append(f"{current_prefix} [INSERT] {current_postfix}")
            context = "\n".join(context_lines)

            # 3. Generate and Update
            bc = generate_backchannel(context)
            item["content"] = bc

    return dialogue


# ================================
# File traversal
# ================================
def collect_files(dataset=None, split=None):
    ds_part = f"text_dialogue_{dataset}" if dataset else "text_dialogue_*"
    split_part = split or "*"
    return list(INPUT_ROOT.glob(f"{ds_part}/{split_part}/*.json"))


# ================================
# Entry point
# ================================
def configure(cfg):
    """Resolve the module-level knobs from config.yaml."""
    global MODEL_NAME, BASE_URL, API_KEY, PROMPT_KIND, TARGET_LANGUAGE
    global INPUT_ROOT, OUTPUT_ROOT, client
    global FALLBACK_BC_VI, FALLBACK_BC_EN
    global BC_MAX_TOKENS, BC_TEMPERATURE, BC_MAX_RETRIES, VALID_MAX_WORDS, MAX_FILE_ATTEMPTS

    llm = cfg_get(cfg, "llm", {})
    s4b = cfg_get(cfg, "stage4b_backchannel", {})
    paths = cfg_get(cfg, "paths", {})

    MODEL_NAME = llm.get("bc_model", MODEL_NAME)
    BASE_URL = llm.get("bc_base_url", BASE_URL)
    API_KEY = llm.get("api_key", API_KEY)
    PROMPT_KIND = s4b.get("prompt_kind", PROMPT_KIND)
    TARGET_LANGUAGE = cfg_get(cfg, "run.target_language", TARGET_LANGUAGE)

    BC_MAX_TOKENS = s4b.get("max_tokens", BC_MAX_TOKENS)
    BC_TEMPERATURE = s4b.get("temperature", BC_TEMPERATURE)
    BC_MAX_RETRIES = s4b.get("max_retries", BC_MAX_RETRIES)
    VALID_MAX_WORDS = s4b.get("valid_max_words", VALID_MAX_WORDS)
    MAX_FILE_ATTEMPTS = s4b.get("max_file_attempts", MAX_FILE_ATTEMPTS)
    fallback = s4b.get("fallback", {})
    if isinstance(fallback, dict):
        if fallback.get("vi"):
            FALLBACK_BC_VI = fallback["vi"]
        if fallback.get("en"):
            FALLBACK_BC_EN = fallback["en"]

    INPUT_ROOT = Path(paths.get("synthesis_root", "data/vi_tt"))
    OUTPUT_ROOT = Path(paths.get("bc_root", "data/vi_tt_bc"))
    client = make_client(MODEL_NAME, API_KEY, BASE_URL)


def main(config_path=None):
    cfg = load_config(config_path)
    configure(cfg)

    datasets = cfg_get(cfg, "run.datasets", [])
    splits = cfg_get(cfg, "run.splits", ["train"])
    total = 0

    for dataset in datasets:
        for split in splits:
            files = collect_files(dataset, split)
            if not files:
                continue
            total += len(files)
            for in_path in tqdm(files, desc=f"Adding backchannels [{dataset}/{split}]"):
                rel_path = in_path.relative_to(INPUT_ROOT)
                out_path = OUTPUT_ROOT / rel_path
                if out_path.exists():
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Defensive: a truncated/empty input (e.g. a write interrupted by
                # a disk-quota error) must not abort the whole stage; skip it.
                try:
                    with open(in_path, "r", encoding="utf-8") as f:
                        dialogue = json.load(f)
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"\n[run_add_bc] SKIP unreadable input {in_path.name}: {exc}")
                    continue

                # A single malformed/empty LLM reply must not abort the whole
                # stage. Retry a few times, then skip the dialogue as lost.
                processed = None
                for attempt in range(MAX_FILE_ATTEMPTS):
                    try:
                        processed = process_dialogue(dialogue)
                        break
                    except Exception as exc:
                        print(
                            f"\n[run_add_bc] WARN {in_path.name} attempt "
                            f"{attempt + 1}/{MAX_FILE_ATTEMPTS} failed: {exc}"
                        )
                if processed is None:
                    print(
                        f"[run_add_bc] SKIP {in_path.name} after {MAX_FILE_ATTEMPTS} "
                        f"failed attempts (no bc added)"
                    )
                    continue

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(processed, f, indent=2, ensure_ascii=False)

    if total == 0:
        raise SystemExit(
            f"No dialogues under {INPUT_ROOT}. The config's paths.synthesis_root must "
            f"contain text_dialogue_<dataset>/<split>/*.json, i.e. the output of "
            f"`python -m src.synthesis.run`."
        )


# ================================
# CLI
# ================================
if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="Add backchannels (Stage 4b).")
    _ap.add_argument("--config", default=None, help="Path to config.yaml")
    main(_ap.parse_args().config)
# ================================
