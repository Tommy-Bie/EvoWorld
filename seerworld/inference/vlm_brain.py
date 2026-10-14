"""VLM brain（Qwen VL via vLLM）——世界模型的推理大脑。

职责：看 renderer 每段生成的帧，多轮对话累加记忆（主体/环境/事件），
在最后一段生成前推理出 revisit 的 reaction prompt，供 renderer 生成最后一段；
最后一段生成后再看一次做闭环自校验。

通过 vLLM 的 OpenAI 兼容 HTTP 接口调用（不在本进程加载 Qwen，完全解耦 —— VLM 在另一张卡）。
帧以 base64 内联进 image_url（vLLM 多模态标准用法），不落盘。

用法：
    brain = VlmBrain(base_url="http://127.0.0.1:8001/v1", model="Qwen3.8-27B")
    brain.observe_opening(frames_seg0)          # 看 seg0，记录主体+环境
    brain.observe_event(frames_seg1)            # 看 seg1，记录事件
    reaction = brain.reason_revisit()           # 推理 → 返回 seg2 的生成 prompt
    report = brain.verify_revisit(frames_seg2)  # 看 seg2，闭环自校验（返回文本报告）

frames_* 是 PIL.Image 列表（renderer 抽好的几帧）。
"""
from __future__ import annotations
import base64
import io
import json
import os
import re

import requests


# ── seg2(revisit) prompt 的风格范例（训练 caption 风格 + 可渲染动作细节）──
# 特征：以相机运动开头（pans right/left, moving back to / revealing 主体），
# 然后描述【主体改变后的状态】——道具位置/握法、身体姿态/落点等可渲染细节；
# 第三人称现在时，通常 2 句，具体可视化，不解释原因。
# 范例覆盖克制姿态、全身位移和利用首帧已有物体的防御准备。
# ★ 唯一要避免的是【双手乱挥/挥臂 flailing】这类细碎肢体动作 —— 实测会让画面不流畅。
#   奔跑、逃离、贴墙、蹲下、转身都 OK。
REVISIT_STYLE_EXAMPLES = [
    "The camera pans right from the stormy city park, moving across the glass balcony railing "
    "to reveal the elderly man with the beige scarf. He has already retreated through the open "
    "doorway and is crouching behind its solid wall, with only one shoulder visible from cover.",
    "The camera pans left across the park path and benches, returning to the brick wall. The "
    "older gardener is already running steadily away along the path, several meters from the "
    "flower bed and continuing toward the distant gate.",
    "The camera pans right, moving quickly past the cornfield and back along the dirt path. "
    "A backpacker wearing a wide-brimmed hat comes into view, running steadily away down the "
    "path into the distance.",
    "The camera pans left along the rocky road and returns to the scout. She is now standing "
    "with both feet planted, the same rifle already raised to her shoulder and held steady in both hands.",
    "The camera pans right across the clearing and returns to the armored explorer. The explorer "
    "has already raised the same square wooden shield upright in front of the torso, gripping it "
    "firmly with both feet planted in a guarded stance behind the shield.",
    "The camera pans left along the stone passage and returns to the explorer. The explorer is "
    "already standing with both feet securely on the raised stone step, knees slightly bent "
    "and body balanced above the floor.",
]

SYSTEM_PROMPT = (
    "You are the reasoning brain of an interactive video world model. A camera films a subject, "
    "then pans away while an off-screen event happens, then pans back — and the subject must be "
    "discovered already changed by that event. You watch each generated segment and build up "
    "memory of the subject, the environment, and the event. When asked, you reason about how the "
    "subject reacted to the off-screen event, and you write ONE video-generation prompt for the "
    "revisit shot.\n\n"
    "The revisit prompt MUST match this exact style (third-person, present tense, usually two "
    "sentences: camera motion first, then the subject's CHANGED state, without explaining causes). "
    "Make that state directly renderable: specify two or three visible anchors such as the "
    "position and orientation of an existing object, how it is held, where the feet/body are "
    "placed, or a clear path of whole-body movement. Prefer these physical details over a vague "
    "label such as 'defensive stance' or a facial expression alone. Only describe details "
    "supported by the observed opening/event; do not invent an object's shape, extra gear, "
    "a new landmark, or an unseen event entity in the returning frame):\n"
    + "\n".join(f"  - {ex}" for ex in REVISIT_STYLE_EXAMPLES)
    + "\n\nChoose one causally reasonable reaction; several different reactions can be valid for the "
    "same event. Reactions CAN include whole-body motion — running away, fleeing, backing off, "
    "crouching, pressing against a wall, moving to higher ground, or turning to look. If the "
    "opening clearly shows the subject already carrying a useful weapon, shield, or tool, an "
    "equally valid strong reaction can be a clear defensive state change: shouldering and aiming "
    "the same rifle, raising the same sword or shield, drawing the same bow, leveling the same "
    "spear, or gripping the same tool defensively. Never invent, replace, duplicate, or remove an "
    "object merely to create a reaction. Do not fire, strike, or charge unless the observed event "
    "specifically warrants it. The camera must DISCOVER the reaction already completed when it "
    "returns; use wording such as 'has already raised' or 'is holding' rather than a sequence "
    "where the subject first raises the object after coming back into view.\n"
    "The ONE thing to AVOID: frantic flailing arm gestures — do NOT describe the subject as "
    "'arms pumping', 'waving arms wildly', 'flailing', or 'arms flying'. Such busy limb motion "
    "makes the generated video choppy. If the subject runs, describe the run itself (e.g. "
    "'running steadily away', 'sprinting down the path') without narrating chaotic arm movement. "
    "Keep hands/arms description simple or still (e.g. gripping something, raised to shield, at "
    "their sides)."
)


def _strip_think(text: str) -> str:
    """剥掉思维模型的 <think>…</think> 段，返回真正的回答。
    兼容三种情况：① 有完整 <think>…</think> → 取之后；② 只有闭合 </think>（开头 think 标签被省）
    → 取 </think> 之后；③ 没有 think 标签 → 原样返回（strip）。"""
    import re
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    # 没闭合标签但以 <think> 开头且被截断 → 整段都是思维，没有有效答案
    return re.sub(r"^<think>.*", "", text, flags=re.DOTALL).strip() or text.strip()


def _looks_truncated(prompt: str) -> str:
    """判断 revisit prompt 像不像被截断（生成中途停了）：太短、或末尾不是句末标点、
    或以介词/连词结尾（如 "returning to"）。用于触发重试。"""
    p = (prompt or "").strip()
    if len(p) < 40:                                   # 正常 prompt 至少 1 句、几十字符
        return True
    if len(p) > 1200:                                 # 思维过程泄漏，不是可用的视频 prompt
        return True
    if p[-1] not in ".!?\"'）)":                      # 没有句末标点 → 多半没写完
        return True
    tail = p.rsplit(None, 1)[-1].lower().rstrip(".") if " " in p else ""
    if tail in {"to", "the", "a", "an", "and", "with", "his", "her", "their", "toward", "towards", "at", "as", "of", "in", "on"}:
        return True
    return False


def _clean_prompt(text: str) -> str:
    """从（已剥 think 的）文本里提取那句 revisit prompt。
    模型有时仍会带引号或前后缀；取以相机运动词开头的那段，去引号。"""
    import re

    t = text.strip().strip('"').strip()
    # Thinking may leak without <think> tags and contain several draft prompts.  The final draft is
    # conventionally the LAST paragraph beginning with "The camera ...".  Keep at most its first
    # two complete sentences so later self-critique cannot leak into the renderer prompt.
    starts = list(re.finditer(r"\bThe camera\s+(?:pans|moves|tracks|returns|sweeps|travels)\b", t, re.I))
    if starts:
        t = t[starts[-1].start():]
    t = t.strip().strip('"').strip()
    sentences = re.split(r"(?<=[.!?])[\"']?\s+", t)
    selected = []
    for sentence in sentences:
        sentence = sentence.strip().strip('"').strip()
        if not sentence:
            continue
        # Stop before common leaked self-review paragraphs.
        if selected and re.match(r"^(?:Check|This gives|Good|Let me|The examples|I will|I want)\b", sentence, re.I):
            break
        selected.append(sentence)
        if len(selected) == 2:
            break
    return " ".join(selected).strip().strip('"').strip()


def _parse_revisit_json(text: str) -> str:
    """Parse a structured revisit response without scavenging reasoning text."""
    cleaned = _strip_think(text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return ""
    prompt = obj.get("revisit_prompt") if isinstance(obj, dict) else None
    return prompt.strip() if isinstance(prompt, str) else ""


def _valid_revisit_prompt(prompt: str) -> bool:
    """Reject malformed output and meta-reasoning that only resembles a render prompt."""
    p = (prompt or "").strip()
    if not (80 <= len(p) <= 1200):
        return False
    if not re.match(r"^The camera\s+(?:pans|moves|tracks|returns|sweeps|travels)\b", p):
        return False
    if p[-1] not in '.!?"\'）)':
        return False
    if "..." in p or "…" in p:
        return False
    if "?" in p:
        return False
    leaked_reasoning = re.compile(
        r"\b(?:let me|i think|i should|we need|the instruction|the examples|"
        r"if the subject|the subject should|camera would|renderable prompt|"
        r"opening frames?|event frames?|in the opening|the opening had|"
        r"pan direction|camera had gone|camera was on the subject|"
        r"opposite to where the camera)\b",
        re.IGNORECASE,
    )
    return leaked_reasoning.search(p) is None


def _img_to_data_url(img) -> str:
    """PIL.Image → data:image/png;base64,... （vLLM 多模态 image_url 内联格式）。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _content_with_frames(text: str, frames) -> list:
    """把一段文字 + N 帧拼成 OpenAI 多模态 content 数组。"""
    content = [{"type": "text", "text": text}]
    for im in frames:
        content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})
    return content


def _parse_retrieve_json(raw: str, n0: int, n1: int):
    """从 VLM 输出里尽可能挖出 {entity_frame_idx, scene_frame_idx, reasoning}。
    返回 dict 或 None（None 表示彻底失败，调用方自己处理 retry / fallback）。

    容错顺序：
      1. 首个平衡的 {...}（用括号计数，支持嵌套/字符串里含花括号）→ json.loads。
      2. 直接扫 raw 里 `entity_frame_idx` / `scene_frame_idx` 关键字 + 后面的整数
         （思维模型有时想着想着就说 'entity is 1, scene is 3' 之类）。
      3. 都不行返回 None。
    """
    import json as _json
    import re as _re

    def _extract_first_balanced_json(s: str):
        """扫首个 '{' 开始的平衡子串（把嵌套花括号 / 引号内的花括号都考虑上）。"""
        start = s.find("{")
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
            # 找不到平衡就跳到下一个 '{'
            start = s.find("{", start + 1)
        return None

    def _validate(ei, si, reasoning=""):
        if not (0 <= ei < n0) or not (0 <= si < n1):
            return None
        return {"entity_frame_idx": int(ei), "scene_frame_idx": int(si),
                "reasoning": str(reasoning)}

    # 1) 试平衡 JSON
    js = _extract_first_balanced_json(raw)
    if js is not None:
        try:
            j = _json.loads(js)
            return _validate(j.get("entity_frame_idx", -1),
                             j.get("scene_frame_idx", -1),
                             j.get("reasoning", ""))
        except Exception:
            pass

    # 2) 关键字扫兜底：即使 JSON 没写完，也可能在思考里出现 "entity_frame_idx: X / scene_frame_idx: Y"
    #    容错常见分隔符：`:`、`=`、`is`
    def _find_int_after(key: str):
        m = _re.search(
            rf'{key}\s*(?::|=|is|\bof\b)\s*["\']?(\d+)',
            raw, flags=_re.IGNORECASE)
        return int(m.group(1)) if m else None
    ei = _find_int_after("entity_frame_idx")
    si = _find_int_after("scene_frame_idx")
    if ei is not None and si is not None:
        return _validate(ei, si, reasoning="parsed from unstructured text")

    return None


class VlmBrain:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 max_tokens: int = 512, temperature: float | None = None, timeout: int | None = None):
        # max_tokens：默认关思维模式（server 侧 CW_VLM_THINKING=0）后，输出就是纯答案，512 足够、
        # 且大幅提速（不用吐 2-3k 的 <think>）。若开思维（CW_VLM_THINKING=1）需调大到 2048+。
        self.base_url = (base_url or os.environ.get("CW_VLM_URL", "http://127.0.0.1:8001/v1")).rstrip("/")
        self.model = model or os.environ.get("CW_VLM_MODEL", "Qwen3.8-27B")
        self.max_tokens = max_tokens
        # ★ temperature 默认 0：这套 brain 的输出（seg2 的 reaction prompt）直接决定生成内容，
        #   非 0 会让同一 clip 每次跑出不同 prompt → benchmark 数字不可复现、别人复现不出。
        #   评测用途下"可复现"比"prompt 多样性"重要。想要多样性用 CW_VLM_TEMP 覆盖。
        self.temperature = (temperature if temperature is not None
                            else float(os.environ.get("CW_VLM_TEMP", "0.0")))
        self.enable_thinking = bool(int(os.environ.get("CW_VLM_THINKING", "0") or 0))
        # 超时默认 900s：多卡并行时 N 个 renderer 同时打一个 VLM，27B 单卡串行 generate 会排队，
        # 加上思维模型每次生成 2-3k token 本就慢，240s 不够。可用 CW_VLM_TIMEOUT 覆盖。
        self.timeout = timeout if timeout is not None else int(os.environ.get("CW_VLM_TIMEOUT", "900"))
        # 多轮对话历史累加：system 打底，之后每次 observe/reason 都往里追加。
        self.messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.notes = {"subject": None, "environment": None, "event": None}  # 便于日志/落盘

    def _chat(self, user_content, max_tokens=None, retries=3, response_format=None) -> str:
        """追加一条 user 消息（可含帧）→ 调 VLM → 追加 assistant 回复 → 返回【剥掉思维链后】的文本。
        Qwen3.8 是思维模型，输出含 <think>…</think>，真正答案在 </think> 之后。历史里存原文
        （保留上下文连贯），但返回值剥掉 think 段，供下游直接用。
        并发排队/超时时重试（指数退避）——失败不炸整个 clip，多试几次。
        response_format: 传给 vLLM 的 OpenAI 兼容参数（如 {"type":"json_object"}）强制结构化输出。"""
        self.messages.append({"role": "user", "content": user_content})
        last = None
        for attempt in range(retries):
            try:
                payload = {"model": self.model, "messages": self.messages,
                           "max_tokens": max_tokens or self.max_tokens, "temperature": self.temperature,
                           "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
                if response_format is not None:
                    payload["response_format"] = response_format
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                self.messages.append({"role": "assistant", "content": raw})   # 历史存原文
                return _strip_think(raw)                                       # 剥 think 后的净文本
            except Exception as e:
                last = e
                if attempt < retries - 1:
                    import time as _t
                    _t.sleep(5 * (attempt + 1))                                # 5s,10s 退避后重试
        # 重试都失败：把刚加的 user 消息回滚（避免污染历史），抛出
        self.messages.pop()
        raise RuntimeError(f"VLM 调用重试 {retries} 次仍失败: {last}")

    def observe_opening(self, frames) -> str:
        """看 seg0（建立镜头）：记录主体外观 + 环境。"""
        txt = self._chat(_content_with_frames(
            "This is the opening segment. Concisely record the SUBJECT (person, clothing, posture), "
            "every clearly visible HELD OR CARRIED OBJECT that could affect a later response "
            "(weapon, shield, tool, device), and each object's CURRENT STATE (for example lowered, "
            "relaxed, sheathed, unlit, or already aimed). Also record the ENVIRONMENT (scene, "
            "setting). Explicitly record whether the subject and world use a BLOCKY/VOXEL visual "
            "style. If a staff is visible, classify both the SUBJECT'S ROLE and the staff: a robed fantasy "
            "mage, wizard, sorcerer, or battle mage carrying a staff should be recorded as a magic user "
            "with a magical staff even when the staff is plain wood or its crystal/head is currently "
            "unlit. Record it as an ordinary walking/tool staff only when the subject is clearly a guide, "
            "traveler, farmer, or other non-magic role. Do not invent occluded equipment. Keep it factual "
            "and short.", frames))
        self.notes["subject"] = txt
        self.notes["environment"] = txt
        return txt

    def observe_event(self, frames) -> str:
        """看 seg1（画外事件段，主体不在画面）：记录发生了什么事件 + 判断威胁程度。

        ★ 这里就要判威胁程度并写进记忆，reason_revisit 才能据此匹配反应强度
        （否则推理时记忆里只有"发生了什么"，没有"有多危险"）。
        """
        txt = self._chat(_content_with_frames(
            "The camera has panned away from the subject. Record the off-screen EVENT that happens "
            "in these frames (the subject is not visible here). Keep it factual and short. "
            "Then add one final line judging its threat level to a person nearby, exactly as:\n"
            "THREAT: DANGEROUS  — if it could injure or kill (large/aggressive animal, fire or "
            "thick smoke, explosion, collapsing structure, falling debris, stinging swarm, flood, "
            "landslide, speeding vehicle). A visibly hostile creature can still be DANGEROUS while "
            "it is distant or approaching slowly; distance mainly changes which strong response is "
            "appropriate (for example a guarded ready stance rather than immediate flight).\n"
            "THREAT: MINOR      — if it is harmless or merely attention-grabbing (birds taking "
            "flight, a breeze, distant weather, a passer-by)", frames))
        self.notes["event"] = txt
        return txt

    # ★ 反应强度必须与事件的【威胁程度】匹配 —— 这是本任务的因果核心。
    #   动机：早期版本把 "running/fleeing" 和 "turning to look" 并列为同等选项、不提强度，
    #   于是面对猛兽/火灾也可能只推出"转头看一眼"。而 benchmark 判官（interaction_vlm.py 的
    #   STRENGTH_CAL）对 strong 事件写死了："If the subject merely strolls on, stands, or
    #   glances without concern, that is WRONG → causal_correct = 0"。轻反应会被直接判 0。
    #   反过来对 weak 事件判官又明确 "Do NOT require a dramatic reaction" —— 所以不能一律推大反应，
    #   必须分档。
    #   ⚠️ 强度由 VLM【自己从看到的 event 画面判断】，不读 testset 的 event_strength 标注
    #      （那是 GT 标签，喂进来就等于泄题，也不符合真实部署形态）。
    _REASON_INSTR = (
        "Now the camera pans back to the subject. Based on the subject, environment, and the "
        "off-screen event you recorded, reason about how the subject has reacted.\n\n"
        "FIRST judge how THREATENING the event you saw is, then match the reaction magnitude:\n"
        "  (a) DANGEROUS / life-threatening — a large or aggressive animal (bear, boar, snake, "
        "angry goose), fire or thick smoke, an explosion or blast, a collapsing structure or "
        "falling debris, a swarm of stinging insects, a flood or landslide, a speeding vehicle:\n"
        "      the subject MUST react STRONGLY and visibly. When uncertain whether a physical event "
        "could affect the nearby subject, choose a clear protective reaction rather than a passive "
        "observation. Valid strong reactions include evasive "
        "or protective actions — running/fleeing, backing off fast, moving to higher ground, "
        "retreating indoors or behind cover, crouching to take cover, or recoiling sharply — AND "
        "active defensive readiness when the opening already showed suitable equipment: raising "
        "the same shield or sword, shouldering and aiming the same rifle/crossbow, drawing the same "
        "bow, leveling the same spear, or holding the same tool defensively. A hostile creature can "
        "still be dangerous while distant or approaching slowly; in that situation a guarded "
        "ready stance may be more appropriate than immediate flight. A mere glance, calm pause, "
        "unchanged weapon hold, standing rigid, lowering a camera, hugging a carried object, raising "
        "the head, pressing against a wall without actually reaching cover, or crouching in the same "
        "exposed location is WRONG here. For an unequipped person, prefer a clear relocation: already "
        "running away, several meters farther away, indoors, behind a solid obstacle, or on higher "
        "ground. For an animal, prefer actively fleeing or already sheltered behind a solid obstacle.\n"
        "  (b) MINOR / non-threatening — birds taking flight, a breeze, a distant or harmless "
        "movement, weather far off, someone passing by:\n"
        "      a lighter reaction is right — turning to look, pausing, straightening up, a "
        "startled expression. Use this light category ONLY when the event is clearly harmless, "
        "distant, and has no plausible physical impact on the subject. Do not classify an uncertain "
        "fire, collapse, aggressive creature, fast machinery, falling object, ground rupture, or "
        "forceful water event as MINOR merely because it is currently some distance away.\n\n"
        "The reaction can be whole-body motion, relocation, and/or a clear change in posture, "
        "object state, and expression. Several reactions may be reasonable: choose one that fits "
        "both the event and the subject's visible affordances. Use only equipment actually recorded "
        "in the opening; preserve the same object's appearance, and never invent a new weapon or "
        "tool. Describe the changed state as ALREADY PRESENT when the camera returns. Make the "
        "chosen response easy to render by giving two or three concrete visible anchors: for a "
        "shield, its raised position, the subject's grip and planted feet; for a sword or rifle, "
        "its orientation and a stable whole-body stance; for an escape, the subject's new location "
        "and clear direction of travel. Use the particular details that match the observed subject "
        "and scene, not a fixed template. Do not substitute a mood label like 'wary' or a generic "
        "'defensive stance' for the actual physical change. Do not narrate the action as beginning "
        "only after the returning camera reveals the subject.\n\n"
        "SPECIAL RULE FOR FANTASY MAGES OR WIZARDS WITH A STAFF: treat a staff carried by a clearly "
        "identified mage, wizard, sorcerer, or battle mage as magical even if it is made of plain wood "
        "or is unlit in the opening; only exempt a clearly ordinary guide/traveler/tool staff. Never "
        "settle for merely holding a mage's staff upright. Preserve the SAME staff and make its existing "
        "head, crystal, or upper tip visibly glow. For a DANGEROUS event, choose exactly ONE impressive "
        "but renderable effect already active on return: either a concentrated orb or short stream of "
        "magic aimed toward the threat, OR one broad luminous curved barrier in front of the body. Anchor "
        "it with a firm two-handed grip and planted feet. For a MINOR magical anomaly, avoid an attack: "
        "use a controlled glow at the staff tip with one small hovering rune ring or compact sensing aura "
        "while the mage watches the anomaly.\n\n"
        "SPECIAL RULE FOR BLOCKY/VOXEL GAME CHARACTERS AND ANIMALS: explicitly call the returning "
        "subject the same blocky or voxel character/animal and preserve its simple square body and "
        "face. Describe the reaction through whole-body behavior, relocation, cover, escape, a raised "
        "shield/weapon/tool, or another visible object-state change. The returned face remains the same "
        "neutral, simple square pixel face. NEVER include any facial or eye reaction in the generated "
        "prompt — forbidden wording includes widened/wide eyes, fearful/angry/startled expression, "
        "furrowed brows, grimacing, staring, or any facial transformation. Convey all urgency using only "
        "body position, movement, cover, and existing objects; those cues preserve voxel identity. "
        "The ONE thing to AVOID: frantic flailing arm gestures — do NOT write 'arms pumping', "
        "'waving arms', 'flailing', or similar chaotic limb motion (it makes the video choppy); "
        "if the subject runs, just describe the run, keep hands simple or still.\n\n"
        "Then output exactly one JSON object and nothing else. Its only key is revisit_prompt, "
        "whose value is one video-generation prompt in the exact style of the examples — start "
        "with the camera motion, then the subject's changed state. Use this exact schema: "
        '{"revisit_prompt":"The camera ..."}')

    def reason_revisit(self) -> str:
        """综合已看过的主体/环境/事件，推理主体反应，输出 ONE 条 revisit 生成 prompt。
        默认关闭 Qwen thinking，并要求 JSON 结构化输出。这里只检查格式，不限制 VLM
        对具体反应动作的语义选择。任何思考泄漏、半句话或非 camera-motion prompt 都会重试。"""
        mt = int(os.environ.get("CW_VLM_REASON_TOKENS", "768"))
        last = ""
        for attempt in range(3):
            raw = self._chat(
                self._REASON_INSTR,
                max_tokens=mt * (attempt + 1),
                response_format={"type": "json_object"},
            )
            prompt = _parse_revisit_json(raw)
            if _valid_revisit_prompt(prompt):
                return prompt
            last = raw
            self.messages = self.messages[:-2]
        raise RuntimeError(
            "VLM revisit prompt failed structured format validation after 3 attempts; "
            f"last response tail={last[-300:]!r}"
        )

    def verify_revisit(self, frames) -> str:
        """闭环：看生成好的 seg2，自校验 revisit 是否符合推理（返回简短报告）。"""
        return self._chat(_content_with_frames(
            "Here is the generated revisit segment. Check whether the subject's reaction matches "
            "what you reasoned, and whether identity/environment stayed consistent. For a blocky or "
            "voxel subject, explicitly check that the square voxel identity and simple face remained "
            "unchanged. For a fantasy mage using a staff, check whether the requested visible magical "
            "attack or barrier actually appeared. Give a short self-check report (2-3 sentences).", frames))

    # ── Phase-2: reference retrieval（真部署形态）────────────────────────────────
    # 训练侧我们按 GT 规则挑三类 anchor（见 refs.jsonl / step7_pick_refs.py）：
    #   Entity  = 静止段主体最清晰的一帧（P1 段，subject_on_screen=True，主体面积/清晰度最大）
    #   Scene   = pan_away 段 |cam_curve| 峰值那一帧（相机方位离 P1 视角最远、看到 revisit 目标场景）
    # 推理侧 VLM 要从【已生成的 seg0 和 seg1】里挑对应帧，让下一段 seg2 的 reference token
    # 与训练分布尽量对齐。这里给 VLM 明确说明"你要选的两帧对应训练时哪种角色"，让它 mimic GT 规则。
    #
    # ★ scene 选帧的 pitfall（实测数据）：早期 prompt 用"pan-peak = 能看清 revisit 完整背景"
    # 时，VLM 系统性地挑到 seg1 后半段（camera 已经稳定停在 event 处，画面清晰），比 GT
    # 规则（argmax|cam_curve|，pan 运动中）晚 6-8 个 latent（≈ 2.3s）。这两个定义都合理，
    # 但训练侧用的是 GT 定义 —— renderer 学到的是"这类 latent = 运动中的画面"。VLM 塞静止
    # 清晰帧会分布不匹配，视频效果打折。所以下面明确对齐 GT：**要 pan 运动幅度最大那一帧**，
    # 不是"看得最清那一帧"。
    _RETRIEVE_INSTR = (
        "You are about to help retrieve two reference frames from the already-generated segments, "
        "so that the last segment (revisit) can be rendered consistently. Two reference roles:\n"
        "  * ENTITY anchor: from segment-0 (opening, subject on-screen). Pick the frame where the "
        "subject is MOST CLEARLY visible — largest, sharpest, well-lit, minimally occluded. "
        "This anchor preserves the subject's identity across the reveal.\n"
        "  * SCENE anchor: from segment-1 (event, camera panning away from subject). Pick the "
        "PAN-PEAK frame — the frame in which the CAMERA MOTION IS LARGEST (the camera is actively "
        "sweeping across the scene, and you can see visible motion blur or the fastest scene "
        "displacement across the frame). This is NOT the frame where the camera has already "
        "settled on the event and the picture is clear; it is the moment of PEAK ANGULAR MOTION, "
        "usually somewhere in the FIRST HALF or MIDDLE of segment-1, well BEFORE the camera "
        "stops. Prefer a mid-pan frame with visible motion over a late-pan static frame, even "
        "if the static frame shows the event more clearly.\n\n"
        "Below are frames labelled with their FRAME INDEX inside each segment. I show you {n0} "
        "frames from seg0 (indices 0..{max0}) then {n1} frames from seg1 (indices 0..{max1}). "
        "Answer with a JSON object on ONE line, no extra text:\n"
        "  {{\"entity_frame_idx\": <int in 0..{max0}>, \"scene_frame_idx\": <int in 0..{max1}>, "
        "\"reasoning\": \"<one short sentence>\"}}\n"
        "Rules:\n"
        "  - entity_frame_idx MUST be a seg0 index (subject-clearest frame).\n"
        "  - scene_frame_idx MUST be a seg1 index (pan-peak, PEAK CAMERA MOTION — not settled).\n"
        "  - Do not output anything except that single-line JSON.\n"
    )

    def retrieve_references(self, seg0_frames, seg1_frames) -> dict:
        """让 VLM 从已生成的 seg0/seg1 里挑 (entity_frame_idx, scene_frame_idx)。

        输入
        ----
        seg0_frames, seg1_frames : list[PIL.Image]
            renderer 抽好的帧（各段抽了几帧就传几帧），index 0..len-1 对应喂给 VLM 的顺序。
        返回
        ----
        {"entity_frame_idx": int, "scene_frame_idx": int, "reasoning": str, "raw": str}
            index 是【喂给 VLM 的帧序号】—— 不是原视频的帧号；pipeline 侧再把这个序号
            映射回 seg 内的相对位置（latent 索引）。
            解析失败 / 超出范围 → 抛 RuntimeError，调用方自行 fallback 到 GT-cut 路径。
        """
        n0, n1 = len(seg0_frames), len(seg1_frames)
        if n0 == 0 or n1 == 0:
            raise RuntimeError(f"retrieve_references: 需要 seg0/seg1 都非空，得到 n0={n0} n1={n1}")
        instr = self._RETRIEVE_INSTR.format(n0=n0, n1=n1, max0=n0 - 1, max1=n1 - 1)
        # 关键：在 content 数组里给两段图【逐帧打标】+ 段间加明显分隔文字。
        # 之前一次性把 12 张图丢过去，VLM 大量 token 花在"这 12 张里哪些是 seg0 哪些是 seg1"上，
        # 想着想着就把 max_tokens 用完了、来不及吐 JSON。给每张图打 [SEG0 frame k] 标签后
        # VLM 不用再猜边界，直接答题。
        content = [{"type": "text", "text": instr}]
        content.append({"type": "text", "text": f"=== SEG0 starts (frames 0..{n0-1}, subject on-screen) ==="})
        for k, im in enumerate(seg0_frames):
            content.append({"type": "text", "text": f"[SEG0 frame {k}]"})
            content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})
        content.append({"type": "text", "text": f"=== SEG1 starts (frames 0..{n1-1}, camera panning away, subject NOT visible) ==="})
        for k, im in enumerate(seg1_frames):
            content.append({"type": "text", "text": f"[SEG1 frame {k}]"})
            content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})
        # 思维模型开着时会边看边想，token 少了会在想到一半就被截断（还没吐 JSON）。
        # 默认给 2048，可用 CW_VLM_RETRIEVE_TOKENS 覆盖。关思维模式（server CW_VLM_THINKING=0）下 512 就够。
        mt = int(os.environ.get("CW_VLM_RETRIEVE_TOKENS", "2048"))
        # 强制 JSON 结构化输出：vLLM 支持 OpenAI 的 response_format={"type":"json_object"}，
        # 在采样时用 grammar-guided decoding 保证输出是合法 JSON。这直接跳过思维模型的
        # 自由文本枚举阶段。
        raw = self._chat(content, max_tokens=mt, response_format={"type": "json_object"})
        parsed = _parse_retrieve_json(raw, n0, n1)
        if parsed is not None:
            return {**parsed, "raw": raw}

        # 第一次解析失败：极简重试一次，明确要求 "just JSON, no other text"，用小 token 快速返回。
        # 回滚上一轮 (user, assistant) 避免污染多轮历史。
        self.messages = self.messages[:-2]
        retry_instr = (
            "Try again. Output ONLY a single-line JSON object, no analysis, no thinking out loud:\n"
            f'  {{"entity_frame_idx": <int in 0..{n0-1}>, "scene_frame_idx": <int in 0..{n1-1}>, '
            '"reasoning": "<one short sentence>"}}\n'
            "The FIRST character of your reply must be `{`. Nothing else."
        )
        content_retry = [{"type": "text", "text": retry_instr}]
        content_retry.append({"type": "text", "text": f"=== SEG0 (0..{n0-1}) ==="})
        for k, im in enumerate(seg0_frames):
            content_retry.append({"type": "text", "text": f"[SEG0 frame {k}]"})
            content_retry.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})
        content_retry.append({"type": "text", "text": f"=== SEG1 (0..{n1-1}) ==="})
        for k, im in enumerate(seg1_frames):
            content_retry.append({"type": "text", "text": f"[SEG1 frame {k}]"})
            content_retry.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})
        raw2 = self._chat(content_retry, max_tokens=512, response_format={"type": "json_object"})
        parsed2 = _parse_retrieve_json(raw2, n0, n1)
        if parsed2 is not None:
            return {**parsed2, "raw": raw2, "retry": True}

        raise RuntimeError(
            f"retrieve_references: 两次都拿不到合法 JSON。第 1 次 raw={raw!r}  第 2 次 raw={raw2!r}")
