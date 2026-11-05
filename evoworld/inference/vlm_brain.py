"""VLM reasoning interface for EvoWorld inference."""
from __future__ import annotations
import base64
import io
import json
import os
import re

import requests


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
    import re
    if "</think>" in text:
        return text.split("</think>")[-1].strip()

    return re.sub(r"^<think>.*", "", text, flags=re.DOTALL).strip() or text.strip()


def _looks_truncated(prompt: str) -> str:
    p = (prompt or "").strip()
    if len(p) < 40:
        return True
    if len(p) > 1200:
        return True
    if p[-1] not in ".!?\"'))":
        return True
    tail = p.rsplit(None, 1)[-1].lower().rstrip(".") if " " in p else ""
    if tail in {"to", "the", "a", "an", "and", "with", "his", "her", "their", "toward", "towards", "at", "as", "of", "in", "on"}:
        return True
    return False


def _clean_prompt(text: str) -> str:
    import re

    t = text.strip().strip('"').strip()
    # Extract the final camera-motion prompt and discard leaked self-review.
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
    cleaned = _strip_think(text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return ""
    prompt = obj.get("revisit_prompt") if isinstance(obj, dict) else None
    return prompt.strip() if isinstance(prompt, str) else ""


def _valid_revisit_prompt(prompt: str) -> bool:
    p = (prompt or "").strip()
    if not (80 <= len(p) <= 1200):
        return False
    if not re.match(r"^The camera\s+(?:pans|moves|tracks|returns|sweeps|travels)\b", p):
        return False
    if p[-1] not in '.!?"\'))':
        return False
    if "..." in p or "..." in p:
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
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _content_with_frames(text: str, frames) -> list:
    content = [{"type": "text", "text": text}]
    for im in frames:
        content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})
    return content


def _parse_retrieve_json(raw: str, n0: int, n1: int):
    import json as _json
    import re as _re

    def _extract_first_balanced_json(s: str):
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

            start = s.find("{", start + 1)
        return None

    def _validate(ei, si, reasoning=""):
        if not (0 <= ei < n0) or not (0 <= si < n1):
            return None
        return {"entity_frame_idx": int(ei), "scene_frame_idx": int(si),
                "reasoning": str(reasoning)}


    js = _extract_first_balanced_json(raw)
    if js is not None:
        try:
            j = _json.loads(js)
            return _validate(j.get("entity_frame_idx", -1),
                             j.get("scene_frame_idx", -1),
                             j.get("reasoning", ""))
        except Exception:
            pass


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


        self.base_url = (base_url or os.environ.get("EW_VLM_URL", "http://127.0.0.1:8001/v1")).rstrip("/")
        self.model = model or os.environ.get("EW_VLM_MODEL", "Qwen3.8-27B")
        self.max_tokens = max_tokens


        self.temperature = (temperature if temperature is not None
                            else float(os.environ.get("EW_VLM_TEMP", "0.0")))
        self.enable_thinking = bool(int(os.environ.get("EW_VLM_THINKING", "0") or 0))


        self.timeout = timeout if timeout is not None else int(os.environ.get("EW_VLM_TIMEOUT", "900"))

        self.messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.notes = {"subject": None, "environment": None, "event": None}

    def _chat(self, user_content, max_tokens=None, retries=3, response_format=None) -> str:
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
                self.messages.append({"role": "assistant", "content": raw})
                return _strip_think(raw)
            except Exception as e:
                last = e
                if attempt < retries - 1:
                    import time as _t
                    _t.sleep(5 * (attempt + 1))

        self.messages.pop()
        raise RuntimeError(f"VLM request failed after {retries} attempts: {last}")

    def observe_opening(self, frames) -> str:
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
        mt = int(os.environ.get("EW_VLM_REASON_TOKENS", "768"))
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
        return self._chat(_content_with_frames(
            "Here is the generated revisit segment. Check whether the subject's reaction matches "
            "what you reasoned, and whether identity/environment stayed consistent. For a blocky or "
            "voxel subject, explicitly check that the square voxel identity and simple face remained "
            "unchanged. For a fantasy mage using a staff, check whether the requested visible magical "
            "attack or barrier actually appeared. Give a short self-check report (2-3 sentences).", frames))


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
        n0, n1 = len(seg0_frames), len(seg1_frames)
        if n0 == 0 or n1 == 0:
            raise RuntimeError(f"retrieve_references requires non-empty seg0 and seg1; n0={n0}, n1={n1}")
        instr = self._RETRIEVE_INSTR.format(n0=n0, n1=n1, max0=n0 - 1, max1=n1 - 1)


        content = [{"type": "text", "text": instr}]
        content.append({"type": "text", "text": f"=== SEG0 starts (frames 0..{n0-1}, subject on-screen) ==="})
        for k, im in enumerate(seg0_frames):
            content.append({"type": "text", "text": f"[SEG0 frame {k}]"})
            content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})
        content.append({"type": "text", "text": f"=== SEG1 starts (frames 0..{n1-1}, camera panning away, subject NOT visible) ==="})
        for k, im in enumerate(seg1_frames):
            content.append({"type": "text", "text": f"[SEG1 frame {k}]"})
            content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})


        mt = int(os.environ.get("EW_VLM_RETRIEVE_TOKENS", "2048"))


        raw = self._chat(content, max_tokens=mt, response_format={"type": "json_object"})
        parsed = _parse_retrieve_json(raw, n0, n1)
        if parsed is not None:
            return {**parsed, "raw": raw}


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
            f"retrieve_references failed to parse JSON twice: first={raw!r}, second={raw2!r}")
