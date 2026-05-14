import asyncio
import json
import os
import re
import time
import secrets as stdlib_secrets
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()
from loguru import logger
import aiohttp
import redis.asyncio as aioredis
from openai import AsyncOpenAI

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import EndFrame, LLMRunFrame, TranscriptionFrame, TTSTextFrame, TextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.runner.types import DailyDialinRequest, RunnerArguments
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.azure.tts import AzureHttpTTSService
from pipecat.services.inworld.tts import InworldTTSService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.azure.tts import AzureHttpTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv()

CONFIG = {
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "backend_url": os.getenv("BACKEND_URL", "http://localhost:8000"),
    "ha_api_key": os.getenv("HA_API_KEY", ""),
    "dashboard_user": os.getenv("DASHBOARD_USER", "admin"),
    "dashboard_password": os.getenv("DASHBOARD_PASSWORD", "changeme123"),
    "dashboard_port": int(os.getenv("DASHBOARD_PORT", "8091")),
    "deepgram_api_key": os.getenv("DEEPGRAM_API_KEY"),
    "mistral_api_key": os.getenv("MISTRAL_API_KEY", ""),
    "deepgram_model": os.getenv("DEEPGRAM_MODEL", "nova-3"),
    "deepgram_language": os.getenv("DEEPGRAM_LANGUAGE", "de"),
    "deepgram_base_url": os.getenv("DEEPGRAM_BASE_URL", ""),
    "inworld_api_key": os.getenv("INWORLD_API_KEY"),
    "inworld_voice_id": os.getenv("INWORLD_VOICE_ID", "default-gir-n2kfw-hbdko0a0q9lw__nadine-neu"),
    "inworld_language": os.getenv("INWORLD_LANGUAGE", "de"),
    "inworld_model": os.getenv("INWORLD_MODEL", "inworld-tts-1"),
    "azure_tts_key": os.getenv("AZURE_TTS_KEY"),
    "azure_tts_region": os.getenv("AZURE_TTS_REGION", "germanywestcentral"),
    "azure_tts_voice": os.getenv("AZURE_TTS_VOICE", "de-DE-KatjaNeural"),
    "llm_api_key": os.getenv("LLM_API_KEY", "dummy"),
    "llm_base_url": os.getenv("LLM_BASE_URL"),
    "llm_model": os.getenv("LLM_MODEL", "qwen3"),
    "verkauf_prompt_path": os.getenv("VERKAUF_PROMPT_PATH", "./verkaufsprompt.txt"),
    "max_call_duration": int(os.getenv("MAX_CALL_DURATION_SECONDS", "600")),
    "smtp_host": os.getenv("SMTP_HOST", ""),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "smtp_user": os.getenv("SMTP_USER", ""),
    "smtp_pass": os.getenv("SMTP_PASS", ""),
    "smtp_from": os.getenv("SMTP_FROM", ""),
    "email_to": os.getenv("EMAIL_TO", ""),
}

DAILY_API_KEY = os.getenv("DAILY_API_KEY")
DAILY_API_URL = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

def load_prompt(path, fallback):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        logger.warning(f"Prompt nicht gefunden: {path}, nutze Fallback")
        return fallback

def sanitize_prompt_for_inworld(prompt):
    prompt = re.sub(r'## SPRACHSTEUERUNG \(Cartesia Sonic-3 TTS\).*?(?=## ZAHLEN|## GESPRÄCHSSTIL|---\s*\n## )', '', prompt, flags=re.DOTALL)
    prompt = re.sub(r'<[^>]+>', '', prompt)
    prompt = re.sub(r'\n---\s*\n---\s*\n', '\n---\n', prompt)
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)
    return prompt.strip()

VERKAUF_PROMPT = sanitize_prompt_for_inworld(load_prompt(
    CONFIG["verkauf_prompt_path"],
    "Du bist ein freundlicher Assistent. Antworte immer auf Deutsch, kurz und praezise."
))

async def get_redis():
    return aioredis.from_url(CONFIG["redis_url"], decode_responses=True)

async def resolve_tenant_config(called_did, sip_user=None):
    headers = {"Content-Type": "application/json"}
    if CONFIG["ha_api_key"]:
        headers["X-API-Key"] = CONFIG["ha_api_key"]
    tc = {}
    try:
        if called_did:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.post(f"{CONFIG['backend_url']}/api/tenants/resolve", json={"did": called_did}, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tc = data.get("config", {})
                        logger.info(f"[Tenant] Resolved tenant by DID {called_did}")
        if not tc and sip_user:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.post(f"{CONFIG['backend_url']}/api/tenants/resolve-by-sip", json={"sip_user": sip_user}, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tc = data.get("config", {})
                        called_did = data.get("did", called_did)
                        logger.info(f"[Tenant] Resolved tenant by SIP user {sip_user} → DID {called_did}")
        if not tc:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{CONFIG['backend_url']}/api/tenants/default-config", headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tc = data.get("config", {})
                        logger.info("[Tenant] Using default tenant config")
    except Exception as e:
        logger.warning(f"[Tenant] Resolve failed: {e}")
    return tc, called_did

async def resolve_dynamic_tools(called_did):
    if not called_did:
        return [], ""
    headers = {"Content-Type": "application/json"}
    if CONFIG["ha_api_key"]:
        headers["X-API-Key"] = CONFIG["ha_api_key"]
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.post(f"{CONFIG['backend_url']}/api/settings/integrations/resolve", json={"did": called_did}, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tool_defs = data.get("tools", [])
                    if tool_defs:
                        lines = ["\n\nVERFÜGBARE API-TOOLS (du kannst sie mit execute_api_tool aufrufen):"]
                        for td in tool_defs:
                            name = td.get("name", "?")
                            desc = td.get("description", "")
                            params = td.get("parameters", {}).get("properties", {})
                            param_desc = ", ".join(params.keys()) if params else "keine"
                            lines.append(f"  • {name}: {desc} (Parameter: {param_desc})")
                        return tool_defs, "\n".join(lines)
    except Exception as e:
        logger.warning(f"[Tools] Resolve failed: {e}")
    return [], ""

async def query_knowledge(query, api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.post(f"{CONFIG['backend_url']}/api/settings/knowledge/query", json={"query": query, "top_k": 3}, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    chunks = [r.get("chunk_text", "") for r in data.get("results", [])]
                    return "\n\n---\n\n".join(chunks) if chunks else "Keine relevanten Informationen gefunden."
                return f"Fehler bei der Wissensabfrage: {resp.status}"
    except Exception as e:
        return f"Wissensabfrage nicht verfügbar: {str(e)[:100]}"

async def generate_summary(messages, llm_key, llm_base, llm_model):
    conversation = "\n".join([f"{'Kunde' if m['role'] == 'user' else 'Bot'}: {m['content']}" for m in messages if m["role"] != "system"])
    if not conversation.strip():
        return "Kein Gesprächsverlauf vorhanden."
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.post(
                f"{llm_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {llm_key}", "Content-Type": "application/json"},
                json={"model": llm_model, "messages": [{"role": "user", "content": f"Fasse das Telefonat auf Deutsch kurz zusammen (max. 5 Sätze). Nenne Grund und Ergebnis.\n\nGESPRÄCH:\n{conversation}\n\nZUSAMMENFASSUNG:"}], "max_tokens": 300, "temperature": 0.3},
            ) as resp:
                data = await resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or conversation[:500]
    except Exception:
        return conversation[:500]

async def send_summary_email(cs):
    if not cs.smtp_host or not cs.email_to:
        logger.info(f"[{cs.call_sid}] No SMTP config, skipping email")
        return
    try:
        summary = await generate_summary(cs.messages, cs.llm_api_key, cs.llm_base_url, cs.llm_model)
        msg = EmailMessage()
        msg["Subject"] = f"Gesprächszusammenfassung - {cs.from_number or cs.call_sid}"
        msg["From"] = cs.smtp_from or cs.smtp_user
        msg["To"] = cs.email_to
        msg.set_content(f"Gesprächszusammenfassung\nDatum: {datetime.now().strftime('%d.%m.%Y %H:%M')}\nDauer: {cs.duration}s\nAnrufer: {cs.from_number}\n\n{summary}")
        with smtplib.SMTP(cs.smtp_host, cs.smtp_port) as s:
            s.starttls()
            if cs.smtp_user:
                s.login(cs.smtp_user, cs.smtp_pass)
            s.send_message(msg)
        logger.info(f"[{cs.call_sid}] Summary email sent to {cs.email_to}")
    except Exception as e:
        logger.warning(f"[{cs.call_sid}] Email failed: {e}")

# =============================================================================
# REDIS PERSISTENCE
# =============================================================================

async def save_call_to_redis(call_data):
    try:
        r = await get_redis()
        key = f"call:{call_data['call_sid']}"
        await r.set(key, json.dumps(call_data))
        await r.lpush("calls:recent", call_data["call_sid"])
        await r.ltrim("calls:recent", 0, 4999)
        await r.sadd("calls:all", call_data["call_sid"])
        logger.info(f"[{call_data['call_sid']}] Call in Redis gespeichert")
    except Exception as e:
        logger.error(f"Redis save_call Fehler: {e}")

async def save_lead_to_redis(lead_id, call_sid, duration, conversation):
    if not lead_id:
        return
    try:
        r = await get_redis()
        key = f"lead:{lead_id}"
        existing = await r.get(key)
        if existing:
            ld = json.loads(existing)
            ld.setdefault("calls", []).append({"call_sid": call_sid, "timestamp": datetime.now().isoformat(), "duration": duration})
            ld["last_call"] = datetime.now().isoformat()
            ld["call_count"] = len(ld["calls"])
        else:
            ld = {"lead_id": lead_id, "status": "UNCLASSIFIED", "first_call": datetime.now().isoformat(), "last_call": datetime.now().isoformat(), "call_count": 1, "calls": []}
        await r.set(key, json.dumps(ld))
        await r.sadd("leads:all", lead_id)
    except Exception as e:
        logger.error(f"Redis save_lead Fehler: {e}")

# =============================================================================
# TRANSCRIPT EVENT BUS
# =============================================================================

class TranscriptEventBus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, call_sid):
        q = asyncio.Queue()
        self._subscribers.setdefault(call_sid, []).append(q)
        return q

    def unsubscribe(self, call_sid, q):
        if call_sid in self._subscribers:
            self._subscribers[call_sid] = [x for x in self._subscribers[call_sid] if x is not q]
            if not self._subscribers[call_sid]:
                del self._subscribers[call_sid]

    def subscribe_all(self):
        q = asyncio.Queue()
        self._subscribers.setdefault("__all__", []).append(q)
        return q

    def unsubscribe_all(self, q):
        self.unsubscribe("__all__", q)

    async def publish(self, call_sid, event):
        event["call_sid"] = call_sid
        event["timestamp"] = datetime.now().isoformat()
        for q in self._subscribers.get(call_sid, []):
            await q.put(event)
        for q in self._subscribers.get("__all__", []):
            await q.put(event)

    def active_calls(self):
        return [k for k in self._subscribers if k != "__all__"]

_event_bus = TranscriptEventBus()

# =============================================================================
# TRANSCRIPT PROCESSORS
# =============================================================================

class TranscriptProcessor(FrameProcessor):
    def __init__(self, session, event_bus, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._event_bus = event_bus

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                entry = {"role": "user", "content": text, "ts": time.time()}
                self._session.live_transcript.append(entry)
                await self._event_bus.publish(self._session.call_sid, {"type": "transcript", "role": "user", "content": text})
                await self._save_live_entry(entry)
                frame.text = ".Client " + text
        await self.push_frame(frame, direction)

    async def _save_live_entry(self, entry):
        try:
            r = await get_redis()
            key = f"live:{self._session.call_sid}"
            await r.rpush(key, json.dumps(entry))
            await r.expire(key, 24 * 60 * 60)
        except Exception as e:
            logger.error(f"Redis live transcript Fehler: {e}")


class AssistantTranscriptProcessor(FrameProcessor):
    def __init__(self, session, event_bus, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._event_bus = event_bus
        self._buffer = ""
        self._last_flush = time.time()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame):
            text = frame.text.strip()
            if text:
                self._buffer += " " + text
                if any(text.endswith(p) for p in ".!?") or (time.time() - self._last_flush > 2):
                    content = self._buffer.strip()
                    if content:
                        entry = {"role": "assistant", "content": content, "ts": time.time()}
                        self._session.live_transcript.append(entry)
                        await self._event_bus.publish(self._session.call_sid, {"type": "transcript", "role": "assistant", "content": content})
                        await self._save_live_entry(entry)
                    self._buffer = ""
                    self._last_flush = time.time()
        await self.push_frame(frame, direction)

    async def _save_live_entry(self, entry):
        try:
            r = await get_redis()
            key = f"live:{self._session.call_sid}"
            await r.rpush(key, json.dumps(entry))
            await r.expire(key, 24 * 60 * 60)
        except Exception as e:
            logger.error(f"Redis live transcript Fehler: {e}")


class SSMLStripProcessor(FrameProcessor):
    _ALL_TAGS_RE = re.compile(r'<[^>]+>')
    _QUOTES_RE = re.compile(r'^["\u201c\u201e]|["\u201d\u201c]$')

    def _clean(self, text):
        text = self._ALL_TAGS_RE.sub('', text)
        text = self._QUOTES_RE.sub('', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSTextFrame, TextFrame)):
            cleaned = self._clean(frame.text)
            if cleaned:
                frame.text = cleaned
            else:
                return
        await self.push_frame(frame, direction)

# =============================================================================
# CALL SESSION
# =============================================================================

class CallSession:
    def __init__(self, call_sid, lead_id, from_number):
        self.call_sid = call_sid
        self.lead_id = lead_id
        self.from_number = from_number
        self.called_did = ""
        self.start_time = time.time()
        self.finalized = False
        self.live_transcript = []
        self.messages = []
        self.tenant_config = {}
        self.meetergo_config = {}
        self.dynamic_tools = []
        self.llm_api_key = CONFIG["llm_api_key"]
        self.llm_base_url = CONFIG["llm_base_url"]
        self.llm_model = CONFIG["llm_model"]
        self.smtp_host = ""
        self.smtp_port = 587
        self.smtp_user = ""
        self.smtp_pass = ""
        self.smtp_from = ""
        self.email_to = ""

    @property
    def duration(self):
        return int(time.time() - self.start_time)

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def to_dict(self):
        return {
            "call_sid": self.call_sid,
            "lead_id": self.lead_id,
            "from": self.from_number,
            "called_did": self.called_did,
            "duration": self.duration,
            "conversation": [m for m in self.messages if m["role"] != "system"],
            "timestamp": datetime.now().isoformat(),
        }

_sessions = {}
_watchdog_tasks = {}

# =============================================================================
# CALL ABSCHLUSS
# =============================================================================

async def finalize_call(session):
    if session.finalized:
        return
    session.finalized = True

    logger.info(f"[{session.call_sid}] Call wird finalisiert (Dauer: {session.duration}s)")
    await _event_bus.publish(session.call_sid, {"type": "call_end", "duration": session.duration})
    await save_call_to_redis(session.to_dict())
    await save_lead_to_redis(session.lead_id, session.call_sid, session.duration, [m for m in session.messages if m["role"] != "system"])
    await send_summary_email(session)
    _sessions.pop(session.call_sid, None)
    task = _watchdog_tasks.pop(session.call_sid, None)
    if task:
        task.cancel()

# =============================================================================
# DYNAMIC TOOLS
# =============================================================================

async def handle_firmenwissen(function_name, tool_call_id, args, llm, context, result_callback):
    logger.info(f"TOOL CALL: {function_name=} {tool_call_id=} {args=}")
    logger.info(f"Known sessions: {list(_sessions.keys())}")
    query = args.get("query", "")
    api_key = session_dynamic_api_key(_sessions.get(tool_call_id, ""), "ha")
    result = await query_knowledge(query, api_key)
    await result_callback(result)

async def handle_check_meetergo_availability(function_name, tool_call_id, args, llm, context, result_callback):
    mc = _sessions.get(tool_call_id, CallSession("dummy", "", "")).meetergo_config if tool_call_id in _sessions else {}
    if mc.get("enabled") != "true":
        await result_callback("Meetergo Kalender ist nicht aktiviert.")
        return
    uid = mc.get("user_id", "")
    key = mc.get("api_key", "")
    mtid = mc.get("meeting_type_id", "")
    if not uid or not key or not mtid:
        await result_callback("Meetergo Zugangsdaten unvollständig.")
        return
    start = args.get("start", datetime.now().strftime("%Y-%m-%d"))
    end = args.get("end", (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"))
    if start == end:
        d = datetime.strptime(start, "%Y-%m-%d") + timedelta(days=1)
        end = d.strftime("%Y-%m-%d")
    try:
        url = f"https://api.meetergo.com/v4/booking-availability?meetingTypeId={mtid}&hostIds={uid}&start={start}&end={end}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(url, headers={"Authorization": f"Bearer {key}", "x-meetergo-api-user-id": uid}) as resp:
                data = await resp.json()
                dates = data.get("dates", [])
                if not dates:
                    await result_callback(f"Keine verfügbaren Termine zwischen {start} und {end}.")
                    return
                all_spots = []
                for d in dates:
                    for spot in d.get("spots", []):
                        all_spots.append((d["date"], spot.get("startTime", "")))
                all_spots_shuffled = all_spots.copy()
                import random
                random.shuffle(all_spots_shuffled)
                result_lines = []
                for date, start_t in all_spots_shuffled[:10]:
                    h = int(start_t[11:13]) + 2
                    if h >= 24:
                        h -= 24
                    result_lines.append(f"{date}: {h:02d}:{start_t[14:16]} Uhr")
                await result_callback("Verfügbare Termine (MEZ/Sommerzeit):\n" + "\n".join(result_lines[:10]))
    except Exception as e:
        await result_callback(f"Kalenderabfrage nicht möglich: {str(e)[:100]}")

async def handle_book_appointment(function_name, tool_call_id, args, llm, context, result_callback):
    cs = _sessions.get(tool_call_id, CallSession("dummy", "", ""))
    mc = cs.meetergo_config if hasattr(cs, 'meetergo_config') else {}
    if mc.get("enabled") != "true":
        await result_callback("Meetergo Kalender ist nicht aktiviert.")
        return
    uid = mc.get("booking_host_id", mc.get("user_id", ""))
    key = mc.get("booking_api_key", mc.get("api_key", ""))
    mtid = mc.get("booking_mtid", mc.get("meeting_type_id", ""))
    if not uid or not key or not mtid:
        await result_callback("Meetergo Zugangsdaten unvollständig.")
        return
    caller_name = args.get("callerName", args.get("caller_firstname", "Anrufer"))
    caller_phone = args.get("callerPhone", cs.from_number if hasattr(cs, 'from_number') else "")
    start_time = args.get("start_time", args.get("startTime", ""))
    grund = args.get("grund", "")
    name_parts = caller_name.strip().split(maxsplit=1)
    firstname = name_parts[0]
    lastname = name_parts[1] if len(name_parts) > 1 else ""
    real_phone = cs.called_did if (hasattr(cs, 'called_did') and cs.called_did and cs.called_did.startswith("+")) else caller_phone
    phone_clean = real_phone.replace("+", "").replace(" ", "")
    email = f"{phone_clean}@gofonia.de"
    mob_text = f" Mobil: {caller_phone}" if caller_phone and caller_phone != real_phone else ""
    grund_text = f". Grund: {grund}" if grund else ""
    context_str = f"{caller_name} hat angerufen von {real_phone}.{mob_text}{grund_text}"
    payload = {
        "attendee": {"email": email, "firstname": firstname, "lastname": lastname, "fullname": caller_name, "phone": real_phone, "receiveReminders": True, "language": "de", "timezone": "Europe/Berlin", "dataPolicyAccepted": True},
        "meetingTypeId": mtid, "hostIds": [uid], "start": start_time, "duration": 30, "channel": "connect", "context": context_str, "source": "gofonia_voice_bot",
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.post("https://api.meetergo.com/v4/booking", headers={"Authorization": f"Bearer {key}", "x-meetergo-api-user-id": uid, "Content-Type": "application/json"}, json=payload) as resp:
                if resp.status < 300:
                    logger.info(f"Meetergo booking created for {caller_name}")
                    await result_callback("Termin gebucht! Bestätigen Sie dem Kunden den Termin.")
                else:
                    err = await resp.text()
                    await result_callback(f"Buchung fehlgeschlagen: HTTP {resp.status} – {err[:200]}")
    except Exception as e:
        await result_callback(f"Buchung nicht möglich: {str(e)[:100]}")

async def handle_execute_api_tool(function_name, tool_call_id, args, llm, context, result_callback):
    cs = _sessions.get(tool_call_id, CallSession("dummy", "", ""))
    tool_name = args.get("tool_name", "")
    tool_args = args.get("arguments", {})
    tool_defs = cs.dynamic_tools if hasattr(cs, 'dynamic_tools') else []
    td = next((t for t in tool_defs if t.get("name") == tool_name), None)
    if not td:
        await result_callback(f"Tool '{tool_name}' nicht gefunden.")
        return
    req = td.get("request", {})
    url = req.get("url", "")
    method = req.get("method", "GET").upper()
    headers = req.get("headers", {})
    body_template = req.get("body", "")
    try:
        for k, v in (tool_args or {}).items():
            body_template = body_template.replace(f"{{{{{k}}}}}", str(v))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            fetch_opts = {"method": method, "headers": headers}
            if method in ("POST", "PUT", "PATCH"):
                fetch_opts["data"] = body_template
                if "Content-Type" not in headers:
                    headers["Content-Type"] = "application/json"
            async with s.request(url=url, **fetch_opts) as resp:
                text = await resp.text()
                await result_callback(text[:1500])
    except Exception as e:
        await result_callback(f"Fehler: {str(e)[:200]}")

def session_dynamic_api_key(session, key_type):
    if not session:
        return CONFIG.get("ha_api_key", "")
    return CONFIG.get("ha_api_key", "")

# =============================================================================
# DASHBOARD
# =============================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pipecat Bot Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');
  :root {
    --bg: #07080c; --surface: #0d0f14; --surface2: #12151c;
    --border: #1a1e2a; --border-bright: #252a3a;
    --accent: #22d97f; --accent-dim: rgba(34,217,127,0.12);
    --accent2: #3b82f6; --accent2-dim: rgba(59,130,246,0.12);
    --text: #e4e7ef; --text2: #9ba3b8; --muted: #4a5068;
    --radius: 10px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'Outfit',sans-serif; min-height:100vh; }
  header { border-bottom:1px solid var(--border); padding:1rem 2rem; display:flex; align-items:center; justify-content:space-between; background:linear-gradient(180deg, var(--surface2) 0%, var(--surface) 100%); position:sticky; top:0; z-index:100; }
  .header-left { display:flex; align-items:center; gap:12px; }
  header h1 { font-family:'JetBrains Mono',monospace; font-size:0.95rem; font-weight:600; color:var(--accent); letter-spacing:0.08em; }
  .header-badge { font-family:'JetBrains Mono',monospace; font-size:0.6rem; padding:3px 8px; border-radius:4px; background:var(--accent-dim); color:var(--accent); border:1px solid rgba(34,217,127,0.2); text-transform:uppercase; letter-spacing:0.12em; }
  .status-dot { width:7px; height:7px; border-radius:50%; background:var(--accent); box-shadow:0 0 8px var(--accent); animation:pulse 2.5s ease-in-out infinite; display:inline-block; margin-right:6px; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)} }
  .live-count { font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--accent); background:var(--accent-dim); padding:4px 10px; border-radius:6px; border:1px solid rgba(34,217,127,0.15); }
  .refresh-btn { background:transparent; border:1px solid var(--border-bright); color:var(--text2); padding:0.35rem 0.9rem; border-radius:6px; cursor:pointer; font-family:'JetBrains Mono',monospace; font-size:0.7rem; transition:all 0.2s; }
  .refresh-btn:hover { border-color:var(--accent); color:var(--accent); background:var(--accent-dim); }
  .tab-nav { display:flex; gap:0; padding:0 2rem; border-bottom:1px solid var(--border); background:var(--surface); }
  .tab-btn { font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:500; text-transform:uppercase; letter-spacing:0.1em; padding:0.85rem 1.4rem; background:transparent; border:none; color:var(--muted); cursor:pointer; border-bottom:2px solid transparent; transition:all 0.2s; }
  .tab-btn:hover { color:var(--text2); }
  .tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
  .tab-content { display:none; }
  .tab-content.active { display:block; }
  .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; padding:1.2rem 2rem; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1rem 1.2rem; transition:border-color 0.2s; }
  .card:hover { border-color:var(--border-bright); }
  .card-label { font-family:'JetBrains Mono',monospace; font-size:0.6rem; font-weight:500; text-transform:uppercase; letter-spacing:0.15em; color:var(--muted); margin-bottom:0.4rem; }
  .card-value { font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:700; line-height:1.1; color:var(--text); }
  .card-value.accent { color:var(--accent2); }
  .card-sub { font-size:0.7rem; color:var(--muted); margin-top:4px; font-family:'JetBrains Mono',monospace; }
  .section { padding:0 2rem 1.5rem; }
  .section-title { font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:0.18em; color:var(--muted); margin-bottom:0.8rem; padding-bottom:0.5rem; border-bottom:1px solid var(--border); }
  .call-entry { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); margin-bottom:0.6rem; overflow:hidden; }
  .call-header { display:flex; align-items:center; justify-content:space-between; padding:0.7rem 1rem; cursor:pointer; }
  .call-header:hover { background:rgba(255,255,255,0.015); }
  .call-meta { display:flex; align-items:center; gap:0.8rem; font-size:0.82rem; }
  .call-meta .lead { font-family:'JetBrains Mono',monospace; color:var(--accent2); font-weight:500; font-size:0.8rem; }
  .call-meta .time { color:var(--muted); font-size:0.75rem; }
  .call-meta .dur { color:var(--text2); font-size:0.75rem; font-family:'JetBrains Mono',monospace; }
  .toggle-arrow { color:var(--muted); font-size:0.85rem; transition:transform 0.2s; }
  .toggle-arrow.open { transform:rotate(90deg); }
  .call-detail { display:none; border-top:1px solid var(--border); padding:1rem; background:var(--surface2); }
  .call-detail.open { display:block; }
  .transcript { margin-top:0.5rem; max-height:400px; overflow-y:auto; }
  .msg { margin-bottom:0.5rem; padding:0.45rem 0.7rem; border-radius:6px; }
  .msg.user { background:rgba(59,130,246,0.05); border-left:3px solid var(--accent2); }
  .msg.assistant { background:rgba(34,217,127,0.04); border-left:3px solid var(--accent); }
  .msg-role { font-family:'JetBrains Mono',monospace; font-size:0.58rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.15rem; }
  .msg-role.user { color:var(--accent2); }
  .msg-role.assistant { color:var(--accent); }
  .msg-text { font-size:0.82rem; line-height:1.5; color:var(--text); }
  .no-calls { text-align:center; color:var(--muted); padding:3rem; font-size:0.85rem; }
  .live-panel { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); margin:0 2rem 1.5rem; overflow:hidden; }
  .live-panel-header { display:flex; align-items:center; justify-content:space-between; padding:0.7rem 1rem; background:var(--surface2); border-bottom:1px solid var(--border); }
  .live-panel-title { font-family:'JetBrains Mono',monospace; font-size:0.68rem; font-weight:600; color:var(--accent); text-transform:uppercase; letter-spacing:0.12em; display:flex; align-items:center; gap:8px; }
  .live-feed { padding:0.8rem 1rem; max-height:500px; overflow-y:auto; scroll-behavior:smooth; }
  .live-msg { padding:0.35rem 0; display:flex; gap:8px; align-items:flex-start; }
  .live-msg-badge { font-family:'JetBrains Mono',monospace; font-size:0.55rem; font-weight:600; padding:2px 6px; border-radius:3px; text-transform:uppercase; white-space:nowrap; margin-top:2px; min-width:36px; text-align:center; }
  .live-msg-badge.user { background:var(--accent2-dim); color:var(--accent2); }
  .live-msg-badge.assistant { background:var(--accent-dim); color:var(--accent); }
  .live-msg-text { font-size:0.82rem; line-height:1.4; color:var(--text); flex:1; }
  .live-msg-sid { font-family:'JetBrains Mono',monospace; font-size:0.55rem; color:var(--muted); white-space:nowrap; }
  .live-empty { color:var(--muted); text-align:center; padding:2rem; font-size:0.82rem; }
  @media (max-width:768px) { .grid { grid-template-columns:repeat(2,1fr); } }
</style>
</head>
<body>
<header>
  <div class="header-left">
    <h1><span class="status-dot"></span>PIPECAT // DASHBOARD</h1>
    <span class="header-badge">Multi-Tenant</span>
  </div>
  <div>
    <span class="live-count" id="live-count">0 LIVE</span>
    <button class="refresh-btn" onclick="loadCalls()" style="margin-left:8px;">REFRESH</button>
  </div>
</header>
<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('overview')">Uebersicht</button>
  <button class="tab-btn" onclick="switchTab('live')">Live Calls</button>
  <button class="tab-btn" onclick="switchTab('history')">Verlauf</button>
</div>
<div class="tab-content active" id="tab-overview">
  <div class="grid">
    <div class="card"><div class="card-label">Calls Total</div><div class="card-value" id="total">&ndash;</div></div>
    <div class="card"><div class="card-label">Dauer avg</div><div class="card-value accent" id="avg-dur">&ndash;</div><div class="card-sub">Sekunden</div></div>
  </div>
  <div class="section"><div class="section-title">Letzte Anrufe</div><div id="calls-list"></div></div>
</div>
<div class="tab-content" id="tab-live">
  <div class="live-panel">
    <div class="live-panel-header"><div class="live-panel-title"><span class="status-dot"></span> Echtzeit Transkript</div><button class="refresh-btn" onclick="clearLiveFeed()">CLEAR</button></div>
    <div class="live-feed" id="live-feed"><div class="live-empty">Warte auf aktive Gespraech...</div></div>
  </div>
</div>
<div class="tab-content" id="tab-history"><div class="section"><div class="section-title">Alle Gespraeeche</div><div id="history-list"></div></div></div>
<script>
function esc(s){if(!s)return'';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML;}
function toggle(id){const el=document.getElementById('detail-'+id);const arrow=document.getElementById('arrow-'+id);if(el)el.classList.toggle('open');if(arrow)arrow.classList.toggle('open');}
function switchTab(name){document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));event.target.classList.add('active');document.getElementById('tab-'+name).classList.add('active');if(name==='live')startSSE();}
function renderCallEntry(c,i,prefix){const t=c.timestamp?new Date(c.timestamp).toLocaleString('de'):'-';const conv=(c.conversation||[]).map(m=>`<div class="msg ${esc(m.role)}"><div class="msg-role ${esc(m.role)}">${m.role==='user'?'KUNDE':'BOT'}</div><div class="msg-text">${esc(m.content)}</div></div>`).join('')||'<div style="color:var(--muted)">Kein Verlauf</div>';const id=prefix+'_'+i;return`<div class="call-entry"><div class="call-header" onclick="toggle('${id}')"><div class="call-meta"><span class="toggle-arrow" id="arrow-${id}">&#9654;</span><span class="lead">${esc(c.lead_id||c.from||'-')}</span><span class="time">${esc(t)}</span><span class="dur">${c.duration||'-'}s</span></div></div><div class="call-detail" id="detail-${id}"><div class="transcript">${conv}</div></div></div>`;}
let allCalls=[];
async function loadCalls(){try{const r=await fetch('/dashboard/api/calls');const data=await r.json();allCalls=data;let total=0,totalDur=0;const entries=data.map((c,i)=>{total++;totalDur+=(c.duration||0);return renderCallEntry(c,i,'ov');});document.getElementById('total').textContent=total;document.getElementById('avg-dur').textContent=total?Math.round(totalDur/total):'-';document.getElementById('calls-list').innerHTML=entries.join('')||'<div class="no-calls">Keine Anrufe</div>';document.getElementById('history-list').innerHTML=entries.join('')||'<div class="no-calls">Keine Anrufe</div>';}catch(e){console.error('Load error:',e);}}
let evtSource=null;
function startSSE(){if(evtSource)return;try{evtSource=new EventSource('/dashboard/api/live-stream');evtSource.onmessage=function(e){const data=JSON.parse(e.data);if(data.type==='transcript')addLiveMsg(data);else if(data.type==='call_start')updateLiveCount(1);else if(data.type==='call_end'){updateLiveCount(-1);loadCalls();}};evtSource.onerror=function(){evtSource.close();evtSource=null;setTimeout(startSSE,3000);};}catch(e){console.error('SSE error:',e);}}
function addLiveMsg(data){const feed=document.getElementById('live-feed');const empty=feed.querySelector('.live-empty');if(empty)empty.remove();const div=document.createElement('div');div.className='live-msg';const shortSid=(data.call_sid||'').slice(-8);div.innerHTML=`<span class="live-msg-badge ${data.role}">${data.role==='user'?'IN':'OUT'}</span><span class="live-msg-text">${esc(data.content)}</span><span class="live-msg-sid">${esc(shortSid)}</span>`;feed.appendChild(div);feed.scrollTop=feed.scrollHeight;while(feed.children.length>200)feed.removeChild(feed.firstChild);}
let liveCountVal=0;
function updateLiveCount(delta){liveCountVal=Math.max(0,liveCountVal+delta);document.getElementById('live-count').textContent=liveCountVal+' LIVE';}
function clearLiveFeed(){document.getElementById('live-feed').innerHTML='<div class="live-empty">Feed geleert</div>';}
loadCalls();setInterval(loadCalls,20000);setTimeout(startSSE,1000);
fetch('/dashboard/api/active-calls').then(r=>r.json()).then(d=>{liveCountVal=d.count||0;document.getElementById('live-count').textContent=liveCountVal+' LIVE';}).catch(()=>{});
</script>
</body>
</html>"""


def register_dashboard_routes(app):
    from fastapi import Depends, HTTPException, status
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials

    security = HTTPBasic()

    def check_auth(credentials=Depends(security)):
        correct_user = stdlib_secrets.compare_digest(credentials.username, CONFIG["dashboard_user"])
        correct_pass = stdlib_secrets.compare_digest(credentials.password, CONFIG["dashboard_password"])
        if not (correct_user and correct_pass):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    @app.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    async def dashboard():
        return DASHBOARD_HTML

    @app.get("/dashboard/api/calls", dependencies=[Depends(check_auth)])
    async def api_calls():
        try:
            r = await get_redis()
            call_sids = await r.lrange("calls:recent", 0, 199)
            calls = []
            for sid in call_sids:
                data = await r.get(f"call:{sid}")
                if data:
                    calls.append(json.loads(data))
            return JSONResponse(calls)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/dashboard/api/lead/{lead_id}", dependencies=[Depends(check_auth)])
    async def api_lead(lead_id):
        try:
            r = await get_redis()
            data = await r.get(f"lead:{lead_id}")
            if not data:
                return JSONResponse({"error": "Lead nicht gefunden"}, status_code=404)
            return JSONResponse(json.loads(data))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/dashboard/api/active-calls", dependencies=[Depends(check_auth)])
    async def api_active_calls():
        return JSONResponse({"count": len(_sessions)})

    @app.get("/dashboard/api/live-stream", dependencies=[Depends(check_auth)])
    async def api_live_stream():
        async def event_generator():
            q = _event_bus.subscribe_all()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type':'heartbeat'})}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                _event_bus.unsubscribe_all(q)
        return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    logger.info("Dashboard Routen registriert")


# =============================================================================
# BOT PIPELINE
# =============================================================================

def _save_context_to_session(context, session, initial_prompt):
    existing_count = len([m for m in session.messages if m["role"] != "system"])
    new_messages = []
    for msg in context.messages:
        if msg["role"] in ("user", "assistant"):
            content = msg.get("content") or ""
            if msg == initial_prompt:
                continue
            new_messages.append({"role": msg["role"], "content": content})
    for msg in new_messages[existing_count:]:
        session.add_message(msg["role"], msg["content"])

async def run_bot(transport, session):
    tc = session.tenant_config
    tenant_stt_key = tc.get("STT_API_KEY", CONFIG["deepgram_api_key"])
    tenant_llm_key = tc.get("LLM_API_KEY", CONFIG["llm_api_key"])
    tenant_llm_base = tc.get("LLM_BASE_URL", CONFIG["llm_base_url"])
    tenant_llm_model = tc.get("LLM_MODEL", CONFIG["llm_model"])
    tts_provider = tc.get("TTS_PROVIDER", "azure").lower()

    stt = AzureSTTService(
        api_key=CONFIG["azure_tts_key"],
        region=CONFIG["azure_tts_region"],
        settings=AzureSTTService.Settings(
            language=Language.DE_DE,
        ),
    )

    tts = AzureHttpTTSService(
        api_key=CONFIG["azure_tts_key"],
        region=CONFIG["azure_tts_region"],
        settings=AzureHttpTTSService.Settings(
            voice="de-DE-SeraphinaMultilingualNeural",
            language="de-DE",
        ),
    )


    llm = OpenAILLMService(
        api_key=tenant_llm_key,
        base_url=tenant_llm_base,
        settings=OpenAILLMService.Settings(
            model=tenant_llm_model,
        ),
    )

    tools_schema = []
    calendar_tools, calendar_handlers = get_calendar_tools(session)
    if calendar_tools:
        for ct in calendar_tools:
            func = ct.get("function", ct) if isinstance(ct, dict) else ct
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            req = params.get("required", [])
            if name:
                tools_schema.append(FunctionSchema(name=name, description=desc, properties=props, required=req))
        for name, handler in calendar_handlers.items():
            llm.register_function(name, handler)

    tools_schema.append(FunctionSchema(
        name="firmenwissen",
        description="Durchsuche das Firmenwissen nach Informationen zu einem bestimmten Thema oder einer Frage. Nutze dies wenn der Kunde etwas zu deinen Produkten, Dienstleistungen oder dem Unternehmen fragt.",
        properties={"query": {"type": "string", "description": "Die Suchanfrage"}},
        required=["query"],
    ))
    llm.register_function("firmenwissen", handle_firmenwissen)

    if session.dynamic_tools:
        tools_schema.append(FunctionSchema(
            name="execute_api_tool",
            description="Fuehre ein API-Tool aus. tool_name = Name des Tools, arguments = JSON-Objekt mit den Parametern.",
            properties={"tool_name": {"type": "string", "description": "Name des Tools"}, "arguments": {"type": "object", "description": "Parameter als JSON"}},
            required=["tool_name"],
        ))
        llm.register_function("execute_api_tool", handle_execute_api_tool)

    task_ref = []

    async def handle_end_call(function_name, tool_call_id, args, llm, context, result_callback):
        _save_context_to_session(context, session, initial_prompt)
        await result_callback(json.dumps({"status": "call_ended"}))
        if task_ref:
            await task_ref[0].queue_frame(EndFrame())

    tools_schema.append(FunctionSchema(
        name="end_call",
        description="Beende das Telefongespraech und lege auf. Nutze dieses Tool NACHDEM du dich verabschiedet hast.",
        properties={},
        required=[],
    ))
    llm.register_function("end_call", handle_end_call)

    messages = session.messages.copy()
    initial_prompt = {"role": "user", "content": "Beginne jetzt das Gespraech. Begruesse den Anrufer gemaess dem Gespraechsablauf."}
    if CONFIG.get("llm_no_think"):
        messages[0]["content"] = "/no_think\n" + messages[0]["content"]
    messages.append(initial_prompt)

    context = LLMContext(messages=messages, tools=ToolsSchema(standard_tools=tools_schema) if tools_schema else None)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    await _event_bus.publish(session.call_sid, {"type": "call_start", "lead_id": session.lead_id})

    # Transcript-Prozessoren
    transcript_processor = TranscriptProcessor(session, _event_bus)
    assistant_transcript_processor = AssistantTranscriptProcessor(session, _event_bus)
    ssml_strip = SSMLStripProcessor()

    pipeline = Pipeline([
        transport.input(),
        stt,
        transcript_processor,
        context_aggregator.user(),
        llm,
        ssml_strip,
        tts,
        assistant_transcript_processor,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    task_ref.append(task)

    async def watchdog():
        while session.call_sid in _sessions:
            await asyncio.sleep(5)
            if session.duration > CONFIG["max_call_duration"]:
                logger.warning(f"[{session.call_sid}] Call duration exceeded {CONFIG['max_call_duration']}s, ending call")
                if task_ref:
                    await task_ref[0].queue_frame(EndFrame())
                return
    _watchdog_tasks[session.call_sid] = asyncio.create_task(watchdog())

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"[{session.call_sid}] Teilnehmer joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await context_aggregator.user().push_context_frame()
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        _save_context_to_session(context, session, initial_prompt)
        await task.cancel()

    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        if state == "left":
            await task.cancel()

    await PipelineRunner().run(task)
    await finalize_call(session)

def get_calendar_tools(session):
    mc = session.meetergo_config
    if mc.get("enabled") != "true" and not CONFIG.get("calendar_enabled"):
        return [], {}
    tools = [
        {"type": "function", "function": {"name": "check_available_slots", "description": "Rufe die naechsten verfuegbaren Termine ab. Nutze dieses Tool BEVOR du dem Anrufer Terminoptionen anbietest.", "parameters": {"type": "object", "properties": {"start": {"type": "string", "description": "Heutiges Datum als ISO 8601"}, "end": {"type": "string", "description": "Datum 7 Tage in der Zukunft als ISO 8601"}}, "required": ["start", "end"]}}},
        {"type": "function", "function": {"name": "book_appointment", "description": "Buche einen Termin nachdem der Anrufer einen Zeitslot gewaehlt hat. Sammle vorher Vorname, Nachname und E-Mail-Adresse.", "parameters": {"type": "object", "properties": {"caller_firstname": {"type": "string", "description": "Vorname"}, "caller_lastname": {"type": "string", "description": "Nachname"}, "caller_email": {"type": "string", "description": "E-Mail-Adresse"}, "caller_phone": {"type": "string", "description": "Mobilnummer im E.164-Format"}, "start_time": {"type": "string", "description": "Gewaehlter Termin als ISO 8601"}, "grund": {"type": "string", "description": "Grund des Anrufs"}}, "required": ["caller_firstname", "caller_lastname", "caller_email", "start_time"]}}},
    ]
    handlers = {"check_available_slots": handle_check_meetergo_availability, "book_appointment": handle_book_appointment}
    return tools, handlers

# =============================================================================
# ENTRY POINT
# =============================================================================

async def bot(runner_args):
    if runner_args.body:
        request = DailyDialinRequest.model_validate(runner_args.body)
        from_field = request.dialin_settings.From or ""
        to_field = getattr(request.dialin_settings, "To", "") or ""
        sip_headers = getattr(request.dialin_settings, "sip_headers", None) or {}
        logger.info(f"[Dialin] Raw webhook - From: {from_field!r}, To: {to_field!r}, sipHeaders: {sip_headers}")

        called_did = ""
        sip_user = ""
        if isinstance(sip_headers, dict):
            x_called = sip_headers.get("X-Called-DID") or sip_headers.get("x-called-did") or ""
            if x_called:
                called_did = x_called

        for field in [to_field, from_field]:
            candidate = field.replace("sip:", "").split("@")[0] if "@sip:" in field or field.startswith("sip:") else field.split("@")[0] if "@" in field else field
            if candidate and not candidate.startswith("+") and "--" in candidate:
                sip_user = candidate
                break

        if not called_did.startswith("+"):
            import re
            dn_match = re.match(r'^"?\+(\d+)"?\s*<', from_field)
            if not called_did and dn_match:
                candidate = "+" + dn_match.group(1)
                if candidate.startswith("+") and len(candidate) > 5:
                    called_did = candidate

        if not called_did.startswith("+"):
            if to_field.startswith("+"):
                called_did = to_field
            else:
                user_part = to_field.replace("sip:", "").split("@")[0] if to_field.startswith("sip:") else to_field
                if user_part.startswith("+"):
                    called_did = user_part

        if not called_did.startswith("+"):
            if from_field.startswith("+"):
                called_did = from_field
            else:
                user_part = from_field.replace("sip:", "").split("@")[0] if from_field.startswith("sip:") else from_field
                if user_part.startswith("+"):
                    called_did = user_part

        logger.info(f"[Dialin] Parsed - called_did: {called_did!r}, sip_user: {sip_user!r}")
        lead_id = from_field.replace("sip:", "").split("@")[0]
        call_sid = request.dialin_settings.call_id
        session = CallSession(call_sid, lead_id, from_field)
        _sessions[call_sid] = session

        tc, called_did = await resolve_tenant_config(called_did, sip_user=sip_user or None)
        session.tenant_config = tc
        session.called_did = called_did
        if tc.get("LLM_API_KEY"):
            session.llm_api_key = tc["LLM_API_KEY"]
        if tc.get("LLM_BASE_URL"):
            session.llm_base_url = tc["LLM_BASE_URL"]
        if tc.get("LLM_MODEL"):
            session.llm_model = tc["LLM_MODEL"]
        if tc.get("SMTP_HOST"):
            session.smtp_host = tc["SMTP_HOST"]
        if tc.get("SMTP_PORT"):
            session.smtp_port = int(tc["SMTP_PORT"]) or 587
        if tc.get("SMTP_USER"):
            session.smtp_user = tc["SMTP_USER"]
        if tc.get("SMTP_PASS"):
            session.smtp_pass = tc["SMTP_PASS"]
        if tc.get("SMTP_FROM"):
            session.smtp_from = tc["SMTP_FROM"]
        if tc.get("EMAIL_TO"):
            session.email_to = tc["EMAIL_TO"]

        session.meetergo_config = {
            "enabled": tc.get("MEETERGO_ENABLED", "true" if CONFIG.get("calendar_enabled") else "false"),
            "user_id": tc.get("MEETERGO_USER_ID", CONFIG.get("meetergo_host_id", "")),
            "api_key": tc.get("MEETERGO_API_KEY", CONFIG.get("meetergo_api_key", "")),
            "meeting_type_id": tc.get("MEETERGO_MEETING_TYPE_ID", CONFIG.get("meetergo_meeting_type_id", "")),
            "booking_api_key": tc.get("MEETERGO_BOOKING_API_KEY", tc.get("MEETERGO_API_KEY", CONFIG.get("meetergo_api_key", ""))),
            "booking_host_id": tc.get("MEETERGO_BOOKING_HOST_ID", tc.get("MEETERGO_USER_ID", CONFIG.get("meetergo_host_id", ""))),
            "booking_mtid": tc.get("MEETERGO_BOOKING_MTID", tc.get("MEETERGO_MEETING_TYPE_ID", CONFIG.get("meetergo_meeting_type_id", ""))),
        }

        dynamic_tools, tool_descriptions = await resolve_dynamic_tools(called_did)
        session.dynamic_tools = dynamic_tools

        system_prompt = tc.get("prompt_system_prompt.txt", tc.get("prompt_system_prompt_txt", VERKAUF_PROMPT))
        today = datetime.now().strftime("%Y-%m-%d")
        system_prompt = f"\n\nHEUTIGES DATUM: {today}\nDie Rufnummer des Anrufers ist {called_did}.\n" + system_prompt
        if tool_descriptions:
            system_prompt += tool_descriptions
        session.messages = [{"role": "system", "content": system_prompt}]

        logger.info(f"[{call_sid}] Neuer Anruf von {from_field} an {called_did} (Lead: {lead_id})")

        transport = DailyTransport(
            runner_args.room_url, runner_args.token, "Assistent",
            DailyParams(
                api_key=request.daily_api_key,
                api_url=request.daily_api_url,
                dialin_settings=DailyDialinSettings(
                    call_id=request.dialin_settings.call_id,
                    call_domain=request.dialin_settings.call_domain,
                ),
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )
    else:
        logger.warning("Kein Body - lokaler Testmodus")
        call_sid = "local-test"
        session = CallSession(call_sid, "test-lead", "local")
        _sessions[call_sid] = session
        session.messages = [{"role": "system", "content": VERKAUF_PROMPT}]
        transport = DailyTransport(
            runner_args.room_url, runner_args.token, "Assistent",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )

    await run_bot(transport, session)


if __name__ == "__main__":
    import threading
    import uvicorn
    from fastapi import FastAPI as DashboardFastAPI

    dashboard_app = DashboardFastAPI()
    register_dashboard_routes(dashboard_app)

    def run_dashboard():
        uvicorn.run(dashboard_app, host="0.0.0.0", port=CONFIG["dashboard_port"], log_level="warning")

    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    logger.info(f"Dashboard gestartet auf Port {CONFIG['dashboard_port']}")

    from pipecat.runner.run import main
    main()