#!/usr/bin/env python3
"""
Pipecat Transfer Agent v2 — Voice AI, strikt getrennt vom Controller.

auto_subscribe=False — der Controller entscheidet, was der Bot hört.
Expliziter Transfer-Flow statt Text-Matching.
"""

import asyncio
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
import aiohttp

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import (
    EndFrame, TTSTextFrame, FunctionCallResultFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.azure.tts import AzureHttpTTSService
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams

# ── Config ────────────────────────────────────────────────────────
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "de")

AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY", "")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION", "germanywestcentral")
AZURE_TTS_VOICE = os.getenv("AZURE_TTS_VOICE", "de-DE-KatjaNeural")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HA_API_KEY = os.getenv("HA_API_KEY", "")
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://127.0.0.1:9100")
TRANSFER_TIMEOUT = int(os.getenv("TRANSFER_TIMEOUT", "60"))


# ══════════════════════════════════════════════════════════════════
#  CONTROLLER CLIENT (robust)
# ══════════════════════════════════════════════════════════════════

async def call_controller(endpoint: str, data: dict, timeout: int = 60) -> dict:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.post(f"{CONTROLLER_URL}{endpoint}", json=data) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.error(f"Controller {endpoint}: HTTP {resp.status}: {text[:200]}")
                    return {"status": "error", "error": f"HTTP {resp.status}"}
                try:
                    return json.loads(text)
                except Exception:
                    return {"status": "error", "error": f"Invalid JSON: {text[:200]}"}
    except asyncio.TimeoutError:
        return {"status": "error", "error": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def poll_controller_state(room: str, target_phase: str, timeout: int = 60) -> dict:
    """Pollt /v1/room/{room}/state bis target_phase erreicht."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
                async with s.get(f"{CONTROLLER_URL}/v1/room/{room}/state") as resp:
                    state = await resp.json()
                    if state.get("phase") == target_phase:
                        return state
                    if state.get("phase") == "failed":
                        return state
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return {"phase": "timeout"}


# ══════════════════════════════════════════════════════════════════
#  BACKEND CLIENT
# ══════════════════════════════════════════════════════════════════

async def fetch_forward_list() -> str:
    headers = {"Content-Type": "application/json"}
    if HA_API_KEY:
        headers["X-API-Key"] = HA_API_KEY
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(f"{BACKEND_URL}/api/settings/forwards", headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if not data:
                        return "Keine Weiterleitungen konfiguriert."
                    lines = [f"{i}) {f['name']} → {f['destination']}" for i, f in enumerate(data, 1)]
                    return "Verfügbare Weiterleitungen:\n" + "\n".join(lines)
                return f"Fehler beim Laden (Status {resp.status})"
    except Exception as e:
        return f"Weiterleitungen nicht verfügbar: {e}"


async def resolve_forward(name: str) -> str:
    headers = {"Content-Type": "application/json"}
    if HA_API_KEY:
        headers["X-API-Key"] = HA_API_KEY
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(f"{BACKEND_URL}/api/settings/forwards", headers=headers) as resp:
                if resp.status == 200:
                    for fwd in await resp.json():
                        if fwd["name"].lower() == name.lower():
                            return fwd["destination"]
    except Exception:
        pass
    return name


# ══════════════════════════════════════════════════════════════════
#  TRANSFER FLOW
# ══════════════════════════════════════════════════════════════════

async def execute_transfer(room_name: str, target: str, announce: bool,
                           conversation_summary: str = "") -> str:
    """Führt Transfer über Controller aus. Pollt auf Status-Änderungen."""

    mode = "warm" if announce else "cold"

    if not target.startswith("+"):
        target = await resolve_forward(target)
    if not target.startswith("+"):
        return f"Keine gültige Rufnummer für '{target}'."

    logger.info(f"[{room_name}] Transfer: {target} mode={mode}")

    # 1. Transfer starten (non-blocking)
    result = await call_controller("/v1/transfer", {
        "room": room_name,
        "target": target,
        "mode": mode,
    })
    if result.get("status") == "error":
        return f"Transfer fehlgeschlagen: {result.get('error', 'unbekannt')}"

    # 2. Auf Agent warten
    state = await poll_controller_state(room_name, "briefing" if announce else "connected",
                                        timeout=TRANSFER_TIMEOUT)

    phase = state.get("phase", "unknown")

    if phase == "failed":
        return "Der Ansprechpartner war nicht erreichbar. Was möchten Sie tun?"

    if phase == "timeout":
        return "Die Verbindung zum Ansprechpartner dauert länger als erwartet."

    if phase == "connected":
        # Cold transfer done
        return f"Durchgestellt zu {target}."

    if phase == "briefing":
        # Warm: Bot soll jetzt briefen
        summary_text = conversation_summary or "der Anrufer möchte mit Ihnen sprechen."
        return (
            f"STATUS: Der Ansprechpartner ist jetzt verbunden und hört NUR dich.\n"
            f"Der Anrufer hört Wartemusik und bekommt nichts mit.\n\n"
            f"Bitte briefe den Ansprechpartner kurz:\n"
            f"{summary_text}\n\n"
            f"Sag am Ende EXAKT: 'Ich stelle jetzt durch.'\n"
            f"Danach wirst du aus dem Raum entfernt."
        )

    return f"Unbekannter Transfer-Status: {phase}"


async def complete_warm_transfer(room_name: str):
    """Briefing abgeschlossen → Controller verbinden + Bot entfernen."""
    await call_controller("/v1/briefing-complete", {"room": room_name})
    await asyncio.sleep(0.5)
    await call_controller("/v1/disconnect-bot", {"room": room_name})


# ══════════════════════════════════════════════════════════════════
#  TRANSFER STAGE — ersetzt Text-Watcher durch expliziten Flow
# ══════════════════════════════════════════════════════════════════

class TransferStage(FrameProcessor):
    """
    Verwaltet den Transfer-Lifecycle explizit — kein Text-Matching.

    Modi:
      - idle: normales Gespräch
      - awaiting_briefing: auf Controller-Signal wartend
      - briefing: Bot brieft Agent
      - done: Transfer abgeschlossen
    """

    def __init__(self, room_name: str, **kwargs):
        super().__init__(**kwargs)
        self._room = room_name
        self._mode = "idle"
        self._task: Optional[PipelineTask] = None

    def set_task(self, task: PipelineTask):
        self._task = task

    async def start_briefing(self, summary: str):
        """Controller hat agent verbunden → Bot soll briefen."""
        self._mode = "briefing"
        logger.info(f"[{self._room}] Briefing phase active")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if self._mode == "briefing":
            if isinstance(frame, TTSTextFrame):
                text = frame.text.strip()
                # Exakter Trigger: Satz endet mit der Abschlussfloskel
                if text.endswith("Ich stelle jetzt durch.") or text.endswith("ich stelle jetzt durch."):
                    self._mode = "done"
                    logger.info(f"[{self._room}] Briefing complete, connecting + disconnecting bot")
                    asyncio.create_task(self._finish_transfer())

        await self.push_frame(frame, direction)

    async def _finish_transfer(self):
        await complete_warm_transfer(self._room)
        # Signal pipeline to stop
        if self._task:
            await self._task.queue_frame(EndFrame())


# ══════════════════════════════════════════════════════════════════
#  FUNCTION TOOLS
# ══════════════════════════════════════════════════════════════════

def make_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "weiterleitungen_abrufen",
                "description": "Liste verfügbare Weiterleitungen. IMMER vor verbinden() aufrufen.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verbinden",
                "description": (
                    "Stellt Anrufer durch. ansagen=true: Warm (du brieft vorher, "
                    "Caller hört nichts vom Briefing). ansagen=false: Cold (direkt). "
                    "ziel = Telefonnummer (+49...) oder Name aus der Weiterleitungsliste."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ziel": {"type": "string", "description": "Rufnummer oder Name"},
                        "ansagen": {"type": "boolean", "description": "true=warm, false=cold"},
                    },
                    "required": ["ziel", "ansagen"],
                },
            },
        },
    ]


# ══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Du bist ein freundlicher Telefonassistent. Sprich Deutsch, kurz und natürlich.

VERHALTEN:
- Sprich wie ein Mensch, keine langen Monologe.
- Will der Anrufer jemanden sprechen: erst weiterleitungen_abrufen(), dann verbinden().

WARM-TRANSFER (ansagen=true):
Nachdem die Verbindung zum Ansprechpartner hergestellt wurde, 
sprichst du NUR mit dem Ansprechpartner. Der Anrufer hört Wartemusik.
Briefe den Ansprechpartner kurz: Wer ruft an, was ist das Anliegen?
Beende dein Briefing EXAKT mit dem Satz: 'Ich stelle jetzt durch.'
Danach wirst du automatisch entfernt und das Gespräch läuft direkt.

COLD-TRANSFER (ansagen=false):
Der Anrufer wird direkt verbunden."""


# ══════════════════════════════════════════════════════════════════
#  AGENT
# ══════════════════════════════════════════════════════════════════

async def run_agent(room_name: str):
    transfer_stage = TransferStage(room_name)

    # Transport — Bot managed Subscriptions NICHT selbst
    transport = LiveKitTransport(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        room_name=room_name,
        bot_identity=f"pipecat-bot-{int(time.time())}",
        params=LiveKitParams(
            auto_subscribe=False,  # Controller steuert, was der Bot hört
        ),
    )

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        model=DEEPGRAM_MODEL,
        language=DEEPGRAM_LANGUAGE,
    )

    llm = OpenAILLMService(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
    )

    tts = AzureHttpTTSService(
        api_key=AZURE_TTS_KEY,
        region=AZURE_TTS_REGION,
        voice=AZURE_TTS_VOICE,
    )

    # Tools registrieren
    tools = make_tools()

    # Korrekte Pipecat-Tool-Registrierung
    llm.register_function("weiterleitungen_abrufen", None)
    llm.register_function("verbinden", None)

    context = LLMContextAggregatorPair(
        llm=llm,
        user_params=LLMUserAggregatorParams(system_prompt=SYSTEM_PROMPT),
    )

    # Korrekte Pipeline-Reihenfolge:
    # Audio → STT → UserContext → LLM → TTS → TransferStage → Output → AssistantContext
    pipeline = Pipeline([
        transport.input(),
        stt,
        context.user(),
        llm,
        tts,
        transfer_stage,
        transport.output(),
        context.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    transfer_stage.set_task(task)

    # Gesprächsspeicher für Briefing-Summary
    conversation_history = []

    @transport.event_handler("on_first_participant_joined")
    async def on_first(transport, participant_id):
        logger.info(f"[{room_name}] First participant: {participant_id}")

    @transport.event_handler("on_participant_joined")
    async def on_joined(transport, participant_id):
        logger.info(f"[{room_name}] Participant joined: {participant_id}")

    # Function handler
    async def handle_function(function_name: str, tool_call_id: str,
                              args: dict, llm_service):
        logger.info(f"[{room_name}] FUNCTION CALL: {function_name} args={args}")

        if function_name == "weiterleitungen_abrufen":
            return await fetch_forward_list()

        if function_name == "verbinden":
            ziel = args.get("ziel", "")
            ansagen = args.get("ansagen", False)

            # Briefing-Summary aus Konversation
            summary = "Der Anrufer möchte mit Ihnen sprechen."
            if conversation_history:
                recent = [m.get("content", "") for m in conversation_history[-6:]
                          if m.get("role") == "user"]
                if recent:
                    summary = f"Anliegen des Anrufers: {' '.join(recent[-3:])}"

            result = await execute_transfer(room_name, ziel, ansagen, summary)

            if ansagen and "STATUS:" in result:
                transfer_stage._mode = "briefing"
                # Entferne die Meta-Anweisung für das LLM — nur die Briefing-Info
                result = result.replace("STATUS: ", "")

            return result

        return f"Unbekannte Funktion: {function_name}"

    # Handler registrieren (Pipecat 1.1 pattern)
    llm.register_function("weiterleitungen_abrufen",
                          lambda name, tid, a: handle_function(name, tid, a, None))
    llm.register_function("verbinden",
                          lambda name, tid, a: handle_function(name, tid, a, None))

    # Conversation tracker
    @transport.event_handler("on_user_started_speaking")
    async def on_user_start(transport):
        pass

    runner = PipelineRunner()
    await runner.run(task)


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transfer_agent.py <room_name>")
        sys.exit(1)
    asyncio.run(run_agent(sys.argv[1]))
