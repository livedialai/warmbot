#!/usr/bin/env python3
"""
Pipecat Transfer Agent — Voice AI für Warm/Cold Transfers.

Der Bot spricht mit dem Anrufer (STT→LLM→TTS) und delegiert
die LiveKit-Call-Control (Track Subscriptions, SIP Outbound)
an den Controller (HTTP localhost:9100).

Transfer-Modi:
  - "cold": Bot sagt Controller → Caller auf Musik → Agent joint → verbunden
  - "warm": Bot brieft Agent mündlich → dann complete → verbunden
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


# ══════════════════════════════════════════════════════════════════
#  CONTROLLER CLIENT
# ══════════════════════════════════════════════════════════════════

async def call_controller(endpoint: str, data: dict) -> dict:
    """Ruft den Controller via HTTP."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
        async with s.post(
            f"{CONTROLLER_URL}{endpoint}",
            json=data,
        ) as resp:
            return await resp.json()


# ══════════════════════════════════════════════════════════════════
#  BACKEND CLIENT (Weiterleitungen)
# ══════════════════════════════════════════════════════════════════

async def fetch_forward_list() -> str:
    """Holt Weiterleitungsliste vom Backend."""
    headers = {"Content-Type": "application/json"}
    if HA_API_KEY:
        headers["X-API-Key"] = HA_API_KEY
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(
                f"{BACKEND_URL}/api/settings/forwards", headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if not data:
                        return "Keine Weiterleitungen konfiguriert."
                    lines = []
                    for i, fwd in enumerate(data, 1):
                        lines.append(f"{i}) {fwd['name']} → {fwd['destination']}")
                    return "Verfügbare Weiterleitungen:\n" + "\n".join(lines)
                return f"Fehler beim Laden (Status {resp.status})"
    except Exception as e:
        return f"Weiterleitungen nicht verfügbar: {e}"


async def resolve_forward_target(name: str) -> str:
    """Löst Weiterleitungs-Namen in Rufnummer auf."""
    headers = {"Content-Type": "application/json"}
    if HA_API_KEY:
        headers["X-API-Key"] = HA_API_KEY
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(
                f"{BACKEND_URL}/api/settings/forwards", headers=headers
            ) as resp:
                if resp.status == 200:
                    for fwd in await resp.json():
                        if fwd["name"].lower() == name.lower():
                            return fwd["destination"]
    except Exception:
        pass
    return name


# ══════════════════════════════════════════════════════════════════
#  TRANSFER LOGIC (ruft Controller)
# ══════════════════════════════════════════════════════════════════

async def execute_transfer(room_name: str, target: str, announce: bool) -> str:
    """
    Sagt dem Controller: starte Transfer.
    Bei Warm: Bot spricht Briefing und meldet dann complete.
    Bei Cold: Controller verbindet direkt.
    """
    mode = "warm" if announce else "cold"

    # Auflösen falls Forward-Name statt Nummer
    if not target.startswith("+"):
        target = await resolve_forward_target(target)
    if not target.startswith("+"):
        return f"Keine gültige Rufnummer für '{target}' gefunden."

    logger.info(f"Transfer: room={room_name} target={target} mode={mode}")

    result = await call_controller("/v1/transfer", {
        "room": room_name,
        "target": target,
        "mode": mode,
    })

    status = result.get("status", "error")

    if status == "connected":
        return f"Durchgestellt zu {target}."
    elif status == "briefing":
        return (
            f"VERBINDUNG HERGESTELLT. Der Ansprechpartner ist jetzt im Raum. "
            f"Bitte briefe kurz: Wer ruft an, was ist das Anliegen? "
            f"Sag danach 'Ich stelle durch' und beende deine Teilnahme."
        )
    elif status == "failed":
        return f"Verbindung zu {target} fehlgeschlagen: {result.get('error', 'unbekannt')}."
    else:
        return f"Unerwarteter Status: {status}"


# ══════════════════════════════════════════════════════════════════
#  FRAME PROCESSOR: Briefing-Detection
# ══════════════════════════════════════════════════════════════════

class BriefingWatcher(FrameProcessor):
    """Erkennt, wann der Bot das Briefing beendet (sagt 'Ich stelle durch')."""

    def __init__(self, room_name: str, **kwargs):
        super().__init__(**kwargs)
        self._room = room_name
        self._briefing_active = False
        self._done = False

    def activate(self):
        self._briefing_active = True
        self._done = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if self._briefing_active and not self._done:
            if isinstance(frame, TTSTextFrame):
                text = frame.text.lower()
                if any(w in text for w in ["durchstelle", "durchstellen", "verbinde jetzt", "stelle durch"]):
                    self._done = True
                    self._briefing_active = False
                    logger.info(f"[{self._room}] Briefing complete detected, calling controller")
                    asyncio.create_task(
                        call_controller("/v1/briefing-complete", {"room": self._room})
                    )

        await self.push_frame(frame, direction)


# ══════════════════════════════════════════════════════════════════
#  FUNCTION TOOLS
# ══════════════════════════════════════════════════════════════════

def make_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "weiterleitungen_abrufen",
                "description": "Liste verfügbare Weiterleitungen. IMMER aufrufen bevor du verbinden() mit einem Namen aufrufst.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verbinden",
                "description": (
                    "Stellt Anrufer durch. ansagen=true: Warm (du brieft vorher). "
                    "ansagen=false: Cold (direkt). ziel = Telefonnummer (+49...) "
                    "oder Name aus der Weiterleitungsliste."
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
#  AGENT SETUP
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Du bist ein freundlicher Telefonassistent. Deutsch, kurz, natürlich.

VERHALTEN:
- Sprich wie ein Mensch. Keine langen Monologe.
- Will der Anrufer jemanden sprechen: erst weiterleitungen_abrufen(), dann verbinden().

TRANSFER:
- weiterleitungen_abrufen() zeigt Optionen.
- verbinden(ziel="Nummer/Name", ansagen=true/false) stellt durch.
  ansagen=true: du kündigst an und brieft den Kollegen.
  ansagen=false: direkt durchstellen.

Bei Warm-Transfer nach der Bestätigung:
Sag dem Kollegen kurz wer anruft und worum es geht, dann 'Ich stelle durch' und verabschiede dich."""


async def run_agent(room_name: str):
    """Hauptfunktion: Pipecat Bot im LiveKit Raum."""

    briefing_watcher = BriefingWatcher(room_name)

    # LiveKit Transport
    transport = LiveKitTransport(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        room_name=room_name,
        bot_identity=f"pipecat-bot-{int(time.time())}",
        params=LiveKitParams(auto_subscribe=True),
    )

    # STT
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        model=DEEPGRAM_MODEL,
        language=DEEPGRAM_LANGUAGE,
    )

    # LLM
    llm = OpenAILLMService(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
    )

    # TTS
    tts = AzureHttpTTSService(
        api_key=AZURE_TTS_KEY,
        region=AZURE_TTS_REGION,
        voice=AZURE_TTS_VOICE,
    )

    # Register tools
    tools = make_tools()
    llm.register_function(None, tools)

    # Context aggregators
    context = LLMContextAggregatorPair(
        llm=llm,
        user_params=LLMUserAggregatorParams(system_prompt=SYSTEM_PROMPT),
    )

    # Pipeline
    pipeline = Pipeline([
        transport.input(),
        context.user(),
        stt,
        llm,
        tts,
        transport.output(),
        context.assistant(),
        briefing_watcher,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Function handler
    async def handle_function(name: str, tool_id: str, args: dict) -> str:
        if name == "weiterleitungen_abrufen":
            return await fetch_forward_list()
        elif name == "verbinden":
            ansagen = args.get("ansagen", False)
            if ansagen:
                briefing_watcher.activate()
            return await execute_transfer(room_name, args.get("ziel", ""), ansagen)
        return f"Unbekannt: {name}"

    llm.register_function_handler(None, handle_function)

    runner = PipelineRunner()
    await runner.run(task)


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transfer_agent.py <room_name>")
        sys.exit(1)
    asyncio.run(run_agent(sys.argv[1]))
