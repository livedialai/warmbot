#!/usr/bin/env python3
"""
Pipecat Transfer Agent v2.1 — Voice AI mit ToolsSchema + ConversationTracker.

auto_subscribe=False — der Controller steuert, was der Bot hört.
Expliziter Transfer-Flow mit Polling.
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
    EndFrame, LLMRunFrame, TTSTextFrame, TranscriptionFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema
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
#  HTTP CLIENTS
# ══════════════════════════════════════════════════════════════════

async def call_controller(endpoint: str, data: dict, timeout: int = 60) -> dict:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.post(f"{CONTROLLER_URL}{endpoint}", json=data) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return {"status": "error", "error": f"HTTP {resp.status}: {text[:200]}"}
                try:
                    return json.loads(text)
                except Exception:
                    return {"status": "error", "error": f"Invalid JSON: {text[:200]}"}
    except asyncio.TimeoutError:
        return {"status": "error", "error": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def poll_controller_state(room: str, target_phase: str, timeout: int = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
                async with s.get(f"{CONTROLLER_URL}/v1/room/{room}/state") as resp:
                    state = await resp.json()
                    phase = state.get("phase", "")
                    if phase == target_phase:
                        return state
                    if phase == "failed":
                        return state
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return {"phase": "timeout"}


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
                    return "Weiterleitungen:\n" + "\n".join(
                        f"{i}) {f['name']} → {f['destination']}" for i, f in enumerate(data, 1)
                    )
                return f"Fehler (Status {resp.status})"
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
#  TRANSFER
# ══════════════════════════════════════════════════════════════════

async def execute_transfer(room_name: str, target: str, announce: bool,
                           summary_text: str = "") -> str:
    mode = "warm" if announce else "cold"
    if not target.startswith("+"):
        target = await resolve_forward(target)
    if not target.startswith("+"):
        return f"Keine gültige Rufnummer für '{target}'."

    result = await call_controller("/v1/transfer", {"room": room_name, "target": target, "mode": mode})
    if result.get("status") == "error":
        return f"Transfer fehlgeschlagen: {result.get('error', 'unbekannt')}"

    state = await poll_controller_state(room_name, "briefing" if announce else "connected", TRANSFER_TIMEOUT)
    phase = state.get("phase", "unknown")

    if phase == "failed":
        return "Der Ansprechpartner war nicht erreichbar."
    if phase == "timeout":
        return "Die Verbindung dauert länger als erwartet."

    if phase == "connected":
        await call_controller("/v1/disconnect-bot", {"room": room_name})
        return f"Durchgestellt zu {target}."

    if phase == "briefing":
        return (
            f"Der Ansprechpartner ist jetzt in der Leitung. "
            f"Der Anrufer hört Wartemusik. "
            f"Briefe jetzt kurz: {summary_text} "
            f"Beende exakt mit: Ich stelle jetzt durch."
        )

    return f"Unbekannter Status: {phase}"


async def complete_warm_transfer(room_name: str):
    result = await call_controller("/v1/briefing-complete", {"room": room_name})
    if result.get("status") != "connected":
        logger.error(f"[{room_name}] briefing-complete failed: {result}")
        return result
    await asyncio.sleep(0.5)
    disc = await call_controller("/v1/disconnect-bot", {"room": room_name})
    logger.info(f"[{room_name}] disconnect-bot: {disc}")
    return result


# ══════════════════════════════════════════════════════════════════
#  CONVERSATION TRACKER
# ══════════════════════════════════════════════════════════════════

class ConversationTracker(FrameProcessor):
    def __init__(self, history: list, **kwargs):
        super().__init__(**kwargs)
        self._history = history

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                self._history.append({"role": "user", "content": text, "ts": time.time()})
        elif isinstance(frame, TTSTextFrame):
            text = frame.text.strip()
            if text:
                self._history.append({"role": "assistant", "content": text, "ts": time.time()})
        await self.push_frame(frame, direction)


# ══════════════════════════════════════════════════════════════════
#  TRANSFER STAGE
# ══════════════════════════════════════════════════════════════════

class TransferStage(FrameProcessor):
    def __init__(self, room_name: str, **kwargs):
        super().__init__(**kwargs)
        self._room = room_name
        self._mode = "idle"
        self._task: Optional[PipelineTask] = None

    def set_task(self, task: PipelineTask):
        self._task = task

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if self._mode == "briefing":
            if isinstance(frame, TTSTextFrame):
                text = frame.text.strip()
                if text.endswith("Ich stelle jetzt durch.") or text.endswith("ich stelle jetzt durch."):
                    self._mode = "done"
                    logger.info(f"[{self._room}] Briefing complete")
                    asyncio.create_task(self._finish())

        await self.push_frame(frame, direction)

    async def _finish(self):
        await complete_warm_transfer(self._room)
        if self._task:
            await self._task.queue_frame(EndFrame())


# ══════════════════════════════════════════════════════════════════
#  AGENT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Du bist ein freundlicher Telefonassistent. Sprich Deutsch, kurz und natürlich.

VERHALTEN:
- Sprich wie ein Mensch, keine langen Monologe.
- Will der Anrufer jemanden sprechen: erst weiterleitungen_abrufen(), dann verbinden().
- Interne Statushinweise nie wörtlich vorlesen.

WARM-TRANSFER (ansagen=true):
Du sprichst NUR mit dem Ansprechpartner. Der Anrufer hört Wartemusik.
Briefe den Ansprechpartner kurz: Wer ruft an, was ist das Anliegen?
Beende dein Briefing EXAKT: 'Ich stelle jetzt durch.'

COLD-TRANSFER (ansagen=false):
Der Anrufer wird direkt verbunden."""


async def run_agent(room_name: str):
    transfer_stage = TransferStage(room_name)
    conversation_history = []
    conversation_tracker = ConversationTracker(conversation_history)

    transport = LiveKitTransport(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        room_name=room_name,
        bot_identity=f"pipecat-bot-{int(time.time())}",
        params=LiveKitParams(auto_subscribe=False),
    )

    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY, model=DEEPGRAM_MODEL, language=DEEPGRAM_LANGUAGE)
    llm = OpenAILLMService(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        settings=OpenAILLMService.Settings(model=LLM_MODEL),
    )

    tts = AzureHttpTTSService(
        api_key=AZURE_TTS_KEY,
        region=AZURE_TTS_REGION,
        settings=AzureHttpTTSService.Settings(
            voice=AZURE_TTS_VOICE,
            language="de-DE",
        ),
    )

    # ToolsSchema
    tools_schema = ToolsSchema(standard_tools=[
        FunctionSchema(
            name="weiterleitungen_abrufen",
            description="Liste verfügbare Weiterleitungen. IMMER vor verbinden() aufrufen.",
            properties={}, required=[],
        ),
        FunctionSchema(
            name="verbinden",
            description="Stellt Anrufer durch. ansagen=true Warm, ansagen=false Cold.",
            properties={
                "ziel": {"type": "string", "description": "Rufnummer oder Name"},
                "ansagen": {"type": "boolean", "description": "true=warm, false=cold"},
            },
            required=["ziel", "ansagen"],
        ),
    ])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Beginne das Gespräch freundlich."},
    ]

    llm_context = LLMContext(messages=messages, tools=tools_schema)

    context = LLMContextAggregatorPair(
        llm_context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    async def handle_function(function_name: str, tool_call_id: str, args: dict,
                              llm, context, result_callback):
        logger.info(f"FUNCTION CALL: {function_name} args={args} room={room_name}")

        if function_name == "weiterleitungen_abrufen":
            await result_callback(await fetch_forward_list())
            return

        if function_name == "verbinden":
            ziel = args.get("ziel", "")
            ansagen = args.get("ansagen", False)

            summary = "Der Anrufer möchte mit Ihnen sprechen."
            if conversation_history:
                user_msgs = [m.get("content", "") for m in conversation_history[-6:]
                             if m.get("role") == "user"]
                if user_msgs:
                    summary = f"Anliegen: {' '.join(user_msgs[-3:])}"

            result = await execute_transfer(room_name, ziel, ansagen, summary)

            if ansagen and "Der Ansprechpartner ist jetzt" in result:
                transfer_stage._mode = "briefing"

            await result_callback(result)
            return

        await result_callback(f"Unbekannte Funktion: {function_name}")

    llm.register_function("weiterleitungen_abrufen", handle_function)
    llm.register_function("verbinden", handle_function)

    pipeline = Pipeline([
        transport.input(),
        stt,
        conversation_tracker,
        context.user(),
        llm,
        tts,
        transfer_stage,
        transport.output(),
        context.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(
        allow_interruptions=True, enable_metrics=True,
        vad_analyzer=SileroVADAnalyzer(),
    ))
    transfer_stage.set_task(task)

    @transport.event_handler("on_first_participant_joined")
    async def on_first(transport, participant_id):
        logger.info(f"[{room_name}] First participant: {participant_id}")
        await context.user().push_context_frame()
        await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transfer_agent.py <room_name>")
        sys.exit(1)
    asyncio.run(run_agent(sys.argv[1]))
