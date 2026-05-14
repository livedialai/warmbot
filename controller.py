#!/usr/bin/env python3
"""
LiveKit Call Controller — Die Regie für Warm/Cold Transfers.

Verwaltet Track-Subscriptions unabhängig von der Pipecat-Pipeline.
Kommunikation mit Pipecat via HTTP (localhost:9100).

Zustände pro Room:
  idle       → Caller hört Bot
  music      → Caller hört nur Musik (Transfer läuft)
  briefing   → Agent hört Bot, Caller hört Musik (Warm)
  connected  → Caller hört Agent, Agent hört Caller
  failed     → Zurück zu idle

API:
  POST /v1/transfer           Pipecat → "starte Transfer"
  POST /v1/briefing-complete  Pipecat → "Briefing fertig, verbinde"
  GET  /v1/room/{room}/state  Debug
"""

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from aiohttp import web
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from livekit import api as lk_api
from livekit.api.room_service import (
    UpdateSubscriptionsRequest,
    ListParticipantsRequest,
)
from livekit.api.sip_service import CreateSIPParticipantRequest

# ── Config ────────────────────────────────────────────────────────
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
LIVEKIT_SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", os.getenv("LIVEKIT_SIP_TRUNK_ID", ""))
CONTROLLER_PORT = int(os.getenv("CONTROLLER_PORT", "9100"))
AGENT_RING_TIMEOUT = int(os.getenv("AGENT_RING_TIMEOUT", "40"))  # Sekunden

HTTP_URL = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://").rstrip("/")
MUSIC_TRACK_NAME = "music-hold"


# ══════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════

class Phase(str, Enum):
    IDLE = "idle"
    MUSIC = "music"
    BRIEFING = "briefing"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class Participant:
    identity: str
    track_sids: list[str] = field(default_factory=list)


@dataclass
class RoomState:
    room_name: str
    phase: Phase = Phase.IDLE

    caller: Optional[Participant] = None
    agent: Optional[Participant] = None
    bot: Optional[Participant] = None
    music: Optional[Participant] = None

    music_track_sid: Optional[str] = None
    agent_number: str = ""
    transfer_mode: str = ""  # "cold" | "warm"


# Global state: room_name → RoomState
_rooms: Dict[str, RoomState] = {}
_lk_api: Optional[lk_api.LiveKitAPI] = None


def get_lk() -> lk_api.LiveKitAPI:
    global _lk_api
    if _lk_api is None:
        _lk_api = lk_api.LiveKitAPI(HTTP_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    return _lk_api


# ══════════════════════════════════════════════════════════════════
#  ROOM STATE DISCOVERY
# ══════════════════════════════════════════════════════════════════

async def discover_room(room_name: str) -> RoomState:
    """Scannt den Raum und baut/aktualisiert RoomState."""
    rs = _rooms.get(room_name, RoomState(room_name=room_name))
    _rooms[room_name] = rs

    try:
        resp = await get_lk().room.list_participants(
            ListParticipantsRequest(room=room_name)
        )

        for p in resp.participants:
            identity = p.identity
            track_sids = [t.sid for t in p.tracks]

            if identity.startswith("music-bot-"):
                rs.music = Participant(identity, track_sids)
                for t in p.tracks:
                    if t.name == MUSIC_TRACK_NAME:
                        rs.music_track_sid = t.sid
                        logger.info(f"[{room_name}] Music track: {t.sid}")

            elif identity.startswith("pipecat-bot-"):
                rs.bot = Participant(identity, track_sids)

            elif identity.startswith("agent-"):
                rs.agent = Participant(identity, track_sids)

            elif not identity.startswith("music-bot-") and not identity.startswith("pipecat-bot-") and not identity.startswith("agent-"):
                # SIP-Participant (Caller)
                # SIP bridge creates identities like "sip_<number>_<random>"
                if rs.caller is None:
                    rs.caller = Participant(identity, track_sids)
                    logger.info(f"[{room_name}] Caller identified: {identity}")

    except Exception as e:
        logger.error(f"[{room_name}] discover_room failed: {e}")

    return rs


# ══════════════════════════════════════════════════════════════════
#  SUBSCRIPTION CONTROL
# ══════════════════════════════════════════════════════════════════

async def set_subscriptions(room: str, identity: str, track_sids: list[str], subscribe: bool = True):
    """Setzt welche Tracks ein Participant subscribet bekommt."""
    if not identity:
        logger.warning(f"[{room}] set_subscriptions: no identity")
        return

    audio_sids = [s for s in track_sids if s]  # filter None
    if not audio_sids:
        # Unsubscribe from everything
        audio_sids = []

    try:
        await get_lk().room.update_subscriptions(
            UpdateSubscriptionsRequest(
                room=room,
                identity=identity,
                track_sids=audio_sids,
                subscribe=subscribe,
            )
        )
        logger.info(f"[{room}] {identity} now hears: {audio_sids}")
    except Exception as e:
        logger.error(f"[{room}] set_subscriptions({identity}): {e}")


async def apply_phase_matrix(rs: RoomState):
    """Wendet die Track-Subscription-Matrix für die aktuelle Phase an."""
    if not rs.caller or not rs.caller.identity:
        logger.warning(f"[{rs.room_name}] No caller, can't apply matrix")
        return

    if rs.phase == Phase.IDLE:
        # Caller → Bot
        if rs.bot and rs.bot.track_sids:
            await set_subscriptions(rs.room_name, rs.caller.identity, rs.bot.track_sids)

    elif rs.phase == Phase.MUSIC:
        # Caller → Musik, Agent → nichts (noch nicht da)
        await set_subscriptions(rs.room_name, rs.caller.identity,
                                [rs.music_track_sid] if rs.music_track_sid else [])

    elif rs.phase == Phase.BRIEFING:
        # Caller → Musik, Agent → Bot (Briefing)
        await set_subscriptions(rs.room_name, rs.caller.identity,
                                [rs.music_track_sid] if rs.music_track_sid else [])
        if rs.agent and rs.bot and rs.bot.track_sids:
            await set_subscriptions(rs.room_name, rs.agent.identity, rs.bot.track_sids)

    elif rs.phase == Phase.CONNECTED:
        # Caller → Agent, Agent → Caller
        if rs.agent and rs.agent.track_sids:
            await set_subscriptions(rs.room_name, rs.caller.identity, rs.agent.track_sids)
        if rs.caller and rs.caller.track_sids:
            await set_subscriptions(rs.room_name, rs.agent.identity, rs.caller.track_sids)

    elif rs.phase == Phase.FAILED:
        # Caller → Bot (zurück zum Gespräch)
        if rs.bot and rs.bot.track_sids:
            await set_subscriptions(rs.room_name, rs.caller.identity, rs.bot.track_sids)


# ══════════════════════════════════════════════════════════════════
#  TRANSFER LOGIC
# ══════════════════════════════════════════════════════════════════

async def execute_transfer(room_name: str, target: str, mode: str):
    """
    Führt Transfer aus:
    1. Raum scannen
    2. Caller auf Musik
    3. Agent via SIP anrufen
    4. Je nach Modus: Cold direkt verbinden, Warm auf Briefing warten
    """
    rs = await discover_room(room_name)
    rs.agent_number = target
    rs.transfer_mode = mode

    if mode == "warm":
        rs.phase = Phase.MUSIC  # Erstmal Musik, später BRIEFING wenn Agent da
    else:
        rs.phase = Phase.MUSIC

    _rooms[room_name] = rs

    # Phase 1: Caller auf Musik
    await apply_phase_matrix(rs)

    # Agent via SIP anrufen
    agent_identity = f"agent-{int(time.time())}"
    try:
        await get_lk().sip.create_sip_participant(
            CreateSIPParticipantRequest(
                room_name=room_name,
                sip_trunk_id=LIVEKIT_SIP_TRUNK_ID,
                sip_call_to=target,
                participant_identity=agent_identity,
                participant_name=f"Agent {target}",
                play_ringtone=True,
            )
        )
        logger.info(f"[{room_name}] SIP call to {target} (identity={agent_identity})")

        # Poll for agent to join (max AGENT_RING_TIMEOUT seconds)
        for attempt in range(AGENT_RING_TIMEOUT * 2):
            await asyncio.sleep(0.5)
            rs = await discover_room(room_name)
            if rs.agent and rs.agent.track_sids:
                logger.info(f"[{room_name}] Agent joined: {rs.agent.identity}")
                break
            # Check if last agent joined
            if attempt % 4 == 0:
                logger.info(f"[{room_name}] Waiting for agent... ({attempt/2:.0f}s)")

        if not rs.agent or not rs.agent.track_sids:
            raise TimeoutError(f"Agent joined but no audio track")

    except Exception as e:
        logger.error(f"[{room_name}] Agent call failed: {e}")
        rs.phase = Phase.FAILED
        await apply_phase_matrix(rs)
        return {"status": "failed", "error": str(e)}

    # Agent ist da — Phase wechseln
    if mode == "cold":
        rs.phase = Phase.CONNECTED
        await apply_phase_matrix(rs)
        logger.info(f"[{room_name}] COLD transfer complete: caller ↔ agent")
        return {"status": "connected"}

    else:  # warm
        rs.phase = Phase.BRIEFING
        await apply_phase_matrix(rs)
        logger.info(f"[{room_name}] WARM transfer: briefing phase, waiting for complete signal")
        return {"status": "briefing"}


async def complete_briefing(room_name: str):
    """Pipecat meldet: Briefing fertig → Caller und Agent verbinden."""
    rs = _rooms.get(room_name)
    if not rs:
        return {"status": "error", "error": "Room not found"}

    if rs.phase != Phase.BRIEFING:
        return {"status": "error", "error": f"Not in briefing phase (current: {rs.phase})"}

    rs.phase = Phase.CONNECTED
    await apply_phase_matrix(rs)
    logger.info(f"[{room_name}] WARM transfer complete: caller ↔ agent")
    return {"status": "connected"}


async def cancel_transfer(room_name: str):
    """Bricht Transfer ab, Caller zurück zum Bot."""
    rs = _rooms.get(room_name)
    if not rs:
        return {"status": "error", "error": "Room not found"}

    rs.phase = Phase.IDLE
    await apply_phase_matrix(rs)
    logger.info(f"[{room_name}] Transfer cancelled, caller back to bot")
    return {"status": "idle"}


# ══════════════════════════════════════════════════════════════════
#  MONITOR: Track Changes Auto-Apply
# ══════════════════════════════════════════════════════════════════

async def monitor_room(room_name: str):
    """Pollt regelmäßig den Raum und wendet die Matrix bei Änderungen an."""
    while room_name in _rooms:
        try:
            old_agent_tracks = None
            rs = _rooms.get(room_name)
            if rs and rs.agent:
                old_agent_tracks = set(rs.agent.track_sids)

            rs = await discover_room(room_name)

            # Agent-Track erschienen? → Matrix anwenden
            if old_agent_tracks is not None and rs.agent:
                new_tracks = set(rs.agent.track_sids)
                if new_tracks and new_tracks != old_agent_tracks:
                    logger.info(f"[{room_name}] Agent tracks changed: {old_agent_tracks} → {new_tracks}")
                    await apply_phase_matrix(rs)

        except Exception as e:
            logger.error(f"[{room_name}] Monitor error: {e}")

        await asyncio.sleep(1)

    logger.info(f"[{room_name}] Monitor stopped (room removed)")


# ══════════════════════════════════════════════════════════════════
#  HTTP API
# ══════════════════════════════════════════════════════════════════

async def handle_transfer(request: web.Request):
    """POST /v1/transfer
    Body: {"room": "...", "target": "+49...", "mode": "cold|warm"}
    """
    try:
        data = await request.json()
        room = data["room"]
        target = data["target"]
        mode = data.get("mode", "cold")

        # Starte Monitor falls nicht aktiv
        if room not in _rooms:
            _rooms[room] = RoomState(room_name=room)
            asyncio.create_task(monitor_room(room))

        result = await execute_transfer(room, target, mode)
        return web.json_response(result)

    except Exception as e:
        logger.error(f"handle_transfer error: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=400)


async def handle_briefing_complete(request: web.Request):
    """POST /v1/briefing-complete
    Body: {"room": "..."}
    """
    try:
        data = await request.json()
        room = data["room"]
        result = await complete_briefing(room)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e)}, status=400)


async def handle_cancel(request: web.Request):
    """POST /v1/cancel
    Body: {"room": "..."}
    """
    try:
        data = await request.json()
        room = data["room"]
        result = await cancel_transfer(room)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e)}, status=400)


async def handle_room_state(request: web.Request):
    """GET /v1/room/{room}/state"""
    room = request.match_info.get("room", "")
    rs = _rooms.get(room)
    if not rs:
        return web.json_response({"error": "room not found"}, status=404)

    return web.json_response({
        "room": rs.room_name,
        "phase": rs.phase.value,
        "caller": rs.caller.identity if rs.caller else None,
        "agent": rs.agent.identity if rs.agent else None,
        "music_track": rs.music_track_sid,
        "transfer_mode": rs.transfer_mode,
    })


async def handle_health(request: web.Request):
    return web.json_response({"status": "ok"})


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

async def main():
    app = web.Application()
    app.router.add_post("/v1/transfer", handle_transfer)
    app.router.add_post("/v1/briefing-complete", handle_briefing_complete)
    app.router.add_post("/v1/cancel", handle_cancel)
    app.router.add_get("/v1/room/{room}/state", handle_room_state)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", CONTROLLER_PORT)
    await site.start()

    logger.info(f"Controller listening on http://127.0.0.1:{CONTROLLER_PORT}")
    logger.info(f"LiveKit: {HTTP_URL}")
    logger.info(f"SIP Trunk: {LIVEKIT_SIP_TRUNK_ID or 'NOT SET'}")

    # Keep running
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, stop_event.set)
    await stop_event.wait()

    # Cleanup
    await runner.cleanup()
    if _lk_api:
        await _lk_api.aclose()


if __name__ == "__main__":
    asyncio.run(main())
