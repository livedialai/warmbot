#!/usr/bin/env python3
"""
LiveKit Call Controller v2 — Harte Subscription-Matrix, async Transfer, voller Snapshot.

Jeder Participant hört EXAKT die freigegebenen Tracks — nichts zusätzlich.
Die Matrix wird bei jeder Zustandsänderung vollständig neu angewendet.

API (HTTP localhost:9100):
  POST /v1/transfer           → startet async Transfer
  POST /v1/briefing-complete  → schließt Warm-Transfer ab
  POST /v1/cancel             → bricht ab, Caller zurück zum Bot
  POST /v1/disconnect-bot     → entfernt Pipecat-Bot aus Raum
  GET  /v1/room/{room}/state  → Raum-Status
"""

import asyncio
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

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
LIVEKIT_SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "")
CONTROLLER_PORT = int(os.getenv("CONTROLLER_PORT", "9100"))
AGENT_RING_TIMEOUT = int(os.getenv("AGENT_RING_TIMEOUT", "40"))

HTTP_URL = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://").rstrip("/")
MUSIC_TRACK_NAME = "music-hold"


# ══════════════════════════════════════════════════════════════════
#  TYPES
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
    track_sids: List[str] = field(default_factory=list)


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
    transfer_mode: str = ""
    transferring: bool = False  # async transfer in progress


_rooms: Dict[str, RoomState] = {}
_lk_api: Optional[lk_api.LiveKitAPI] = None


def get_lk() -> lk_api.LiveKitAPI:
    global _lk_api
    if _lk_api is None:
        _lk_api = lk_api.LiveKitAPI(HTTP_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    return _lk_api


# ══════════════════════════════════════════════════════════════════
#  ROOM DISCOVERY — aktualisiert ALLE Participants
# ══════════════════════════════════════════════════════════════════

async def discover_room(room_name: str) -> RoomState:
    """Scannt den Raum und aktualisiert ALLE Participants (nicht nur erste Sichtung)."""
    rs = _rooms.get(room_name)
    if not rs:
        rs = RoomState(room_name=room_name)
        _rooms[room_name] = rs

    # Reset current state from room
    rs.caller = None
    rs.agent = None
    rs.bot = None
    rs.music = None
    rs.music_track_sid = None

    # Resolve active agent identity
    agent_identity = rs.agent_number and f"agent-{rs.agent_number}" or ""

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

            elif identity.startswith("pipecat-bot-"):
                rs.bot = Participant(identity, track_sids)

            elif identity.startswith("agent-"):
                # Only recognize the CURRENT agent for this transfer
                if rs.agent_number and identity == f"agent-{int(time.time()) if rs.agent_number else ''}":
                    pass  # identity comes from SIP create, match below
                rs.agent = Participant(identity, track_sids)

            else:
                # Any other = Caller (SIP participant)
                rs.caller = Participant(identity, track_sids)

    except Exception as e:
        logger.error(f"[{room_name}] discover_room: {e}")

    logger.debug(
        f"[{room_name}] Discovered: caller={rs.caller.identity if rs.caller else '?'} "
        f"bot={rs.bot.identity if rs.bot else '?'} "
        f"agent={rs.agent.identity if rs.agent else '?'} "
        f"music={rs.music_track_sid or '?'}"
    )
    return rs


# ══════════════════════════════════════════════════════════════════
#  HARTE SUBSCRIPTION-MATRIX
# ══════════════════════════════════════════════════════════════════

def all_known_tracks(rs: RoomState) -> List[str]:
    """Alle bekannten Track-SIDs im Raum."""
    tracks: List[str] = []
    for p in [rs.caller, rs.agent, rs.bot, rs.music]:
        if p:
            tracks.extend(p.track_sids)
    if rs.music_track_sid:
        tracks.append(rs.music_track_sid)
    return list(set(t for t in tracks if t))


async def set_subscriptions(room: str, identity: str,
                            track_sids: List[str], subscribe: bool):
    """Setzt Subscriptions für einen Participant."""
    if not identity or not track_sids:
        return
    try:
        await get_lk().room.update_subscriptions(
            UpdateSubscriptionsRequest(
                room=room,
                identity=identity,
                track_sids=track_sids,
                subscribe=subscribe,
            )
        )
        action = "subscribed" if subscribe else "unsubscribed"
        logger.info(f"[{room}] {identity} {action}: {track_sids}")
    except Exception as e:
        logger.error(f"[{room}] set_subscriptions({identity}): {e}")


async def set_exact_hears(rs: RoomState, listener: Optional[Participant],
                          desired_sids: List[str]):
    """Garantiert: Listener hört EXAKT desired_sids — nichts anderes."""
    if not listener or not listener.identity:
        return

    all_tracks = all_known_tracks(rs)
    desired = [s for s in desired_sids if s]
    unwanted = [s for s in all_tracks if s not in desired]

    if unwanted:
        await set_subscriptions(rs.room_name, listener.identity, unwanted, subscribe=False)
    if desired:
        await set_subscriptions(rs.room_name, listener.identity, desired, subscribe=True)


# ══════════════════════════════════════════════════════════════════
#  VOLLSTÄNDIGE PHASEN-MATRIX
# ══════════════════════════════════════════════════════════════════

async def apply_phase_matrix(rs: RoomState):
    """Wendet die komplette Hör-Matrix für ALLE Participants an."""

    m = rs.music_track_sid
    c_tracks = rs.caller.track_sids if rs.caller else []
    a_tracks = rs.agent.track_sids if rs.agent else []
    b_tracks = rs.bot.track_sids if rs.bot else []

    if rs.phase == Phase.IDLE:
        # Caller ↔ Bot, Music hört nichts, Agent nicht da
        await set_exact_hears(rs, rs.caller, b_tracks)
        await set_exact_hears(rs, rs.bot, c_tracks)
        await set_exact_hears(rs, rs.music, [])

    elif rs.phase == Phase.MUSIC:
        # Caller → Musik, Bot → Caller (optional, für Unterbrechbarkeit), Music → nichts
        await set_exact_hears(rs, rs.caller, [m] if m else [])
        await set_exact_hears(rs, rs.bot, c_tracks)
        await set_exact_hears(rs, rs.music, [])

    elif rs.phase == Phase.BRIEFING:
        # Caller → Musik, Agent ↔ Bot, Music → nichts
        await set_exact_hears(rs, rs.caller, [m] if m else [])
        await set_exact_hears(rs, rs.agent, b_tracks)
        await set_exact_hears(rs, rs.bot, a_tracks)
        await set_exact_hears(rs, rs.music, [])

    elif rs.phase == Phase.CONNECTED:
        # Caller ↔ Agent, Bot → nichts, Music → nichts
        await set_exact_hears(rs, rs.caller, a_tracks)
        await set_exact_hears(rs, rs.agent, c_tracks)
        await set_exact_hears(rs, rs.bot, [])
        await set_exact_hears(rs, rs.music, [])

    elif rs.phase == Phase.FAILED:
        # Caller ↔ Bot, alles andere stumm
        await set_exact_hears(rs, rs.caller, b_tracks)
        await set_exact_hears(rs, rs.bot, c_tracks)
        await set_exact_hears(rs, rs.agent, [])
        await set_exact_hears(rs, rs.music, [])

    logger.info(f"[{rs.room_name}] Matrix applied: phase={rs.phase.value}")


def snapshot(rs: RoomState) -> str:
    """Vergleichbarer Snapshot für Change Detection."""
    parts = [
        rs.phase.value,
        rs.caller.identity if rs.caller else "",
        ",".join(sorted(rs.caller.track_sids)) if rs.caller else "",
        rs.agent.identity if rs.agent else "",
        ",".join(sorted(rs.agent.track_sids)) if rs.agent else "",
        rs.bot.identity if rs.bot else "",
        ",".join(sorted(rs.bot.track_sids)) if rs.bot else "",
        rs.music.identity if rs.music else "",
        rs.music_track_sid or "",
    ]
    return "|".join(parts)


# ══════════════════════════════════════════════════════════════════
#  ASYNC TRANSFER — blockiert HTTP nicht
# ══════════════════════════════════════════════════════════════════

async def execute_transfer(room_name: str, target: str, mode: str):
    """
    Führt Transfer async aus. Wird von handle_transfer via create_task gestartet.

    Ablauf:
    1. Phase → MUSIC, Matrix anwenden
    2. Agent via SIP anrufen
    3. Pollen bis Agent da ist
    4. Cold → CONNECTED, Warm → BRIEFING
    """

    rs = _rooms.get(room_name)
    if not rs:
        logger.error(f"[{room_name}] Room not found for transfer")
        return

    # Reset für neuen Transfer
    rs.agent = None
    rs.agent_number = target
    rs.transfer_mode = mode
    rs.transferring = True

    rs.phase = Phase.MUSIC
    await apply_phase_matrix(rs)

    # Agent via SIP anrufen
    agent_identity = f"agent-{uuid.uuid4().hex[:12]}"
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
        logger.info(f"[{room_name}] SIP → {target} ({agent_identity})")

    except Exception as e:
        logger.error(f"[{room_name}] SIP create failed: {e}")
        rs.phase = Phase.FAILED
        rs.transferring = False
        await apply_phase_matrix(rs)
        return

    # Poll for agent (bis AGENT_RING_TIMEOUT)
    for attempt in range(AGENT_RING_TIMEOUT * 2):
        await asyncio.sleep(0.5)

        # Check if agent joined
        try:
            resp = await get_lk().room.list_participants(
                ListParticipantsRequest(room=room_name)
            )
            for p in resp.participants:
                if p.identity == agent_identity:
                    for t in p.tracks:
                        if str(t.type) == "AUDIO" or "audio" in str(getattr(t, 'mime_type', '')):
                            # Agent ist da — discover und Matrix anwenden
                            rs = await discover_room(room_name)

                            if mode == "cold":
                                rs.phase = Phase.CONNECTED
                            else:
                                rs.phase = Phase.BRIEFING

                            rs.transferring = False
                            await apply_phase_matrix(rs)
                            logger.info(f"[{room_name}] Transfer phase: {rs.phase.value}")
                            return
        except Exception:
            pass

    # Timeout
    logger.warning(f"[{room_name}] Agent {target} did not answer")
    rs.phase = Phase.FAILED
    rs.transferring = False
    await apply_phase_matrix(rs)


# ══════════════════════════════════════════════════════════════════
#  TRANSFER ACTIONS
# ══════════════════════════════════════════════════════════════════

async def complete_briefing(room_name: str):
    rs = _rooms.get(room_name)
    if not rs or rs.phase != Phase.BRIEFING:
        return {"status": "error", "error": "not in briefing phase"}
    rs.phase = Phase.CONNECTED
    await apply_phase_matrix(rs)
    return {"status": "connected"}


async def cancel_transfer(room_name: str):
    rs = _rooms.get(room_name)
    if not rs:
        return {"status": "error", "error": "room not found"}
    rs.phase = Phase.IDLE
    rs.transferring = False
    await apply_phase_matrix(rs)
    return {"status": "idle"}


async def disconnect_bot(room_name: str):
    """Entfernt den Pipecat-Bot aus dem Raum (serverseitig)."""
    rs = _rooms.get(room_name)
    if not rs or not rs.bot:
        return {"status": "error", "error": "no bot in room"}
    try:
        await get_lk().room.remove_participant(
            lk_api.room_service.RoomParticipantIdentity(
                room=room_name,
                identity=rs.bot.identity,
            )
        )
        logger.info(f"[{room_name}] Bot removed: {rs.bot.identity}")
        rs.bot = None
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[{room_name}] Bot removal failed: {e}")
        return {"status": "error", "error": str(e)}


# ══════════════════════════════════════════════════════════════════
#  MONITOR — Snapshot-Vergleich
# ══════════════════════════════════════════════════════════════════

async def monitor_room(room_name: str):
    """Pollt den Raum und wendet Matrix bei jeder Änderung an."""
    last_snapshot = ""

    while room_name in _rooms:
        try:
            rs = await discover_room(room_name)
            current = snapshot(rs)

            if current != last_snapshot:
                # Auto-apply wenn nicht mitten im async transfer
                if not rs.transferring:
                    await apply_phase_matrix(rs)
                last_snapshot = current

        except Exception as e:
            logger.error(f"[{room_name}] Monitor: {e}")

        await asyncio.sleep(0.5)

    logger.info(f"[{room_name}] Monitor stopped")


# ══════════════════════════════════════════════════════════════════
#  HTTP API
# ══════════════════════════════════════════════════════════════════

async def handle_transfer(request: web.Request):
    """POST /v1/transfer — non-blocking."""
    try:
        data = await request.json()
        room = data["room"]
        target = data["target"]
        mode = data.get("mode", "cold")

        if room not in _rooms:
            _rooms[room] = RoomState(room_name=room)
            asyncio.create_task(monitor_room(room))

        rs = _rooms[room]

        # Wenn schon ein Transfer läuft
        if rs.transferring:
            return web.json_response({"status": "error", "error": "transfer in progress"})

        # Async starten
        asyncio.create_task(execute_transfer(room, target, mode))
        return web.json_response({"status": "transfer_started"})

    except Exception as e:
        logger.error(f"handle_transfer: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=400)


async def handle_briefing_complete(request: web.Request):
    try:
        data = await request.json()
        return web.json_response(await complete_briefing(data["room"]))
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e)}, status=400)


async def handle_cancel(request: web.Request):
    try:
        data = await request.json()
        return web.json_response(await cancel_transfer(data["room"]))
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e)}, status=400)


async def handle_disconnect_bot(request: web.Request):
    try:
        data = await request.json()
        return web.json_response(await disconnect_bot(data["room"]))
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e)}, status=400)


async def handle_room_state(request: web.Request):
    room = request.match_info.get("room", "")
    rs = _rooms.get(room)
    if not rs:
        return web.json_response({"error": "room not found"}, status=404)
    return web.json_response({
        "room": rs.room_name,
        "phase": rs.phase.value,
        "transferring": rs.transferring,
        "transfer_mode": rs.transfer_mode,
        "agent_number": rs.agent_number,
        "caller": rs.caller.identity if rs.caller else None,
        "caller_tracks": rs.caller.track_sids if rs.caller else [],
        "agent": rs.agent.identity if rs.agent else None,
        "agent_tracks": rs.agent.track_sids if rs.agent else [],
        "bot": rs.bot.identity if rs.bot else None,
        "bot_tracks": rs.bot.track_sids if rs.bot else [],
        "music_track": rs.music_track_sid,
    })


async def handle_health(request: web.Request):
    return web.json_response({"status": "ok"})


# ══════════════════════════════════════════════════════════════════

async def main():
    app = web.Application()
    app.router.add_post("/v1/transfer", handle_transfer)
    app.router.add_post("/v1/briefing-complete", handle_briefing_complete)
    app.router.add_post("/v1/cancel", handle_cancel)
    app.router.add_post("/v1/disconnect-bot", handle_disconnect_bot)
    app.router.add_get("/v1/room/{room}/state", handle_room_state)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", CONTROLLER_PORT)
    await site.start()

    logger.info(f"Controller v2 → http://127.0.0.1:{CONTROLLER_PORT}")
    logger.info(f"LiveKit: {HTTP_URL}")
    logger.info(f"SIP Trunk: {LIVEKIT_SIP_TRUNK_ID or '⚠ NOT SET'}")

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(sig, stop.set)
    await stop.wait()

    await runner.cleanup()
    if _lk_api:
        await _lk_api.aclose()


if __name__ == "__main__":
    asyncio.run(main())
