#!/usr/bin/env python3
"""
LiveKit Music Bot — Dauerschleifen-Musik-Stream für Warm Transfers.

Der Bot joint einen Raum und published ununterbrochen eine WAV-Datei
als Audio-Track. Wer den Track hört, wird über Track-Permissions
vom Transfer-Agenten gesteuert.

Nutzung:
  python music_bot.py --room call-abc123
  python music_bot.py --room call-abc123 --loop-file /path/to/music.wav
"""

import asyncio
import os
import signal
import sys
import time
import argparse
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from livekit import api, rtc

# ── Config ────────────────────────────────────────────────────────
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
DEFAULT_WAV = os.path.join(os.path.dirname(__file__), "music_hold.wav")

# Sample rate for the WAV file (must be 48000 for LiveKit)
SAMPLE_RATE = 48000
NUM_CHANNELS = 1

WAV_HEADER_SIZE = 44


async def generate_music_frames(wav_path: str) -> asyncio.Queue:
    """Liest WAV-Datei und produziert Audio-Frames im Loop."""
    if not os.path.exists(wav_path):
        print(f"WAV not found: {wav_path}, generating silence")
        # Generate 1 second of silence as WAV
        import struct, wave, tempfile
        wav_path = os.path.join(tempfile.gettempdir(), "silence_48000.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"\x00\x00" * SAMPLE_RATE)

    with open(wav_path, "rb") as f:
        raw = f.read()

    # Skip WAV header (44 bytes)
    if raw[:4] == b"RIFF":
        audio_data = raw[WAV_HEADER_SIZE:]
    else:
        audio_data = raw

    # 20ms frames at 16-bit mono = 48000 * 2 bytes / 1000 * 20 = 1920 bytes
    FRAME_SAMPLES = SAMPLE_RATE // 50  # 20ms
    FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit = 2 bytes/sample

    frames = []
    for i in range(0, len(audio_data) - FRAME_BYTES + 1, FRAME_BYTES):
        frames.append(audio_data[i : i + FRAME_BYTES])

    if not frames:
        frames = [b"\x00\x00" * FRAME_SAMPLES]

    return frames, FRAME_SAMPLES


async def run_music_bot(room_name: str, wav_path: str):
    """Hauptlogik: Raum betreten und Musik-Loop senden."""

    # Token generieren
    token = (
        api.AccessToken()
        .with_identity(f"music-bot-{room_name}")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = rtc.Room()
    frames, frame_samples = await generate_music_frames(wav_path)

    audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("music-hold", audio_source)
    track_id: Optional[str] = None

    frame_idx = 0
    running = True

    @room.on("track_published")
    def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        nonlocal track_id
        if publication.sid:
            track_id = publication.sid
            print(f"Music track published: {track_id}")

    @room.on("disconnected")
    def on_disconnected(reason=None):
        nonlocal running
        running = False
        print(f"Music bot disconnected from {room_name}: {reason}")

    def on_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        await room.connect(LIVEKIT_URL, token)
        print(f"Music bot joined room: {room_name}")

        # Publish audio track
        publication = await room.local_participant.publish_track(track)
        print(f"Published music track: {publication.sid}")

        # Audio frame loop — 20ms frames
        frame_duration = 20 / 1000  # 20ms
        while running and room.is_connected():
            chunk = frames[frame_idx % len(frames)]
            frame_idx += 1

            audio_frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=frame_samples,
            )
            await audio_source.capture_frame(audio_frame)
            await asyncio.sleep(frame_duration)

    except Exception as e:
        print(f"Music bot error: {e}")
    finally:
        if room.is_connected():
            await room.disconnect()
        print(f"Music bot left room: {room_name}")


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiveKit Hold Music Bot")
    parser.add_argument("--room", required=True, help="LiveKit room name")
    parser.add_argument("--loop-file", default=DEFAULT_WAV, help="Path to WAV file (48000Hz mono 16bit)")
    args = parser.parse_args()

    asyncio.run(run_music_bot(args.room, args.loop_file))
