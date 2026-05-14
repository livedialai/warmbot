#!/bin/bash
# Startet das Transfer-System: Controller + Musik-Bot + Pipecat Agent
# Usage: ./run.sh <room_name> [wav_path]

set -e

ROOM="${1:?Usage: $0 <room_name> [wav_path]}"
WAV="${2:-./music_hold.wav}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Transfer System — Room: $ROOM ==="

# 1. Controller (Hintergrund)
echo "[1/3] Starting LiveKit Controller..."
python3 "$SCRIPT_DIR/controller.py" &
CONTROLLER_PID=$!
sleep 2
echo "  Controller PID: $CONTROLLER_PID"

# 2. Music Bot (Hintergrund)
echo "[2/3] Starting Music Bot..."
python3 "$SCRIPT_DIR/music_bot.py" --room "$ROOM" --loop-file "$WAV" &
MUSIC_PID=$!
sleep 3
echo "  Music Bot PID: $MUSIC_PID"

# 3. Pipecat Agent (Vordergrund)
echo "[3/3] Starting Pipecat Agent..."
python3 "$SCRIPT_DIR/transfer_agent.py" "$ROOM" &
AGENT_PID=$!
echo "  Agent PID: $AGENT_PID"

cleanup() {
    echo "Shutting down..."
    kill $AGENT_PID 2>/dev/null || true
    kill $MUSIC_PID 2>/dev/null || true
    kill $CONTROLLER_PID 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
