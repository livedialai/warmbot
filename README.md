# WarmBot — Warm/Cold Call Transfers mit LiveKit + Pipecat

Track-Permission-basierte Durchstellung ohne moveParticipant, ohne Asterisk, ohne SIP REFER.

## Architektur

```
LiveKit Raum
├── Musik-Bot      ← Dauerschleife WAV
├── Pipecat Bot    ← KI-Agent (STT→LLM→TTS)
├── Anrufer (SIP)  ← voip2gsm → LiveKit SIP Bridge
├── Agent (SIP)    ← via CreateSIPParticipant
│
LiveKit Call Controller (HTTP :9100)
├── update_subscriptions
├── CreateSIPParticipant
├── Participant-Monitoring
└── Phasen-Management
```

## Komponenten

| Datei | Rolle |
|---|---|
| `controller.py` | Regie: Track-Subscriptions, SIP-Outbound, Phasen |
| `transfer_agent.py` | Pipecat Voice Bot: STT/LLM/TTS + Transfer-Tools |
| `music_bot.py` | Audio-Publisher: WAV-Loop in den Raum |
| `run.sh` | Startet alle drei Komponenten |

## Setup

```bash
cp .env.example .env
# Keys eintragen
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Start

```bash
./run.sh <room_name>
```

## API (Controller)

| Endpoint | Beschreibung |
|---|---|
| `POST /v1/transfer` | Transfer starten `{"room":"...", "target":"+49...", "mode":"cold|warm"}` |
| `POST /v1/briefing-complete` | Warm-Transfer abschließen |
| `POST /v1/cancel` | Transfer abbrechen |
| `GET /v1/room/{room}/state` | Raum-Status |

## Transfer-Modi

**Cold:** Caller → Musik → Agent joint → Caller auf Agent → Bot raus

**Warm:** Caller → Musik → Agent joint → Bot brieft Agent → Caller auf Agent → Bot raus
