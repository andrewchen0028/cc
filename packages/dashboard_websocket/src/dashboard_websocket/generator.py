"""
WebSocket server that streams a random walk at ~2 Hz.

Run standalone:
    python -m dashboard_websocket.generator
"""

import asyncio
import json
import random
from datetime import datetime, timezone

import websockets


async def _stream(ws: websockets.ServerConnection) -> None:
    value = 100.0
    while True:
        value += random.gauss(0, 1)
        payload = json.dumps({
            "t": datetime.now(timezone.utc).isoformat(),
            "v": round(value, 4),
        })
        await ws.send(payload)
        await asyncio.sleep(0.5)


async def main() -> None:
    print("Generator listening on ws://localhost:8765")
    async with websockets.serve(_stream, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
