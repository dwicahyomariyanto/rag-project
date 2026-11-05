"""Contoh klien SSE untuk endpoint /api/chat/stream.

Jalankan (dari root project, server harus berjalan):

    python examples/stream_client.py "Berapa hari cuti tahunan?"
"""

from __future__ import annotations

import json
import sys

import httpx

BASE_URL = "http://localhost:8000"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Berapa hari cuti tahunan?"
    print(f"Pertanyaan: {question}\n")

    with httpx.stream(
        "POST",
        f"{BASE_URL}/api/chat/stream",
        json={"question": question},
        timeout=120,
    ) as response:
        event = "message"
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
                if event == "sources":
                    print("Sumber konteks:")
                    for citation in data["citations"]:
                        print(
                            f"  [{citation['id']}] {citation['source_name']} "
                            f"hal. {citation['page_start']}"
                        )
                    print("\nJawaban:\n")
                elif event == "token":
                    print(data["delta"], end="", flush=True)
                elif event == "done":
                    stats = data["stats"]
                    print(
                        f"\n\n--- selesai | retrieval {stats['retrieval_ms']} ms | "
                        f"generasi {stats['generation_ms']} ms ---"
                    )
                elif event == "error":
                    print(f"\n[ERROR] {data['detail']}")


if __name__ == "__main__":
    main()
