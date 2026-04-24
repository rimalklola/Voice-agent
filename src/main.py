from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    load_dotenv()
    from src.utils.realtime import create_app
    from src.utils.whatsapp_calling import whatsapp_server

    app = create_app()
    port = int(os.getenv("PORT", "3000"))

    # Start the WhatsApp SIP server in its own threads (pyVoIP is sync/threaded).
    # It listens for incoming SIP INVITEs from Meta and bridges each call to Gemini Live.
    # Set META_SIP_USERNAME + META_SIP_PASSWORD to enable; leave blank to skip.
    whatsapp_server.start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        whatsapp_server.stop()


if __name__ == "__main__":
    main()