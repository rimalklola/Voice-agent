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
    from src.utils.sip_server import sip_server
    from src.utils.whatsapp_calling import whatsapp_server

    app = create_app()
    port = int(os.getenv("PORT", "3000"))

    # Generic SIP server — accepts calls from MicroSIP and any SIP client.
    sip_server.start()

    # WhatsApp Business SIP — only starts when META_SIP_USERNAME is configured.
    whatsapp_server.start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        sip_server.stop()
        whatsapp_server.stop()


if __name__ == "__main__":
    main()