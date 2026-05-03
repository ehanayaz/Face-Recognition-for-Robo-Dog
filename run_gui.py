#!/usr/bin/env python3
"""Start the web GUI.

Default bind is loopback only (127.0.0.1). To open from a phone on the same Wi‑Fi:

  PCA_FACE_HOST=0.0.0.0 python run_gui.py

Then browse to http://<this-computer-LAN-IP>:8000 on the mobile device.

Optional: PCA_FACE_PORT=8000  (change if the port is busy)

Your OS firewall must allow inbound TCP on that port when using 0.0.0.0.
"""

import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("PCA_FACE_HOST", "127.0.0.1")
    port = int(os.environ.get("PCA_FACE_PORT", "8000"))
    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=False,
    )
