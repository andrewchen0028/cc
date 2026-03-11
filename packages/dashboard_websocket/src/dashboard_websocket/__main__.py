"""
Launch generator and Dash app together.

    python -m dashboard_websocket
"""

import subprocess
import sys
import time


def main() -> None:
    gen = subprocess.Popen([sys.executable, "-m", "dashboard_websocket.generator"])
    time.sleep(0.5)  # let the server start
    try:
        subprocess.run([sys.executable, "-m", "dashboard_websocket.app"])
    finally:
        gen.terminate()
        gen.wait()


if __name__ == "__main__":
    main()
