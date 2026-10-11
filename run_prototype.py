import os
import sys
import webbrowser
import time
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING QUANTUM AI MEDICAL IMAGING SYSTEM")
    print("Hybrid Classical-Quantum Capstone Prototype")
    print("=" * 60)
    print("Backend URL: http://localhost:8000")
    print("Opening browser dashboard in 2 seconds...")

    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:8000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Change working directory to backend
    os.chdir(os.path.abspath("C:/SAH/backend"))
    sys.path.append(os.path.abspath("C:/SAH/backend"))

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
