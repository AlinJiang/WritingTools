import os
import shutil
import subprocess
import sys

# Read-only data files the app loads at runtime via resource_path(). Under
# --onefile PyInstaller unpacks these into sys._MEIPASS; without bundling them
# the frozen exe crashes on startup (e.g. missing options.json -> options=None).
# --add-data uses "src<SEP>dest" where SEP is ';' on Windows, ':' elsewhere.
_DATA_ENTRIES = [
    ("options.json", "."),
    ("icons", "icons"),
    ("locales", "locales"),
    ("background.png", "."),
    ("background_dark.png", "."),
    ("background_popup.png", "."),
    ("background_popup_dark.png", "."),
]


def _add_data_args():
    sep = os.pathsep  # ';' on Windows, ':' on macOS/Linux
    args = []
    for src, dest in _DATA_ENTRIES:
        if os.path.exists(src):
            args += ["--add-data", f"{src}{sep}{dest}"]
        else:
            print(f"WARNING: data file/dir not found, skipping: {src}")
    return args


def run_pyinstaller_build():
    pyinstaller_command = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--icon=icons/app_icon.ico",
        "--name=Writing Tools",
        "--clean",
        "--noconfirm",
        *_add_data_args(),
        # Exclude unnecessary modules
        "--exclude-module", "tkinter",
        "--exclude-module", "unittest",
        "--exclude-module", "IPython",
        "--exclude-module", "jedi",
        "--exclude-module", "email_validator",
        # NOTE: do NOT exclude `cryptography` — google-genai's auth chain pulls
        # it in heavily during `genai.Client()` construction. Excluding it
        # makes the compiled exe crash on startup.
        "--exclude-module", "psutil",
        "--exclude-module", "pyzmq",
        "--exclude-module", "tornado",
        # Exclude modules related to PySide6 that are not used
        "--exclude-module", "PySide6.QtNetwork",
        "--exclude-module", "PySide6.QtXml",
        "--exclude-module", "PySide6.QtQml",
        "--exclude-module", "PySide6.QtQuick",
        "--exclude-module", "PySide6.QtQuickWidgets",
        "--exclude-module", "PySide6.QtPrintSupport",
        "--exclude-module", "PySide6.QtSql",
        "--exclude-module", "PySide6.QtTest",
        "--exclude-module", "PySide6.QtSvg",
        "--exclude-module", "PySide6.QtSvgWidgets",
        "--exclude-module", "PySide6.QtHelp",
        "--exclude-module", "PySide6.QtMultimedia",
        "--exclude-module", "PySide6.QtMultimediaWidgets",
        "--exclude-module", "PySide6.QtOpenGL",
        "--exclude-module", "PySide6.QtOpenGLWidgets",
        "--exclude-module", "PySide6.QtPositioning",
        "--exclude-module", "PySide6.QtLocation",
        "--exclude-module", "PySide6.QtSerialPort",
        "--exclude-module", "PySide6.QtWebChannel",
        "--exclude-module", "PySide6.QtWebSockets",
        "--exclude-module", "PySide6.QtWinExtras",
        "--exclude-module", "PySide6.QtNetworkAuth",
        "--exclude-module", "PySide6.QtRemoteObjects",
        "--exclude-module", "PySide6.QtTextToSpeech",
        "--exclude-module", "PySide6.QtWebEngineCore",
        "--exclude-module", "PySide6.QtWebEngineWidgets",
        "--exclude-module", "PySide6.QtWebEngine",
        "--exclude-module", "PySide6.QtBluetooth",
        "--exclude-module", "PySide6.QtNfc",
        "--exclude-module", "PySide6.QtWebView",
        "--exclude-module", "PySide6.QtCharts",
        "--exclude-module", "PySide6.QtDataVisualization",
        "--exclude-module", "PySide6.QtPdf",
        "--exclude-module", "PySide6.QtPdfWidgets",
        "--exclude-module", "PySide6.QtQuick3D",
        "--exclude-module", "PySide6.QtQuickControls2",
        "--exclude-module", "PySide6.QtQuickParticles",
        "--exclude-module", "PySide6.QtQuickTest",
        "--exclude-module", "PySide6.QtQuickWidgets",
        "--exclude-module", "PySide6.QtSensors",
        "--exclude-module", "PySide6.QtStateMachine",
        "--exclude-module", "PySide6.Qt3DCore",
        "--exclude-module", "PySide6.Qt3DRender",
        "--exclude-module", "PySide6.Qt3DInput",
        "--exclude-module", "PySide6.Qt3DLogic",
        "--exclude-module", "PySide6.Qt3DAnimation",
        "--exclude-module", "PySide6.Qt3DExtras",
        "main.py"
    ]

    def _rmtree(path):
        # Cross-platform (the old `rmdir /s /q` only worked on Windows). Do NOT
        # swallow errors: a locked 'dist' (old exe still running) must surface
        # here with a clear message rather than letting PyInstaller fail later
        # with a cryptic "Access is denied" when it tries to overwrite the exe.
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path)
        except PermissionError:
            raise PermissionError(
                f"Could not remove '{path}' — a previous build's app is likely "
                f"still running and holding a file lock. Quit 'Writing Tools' "
                f"(system tray) and rebuild. On Windows: taskkill /F /IM \"Writing Tools.exe\""
            )

    def _kill_running_app():
        # On Windows, a running --onefile exe locks dist\Writing Tools.exe and
        # blocks the rebuild. Best-effort terminate it first.
        if sys.platform.startswith('win'):
            subprocess.run(
                ['taskkill', '/F', '/IM', 'Writing Tools.exe'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    try:
        # Terminate any running instance so its exe isn't locked, then clean.
        _kill_running_app()
        _rmtree('dist')
        _rmtree('build')
        _rmtree('__pycache__')

        # Run PyInstaller
        subprocess.run(pyinstaller_command, check=True)
        print("Build completed successfully!")

        # Clean up intermediate files (the data files are bundled into the
        # executable via --add-data above, so nothing to copy manually).
        _rmtree('build')
        _rmtree('__pycache__')

    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pyinstaller_build()