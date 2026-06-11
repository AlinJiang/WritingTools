"""
Path helpers that work both when running from source and when frozen by
PyInstaller (--onefile).

Two distinct path roots are needed:

1. resource_path(rel) — read-only assets bundled into the build (icons,
   background images, locales, the default options.json). Under a PyInstaller
   --onefile build these are unpacked to a temp dir exposed as sys._MEIPASS;
   from source they sit next to the .py files. Using sys.argv[0] for these is
   wrong under onefile — it points at the exe, not the unpacked bundle.

2. user_data_path(rel) — writable files the app reads AND writes at runtime
   (config.json, the user's edited options.json). These must live in a stable,
   writable location next to the executable (or the source tree), NOT in the
   _MEIPASS temp dir, which is deleted when the app exits.
"""

import os
import sys


def _meipass_base() -> str:
    """The directory bundled read-only resources live in."""
    # PyInstaller sets sys._MEIPASS to the unpacked-bundle dir under --onefile,
    # and to the app dir under --onedir.
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return bundled
    # Running from source: resources sit next to this module.
    return os.path.dirname(os.path.abspath(__file__))


def _writable_base() -> str:
    """The directory user-writable files live in."""
    if getattr(sys, "frozen", False):
        # Frozen: write next to the executable so settings persist across runs.
        return os.path.dirname(sys.executable)
    # Running from source: alongside the .py files.
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(*rel_parts: str) -> str:
    """Absolute path to a bundled read-only resource (icons, locales, etc.)."""
    return os.path.join(_meipass_base(), *rel_parts)


def user_data_path(*rel_parts: str) -> str:
    """Absolute path to a writable runtime file (config.json, options.json)."""
    return os.path.join(_writable_base(), *rel_parts)
