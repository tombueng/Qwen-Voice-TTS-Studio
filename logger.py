"""Shared logging infrastructure for Qwen Voice TTS Studio.

Uses print(flush=True) instead of Python's logging module because Gradio and
transformers routinely reconfigure the root logger and redirect stderr, which
swallows normal logging output.  print() to stdout is always visible.
"""

import ctypes as _ctypes

# Enable ANSI/VT100 colour codes on Windows 10+
# (ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
try:
    _k32 = _ctypes.windll.kernel32
    _k32.SetConsoleMode(_k32.GetStdHandle(-11), 0x0001 | 0x0004)
except Exception:
    pass


# ── Colour helpers ─────────────────────────────────────────────────────────────

def dim(s):    return f"\033[2m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def cyan(s):   return f"\033[36m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def magenta(s): return f"\033[35m{s}\033[0m"
def white(s):  return f"\033[97m{s}\033[0m"


# ── Logger ─────────────────────────────────────────────────────────────────────

class _Log:
    """Thin print()-based logger — bypasses the logging module entirely."""

    # icon, label colour, message colour
    _LVL = {
        #        icon    label_colour    label    msg_colour
        "DBG":  ("🔍",  "\033[2;37m",  "DBG  ", "\033[2;37m"),
        "INFO": ("💬",  "\033[0;96m",  "INFO ", "\033[0;97m"),
        "OK":   ("✅",  "\033[0;92m",  "OK   ", "\033[0;92m"),
        "WARN": ("⚠️ ", "\033[0;33m",  "WARN ", "\033[0;33m"),
        "ERR":  ("❌",  "\033[0;91m",  "ERROR", "\033[0;91m"),
        "TIME": ("⏱️ ", "\033[0;35m",  "TIME ", "\033[0;35m"),
    }

    def _emit(self, lvl: str, msg: str):
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%H:%M:%S")
        icon, label_colour, label, msg_colour = self._LVL[lvl]
        ts_str  = f"\033[2;37m{ts}\033[0m"
        lvl_str = f"{label_colour}{label}\033[0m"
        msg_str = f"{msg_colour}{msg}\033[0m"
        print(f"{ts_str}  {icon}  {lvl_str}  {msg_str}", flush=True)

    def debug(self, msg):   self._emit("DBG",  str(msg))
    def info(self, msg):    self._emit("INFO", str(msg))
    def ok(self, msg):      self._emit("OK",   str(msg))
    def warning(self, msg): self._emit("WARN", str(msg))
    def error(self, msg):   self._emit("ERR",  str(msg))
    def timing(self, msg):  self._emit("TIME", str(msg))


log = _Log()


# ── Section banners ────────────────────────────────────────────────────────────

_SECTION_ICONS = {
    "TTS Generate":         "🎙️",
    "Clone Voice":          "🧬",
    "Clone Voice (JSON)":   "🧬",
    "Design Voice":         "🎨",
    "Design Voice (JSON)":  "🎨",
    "Transcribe":           "📝",
    "System Configuration": "⚙️",
    "Dependency Check":     "📦",
    "Pre-loading Models":   "🤖",
}

_DEFAULT_ICON = "▶"


def section(title: str, width: int = 62):
    """Print a colourful section-start banner."""
    icon = _SECTION_ICONS.get(title, _DEFAULT_ICON)
    header = f" {icon}  {title} "
    bar = "─" * max(0, width - len(header))
    print(f"\033[1;96m╭{'─' * (width)}\033[0m", flush=True)
    print(f"\033[1;96m│\033[0m\033[1;97m{header}\033[2;37m{bar}\033[0m", flush=True)
    print(f"\033[1;96m╰{'─' * (width)}\033[0m", flush=True)


def section_end(label: str = ""):
    """Print a section-end banner."""
    if label in ("done", "ok"):
        colour = "\033[0;92m"   # green
        icon   = "✅"
    elif label in ("error", "failed", "FAILED"):
        colour = "\033[0;91m"   # red
        icon   = "❌"
    else:
        colour = "\033[2;37m"
        icon   = "·"
    suffix = f"  {icon}  {colour}{label}\033[0m" if label else ""
    print(f"\033[1;96m{'─' * 20}\033[0m{suffix}", flush=True)
