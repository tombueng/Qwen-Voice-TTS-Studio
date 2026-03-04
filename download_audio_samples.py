"""Download ~50 public-domain audio samples from Wikimedia Commons.

All files are CC-BY / CC0 / public domain.
Output directory: audio/

Run once:  python download_audio_samples.py
Re-running is safe -- already-downloaded files are skipped.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
UA  = "QwenVoiceTTSStudio/1.0 (audio-sample-downloader)"

# ── Target list ────────────────────────────────────────────────────────────────
# (output_filename, search_query, pick_index)
# pick_index selects which result to grab (0=first, 1=second, ...)
# Prefer .ogg / .wav / .flac; very large files (>20 MB) are filtered out.

TARGETS = [
    # Ambience / backgrounds
    ("forest_ambient.ogg",    "forest ambience sound",       0),
    ("ocean_waves.ogg",       "ocean waves sound",           0),
    ("ocean_waves2.ogg",      "ocean waves sound",           1),
    ("wind_howling.ogg",      "wind howling sound",          0),
    ("thunderstorm.ogg",      "thunderstorm rain sound",     0),
    ("river_stream.ogg",      "river stream water sound",    0),
    ("fireplace.ogg",         "fireplace crackling sound",   0),
    ("crickets_night.ogg",    "crickets night sound",        0),
    ("crickets_night2.ogg",   "crickets night sound",        1),
    ("rain_heavy.ogg",        "heavy rain sound",            0),
    ("rain_light.ogg",        "light rain sound",            0),

    # City / transport
    ("city_traffic.ogg",      "city traffic sound",          0),
    ("city_traffic2.ogg",     "city traffic ambient",        1),
    ("car_horn.ogg",          "car horn beep",               0),
    ("car_horn2.ogg",         "car horn honk",               1),
    ("car_engine.ogg",        "car engine running sound",    0),
    ("car_passing.ogg",       "car passing by sound",        0),
    ("train_passing.ogg",     "train passing sound",         0),
    ("train_station.ogg",     "train station ambient sound", 0),
    ("airplane_overhead.ogg", "airplane jet sound",          0),
    ("helicopter.ogg",        "helicopter sound",            0),
    ("bicycle_bell.ogg",      "bicycle bell ring",           0),

    # Indoor effects
    ("door_knock.ogg",        "door knock sound",            0),
    ("door_creak.ogg",        "door creak sound",            0),
    ("door_slam.ogg",         "door slam sound",             0),
    ("footsteps_gravel.ogg",  "footsteps gravel walk",       0),
    ("footsteps_wood.ogg",    "footsteps wooden floor",      0),
    ("glass_break.ogg",       "glass breaking shatter",      0),
    ("typewriter.ogg",        "typewriter sound",            0),
    ("clock_ticking.ogg",     "clock ticking sound",         0),
    ("phone_ring.ogg",        "telephone ringing sound",     0),
    ("keyboard_typing.ogg",   "keyboard typing sound",       0),
    ("cash_register.ogg",     "cash register sound",         0),

    # Weather
    ("thunder_crack.ogg",     "thunder crack sound",         0),
    ("thunder_rumble.ogg",    "thunder rumble sound",        1),
    ("wind_storm.ogg",        "windstorm sound",             0),

    # Crowd / human
    ("crowd_cheering.ogg",    "crowd cheering applause",     0),
    ("crowd_murmur.ogg",      "crowd murmur talking",        0),
    ("audience_laugh.ogg",    "audience laughing sound",     0),
    ("baby_cry.ogg",          "baby crying sound",           0),
    ("sneeze.ogg",            "sneeze sound",                0),
    ("cough.ogg",             "cough sound",                 0),
    ("children_playing.ogg",  "children playing sound",      0),

    # Animals
    ("dog_bark.ogg",          "dog barking sound",           0),
    ("dog_bark2.ogg",         "dog barking sound",           1),
    ("cat_meow.ogg",          "cat meowing sound",           0),
    ("rooster.ogg",           "rooster crow sound",          0),
    ("crow_caw.ogg",          "crow cawing sound",           0),
    ("wolf_howl.ogg",         "wolf howling sound",          0),
    ("horse_whinny.ogg",      "horse whinny neigh sound",    0),
    ("horse_gallop.ogg",      "horse galloping hooves",      0),
    ("frog_croak.ogg",        "frog croaking sound",         0),
    ("owl_hoot.ogg",          "owl hooting sound",           0),
    ("cow_moo.ogg",           "cow mooing sound",            0),

    # Music / stingers
    ("church_bell.ogg",       "church bell ringing sound",   0),
    ("church_bell2.ogg",      "church bell ringing sound",   1),
    ("gong.ogg",              "gong strike sound",           0),
    ("fanfare.ogg",           "fanfare trumpet sound",       0),
    ("dramatic_sting.ogg",    "dramatic orchestra sting",    0),
]

AUDIO_EXT = {"ogg", "wav", "flac", "oga", "mp3"}
MAX_BYTES  = 20 * 1024 * 1024   # skip files larger than 20 MB


# ── Wikimedia Commons API helpers ──────────────────────────────────────────────

def _api(params: dict) -> dict:
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def search_audio_files(query: str, limit: int = 20) -> list[str]:
    """Search Wikimedia Commons File namespace for audio files matching query.
    Returns a list of file titles, e.g. ['File:Rain.ogg', ...].
    """
    data = _api({
        "action":      "query",
        "list":        "search",
        "srsearch":    query,
        "srnamespace": "6",          # File namespace
        "srlimit":     str(limit),
        "srinfo":      "",
        "srprop":      "snippet",
    })
    titles = []
    for hit in data.get("query", {}).get("search", []):
        title = hit["title"]
        ext   = title.rsplit(".", 1)[-1].lower() if "." in title else ""
        if ext in AUDIO_EXT:
            titles.append(title)
    return titles


def file_info(title: str):
    """Return (url, size_bytes) for a Commons file, or (None, 0) on failure."""
    data = _api({
        "action": "query",
        "titles": title,
        "prop":   "imageinfo",
        "iiprop": "url|size",
    })
    for page in data.get("query", {}).get("pages", {}).values():
        for info in page.get("imageinfo", []):
            return info.get("url"), info.get("size", 0)
    return None, 0


def download_file(url: str, dest: Path, retries: int = 5) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(dest, "wb") as f:
                total = 0
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            return total > 0
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = 10 * (attempt + 1)
                print(f"    429 rate-limited, waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                print(f"    FAIL HTTP {exc.code}: {exc}")
                break
        except Exception as exc:
            print(f"    FAIL download error: {exc}")
            break
    if dest.exists():
        dest.unlink()
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    query_cache: dict[str, list[str]] = {}
    ok = skip = fail = 0

    for output_name, query, pick_idx in TARGETS:
        dest = AUDIO_DIR / output_name

        if dest.exists():
            print(f"  SKIP  {output_name}")
            skip += 1
            continue

        # Fetch search results (cached per query string)
        if query not in query_cache:
            try:
                time.sleep(0.25)
                query_cache[query] = search_audio_files(query, limit=25)
                print(f"  SRCH  '{query}' -> {len(query_cache[query])} result(s)")
            except Exception as exc:
                print(f"  FAIL  search '{query}': {exc}")
                query_cache[query] = []

        titles = query_cache[query]

        # Find a suitable file starting at pick_idx, skipping oversized files
        chosen_title = None
        chosen_url   = None
        checked      = 0
        for title in titles[pick_idx:]:
            checked += 1
            try:
                time.sleep(0.15)
                url, size = file_info(title)
            except Exception as exc:
                print(f"    SKIP  {title}: info error {exc}")
                continue
            if not url:
                continue
            if size > MAX_BYTES:
                print(f"    SKIP  {title}  ({size//1024//1024} MB, too large)")
                continue
            chosen_title = title
            chosen_url   = url
            break
            if checked > 6:
                break

        if not chosen_url:
            print(f"  FAIL  {output_name}: no suitable file found for '{query}'[{pick_idx}:]")
            fail += 1
            continue

        size_hint = f"  {size//1024} KB" if size else ""
        print(f"  GET   {output_name}  <- {chosen_title}{size_hint}")
        time.sleep(1.5)   # be polite to upload.wikimedia.org
        if download_file(chosen_url, dest):
            actual_kb = dest.stat().st_size // 1024
            print(f"  OK    saved {output_name}  ({actual_kb} KB)")
            ok += 1
        else:
            fail += 1

    print()
    print(f"Done -- {ok} downloaded  {skip} skipped  {fail} failed")
    print(f"Files in: {AUDIO_DIR}")


if __name__ == "__main__":
    main()
