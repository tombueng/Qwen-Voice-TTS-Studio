"""Download ~100 sound effects from Pixabay (free, royalty-free).

Requirements:
  - Free Pixabay API key — register at https://pixabay.com/api/docs/
    and copy your key from the "Your API Key" section.

Usage:
  python download_pixabay_sounds.py YOUR_API_KEY
  -- or --
  set PIXABAY_KEY=YOUR_API_KEY
  python download_pixabay_sounds.py

Output directory: audio/
Re-running is safe -- already-downloaded files are skipped.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

API_BASE = "https://pixabay.com/api/sounds/"
UA = "QwenVoiceTTSStudio/1.0 (pixabay-sound-downloader)"

# ── Search targets ─────────────────────────────────────────────────────────────
# (output_filename, query)
# One file will be downloaded per entry (the highest-rated result).

TARGETS = [
    # Nature / ambience
    ("ambient_forest.mp3",       "forest ambience"),
    ("ambient_ocean.mp3",        "ocean waves"),
    ("ambient_rain.mp3",         "rain"),
    ("ambient_rain_heavy.mp3",   "heavy rain"),
    ("ambient_rain_light.mp3",   "light rain"),
    ("ambient_thunder.mp3",      "thunderstorm"),
    ("ambient_wind.mp3",         "wind"),
    ("ambient_river.mp3",        "river stream"),
    ("ambient_fireplace.mp3",    "fireplace crackling"),
    ("ambient_crickets.mp3",     "crickets night"),
    ("ambient_birds.mp3",        "birds singing"),
    ("ambient_jungle.mp3",       "jungle ambience"),
    ("ambient_cave.mp3",         "cave dripping"),
    ("ambient_underwater.mp3",   "underwater bubbles"),

    # City / transport
    ("city_traffic.mp3",         "city traffic"),
    ("city_crowd.mp3",           "city crowd"),
    ("city_market.mp3",          "market crowd"),
    ("car_horn.mp3",             "car horn"),
    ("car_engine.mp3",           "car engine"),
    ("car_pass.mp3",             "car passing"),
    ("car_brake.mp3",            "car brake screech"),
    ("car_crash.mp3",            "car crash"),
    ("train_pass.mp3",           "train passing"),
    ("train_station.mp3",        "train station"),
    ("airplane.mp3",             "airplane jet"),
    ("helicopter.mp3",           "helicopter"),
    ("bicycle_bell.mp3",         "bicycle bell"),
    ("motorcycle.mp3",           "motorcycle"),

    # Indoor / office / home
    ("door_knock.mp3",           "door knock"),
    ("door_creak.mp3",           "door creak"),
    ("door_slam.mp3",            "door slam"),
    ("door_bell.mp3",            "doorbell"),
    ("footsteps_wood.mp3",       "footsteps wood"),
    ("footsteps_gravel.mp3",     "footsteps gravel"),
    ("glass_break.mp3",          "glass breaking"),
    ("typewriter.mp3",           "typewriter"),
    ("clock_ticking.mp3",        "clock ticking"),
    ("phone_ring.mp3",           "phone ringing"),
    ("phone_notification.mp3",   "phone notification"),
    ("keyboard_typing.mp3",      "keyboard typing"),
    ("cash_register.mp3",        "cash register"),
    ("microwave_beep.mp3",       "microwave beep"),
    ("tv_static.mp3",            "tv static noise"),

    # Weather
    ("weather_thunder.mp3",      "thunder crack"),
    ("weather_hail.mp3",         "hail"),
    ("weather_storm.mp3",        "storm wind"),
    ("weather_blizzard.mp3",     "blizzard wind"),

    # Crowd / human
    ("crowd_cheer.mp3",          "crowd cheering"),
    ("crowd_murmur.mp3",         "crowd murmur"),
    ("crowd_laugh.mp3",          "audience laughing"),
    ("crowd_applause.mp3",       "applause clapping"),
    ("crowd_boo.mp3",            "crowd booing"),
    ("baby_cry.mp3",             "baby crying"),
    ("sneeze.mp3",               "sneeze"),
    ("cough.mp3",                "cough"),
    ("children_play.mp3",        "children playing"),
    ("people_talking.mp3",       "people talking indoors"),
    ("whisper.mp3",              "whisper"),

    # Animals
    ("animal_dog_bark.mp3",      "dog barking"),
    ("animal_dog_growl.mp3",     "dog growling"),
    ("animal_cat_meow.mp3",      "cat meowing"),
    ("animal_cat_purr.mp3",      "cat purring"),
    ("animal_rooster.mp3",       "rooster crow"),
    ("animal_cow.mp3",           "cow mooing"),
    ("animal_horse.mp3",         "horse whinny"),
    ("animal_horse_gallop.mp3",  "horse galloping"),
    ("animal_wolf.mp3",          "wolf howl"),
    ("animal_lion.mp3",          "lion roar"),
    ("animal_elephant.mp3",      "elephant trumpet"),
    ("animal_frog.mp3",          "frog croaking"),
    ("animal_owl.mp3",           "owl hooting"),
    ("animal_crow.mp3",          "crow cawing"),
    ("animal_duck.mp3",          "duck quacking"),
    ("animal_bee.mp3",           "bee buzzing"),

    # UI / interface
    ("ui_click.mp3",             "mouse click"),
    ("ui_error.mp3",             "error beep"),
    ("ui_success.mp3",           "success chime"),
    ("ui_notification.mp3",      "notification sound"),
    ("ui_swoosh.mp3",            "swoosh sound"),

    # Impacts / effects
    ("impact_explosion.mp3",     "explosion"),
    ("impact_gunshot.mp3",       "gunshot"),
    ("impact_punch.mp3",         "punch hit"),
    ("impact_metal.mp3",         "metal impact"),
    ("impact_wood.mp3",          "wood impact"),
    ("impact_thud.mp3",          "thud impact"),

    # Music stingers / ambience
    ("music_church_bell.mp3",    "church bell"),
    ("music_gong.mp3",           "gong"),
    ("music_fanfare.mp3",        "fanfare trumpet"),
    ("music_dramatic.mp3",       "dramatic sting"),
    ("music_horror.mp3",         "horror ambience"),
    ("music_suspense.mp3",       "suspense music"),
    ("music_upbeat.mp3",         "upbeat music loop"),
    ("music_jazz.mp3",           "jazz cafe"),
    ("music_piano.mp3",          "piano melody"),

    # Sci-fi / game
    ("scifi_laser.mp3",          "laser shot"),
    ("scifi_spaceship.mp3",      "spaceship hum"),
    ("scifi_alarm.mp3",          "alarm siren"),
    ("scifi_robot.mp3",          "robot sound"),
    ("scifi_teleport.mp3",       "teleport sound"),
]


# ── API helpers ────────────────────────────────────────────────────────────────

def search_sound(api_key: str, query: str) -> dict | None:
    """Return the first (highest-rated) sound hit for the query, or None."""
    params = {
        "key":       api_key,
        "q":         query,
        "per_page":  "3",
        "page":      "1",
    }
    url = API_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        hits = data.get("hits", [])
        return hits[0] if hits else None
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            print("ERROR: Invalid API key — get a free key at https://pixabay.com/api/docs/")
            sys.exit(1)
        print(f"  HTTP {exc.code} for query '{query}'")
        return None
    except Exception as exc:
        print(f"  Error searching '{query}': {exc}")
        return None


def download_file(url: str, dest: Path, retries: int = 4) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
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
                wait = 15 * (attempt + 1)
                print(f"    429 rate-limited -- waiting {wait}s (attempt {attempt + 1}/{retries})...")
                time.sleep(wait)
            else:
                print(f"    FAIL HTTP {exc.code}")
                break
        except Exception as exc:
            print(f"    FAIL {exc}")
            break
    if dest.exists():
        dest.unlink()
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Resolve API key
    api_key = None
    if len(sys.argv) > 1:
        api_key = sys.argv[1].strip()
    if not api_key:
        api_key = os.environ.get("PIXABAY_KEY", "").strip()
    if not api_key:
        print(__doc__)
        print("ERROR: No API key provided.")
        print("  Usage:  python download_pixabay_sounds.py YOUR_API_KEY")
        print("  Or set: PIXABAY_KEY=YOUR_API_KEY")
        sys.exit(1)

    print(f"Pixabay sound downloader -- {len(TARGETS)} targets -> {AUDIO_DIR}")
    print()

    ok = skip = fail = 0

    for output_name, query in TARGETS:
        dest = AUDIO_DIR / output_name

        if dest.exists():
            print(f"  SKIP  {output_name}")
            skip += 1
            continue

        # Search
        time.sleep(0.3)
        hit = search_sound(api_key, query)
        if not hit:
            print(f"  FAIL  {output_name}: no results for '{query}'")
            fail += 1
            continue

        # Pick the HQ preview URL
        previews = hit.get("previews", {})
        audio_url = (
            previews.get("preview-hq-mp3")
            or previews.get("preview-lq-mp3")
        )
        if not audio_url:
            print(f"  FAIL  {output_name}: no preview URL in hit")
            fail += 1
            continue

        size_kb = hit.get("filesize", 0) // 1024
        tags    = hit.get("tags", "")[:50]
        print(f"  GET   {output_name}  [{size_kb} KB]  tags: {tags}")

        time.sleep(1.0)   # be polite
        if download_file(audio_url, dest):
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
