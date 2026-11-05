"""The README screenshot and the social-preview image come from ONE capture, each from
the state it is documented to show.

frontend/public/og.png is what every shared link shows, so it must not drift from what
the engine serves. `make capture` (e2e/capture.mjs) reads what the page rendered from the
DOM around each screenshot and writes it INSIDE the PNG, with an id shared by both images
of the run. These tests read it back out of the image, so editing the sidecar JSON cannot
make a stale image pass.

There is no OCR here. The numbers in the pixels are bound to the record by the file's
hash and by the capture's two reads of the page around the screenshot.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD_KEY = b"creditlens-capture"


def _manifest() -> dict:
    return json.loads((ROOT / "docs" / "capture.json").read_text(encoding="utf-8"))


def _embedded_record(png: bytes) -> dict:
    """The capture record carried in the PNG's own tEXt chunk."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    at = 8
    while at < len(png):
        length, kind = struct.unpack(">I4s", png[at:at + 8])
        if kind == b"tEXt":
            key, _, text = png[at + 8:at + 8 + length].partition(b"\0")
            if key == RECORD_KEY:
                return json.loads(text.decode("latin1"))
        at += 12 + length
    raise AssertionError("the image carries no capture record: re-take it with `make capture`")


def _check_pairing(manifest: dict, read=lambda path: (ROOT / path).read_bytes()) -> None:
    """Each image is the file the capture wrote, and carries that capture's record."""
    for key in ("screenshot", "og"):
        image = manifest[key]
        data = read(image["path"])
        assert hashlib.sha256(data).hexdigest() == image["sha256"], f"{image['path']} changed without `make capture`"
        record = _embedded_record(data)
        assert record["capture_id"] == manifest["capture_id"], f"{image['path']} is from another capture"
        assert record["state"] == image["state"], image["path"]
        assert record["shows"] == image["shows"], f"capture.json does not say what {image['path']} recorded"
        assert record["captured"] == manifest["captured"], image["path"]


def test_both_images_are_the_ones_one_capture_recorded():
    m = _manifest()
    _check_pairing(m)
    assert m["screenshot"]["path"] == "docs/screenshot.png" and m["og"]["path"] == "frontend/public/og.png"


@pytest.mark.parametrize("edit", [
    lambda m: m["screenshot"]["shows"].update(credit_score=m["screenshot"]["shows"]["credit_score"] - 1),
    lambda m: m["og"]["shows"].update(risk_band="Elevated"),
    lambda m: m["og"].update(state="live engine"),
    lambda m: m.update(capture_id="0" * 16),
    lambda m: m["og"].update(sha256="0" * 64),
], ids=["screenshot's numbers", "og's numbers", "og's state", "the capture id", "a hash"])
def test_editing_the_sidecar_cannot_make_an_image_pass(edit):
    manifest = _manifest()
    edit(manifest)
    with pytest.raises(AssertionError):
        _check_pairing(manifest)


def test_an_image_from_another_capture_does_not_pair():
    """Swapped in, and its hash updated in the sidecar: its record still says otherwise."""
    manifest = _manifest()
    original = (ROOT / manifest["og"]["path"]).read_bytes()
    record = _embedded_record(original)
    text = RECORD_KEY + b"\0" + json.dumps({**record, "capture_id": "f" * 16}).encode("latin1")
    chunk = struct.pack(">I4s", len(text), b"tEXt") + text + b"\0\0\0\0"
    swapped = original[:33] + chunk + original[33:]  # read first: this record wins
    manifest["og"]["sha256"] = hashlib.sha256(swapped).hexdigest()

    def read(path: str) -> bytes:
        return swapped if path == manifest["og"]["path"] else (ROOT / path).read_bytes()

    with pytest.raises(AssertionError, match="another capture"):
        _check_pairing(manifest, read)


def test_an_image_with_no_record_does_not_pass():
    with pytest.raises(AssertionError, match="no capture record"):
        _embedded_record(b"\x89PNG\r\n\x1a\n" + struct.pack(">I4s", 0, b"IEND") + b"\0\0\0\0")


def test_each_image_shows_what_its_state_serves_today():
    """The screenshot is the live engine's answer; the social image is the no-backend
    page's, which is the engine's recording, and they agree."""
    from fastapi.testclient import TestClient

    from creditlens.api.app import _startup_train, app

    m = _manifest()
    smoke = (ROOT / "e2e" / "smoke.mjs").read_text(encoding="utf-8")
    payload = re.search(r"const DEFAULT_APPLICANT = (\{.*?\});", smoke, re.S).group(1)
    applicant = json.loads(re.sub(r"(\w+):", r'"\1":', re.sub(r",\s*\}", "}", payload)))
    with TestClient(app) as c:
        _startup_train()
        score = c.post("/score", json=applicant).json()
        report = c.get("/fairness/report").json()
    band = next(b["label"] for b in report["score_bands"]
                if b["min_score"] <= score["credit_score"] <= b["max_score"])
    engine = {"credit_score": score["credit_score"], "decision": score["decision"].upper(),
              "default_probability": f"{score['default_probability'] * 100:.1f}%", "risk_band": band,
              "fairness_verdict": report["overall_verdict"]}
    assert m["screenshot"]["state"] == "live engine" and m["screenshot"]["shows"] == engine
    assert m["og"]["state"] == "no backend" and m["og"]["shows"] == engine

    snapshot = json.loads((ROOT / "frontend" / "lib" / "engineSnapshot.json").read_text(encoding="utf-8"))
    default = next(v for v in snapshot["vectors"] if v["name"] == "default")
    assert default["credit_score"] == m["og"]["shows"]["credit_score"]
    assert int(re.search(r"const EXPECT_SCORE = (\d+);", smoke).group(1)) == engine["credit_score"]


def test_the_social_image_is_1200x630_at_2x_and_the_layout_points_at_it():
    width, height = struct.unpack(">II", (ROOT / "frontend/public/og.png").read_bytes()[16:24])
    assert (width, height) == (2400, 1260)
    layout = (ROOT / "frontend" / "app" / "layout.tsx").read_text(encoding="utf-8")
    assert 'url: "/og.png"' in layout and "width: 1200" in layout and "height: 630" in layout


def test_the_readme_says_what_each_image_is():
    readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    assert f"Captured {_manifest()['captured']} from a production build against the real engine" in readme
    assert "The social-preview image is captured from a build with no backend" in readme


def test_the_capture_script_writes_both_images_in_one_run():
    script = (ROOT / "e2e" / "capture.mjs").read_text(encoding="utf-8")
    assert 'const SHOT = "docs/screenshot.png"' in script and 'const OG = "frontend/public/og.png"' in script
    assert f'const RECORD_KEY = "{RECORD_KEY.decode()}"' in script
    assert 'badge: "Live engine"' in script and 'badge: "Demo data"' in script
    assert "capture:" in (ROOT / "Makefile").read_text(encoding="utf-8")
