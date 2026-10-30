"""save_profiles must be atomic (security audit 2026-10-30, L-8).

vezir SIGKILLs the whole millet process group on a watchdog timeout.  A
plain write_text() interrupted mid-way left a truncated team voiceprint DB
— a running average with no provenance, so unrecoverable.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from millet import voiceprint as _vp


def _profiles(n: int = 2) -> dict[str, _vp.SpeakerProfile]:
    return {
        f"p{i}": _vp.SpeakerProfile(f"p{i}", np.ones(256, dtype=np.float32) / 16, i + 1)
        for i in range(n)
    }


def test_save_profiles_round_trips(tmp_path):
    path = tmp_path / "speaker_profiles.json"
    _vp.save_profiles(_profiles(), profiles_path=path)
    loaded = _vp.load_profiles(path)
    assert set(loaded) == {"p0", "p1"}
    assert loaded["p1"].n_sessions == 2
    # No temp file left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["speaker_profiles.json"]


def test_save_profiles_is_atomic_on_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "speaker_profiles.json"
    _vp.save_profiles(_profiles(1), profiles_path=path)
    original = path.read_bytes()

    real_replace = os.replace

    def boom(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        _vp.save_profiles(_profiles(3), profiles_path=path)
    monkeypatch.setattr(os, "replace", real_replace)

    # Old content intact, parseable, and the temp file cleaned up.
    assert path.read_bytes() == original
    assert set(json.loads(path.read_text())) == {"p0"}
    assert [p.name for p in tmp_path.iterdir()] == ["speaker_profiles.json"]


def test_save_profiles_preserves_mode(tmp_path):
    path = tmp_path / "speaker_profiles.json"
    _vp.save_profiles(_profiles(1), profiles_path=path)
    path.chmod(0o600)
    _vp.save_profiles(_profiles(2), profiles_path=path)
    assert path.stat().st_mode & 0o777 == 0o600
