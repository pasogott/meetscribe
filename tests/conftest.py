"""Shared test fixtures for millet tests."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from millet.transcribe import Segment, Speaker, Transcript

# ─── Transcript fixture ────────────────────────────────────────────────────

def _make_segments() -> list[Segment]:
    """Build 6 segments across 3 speakers with known timestamps."""
    return [
        Segment(start=0.0, end=5.5, text="Hello everyone, welcome.", speaker="YOU"),
        Segment(start=5.5, end=12.0, text="Thanks for joining us today.", speaker="REMOTE_1"),
        Segment(start=12.0, end=20.3, text="I have a few items on the agenda.", speaker="REMOTE_2"),
        Segment(start=20.3, end=28.0, text="Let me share my screen.", speaker="YOU"),
        Segment(start=28.0, end=35.7, text="Looks good, I can see the slides.", speaker="REMOTE_1"),
        Segment(start=35.7, end=42.0, text="Can we start with the budget review?", speaker="REMOTE_2"),
    ]


def _make_speakers() -> list[Speaker]:
    return [
        Speaker(id="YOU", label="YOU"),
        Speaker(id="REMOTE_1", label="REMOTE_1"),
        Speaker(id="REMOTE_2", label="REMOTE_2"),
    ]


@pytest.fixture
def transcript() -> Transcript:
    """A Transcript with 6 segments, 3 speakers, 42s duration."""
    return Transcript(
        segments=_make_segments(),
        speakers=_make_speakers(),
        language="en",
        audio_file="meeting-20260314-100000.wav",
        duration=42.0,
    )


# ─── Stereo WAV fixture ────────────────────────────────────────────────────

def _generate_stereo_wav(path: Path, duration: float = 5.0, sr: int = 16000) -> Path:
    """Create a synthetic stereo WAV with distinguishable channels.

    Left channel (mic): loud sine wave at 440Hz
    Right channel (system): quiet sine wave at 880Hz

    This lets tests verify that channel extraction and energy analysis
    correctly distinguish mic from system audio.
    """
    n_frames = int(duration * sr)
    t = np.linspace(0, duration, n_frames, dtype=np.float32)

    # Mic channel: loud (amplitude 20000)
    mic = (20000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    # System channel: quiet (amplitude 3000)
    system = (3000 * np.sin(2 * np.pi * 880 * t)).astype(np.int16)

    # Interleave for stereo
    stereo = np.column_stack((mic, system)).flatten()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())

    return path


@pytest.fixture
def stereo_wav(tmp_path) -> Path:
    """A 5-second stereo WAV: loud mic (left), quiet system (right)."""
    return _generate_stereo_wav(tmp_path / "test-audio.wav")


def _generate_stereo_wav_speakers(
    path: Path,
    segments: list[Segment],
    sr: int = 16000,
) -> Path:
    """Create a stereo WAV where each speaker's segments have energy on
    the expected channel.

    YOU segments: loud on mic (ch0), quiet on system (ch1)
    REMOTE_* segments: quiet on mic (ch0), loud on system (ch1)
    """
    duration = max(s.end for s in segments) + 0.5
    n_frames = int(duration * sr)

    mic = np.zeros(n_frames, dtype=np.float32)
    system = np.zeros(n_frames, dtype=np.float32)

    for seg in segments:
        start = int(seg.start * sr)
        end = min(int(seg.end * sr), n_frames)
        t = np.arange(end - start, dtype=np.float32) / sr

        if seg.speaker and seg.speaker.startswith("REMOTE"):
            # REMOTE: loud on system, quiet on mic
            system[start:end] += 18000 * np.sin(2 * np.pi * 880 * t)
            mic[start:end] += 1500 * np.sin(2 * np.pi * 440 * t)
        else:
            # YOU: loud on mic, quiet on system
            mic[start:end] += 18000 * np.sin(2 * np.pi * 440 * t)
            system[start:end] += 1500 * np.sin(2 * np.pi * 880 * t)

    mic_i16 = np.clip(mic, -32768, 32767).astype(np.int16)
    sys_i16 = np.clip(system, -32768, 32767).astype(np.int16)
    stereo = np.column_stack((mic_i16, sys_i16)).flatten()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())

    return path


@pytest.fixture
def stereo_wav_with_speakers(tmp_path, transcript) -> Path:
    """Stereo WAV whose channel energy matches the transcript speakers."""
    return _generate_stereo_wav_speakers(
        tmp_path / "meeting-20260314-100000.wav",
        transcript.segments,
    )


# ─── Session directory fixture ──────────────────────────────────────────────

@pytest.fixture
def session_dir(tmp_path, transcript) -> Path:
    """A complete fake session directory with all output files.

    Contains:
      - meeting-20260314-100000.session.json
      - meeting-20260314-100000.json  (transcript)
      - meeting-20260314-100000.txt
      - meeting-20260314-100000.srt
      - meeting-20260314-100000.summary.md
      - meeting-20260314-100000.wav  (stereo)
    """
    sdir = tmp_path / "meeting-20260314-100000"
    sdir.mkdir()
    basename = "meeting-20260314-100000"

    # Session metadata
    session_meta = {
        "started_at": "2026-03-14T10:00:00.000000",
        "mic_source": "test-mic",
        "monitor_source": "test-monitor",
        "virtual_sink": False,
        "output_file": str(sdir / f"{basename}.wav"),
        "stopped_at": "2026-03-14T10:00:42.000000",
        "restart_count": 0,
        "chunk_count": 1,
        "failed": False,
        "file_exists": True,
        "file_size_bytes": 12345,
    }
    (sdir / f"{basename}.session.json").write_text(
        json.dumps(session_meta, indent=2), encoding="utf-8",
    )

    # Save transcript files (.json, .txt, .srt)
    transcript.save(sdir, basename=basename)

    # Summary
    summary_text = """\
## Meeting Overview
A test meeting with three participants discussing the agenda and budget.

## Key Topics Discussed
- Agenda items for the meeting
- Budget review planning

## Action Items
- [ ] YOU to share screen

## Decisions Made
- None explicitly stated

## Open Questions / Follow-ups
- Budget review details"""
    (sdir / f"{basename}.summary.md").write_text(summary_text, encoding="utf-8")

    # Stereo WAV with speaker-appropriate energy
    _generate_stereo_wav_speakers(
        sdir / f"{basename}.wav",
        transcript.segments,
    )

    return sdir


# ─── Local git remote (shared by test_sync_core / test_sync_hardening) ──────


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


@pytest.fixture
def git_identity(monkeypatch):
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")
    # Never pick up the developer's git config / credential helpers.
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def local_remote(tmp_path, git_identity):
    """A seeded local bare repo usable as a sync target."""
    bare = tmp_path / "remote.git"
    _git("init", "--bare", "--initial-branch=main", str(bare))
    seed = tmp_path / "seed"
    _git("clone", str(bare), str(seed))
    (seed / "README.md").write_text("seed\n")
    _git("add", ".", cwd=seed)
    _git("commit", "-m", "init", cwd=seed)
    _git("push", "origin", "HEAD:main", cwd=seed)
    return bare
