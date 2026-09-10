"""Tests for millet.label — speaker labeling module."""

from __future__ import annotations

import json
import wave
from unittest.mock import patch

import numpy as np

from millet.label import (
    REMOTE_ABSORB_MAX_SEGMENTS,
    TINY_SPEAKER_MAX_SECONDS,
    TINY_SPEAKER_MAX_SEGMENTS,
    SpeakerInfo,
    _detect_speaker_channels,
    _find_session_files,
    _load_transcript,
    absorb_tiny_speakers,
    absorb_unresolved_remote,
    apply_labels,
    extract_speaker_clip,
    get_speakers,
    relabel_transcript_in_memory,
)
from millet.transcribe import Segment, Speaker, Transcript

# ─── _find_session_files() ─────────────────────────────────────────────────

class TestFindSessionFiles:
    def test_finds_all_file_types(self, session_dir):
        files = _find_session_files(session_dir)
        assert "json" in files
        assert "session" in files
        assert "wav" in files
        assert "summary" in files

    def test_ignores_translation_files(self, session_dir):
        # Create a translation file that should NOT be picked up as "json"
        (session_dir / "meeting-20260314-100000.translation.en.json").write_text("{}")
        files = _find_session_files(session_dir)
        assert ".translation." not in files["json"].name

    def test_ignores_autoid_sidecar(self, session_dir):
        # The auto-id sidecar must NOT be mistaken for the transcript JSON.
        (session_dir / "meeting-20260314-100000.autoid.json").write_text(
            '{"version": 1, "suggestions": {}}'
        )
        files = _find_session_files(session_dir)
        assert ".autoid." not in files["json"].name

    def test_empty_dir(self, tmp_path):
        files = _find_session_files(tmp_path)
        assert "json" not in files
        assert "wav" not in files

    def test_finds_mp3_when_no_wav_or_ogg(self, tmp_path):
        # MP3-only session dir: the audio is discovered under the "wav" key
        # (kept for backward compat).
        (tmp_path / "meeting.mp3").write_bytes(b"ID3" + b"\x00" * 16)
        files = _find_session_files(tmp_path)
        assert files["wav"].name == "meeting.mp3"

    def test_prefers_ogg_over_mp3(self, tmp_path):
        # When both OGG and MP3 are present, OGG wins.
        (tmp_path / "meeting.ogg").write_bytes(b"OggS" + b"\x00" * 16)
        (tmp_path / "meeting.mp3").write_bytes(b"ID3" + b"\x00" * 16)
        files = _find_session_files(tmp_path)
        assert files["wav"].name == "meeting.ogg"

    def test_ignores_frontmatter_json(self, session_dir):
        # The frontmatter sidecar has no "segments" and must NOT be the json.
        (session_dir / "meeting-20260314-100000.frontmatter.json").write_text(
            '{"schema_version": 1}'
        )
        files = _find_session_files(session_dir)
        assert ".frontmatter." not in files["json"].name

    def test_ignores_bare_session_json(self, tmp_path):
        # A vezir-pulled dir: friendly transcript.json + a bare session.json
        # (whose name lacks the ``.session.`` substring) + frontmatter.json.
        # session.json sorts LAST but must not shadow the transcript.
        (tmp_path / "transcript.json").write_text(
            '{"segments": [], "speakers": []}'
        )
        (tmp_path / "session.json").write_text('{"session_id": "X"}')
        (tmp_path / "frontmatter.json").write_text('{"schema_version": 1}')
        files = _find_session_files(tmp_path)
        assert files["json"].name == "transcript.json"
        assert files["session"].name == "session.json"

    def test_prefers_transcript_json_over_late_sorting_sidecars(self, tmp_path):
        # Even with several non-transcript JSONs that sort after it, the
        # canonical transcript.json is selected.
        (tmp_path / "transcript.json").write_text(
            '{"segments": [], "speakers": []}'
        )
        (tmp_path / "zzz.summary.meta.json").write_text("{}")
        (tmp_path / "session.json").write_text("{}")
        files = _find_session_files(tmp_path)
        assert files["json"].name == "transcript.json"

    def test_prefers_dirname_json(self, tmp_path):
        # Server stem convention: <dirname>.json wins over a generic decoy.
        sd = tmp_path / "meeting-20260314-100000"
        sd.mkdir()
        (sd / "meeting-20260314-100000.json").write_text(
            '{"segments": [], "speakers": []}'
        )
        (sd / "meeting-20260314-100000.frontmatter.json").write_text("{}")
        files = _find_session_files(sd)
        assert files["json"].name == "meeting-20260314-100000.json"


# ─── _write_autoid_sidecar() ───────────────────────────────────────────────


class TestAutoidSidecar:
    def test_writes_keyed_by_id_with_confidence(self, tmp_path):
        from types import SimpleNamespace

        from millet.cli.label import _write_autoid_sidecar

        json_path = tmp_path / "meeting-20260314-100000.json"
        json_path.write_text("{}")
        matches = {
            "Openoms": SimpleNamespace(name="Openoms", confidence=0.93),
            "Kemal": SimpleNamespace(name="Kemal", confidence=0.8827),
        }
        _write_autoid_sidecar(json_path, matches)

        sidecar = tmp_path / "meeting-20260314-100000.autoid.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["version"] == 1
        assert data["suggestions"]["Openoms"]["name"] == "Openoms"
        assert data["suggestions"]["Openoms"]["confidence"] == 0.93
        # confidence rounded to 4 dp
        assert data["suggestions"]["Kemal"]["confidence"] == 0.8827

    def test_empty_matches_writes_empty_suggestions(self, tmp_path):
        from millet.cli.label import _write_autoid_sidecar

        json_path = tmp_path / "m.json"
        json_path.write_text("{}")
        _write_autoid_sidecar(json_path, {})
        data = json.loads((tmp_path / "m.autoid.json").read_text())
        assert data["suggestions"] == {}


# ─── _load_transcript() ────────────────────────────────────────────────────

class TestLoadTranscript:
    def test_round_trip(self, session_dir):
        files = _find_session_files(session_dir)
        t = _load_transcript(files["json"])
        assert len(t.segments) == 6
        assert len(t.speakers) == 3
        assert t.language == "en"

    def test_segment_data(self, session_dir):
        files = _find_session_files(session_dir)
        t = _load_transcript(files["json"])
        seg = t.segments[0]
        assert seg.start == 0.0
        assert seg.speaker == "YOU"
        assert "Hello" in seg.text


# ─── _detect_speaker_channels() ────────────────────────────────────────────

class TestDetectSpeakerChannels:
    def test_you_on_mic(self, transcript, stereo_wav_with_speakers):
        channels = _detect_speaker_channels(
            stereo_wav_with_speakers, transcript.segments, transcript.speakers,
        )
        assert channels["YOU"] == "mic"

    def test_remote_on_system(self, transcript, stereo_wav_with_speakers):
        channels = _detect_speaker_channels(
            stereo_wav_with_speakers, transcript.segments, transcript.speakers,
        )
        assert channels["REMOTE_1"] == "system"
        assert channels["REMOTE_2"] == "system"

    def test_sensitive_mic_you_still_on_mic(self, tmp_path):
        sr = 16000
        n_each = int(3.0 * sr)
        t = np.linspace(0, 3.0, n_each, dtype=np.float32)

        mic_you = (5000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        sys_you = (7500 * np.sin(2 * np.pi * 880 * t)).astype(np.int16)
        mic_rem = (500 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
        sys_rem = (20000 * np.sin(2 * np.pi * 1100 * t)).astype(np.int16)

        mic = np.concatenate([mic_you, mic_rem])
        system = np.concatenate([sys_you, sys_rem])
        stereo = np.column_stack((mic, system)).flatten()

        path = tmp_path / "sensitive-mic.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        segments = [
            Segment(start=0.0, end=3.0, text="local", speaker="YOU"),
            Segment(start=3.0, end=6.0, text="remote", speaker="REMOTE"),
        ]
        speakers = [Speaker(id="YOU"), Speaker(id="REMOTE")]

        channels = _detect_speaker_channels(path, segments, speakers)
        assert channels["YOU"] == "mic", (
            f"YOU mic_ratio < 0.5 but margin should keep it on mic; got {channels}"
        )

    def test_mono_defaults_to_mic(self, tmp_path, transcript):
        # Create a mono WAV
        mono_path = tmp_path / "mono.wav"
        with wave.open(str(mono_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

        channels = _detect_speaker_channels(
            mono_path, transcript.segments, transcript.speakers,
        )
        for sp in transcript.speakers:
            assert channels[sp.id] == "mic"


# ─── get_speakers() ────────────────────────────────────────────────────────

class TestGetSpeakers:
    def test_returns_all_speakers(self, session_dir):
        speakers = get_speakers(session_dir)
        assert len(speakers) == 3
        ids = {sp.id for sp in speakers}
        assert ids == {"YOU", "REMOTE_1", "REMOTE_2"}

    def test_speaker_info_fields(self, session_dir):
        speakers = get_speakers(session_dir)
        for sp in speakers:
            assert isinstance(sp, SpeakerInfo)
            assert sp.channel in ("mic", "system")
            assert sp.segment_count > 0
            assert sp.sample_text
            assert sp.sample_start >= 0
            assert sp.sample_end > sp.sample_start


# ─── extract_speaker_clip() ────────────────────────────────────────────────

class TestExtractSpeakerClip:
    def test_produces_mono_wav(self, session_dir):
        speakers = get_speakers(session_dir)
        files = _find_session_files(session_dir)
        sp = speakers[0]

        clip_path = extract_speaker_clip(files["wav"], sp)
        try:
            with wave.open(str(clip_path), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getframerate() == 16000
        finally:
            clip_path.unlink(missing_ok=True)

    def test_clip_duration_capped(self, session_dir):
        speakers = get_speakers(session_dir)
        files = _find_session_files(session_dir)
        sp = speakers[0]

        clip_path = extract_speaker_clip(files["wav"], sp, max_duration=2.0)
        try:
            with wave.open(str(clip_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                assert duration <= 2.1  # small tolerance
        finally:
            clip_path.unlink(missing_ok=True)

    def test_quiet_clip_is_normalized(self, tmp_path):
        """A very quiet mic channel should be peak-normalized so the clip is audible."""
        sr = 16000
        duration = 3.0
        n_frames = int(duration * sr)
        t = np.arange(n_frames, dtype=np.float32) / sr

        # Mic: very quiet (amplitude 500 out of 32767 — ~1.5% full-scale)
        mic = (500 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        # System: normal level
        system = (18000 * np.sin(2 * np.pi * 880 * t)).astype(np.int16)

        wav_path = tmp_path / "quiet-mic.wav"
        stereo = np.column_stack((mic, system)).flatten()
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        sp = SpeakerInfo(
            id="YOU", channel="mic",
            sample_text="test", sample_start=0.0, sample_end=3.0,
            segment_count=1,
        )

        clip_path = extract_speaker_clip(wav_path, sp)
        try:
            with wave.open(str(clip_path), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16)
            peak = int(np.max(np.abs(samples)))

            # After normalization the peak should be near 70% of 32767 (~22937)
            assert peak > 15000, f"Clip should be normalized louder, got peak={peak}"
            assert peak <= 32767, f"Clip should not clip, got peak={peak}"
        finally:
            clip_path.unlink(missing_ok=True)

    def test_already_loud_clip_not_clipped(self, tmp_path):
        """A clip whose peak already exceeds the target should not be altered."""
        sr = 16000
        duration = 3.0
        n_frames = int(duration * sr)
        t = np.arange(n_frames, dtype=np.float32) / sr

        # Mic: already at 30000 — above the 70% target (~22937)
        mic = (30000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        system = (5000 * np.sin(2 * np.pi * 880 * t)).astype(np.int16)

        wav_path = tmp_path / "loud-mic.wav"
        stereo = np.column_stack((mic, system)).flatten()
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        sp = SpeakerInfo(
            id="YOU", channel="mic",
            sample_text="test", sample_start=0.0, sample_end=3.0,
            segment_count=1,
        )

        clip_path = extract_speaker_clip(wav_path, sp)
        try:
            with wave.open(str(clip_path), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16)
            peak = int(np.max(np.abs(samples)))

            # Peak was 30000, above the 22937 target — should stay unchanged.
            assert 29000 <= peak <= 31000, (
                f"Already-loud clip should not be re-scaled, got peak={peak}"
            )
        finally:
            clip_path.unlink(missing_ok=True)


# ─── relabel_transcript_in_memory() ────────────────────────────────────────

class TestRelabelTranscriptInMemory:
    def test_remaps_segments(self, transcript):
        label_map = {"YOU": "Alice", "REMOTE_1": "Bob"}
        new_t = relabel_transcript_in_memory(transcript, label_map)
        alice_segs = [s for s in new_t.segments if s.speaker == "Alice"]
        assert len(alice_segs) == 2

    def test_remaps_speakers(self, transcript):
        label_map = {"YOU": "Alice", "REMOTE_1": "Bob"}
        new_t = relabel_transcript_in_memory(transcript, label_map)
        ids = {sp.id for sp in new_t.speakers}
        assert "Alice" in ids
        assert "Bob" in ids
        # REMOTE_2 unchanged
        assert "REMOTE_2" in ids

    def test_empty_map_returns_copy(self, transcript):
        new_t = relabel_transcript_in_memory(transcript, {})
        assert new_t is transcript  # no-op returns same object

    def test_preserves_language_and_metadata(self, transcript):
        label_map = {"YOU": "Alice"}
        new_t = relabel_transcript_in_memory(transcript, label_map)
        assert new_t.language == "en"
        assert new_t.duration == 42.0
        assert new_t.audio_file == transcript.audio_file


# ─── apply_labels() ────────────────────────────────────────────────────────

class TestApplyLabels:
    def test_updates_txt(self, session_dir):
        apply_labels(session_dir, {"YOU": "Alice"}, regenerate_summary=False)
        txt = (session_dir / "meeting-20260314-100000.txt").read_text()
        assert "Alice:" in txt
        assert "YOU:" not in txt

    def test_updates_srt(self, session_dir):
        apply_labels(session_dir, {"YOU": "Alice"}, regenerate_summary=False)
        srt = (session_dir / "meeting-20260314-100000.srt").read_text()
        assert "[Alice]" in srt
        assert "[YOU]" not in srt

    def test_updates_json_speakers(self, session_dir):
        apply_labels(session_dir, {"YOU": "Alice"}, regenerate_summary=False)
        data = json.loads(
            (session_dir / "meeting-20260314-100000.json").read_text()
        )
        ids = {sp["id"] for sp in data["speakers"]}
        assert "Alice" in ids

    def test_stores_label_map_in_session(self, session_dir):
        label_map = {"YOU": "Alice", "REMOTE_1": "Bob"}
        apply_labels(session_dir, label_map, regenerate_summary=False)
        meta = json.loads(
            (session_dir / "meeting-20260314-100000.session.json").read_text()
        )
        assert meta["speaker_labels"] == label_map

    def test_find_replace_summary(self, session_dir):
        """With regenerate_summary=False, apply_labels should do
        find-and-replace on the existing summary file."""
        # Put a speaker name in the summary so we can verify replacement
        summary_path = session_dir / "meeting-20260314-100000.summary.md"
        text = summary_path.read_text()
        text = text.replace("the meeting", "YOU discussed the meeting")
        summary_path.write_text(text)

        apply_labels(session_dir, {"YOU": "Alice"}, regenerate_summary=False)
        new_text = summary_path.read_text()
        assert "Alice discussed the meeting" in new_text

    def test_find_replace_summary_overlapping_labels(self, session_dir):
        """Overlapping labels must not corrupt each other: naive dict-order
        substring replacement turned REMOTE_1 into <REMOTE's name>_1."""
        summary_path = session_dir / "meeting-20260314-100000.summary.md"
        summary_path.write_text(
            "REMOTE said hi. REMOTE_1 agreed. REMOTE_2 disagreed.\n"
        )
        # Insertion order deliberately puts the short key first.
        apply_labels(
            session_dir,
            {"REMOTE": "Alice", "REMOTE_1": "Bob", "REMOTE_2": "Carol"},
            regenerate_summary=False,
        )
        new_text = summary_path.read_text()
        assert "Alice said hi." in new_text
        assert "Bob agreed." in new_text
        assert "Carol disagreed." in new_text
        assert "Alice_1" not in new_text
        assert "Alice_2" not in new_text

    def test_find_replace_summary_swap(self, session_dir):
        """A swap {A:B, B:A} must swap, not collapse both to one name.

        Regression: sequential per-key re.sub chained on the same string —
        replacing Alaaddin->Kemal first, then Kemal->Alaaddin re-caught the
        just-written Kemals and collapsed BOTH speakers to a single name in
        the summary/frontmatter/PDF. The relabel must be a single atomic pass.
        """
        summary_path = session_dir / "meeting-20260314-100000.summary.md"
        summary_path.write_text(
            "Alaaddin opened. Kemal replied. Alaaddin closed.\n"
        )
        apply_labels(
            session_dir,
            {"Alaaddin": "Kemal", "Kemal": "Alaaddin"},
            regenerate_summary=False,
        )
        new_text = summary_path.read_text()
        assert "Kemal opened." in new_text
        assert "Alaaddin replied." in new_text
        assert "Kemal closed." in new_text

    def test_find_replace_summary_chain_single_pass(self, session_dir):
        """{A:B, B:C} must not chain A->B->C: each source position is
        rewritten exactly once from the original text."""
        summary_path = session_dir / "meeting-20260314-100000.summary.md"
        summary_path.write_text("Ann met Bob.\n")
        apply_labels(
            session_dir,
            {"Ann": "Bob", "Bob": "Cara"},
            regenerate_summary=False,
        )
        new_text = summary_path.read_text()
        assert new_text.strip() == "Bob met Cara."

    def test_find_replace_summary_word_boundary(self, session_dir):
        """A label must only match as a whole word — 'YOU' must not fire
        inside 'YOUR' (uppercase prose)."""
        summary_path = session_dir / "meeting-20260314-100000.summary.md"
        summary_path.write_text("YOU said: BRING YOUR OWN DEVICE.\n")
        apply_labels(session_dir, {"YOU": "Alice"}, regenerate_summary=False)
        new_text = summary_path.read_text()
        assert "Alice said" in new_text
        assert "YOUR OWN DEVICE" in new_text

    def test_accepts_summary_preset_kwarg(self, session_dir):
        """Regression: cli.py `meet label` passes summary_preset= to
        apply_labels().  If the kwarg isn't in the signature, every
        auto-label call from `meet label --auto` raises TypeError and
        the relabel never lands on disk.  See git history for context.

        Validates the signature itself; runs with regenerate_summary=False
        so the LLM path is bypassed and we don't need a live backend."""
        apply_labels(
            session_dir,
            {"YOU": "Alice"},
            regenerate_summary=False,
            summary_preset="high-quality",
        )
        txt = (session_dir / "meeting-20260314-100000.txt").read_text()
        assert "Alice:" in txt


class TestApplyLabelsSummaryLanguage:
    """summary_language override writes an ADDITIONAL <name>.summary.<lang>.md
    without clobbering the primary <name>.summary.md."""

    def _fake_summary(self):
        from millet.summarize import MeetingSummary
        return MeetingSummary(
            markdown="## Zusammenfassung\nEin Test.",
            backend="test",
            model="test-model",
            elapsed_seconds=0.1,
            data=None,
            data_error=None,
        )

    def test_language_override_creates_additional_file(self, session_dir):
        basename = "meeting-20260314-100000"
        primary = session_dir / f"{basename}.summary.md"
        primary_before = primary.read_text()

        with patch("millet.summarize.is_backend_available", return_value=True), \
             patch("millet.summarize.summarize", return_value=self._fake_summary()), \
             patch("millet.transcribe.ensure_gpu_available", lambda *a, **k: None):
            apply_labels(
                session_dir,
                {},
                regenerate_summary=True,
                summary_language="de",
            )

        # Additional language file created…
        de = session_dir / f"{basename}.summary.de.md"
        assert de.exists()
        assert "Zusammenfassung" in de.read_text()
        # …and its sidecars are suffixed (don't clobber the primary's).
        assert (session_dir / f"{basename}.summary.de.meta.json").exists()
        assert (session_dir / f"{basename}.de.frontmatter.json").exists()
        # Primary summary is untouched.
        assert primary.read_text() == primary_before

    def test_no_language_writes_primary(self, session_dir):
        basename = "meeting-20260314-100000"
        with patch("millet.summarize.is_backend_available", return_value=True), \
             patch("millet.summarize.summarize", return_value=self._fake_summary()), \
             patch("millet.transcribe.ensure_gpu_available", lambda *a, **k: None):
            apply_labels(
                session_dir,
                {},
                regenerate_summary=True,
            )
        # No language → primary rewritten, no .summary.<lang>.md created.
        assert (session_dir / f"{basename}.summary.md").exists()
        assert not list(session_dir.glob(f"{basename}.summary.??.md"))


class TestApplyLabelsSummaryTemplate:
    """summary_template writes <name>.<template>.md instead of clobbering
    the primary <name>.summary.md."""

    def _fake_summary(self):
        from millet.summarize import MeetingSummary
        return MeetingSummary(
            markdown="## Overview\nA plan.",
            backend="test",
            model="test-model",
            elapsed_seconds=0.1,
            template="iteration-plan",
            data=None,
            data_error=None,
        )

    def test_template_writes_own_artifact(self, session_dir):
        basename = "meeting-20260314-100000"
        primary = session_dir / f"{basename}.summary.md"
        primary_before = primary.read_text()

        with patch("millet.summarize.is_backend_available", return_value=True), \
             patch("millet.summarize.summarize", return_value=self._fake_summary()), \
             patch("millet.transcribe.ensure_gpu_available", lambda *a, **k: None):
            result_files = apply_labels(
                session_dir,
                {},
                regenerate_summary=True,
                summary_template="iteration-plan",
            )

        plan = session_dir / f"{basename}.iteration-plan.md"
        assert plan.exists()
        assert (session_dir / f"{basename}.iteration-plan.meta.json").exists()
        assert result_files.get("iteration-plan") == plan
        # Primary summary is untouched.
        assert primary.read_text() == primary_before


# ─── relabel_transcript_in_memory: same-name speaker collapse (0.12.11) ──────
# Diarization over-segments one person into several clusters; once they resolve
# to the same name (via many-to-one voiceprint match or a human naming each),
# they must collapse into ONE speaker entry instead of "Destiny, Destiny, …".

class TestRelabelSpeakerDedup:
    def _txn(self, speakers, segments):
        from millet.transcribe import Transcript
        return Transcript(
            segments=segments,
            speakers=speakers,
            language="en",
            audio_file="x.ogg",
            duration=10.0,
        )

    def test_label_map_collapses_duplicate_names(self):
        # Three raw clusters, two of which a human (or many-to-one) maps to
        # Destiny; the result must have a single Destiny speaker.
        txn = self._txn(
            speakers=[
                Speaker(id="SPEAKER_00", label="SPEAKER_00"),
                Speaker(id="SPEAKER_01", label="SPEAKER_01"),
                Speaker(id="SPEAKER_02", label="SPEAKER_02"),
            ],
            segments=[
                Segment(start=0.0, end=2.0, text="a", speaker="SPEAKER_00"),
                Segment(start=2.0, end=4.0, text="b", speaker="SPEAKER_01"),
                Segment(start=4.0, end=6.0, text="c", speaker="SPEAKER_02"),
            ],
        )
        out = relabel_transcript_in_memory(
            txn,
            {"SPEAKER_00": "Destiny", "SPEAKER_01": "Andrej", "SPEAKER_02": "Destiny"},
        )
        ids = [s.id for s in out.speakers]
        assert ids == ["Destiny", "Andrej"]  # de-duped, first-seen order
        # All Destiny segments now share the one id.
        assert {seg.speaker for seg in out.segments} == {"Destiny", "Andrej"}
        assert sum(1 for seg in out.segments if seg.speaker == "Destiny") == 2

    def test_empty_map_dedupes_preexisting_duplicates(self):
        # Backfill case: a transcript already carrying duplicate "Destiny"
        # entries collapses with an empty label_map.
        txn = self._txn(
            speakers=[
                Speaker(id="Destiny", label="Destiny"),
                Speaker(id="Destiny", label="Destiny"),
                Speaker(id="Andrej", label="Andrej"),
                Speaker(id="Destiny", label="Destiny"),
            ],
            segments=[
                Segment(start=0.0, end=2.0, text="a", speaker="Destiny"),
                Segment(start=2.0, end=4.0, text="b", speaker="Andrej"),
            ],
        )
        out = relabel_transcript_in_memory(txn, {})
        assert [s.id for s in out.speakers] == ["Destiny", "Andrej"]

    def test_empty_map_no_dupes_is_noop(self):
        txn = self._txn(
            speakers=[
                Speaker(id="Destiny", label="Destiny"),
                Speaker(id="Andrej", label="Andrej"),
            ],
            segments=[Segment(start=0.0, end=2.0, text="a", speaker="Destiny")],
        )
        out = relabel_transcript_in_memory(txn, {})
        assert out is txn  # unchanged object (fast path)

    def test_no_dupes_with_map_relabels_normally(self):
        txn = self._txn(
            speakers=[
                Speaker(id="YOU", label="YOU"),
                Speaker(id="REMOTE_1", label="REMOTE_1"),
            ],
            segments=[
                Segment(start=0.0, end=2.0, text="a", speaker="YOU"),
                Segment(start=2.0, end=4.0, text="b", speaker="REMOTE_1"),
            ],
        )
        out = relabel_transcript_in_memory(txn, {"YOU": "Kasita", "REMOTE_1": "Ahmad"})
        assert [s.id for s in out.speakers] == ["Kasita", "Ahmad"]


# ─── absorb_unresolved_remote: rescue the leftover REMOTE bucket (A1, 0.12.12) ─
# The dual-diarize path leaves a small literal REMOTE bucket (unassigned system
# segments + non-overlapping mic bleed) AFTER consolidation, so it never merges
# and rarely voiceprint-matches.  This forces needs_labeling even when every
# real participant was identified.  absorb_unresolved_remote folds a SMALL such
# leftover onto the named speaker it overlaps most in time.

class TestAbsorbUnresolvedRemote:
    def _txn(self, segments):
        speakers = []
        seen = set()
        for s in segments:
            if s.speaker and s.speaker not in seen:
                seen.add(s.speaker)
                speakers.append(Speaker(id=s.speaker, label=s.speaker))
        return Transcript(
            segments=segments, speakers=speakers, language="en",
            audio_file="x.ogg", duration=100.0,
        )

    def test_small_remote_absorbed_into_overlapping_named(self):
        # REMOTE one-liners interleaved with Openoms' speech → absorbed to Openoms.
        segs = [
            Segment(start=0.0, end=10.0, text="long openoms turn", speaker="Openoms"),
            Segment(start=10.2, end=10.6, text="yeah", speaker="REMOTE"),
            Segment(start=11.0, end=20.0, text="more openoms", speaker="Openoms"),
            Segment(start=20.1, end=20.5, text="ok", speaker="REMOTE"),
            Segment(start=30.0, end=40.0, text="hoang turn", speaker="Hoang"),
        ]
        txn = self._txn(segs)
        absorb = absorb_unresolved_remote(txn, {"Openoms", "Hoang"})
        assert absorb == {"REMOTE": "Openoms"}

    def test_remote_absorbed_into_nearest_when_no_overlap(self):
        # REMOTE one-liner between turns, no direct overlap → nearest named wins.
        segs = [
            Segment(start=0.0, end=10.0, text="openoms", speaker="Openoms"),
            Segment(start=50.0, end=50.4, text="yeah", speaker="REMOTE"),  # nearer Hoang
            Segment(start=51.0, end=60.0, text="hoang", speaker="Hoang"),
        ]
        txn = self._txn(segs)
        absorb = absorb_unresolved_remote(txn, {"Openoms", "Hoang"})
        assert absorb == {"REMOTE": "Hoang"}

    def test_large_remote_not_absorbed(self):
        # A substantial unknown (> max seconds) stays raw for human review.
        segs = [Segment(start=0.0, end=5.0, text="openoms", speaker="Openoms")]
        # 40s of REMOTE speech across many segments → over the 30s cap.
        for i in range(20):
            segs.append(Segment(start=100.0 + i * 3, end=100.0 + i * 3 + 2.0,
                                text="unknown person talking", speaker="REMOTE"))
        txn = self._txn(segs)
        absorb = absorb_unresolved_remote(txn, {"Openoms"})
        assert absorb == {}

    def test_too_many_remote_segments_not_absorbed(self):
        segs = [Segment(start=0.0, end=5.0, text="openoms", speaker="Openoms")]
        # Many tiny segments (> max count) but little total time.
        for i in range(REMOTE_ABSORB_MAX_SEGMENTS + 5):
            segs.append(Segment(start=100.0 + i, end=100.0 + i + 0.2,
                                text="x", speaker="REMOTE"))
        txn = self._txn(segs)
        absorb = absorb_unresolved_remote(txn, {"Openoms"})
        assert absorb == {}

    def test_no_named_speakers_no_absorb(self):
        segs = [Segment(start=0.0, end=2.0, text="yeah", speaker="REMOTE")]
        txn = self._txn(segs)
        assert absorb_unresolved_remote(txn, set()) == {}

    def test_resolved_remote_not_touched(self):
        # If REMOTE was already matched (in resolved_ids), it's not raw → skip.
        segs = [
            Segment(start=0.0, end=10.0, text="openoms", speaker="Openoms"),
            Segment(start=10.2, end=10.6, text="yeah", speaker="REMOTE"),
        ]
        txn = self._txn(segs)
        absorb = absorb_unresolved_remote(txn, {"Openoms", "REMOTE"})
        assert absorb == {}

    def test_raw_speaker_n_also_absorbed(self):
        # A leftover raw SPEAKER_n (not just literal REMOTE) is eligible too.
        segs = [
            Segment(start=0.0, end=10.0, text="openoms", speaker="Openoms"),
            Segment(start=10.2, end=10.6, text="yeah", speaker="SPEAKER_03"),
        ]
        txn = self._txn(segs)
        absorb = absorb_unresolved_remote(txn, {"Openoms"})
        assert absorb == {"SPEAKER_03": "Openoms"}


# ─── absorb_tiny_speakers: fold spurious noise into the DOMINANT speaker (A2) ─
# Complements absorb_unresolved_remote for the common case where the dominant
# speaker is itself an unmatched SPEAKER_n: a single 1-3 segment / few-second
# REMOTE blip should fold into the real conversation rather than force review.

class TestAbsorbTinySpeakers:
    def _txn(self, segments):
        speakers = []
        seen = set()
        for s in segments:
            if s.speaker and s.speaker not in seen:
                seen.add(s.speaker)
                speakers.append(Speaker(id=s.speaker, label=s.speaker))
        return Transcript(
            segments=segments, speakers=speakers, language="en",
            audio_file="x.ogg", duration=1000.0,
        )

    def test_tiny_remote_folds_into_unmatched_dominant(self):
        # The 01KV8G41 case: big UNMATCHED SPEAKER_00 + tiny REMOTE.
        segs = [
            Segment(start=0.0, end=300.0, text="long unmatched turn", speaker="SPEAKER_00"),
            Segment(start=312.6, end=313.7, text="Thank you.", speaker="REMOTE"),
            Segment(start=864.0, end=866.1, text="Fine.", speaker="REMOTE"),
        ]
        txn = self._txn(segs)
        # No named speaker → absorb_unresolved_remote can't help.
        assert absorb_unresolved_remote(txn, set()) == {}
        # But the tiny REMOTE folds into the dominant raw speaker.
        assert absorb_tiny_speakers(txn, set()) == {"REMOTE": "SPEAKER_00"}

    def test_tiny_remote_folds_into_named_when_present(self):
        segs = [
            Segment(start=0.0, end=300.0, text="long", speaker="Openoms"),
            Segment(start=310.0, end=310.2, text="yeah", speaker="REMOTE"),
        ]
        txn = self._txn(segs)
        assert absorb_tiny_speakers(txn, {"Openoms"}) == {"REMOTE": "Openoms"}

    def test_substantial_unmatched_not_folded(self):
        # A genuine unlabeled participant (above thresholds) stays raw.
        segs = [
            Segment(start=0.0, end=300.0, text="main", speaker="SPEAKER_00"),
            # SPEAKER_01: well above 5.0s and 3 segments.
            Segment(start=300.0, end=320.0, text="real other person", speaker="SPEAKER_01"),
            Segment(start=330.0, end=345.0, text="more", speaker="SPEAKER_01"),
            Segment(start=350.0, end=360.0, text="and more", speaker="SPEAKER_01"),
            Segment(start=370.0, end=380.0, text="still talking", speaker="SPEAKER_01"),
        ]
        txn = self._txn(segs)
        assert absorb_tiny_speakers(txn, set()) == {}

    def test_all_tiny_nothing_folded(self):
        # No real conversation to fold into → bail (let the worker decide).
        segs = [
            Segment(start=0.0, end=0.2, text="x", speaker="REMOTE"),
            Segment(start=5.0, end=5.3, text="y", speaker="SPEAKER_00"),
        ]
        txn = self._txn(segs)
        assert absorb_tiny_speakers(txn, set()) == {}

    def test_named_speaker_never_folded_even_if_tiny(self):
        # A resolved/named tiny speaker is not a noise placeholder.
        segs = [
            Segment(start=0.0, end=300.0, text="main", speaker="SPEAKER_00"),
            Segment(start=310.0, end=310.2, text="hi", speaker="Alice"),
        ]
        txn = self._txn(segs)
        # Alice is resolved → not eligible; nothing else tiny+raw to fold.
        assert absorb_tiny_speakers(txn, {"Alice"}) == {}

    def test_boundary_at_thresholds_is_tiny(self):
        # Exactly 5.0s across exactly 3 segments → still tiny (<=).
        segs = [
            Segment(start=0.0, end=300.0, text="main", speaker="SPEAKER_00"),
            Segment(start=400.0, end=402.0, text="a", speaker="REMOTE"),
            Segment(start=410.0, end=412.0, text="b", speaker="REMOTE"),
            Segment(start=420.0, end=421.0, text="c", speaker="REMOTE"),
        ]
        txn = self._txn(segs)
        assert TINY_SPEAKER_MAX_SECONDS == 5.0
        assert TINY_SPEAKER_MAX_SEGMENTS == 3
        assert absorb_tiny_speakers(txn, set()) == {"REMOTE": "SPEAKER_00"}

    def test_just_over_segment_threshold_not_tiny(self):
        # 4 short segments (> 3) but tiny duration → not folded (count gate).
        segs = [Segment(start=0.0, end=300.0, text="main", speaker="SPEAKER_00")]
        for i in range(4):
            segs.append(Segment(start=400.0 + i, end=400.0 + i + 0.2,
                                text="x", speaker="REMOTE"))
        txn = self._txn(segs)
        assert absorb_tiny_speakers(txn, set()) == {}


# ─── millet label --apply-json (non-interactive embedder mode) ──────────────

class TestLabelApplyJson:
    """The subprocess boundary for embedders (vezir): apply a label map
    from JSON with no prompting, no audio, no terminal dependence."""

    def _invoke(self, args):
        from click.testing import CliRunner

        from millet.cli.label import label as label_cmd

        return CliRunner().invoke(label_cmd, args)

    def test_applies_labels_from_file(self, session_dir, tmp_path):
        payload = tmp_path / "labels.json"
        payload.write_text(json.dumps({"labels": {"YOU": "Alice"}}))
        result = self._invoke(
            [str(session_dir), "--apply-json", str(payload), "--no-summary"]
        )
        assert result.exit_code == 0, result.output
        txt = (session_dir / "meeting-20260314-100000.txt").read_text()
        assert "Alice:" in txt
        assert "YOU:" not in txt

    def test_accepts_plain_map_without_envelope(self, session_dir, tmp_path):
        payload = tmp_path / "labels.json"
        payload.write_text(json.dumps({"REMOTE_1": "Bob"}))
        result = self._invoke(
            [str(session_dir), "--apply-json", str(payload), "--no-summary"]
        )
        assert result.exit_code == 0, result.output
        txt = (session_dir / "meeting-20260314-100000.txt").read_text()
        assert "Bob:" in txt

    def test_reads_map_from_stdin(self, session_dir):
        from click.testing import CliRunner

        from millet.cli.label import label as label_cmd

        result = CliRunner().invoke(
            label_cmd,
            [str(session_dir), "--apply-json", "-", "--no-summary"],
            input=json.dumps({"labels": {"YOU": "Alice"}}),
        )
        assert result.exit_code == 0, result.output
        txt = (session_dir / "meeting-20260314-100000.txt").read_text()
        assert "Alice:" in txt

    def test_never_prompts(self, session_dir, tmp_path):
        """Even with unlabeled speakers left over, apply-json mode must not
        touch stdin for prompting (stdin may be closed in the vezir worker)."""
        payload = tmp_path / "labels.json"
        payload.write_text(json.dumps({"labels": {"YOU": "Alice"}}))
        with patch("click.prompt", side_effect=AssertionError("prompted!")):
            result = self._invoke(
                [str(session_dir), "--apply-json", str(payload), "--no-summary"]
            )
        assert result.exit_code == 0, result.output

    def test_invalid_json_exits_nonzero(self, session_dir, tmp_path):
        payload = tmp_path / "labels.json"
        payload.write_text("{not json")
        result = self._invoke(
            [str(session_dir), "--apply-json", str(payload), "--no-summary"]
        )
        assert result.exit_code == 1
        assert "could not parse" in result.output

    def test_summary_template_forwarded(self, session_dir, tmp_path):
        """--summary-template must reach apply_labels (vezir retry-summary path)."""
        payload = tmp_path / "labels.json"
        payload.write_text(json.dumps({"labels": {}}))
        with patch("millet.label.apply_labels", return_value={}) as mock_apply:
            result = self._invoke(
                [
                    str(session_dir),
                    "--apply-json", str(payload),
                    "--summary-template", "iteration-plan",
                ]
            )
        assert result.exit_code == 0, result.output
        assert mock_apply.call_args.kwargs["summary_template"] == "iteration-plan"

    def test_non_dict_json_exits_nonzero(self, session_dir, tmp_path):
        payload = tmp_path / "labels.json"
        payload.write_text(json.dumps(["not", "a", "map"]))
        result = self._invoke(
            [str(session_dir), "--apply-json", str(payload), "--no-summary"]
        )
        assert result.exit_code == 1

    def test_empty_map_no_summary_is_a_noop(self, session_dir, tmp_path):
        payload = tmp_path / "labels.json"
        payload.write_text(json.dumps({"labels": {}}))
        before = (session_dir / "meeting-20260314-100000.txt").read_text()
        result = self._invoke(
            [str(session_dir), "--apply-json", str(payload), "--no-summary"]
        )
        assert result.exit_code == 0, result.output
        assert "No labels to apply" in result.output
        after = (session_dir / "meeting-20260314-100000.txt").read_text()
        assert before == after

    def test_apply_failure_exits_nonzero(self, session_dir, tmp_path):
        payload = tmp_path / "labels.json"
        payload.write_text(json.dumps({"labels": {"YOU": "Alice"}}))
        with patch(
            "millet.label.apply_labels",
            side_effect=RuntimeError("disk full"),
        ):
            result = self._invoke(
                [str(session_dir), "--apply-json", str(payload), "--no-summary"]
            )
        assert result.exit_code == 1
        assert "apply failed" in result.output

    def test_update_profiles_is_nonfatal_on_failure(self, session_dir, tmp_path):
        """A voiceprint update failure must not fail the command — labels
        are already applied at that point (embedder contract)."""
        payload = tmp_path / "labels.json"
        payload.write_text(json.dumps({"labels": {"YOU": "Alice"}}))
        with patch(
            "millet.voiceprint.update_profiles_from_confirmed_labels",
            side_effect=RuntimeError("no model"),
        ):
            result = self._invoke(
                [
                    str(session_dir),
                    "--apply-json", str(payload),
                    "--no-summary", "--update-profiles",
                ]
            )
        assert result.exit_code == 0, result.output
        assert "profile update failed" in result.output
        # ...but the labels landed.
        txt = (session_dir / "meeting-20260314-100000.txt").read_text()
        assert "Alice:" in txt
