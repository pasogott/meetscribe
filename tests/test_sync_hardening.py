"""Security-audit (2026-10-30) regression tests for millet.sync.

* H-3  clone reuse must verify ``origin`` matches the configured remote —
       two remotes sharing a basename, or an old and a new remote with the
       same name, must never share a clone and push to the wrong repository.
* M-5  ``*.session.json`` is user-supplied on vezir's import path: a crafted
       ``started_at`` / ``session_id`` must not place the meeting folder (or
       any artifact) outside the clone, and nothing may be written before
       containment is checked.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from millet import sync

# ``git_identity`` / ``local_remote`` fixtures and ``_git`` live in conftest.
from tests.conftest import _git
from tests.test_sync_core import _make_session

# ─── H-3: origin check on clone reuse ─────────────────────────────────────────


@pytest.mark.parametrize(
    "a, b",
    [
        ("https://github.com/org/repo.git", "https://github.com/org/repo"),
        ("https://github.com/org/repo.git", "https://GitHub.com/Org/Repo.git/"),
        ("https://user:tok@github.com/org/repo.git", "https://github.com/org/repo.git"),
        ("git@github.com:org/repo.git", "ssh://git@github.com/org/repo"),
    ],
)
def test_same_remote_normalizes_equivalent_forms(a, b):
    assert sync._same_remote(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("https://github.com/orgA/meetings.git", "https://github.com/orgB/meetings.git"),
        ("https://github.com/org/meetings.git", "https://gitlab.com/org/meetings.git"),
        ("", "https://github.com/org/repo.git"),
    ],
)
def test_same_remote_distinguishes_different_repos(a, b):
    assert not sync._same_remote(a, b)


def _configure(monkeypatch, tmp_path: Path, repo_url: str) -> None:
    monkeypatch.setattr(sync, "CLONE_BASE_DIR", tmp_path / "clones")
    monkeypatch.setattr(
        sync, "load_sync_config",
        lambda team=None, config_path=None: {"repo_url": repo_url, "meetings": []},
    )


def test_clone_reuse_with_wrong_origin_is_refused(monkeypatch, tmp_path, local_remote):
    """Two remotes with the same basename resolve to the same flat clone dir.

    The second team's sync must refuse rather than ``pull``/``push`` against
    the first team's origin.
    """
    other_remote = tmp_path / "other" / "remote.git"  # same basename: remote.git
    other_remote.parent.mkdir()
    _git("init", "--bare", "--initial-branch=main", str(other_remote))

    # First sync establishes the clone against local_remote.
    _configure(monkeypatch, tmp_path, str(local_remote))
    clone = sync.ensure_repo_cloned(progress_callback=lambda m: None)
    assert clone.exists()
    assert sync._clone_dir_for(str(other_remote), None) == clone  # the collision

    # Now the config points at the OTHER repo with the same basename.
    _configure(monkeypatch, tmp_path, str(other_remote))
    with pytest.raises(RuntimeError, match="points at .* not the configured"):
        sync.ensure_repo_cloned(progress_callback=lambda m: None)

    # And a full sync_session never reaches the copy/commit/push stage.
    sdir = _make_session(tmp_path, "meeting-20260706-100000", "01TESTULID")
    match = sync.MeetingMatch(name="Weekly", folder="weekly")
    with pytest.raises(RuntimeError, match="not the configured"):
        sync.sync_session(sdir, match, progress_callback=lambda m: None)

    check = tmp_path / "check"
    _git("clone", str(local_remote), str(check))
    assert not (check / "meetings").exists(), "nothing may land in the wrong repo"


def test_clone_reuse_with_matching_origin_still_works(monkeypatch, tmp_path, local_remote):
    _configure(monkeypatch, tmp_path, str(local_remote))
    first = sync.ensure_repo_cloned(progress_callback=lambda m: None)
    # Equivalent spelling of the same remote must not trip the guard.
    _configure(monkeypatch, tmp_path, str(local_remote) + "/")
    second = sync.ensure_repo_cloned(progress_callback=lambda m: None)
    assert first == second


def test_team_namespace_gives_each_team_its_own_clone(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "CLONE_BASE_DIR", tmp_path / "clones")
    url_a = "https://github.com/org-a/meetings.git"
    url_b = "https://github.com/org-b/meetings.git"
    assert sync._clone_dir_for(url_a, None) == sync._clone_dir_for(url_b, None)
    assert sync._clone_dir_for(url_a, "teama") != sync._clone_dir_for(url_b, "teamb")


# ─── M-5: session.json cannot steer the target path ──────────────────────────


def _session_with_meta(tmp_path: Path, name: str, meta: dict) -> Path:
    sdir = tmp_path / name
    sdir.mkdir()
    (sdir / f"{name}.session.json").write_text(json.dumps(meta), encoding="utf-8")
    (sdir / f"{name}.summary.md").write_text("# Summary\n")
    (sdir / f"{name}.txt").write_text("transcript\n")
    return sdir


@pytest.mark.parametrize(
    "started_at",
    ["../../../a", "2026-07-06T10:00:00+00:00", "2026-07-06", "garbage", 42, None, ""],
)
def test_date_from_session_never_yields_a_path(tmp_path, started_at):
    sdir = _session_with_meta(tmp_path, "sess", {"started_at": started_at})
    out = sync._date_from_session(sdir)
    assert len(out) == 10 and out[4] == "-" and out[7] == "-"
    assert "/" not in out and ".." not in out
    if started_at in ("2026-07-06T10:00:00+00:00", "2026-07-06"):
        assert out == "2026-07-06"


@pytest.mark.parametrize(
    "bad",
    ["../../x", "a/b", "..", ".", "", "x\x00y", "abc/../../..", "/abs", " ", 123],
)
def test_unsafe_session_id_is_ignored(tmp_path, bad):
    sdir = _session_with_meta(tmp_path, "01GOODULID", {"session_id": bad})
    # Falls back to the (safe) directory name, never the crafted value.
    assert sync._session_id_for(sdir) == "01GOODULID"


def test_resolve_target_dir_cannot_escape(tmp_path):
    base = tmp_path / "repo" / "meetings" / "2026-07-06_weekly"
    base.mkdir(parents=True)
    (base / sync.SESSION_ID_MARKER).write_text("OTHER\n")
    # A crafted id is rejected upstream; here we assert the suffix logic
    # itself keeps the candidate a sibling of base_dir.
    out = sync._resolve_target_dir(base, "01TESTULID", lambda m: None)
    assert out.parent == base.parent


def test_assert_inside_rejects_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    sync._assert_inside(root / "meetings" / "x", root, "ok")
    sync._assert_inside(root, root, "ok")
    with pytest.raises(RuntimeError, match="outside the sync repo"):
        sync._assert_inside(root / ".." / "elsewhere", root, "bad")
    with pytest.raises(RuntimeError, match="outside the sync repo"):
        sync._assert_inside(tmp_path / "repo-sibling", root, "bad")


def test_sync_session_rejects_cloned_sync_root_symlink(
    monkeypatch, tmp_path, git_identity
):
    bare = tmp_path / "escape.git"
    seed = tmp_path / "escape-seed"
    outside = tmp_path / "outside"
    outside.mkdir()
    _git("init", "--bare", "--initial-branch=main", str(bare))
    _git("clone", str(bare), str(seed))
    (seed / "meetings").symlink_to(outside)
    _git("add", "meetings", cwd=seed)
    _git("commit", "-m", "malicious sync root", cwd=seed)
    _git("push", "origin", "HEAD:main", cwd=seed)
    _configure(monkeypatch, tmp_path, str(bare))
    sdir = _session_with_meta(
        tmp_path,
        "meeting-20260706-100000",
        {"started_at": "2026-07-06T10:00:00+00:00", "session_id": "01TESTULID"},
    )

    with pytest.raises(RuntimeError, match="sync root outside the sync repo"):
        sync.sync_session(
            sdir,
            sync.MeetingMatch(name="Weekly", folder="weekly"),
            progress_callback=lambda m: None,
        )

    assert list(outside.iterdir()) == []


def test_sync_session_with_crafted_session_json_writes_nothing_outside(
    monkeypatch, tmp_path, local_remote
):
    """End-to-end: crafted started_at + session_id, real local git remote."""
    _configure(monkeypatch, tmp_path, str(local_remote))
    sdir = _session_with_meta(
        tmp_path, "meeting-crafted",
        {"started_at": "../../../pwn", "session_id": "../../../../../tmp/pwned"},
    )
    match = sync.MeetingMatch(name="Weekly", folder="weekly")
    before = {p for p in tmp_path.rglob("*")}

    copied = sync.sync_session(sdir, match, progress_callback=lambda m: None)

    clone = sync.CLONE_BASE_DIR / "remote"
    # Everything that was written landed inside the clone's meetings dir.
    for p in copied:
        assert (clone / "meetings") in p.parents
    # The only new filesystem entries are the clone itself and the bare
    # remote's git objects from the push — nothing steered by session.json.
    new = {p for p in tmp_path.rglob("*")} - before
    allowed = (sync.CLONE_BASE_DIR, local_remote)
    outside = [p for p in new if not any(p == a or a in p.parents for a in allowed)]
    assert not outside, f"files written outside the clone/remote: {outside}"
    assert not (tmp_path / "pwn_weekly").exists()
    assert not Path("/tmp/pwned").exists()


def test_sync_session_with_crafted_session_json_pushes_sane_folder(
    monkeypatch, tmp_path, local_remote
):
    _configure(monkeypatch, tmp_path, str(local_remote))
    sdir = _session_with_meta(
        tmp_path, "meeting-crafted",
        {"started_at": "../../../pwn", "session_id": "../../x"},
    )
    match = sync.MeetingMatch(name="Weekly", folder="weekly")
    sync.sync_session(sdir, match, progress_callback=lambda m: None)

    check = tmp_path / "check"
    _git("clone", str(local_remote), str(check))
    folders = [p.name for p in (check / "meetings").iterdir() if p.is_dir()]
    assert len(folders) == 1
    # Unparseable started_at -> today's date; unsafe id -> dir name fallback.
    assert folders[0].endswith("_weekly") or "_weekly-" in folders[0]
    assert ".." not in folders[0] and "/" not in folders[0]
    out = subprocess.run(
        ["git", "ls-files"], cwd=check, capture_output=True, text=True, check=True
    ).stdout
    assert all(line.startswith(("meetings/", "README")) for line in out.splitlines())


# ─── L-11: URL policy, host-aware DNS gate, non-interactive git ─────────────


@pytest.mark.parametrize(
    "url, host",
    [
        ("https://github.com/org/repo.git", "github.com"),
        ("https://user:tok@gitlab.example.org/org/repo.git", "gitlab.example.org"),
        ("ssh://git@codeberg.org:22/org/repo", "codeberg.org"),
        ("git@github.com:org/repo.git", "github.com"),
        ("file:///srv/git/repo.git", None),
        ("/srv/git/repo.git", None),
    ],
)
def test_host_from_url(url, host):
    assert sync._host_from_url(url) == host


@pytest.mark.parametrize(
    "bad",
    [
        "git://github.com/org/repo.git",
        "ext::sh -c id",
        "--upload-pack=id",
        "https://github.com/org/repo.git\n--evil",
        "https://github.com/org/re po.git",
    ],
)
def test_validate_repo_url_rejects(bad):
    with pytest.raises(RuntimeError):
        sync._validate_repo_url(bad)


@pytest.mark.parametrize(
    "good",
    [
        "https://github.com/org/repo.git",
        "ssh://git@github.com/org/repo.git",
        "git@github.com:org/repo.git",
        "file:///srv/git/repo.git",
        "/srv/git/repo.git",
    ],
)
def test_validate_repo_url_accepts(good):
    sync._validate_repo_url(good)


def test_git_env_never_prompts(monkeypatch):
    monkeypatch.delenv("GIT_TERMINAL_PROMPT", raising=False)
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.delenv("GIT_SSH", raising=False)
    env = sync._git_env()
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "BatchMode=yes" in env["GIT_SSH_COMMAND"]

    monkeypatch.setenv(
        "GIT_SSH_COMMAND", "ssh -o BatchMode=no -i /custom/key"
    )
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    env = sync._git_env()
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "BatchMode=no" not in env["GIT_SSH_COMMAND"]
    assert "-i /custom/key" in env["GIT_SSH_COMMAND"]
    assert env["GIT_SSH_COMMAND"].endswith("-o BatchMode=yes")


def test_https_clone_without_credentials_fails_fast_not_hangs(tmp_path, monkeypatch):
    """A credential prompt used to hang until vezir's 15-min watchdog fired."""
    import time

    monkeypatch.delenv("GIT_TERMINAL_PROMPT", raising=False)
    # A loopback URL nothing listens on: fails at connect, never at a prompt.
    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        sync._run_network(
            ["git", "clone", "--", "https://127.0.0.1:9/org/repo.git", str(tmp_path / "c")],
            retries=1, delay=0,
        )
    assert time.monotonic() - t0 < 30


def test_run_network_bounds_hanging_dns(monkeypatch, tmp_path):
    import sys
    import time

    started = []

    def failed_run(cmd, **kwargs):
        if cmd[0] == sys.executable:
            started.append(cmd)
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 1, "", "network unavailable")

    monkeypatch.setattr(sync.subprocess, "run", failed_run)
    monkeypatch.setattr(sync, "_DNS_TIMEOUT", 0.02)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="network unavailable"):
        sync._run_network(
            [
                "git",
                "clone",
                "--",
                "https://gitlab.example.org/o/r.git",
                str(tmp_path / "c"),
            ],
            retries=1,
            delay=0,
        )

    assert started
    assert time.monotonic() - t0 < 1


def test_dns_resolution_failure_is_retryable(monkeypatch):
    import socket

    monkeypatch.setattr(
        sync.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", "NXDOMAIN"),
    )
    with pytest.raises(socket.gaierror):
        sync._getaddrinfo_with_timeout("missing.example", 443)


def test_run_network_gates_on_remote_host_not_github(monkeypatch, tmp_path):
    import sys

    looked_up: list[str] = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == sys.executable:
            looked_up.append(cmd[-2])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "no repository")

    monkeypatch.setattr(sync.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        sync._run_network(
            ["git", "clone", "--", "https://gitlab.example.org/o/r.git", str(tmp_path / "c")],
            retries=1, delay=0,
        )
    assert looked_up == ["gitlab.example.org"]


# ─── L-7: transcript.json must not carry the server's absolute audio path ────


def test_transcript_json_stores_audio_basename_only(tmp_path):
    from millet.transcribe import Transcript

    t = Transcript(
        audio_file="/home/op/vezir-data/sessions/01ULID/01ULID.wav",
        language="en", duration=1.0, speakers=[], segments=[],
    )
    data = json.loads(t.to_json())
    assert data["audio_file"] == "01ULID.wav"
    assert "/home/" not in t.to_json()
    # Readers only ever derive stem/date from it; the round-trip keeps that.
    assert Path(data["audio_file"]).stem == "01ULID"
