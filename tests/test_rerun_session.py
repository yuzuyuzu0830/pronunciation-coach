from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import rerun_session


def test_load_records_skips_blank_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "trial_log.jsonl"
    record = {"app_session_id": "session-1"}
    log_path.write_text(f"{json.dumps(record)}\n\n", encoding="utf-8")

    assert rerun_session.load_records(log_path) == [record]


def test_load_records_identifies_invalid_json_line(tmp_path: Path) -> None:
    log_path = tmp_path / "trial_log.jsonl"
    log_path.write_text('{"app_session_id": "session-1"}\ninvalid\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"trial_log\.jsonl.*line 2"):
        rerun_session.load_records(log_path)


def test_load_records_rejects_non_object_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "trial_log.jsonl"
    log_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected a JSON object.*line 1"):
        rerun_session.load_records(log_path)


def test_rerun_rejects_existing_log_before_loading_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "rerun"
    out_dir.mkdir()
    (out_dir / "trial_log.jsonl").touch()

    def fail_if_called(*, trial_logs_dir: Path) -> None:
        del trial_logs_dir
        pytest.fail("load_models must not run when the output log already exists")

    monkeypatch.setattr(rerun_session, "load_models", fail_if_called)

    with pytest.raises(SystemExit) as exc_info:
        rerun_session.rerun(
            [], source_dir=tmp_path, out_dir=out_dir, session_prefix="session-1"
        )

    assert exc_info.value.code == 1
