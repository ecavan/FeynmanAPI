from __future__ import annotations

from pathlib import Path

import pytest

from feynman_engine import __main__ as cli
import feynman_engine.form as form_mod
import feynman_engine.looptools as looptools_mod
import feynman_engine.qgraf as qgraf_mod


def test_setup_runs_all_installers_in_order(monkeypatch, capsys, tmp_path: Path):
    calls: list[str] = []

    def fake_qgraf(*, force: bool = False):
        calls.append(f"qgraf:{force}")
        return tmp_path / "qgraf"

    def fake_form(*, force: bool = False):
        calls.append(f"form:{force}")
        return tmp_path / "form"

    def fake_looptools(*, force: bool = False):
        calls.append(f"looptools:{force}")
        return tmp_path / "liblooptools.so"

    monkeypatch.setattr(qgraf_mod, "build_qgraf", fake_qgraf)
    monkeypatch.setattr(form_mod, "build_form", fake_form)
    monkeypatch.setattr(looptools_mod, "build_looptools", fake_looptools)
    monkeypatch.setattr("sys.argv", ["feynman", "setup", "--force"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert calls == ["qgraf:True", "form:True", "looptools:True"]
    assert "Setup summary:" in out
    assert "QGRAF: ok" in out
    assert "FORM: ok" in out
    assert "LoopTools: ok" in out


def test_setup_can_skip_looptools(monkeypatch, capsys, tmp_path: Path):
    calls: list[str] = []

    def fake_qgraf(*, force: bool = False):
        calls.append("qgraf")
        return tmp_path / "qgraf"

    def fake_form(*, force: bool = False):
        calls.append("form")
        return tmp_path / "form"

    def fail_if_called(*, force: bool = False):  # pragma: no cover - safety check
        raise AssertionError("LoopTools should have been skipped")

    monkeypatch.setattr(qgraf_mod, "build_qgraf", fake_qgraf)
    monkeypatch.setattr(form_mod, "build_form", fake_form)
    monkeypatch.setattr(looptools_mod, "build_looptools", fail_if_called)
    monkeypatch.setattr("sys.argv", ["feynman", "setup", "--skip-looptools"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert calls == ["qgraf", "form"]
    assert "LoopTools: skipped - not requested" in out


def test_setup_reports_failure_and_exits_nonzero(monkeypatch, capsys, tmp_path: Path):
    def fake_qgraf(*, force: bool = False):
        return tmp_path / "qgraf"

    def fake_form(*, force: bool = False):
        raise RuntimeError("missing compiler")

    def fake_looptools(*, force: bool = False):
        return tmp_path / "liblooptools.so"

    monkeypatch.setattr(qgraf_mod, "build_qgraf", fake_qgraf)
    monkeypatch.setattr(form_mod, "build_form", fake_form)
    monkeypatch.setattr(looptools_mod, "build_looptools", fake_looptools)
    monkeypatch.setattr("sys.argv", ["feynman", "install-all"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    out = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "FORM: failed - missing compiler" in out
    assert "One or more setup steps failed." in out


def test_doctor_json_prints_diagnostics(monkeypatch, capsys):
    fake = {
        "backend": "qgraf (/tmp/qgraf)",
        "qgraf": {"available": True, "binary_path": "/tmp/qgraf", "source_available": True},
        "form": {"available": True, "binary_path": "/tmp/form", "source_available": True},
        "looptools": {"available": False, "library_path": None},
        "rendering": {
            "lualatex_available": False,
            "lualatex_path": None,
            "pdf2svg_available": True,
            "pdf2svg_path": "/tmp/pdf2svg",
        },
        "toolchain": {"gfortran_path": "/tmp/gfortran", "make_path": "/tmp/make", "cc_path": "/tmp/cc"},
    }

    monkeypatch.setattr("feynman_engine.diagnostics.collect_diagnostics", lambda: fake)
    monkeypatch.setattr("sys.argv", ["feynman", "doctor", "--json"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert '"backend": "qgraf (/tmp/qgraf)"' in out
    assert '"available": false' in out.lower()


def test_doctor_human_output_returns_nonzero_when_core_tools_missing(monkeypatch, capsys):
    fake = {
        "backend": "qgraf-unavailable",
        "qgraf": {"available": False, "binary_path": None, "source_available": True},
        "form": {"available": False, "binary_path": None, "source_available": True},
        "looptools": {"available": False, "library_path": None},
        "rendering": {
            "lualatex_available": False,
            "lualatex_path": None,
            "pdf2svg_available": False,
            "pdf2svg_path": None,
        },
        "toolchain": {"gfortran_path": None, "make_path": None, "cc_path": None},
    }

    monkeypatch.setattr("feynman_engine.diagnostics.collect_diagnostics", lambda: fake)
    monkeypatch.setattr("sys.argv", ["feynman", "doctor"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    out = capsys.readouterr().out
    assert excinfo.value.code == 1
    assert "FeynmanEngine doctor" in out
    assert "Recommendation: run `feynman setup`" in out
