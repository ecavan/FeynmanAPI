from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feynman_engine.api.app import app


def _frontend_examples() -> list[tuple[str, str, int]]:
    html = (Path(__file__).resolve().parent.parent / "frontend" / "index.html").read_text()
    pattern = re.compile(
        r'<button class="example-btn"[^>]*data-process="([^"]+)"'
        r'[^>]*data-theory="([^"]+)"[^>]*data-loops="([^"]+)"'
    )

    examples: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for process, theory, loops in pattern.findall(html):
        item = (process, theory, int(loops))
        if item not in seen:
            seen.add(item)
            examples.append(item)
    return examples


EXAMPLES = _frontend_examples()


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize(("process", "theory", "loops"), EXAMPLES)
def test_frontend_examples_generate_and_report_availability(
    client: TestClient,
    process: str,
    theory: str,
    loops: int,
):
    generate = client.post(
        "/api/generate",
        json={
            "process": process,
            "theory": theory,
            "loops": loops,
            "output_format": "tikz",
            "filters": {
                "no_tadpoles": True,
                "one_pi": False,
                "connected": True,
                "unique_topologies": False,
            },
        },
    )
    assert generate.status_code == 200, generate.text

    summary = generate.json()["summary"]

    amplitude = client.get(
        "/api/amplitude",
        params={"process": process, "theory": theory, "loops": loops},
    )
    assert amplitude.status_code == 200, amplitude.text
    payload = amplitude.json()
    assert "has_msq" in payload
    assert "has_integral" in payload

    # Every sidebar example must either generate diagrams or have a curated |M|².
    has_diagrams = summary["total_diagrams"] > 0
    has_amplitude = payload["has_msq"]
    assert has_diagrams or has_amplitude, (
        f"Example '{process}' [{theory}, loops={loops}] has neither diagrams nor |M|²"
    )

    assert payload["has_msq"] == payload["supported"]
    assert payload["has_integral"] == bool(payload.get("integral_latex"))
    if payload["has_msq"]:
        assert payload["msq_sympy"]


def test_curated_amplitudes_backfill_integral_representation(client: TestClient):
    for process, theory in [
        ("e+ e- -> gamma gamma", "QED"),
        ("u u~ -> g g", "QCD"),
        ("e+ e- -> Z H", "EW"),
    ]:
        response = client.get("/api/amplitude", params={"process": process, "theory": theory})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["has_msq"] is True
        assert payload["has_integral"] is True
        assert payload["integral_latex"]


def test_valid_tree_process_without_exact_backend_returns_approximate_payload(client: TestClient):
    response = client.get("/api/amplitude", params={"process": "e- nu_e~ -> W- Z", "theory": "EW"})
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["supported"] is True
    assert payload["has_msq"] is True
    assert payload["has_integral"] is True
    assert payload["approximation_level"] == "approximate-pointwise"
    assert payload["evaluation_point"]


def test_decay_process_returns_exact_symbolic(client: TestClient):
    """Z → e+e- now has an exact analytic decay backend."""
    response = client.get("/api/amplitude", params={"process": "Z -> e+ e-", "theory": "EW"})
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["supported"] is True
    assert payload["has_msq"] is True
    assert payload["has_integral"] is True
    assert payload["approximation_level"] == "exact-symbolic"


def test_non_example_qed_tau_process_generates_and_returns_msq(client: TestClient):
    generate = client.post(
        "/api/generate",
        json={
            "process": "e+ e- -> tau+ tau-",
            "theory": "QED",
            "loops": 0,
            "output_format": "tikz",
            "filters": {
                "no_tadpoles": True,
                "one_pi": False,
                "connected": True,
                "unique_topologies": False,
            },
        },
    )
    assert generate.status_code == 200, generate.text
    assert generate.json()["summary"]["total_diagrams"] > 0

    response = client.get("/api/amplitude", params={"process": "e+ e- -> tau+ tau-", "theory": "QED"})
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["has_msq"] is True
    assert payload["has_integral"] is True


def test_loop_process_returns_pointwise_proxy_when_available(client: TestClient):
    response = client.get(
        "/api/amplitude",
        params={"process": "e+ e- -> mu+ mu-", "theory": "QED", "loops": 1},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["supported"] is True
    assert payload["has_msq"] is True
    assert payload["has_integral"] is True
    assert payload["approximation_level"] == "approximate-pointwise"
    assert payload["evaluation_point"]


def test_valid_but_unsupported_loop_process_returns_unavailable_payload(client: TestClient):
    response = client.get(
        "/api/amplitude/loop-integral",
        params={"process": "e- nu_e -> W- Z", "theory": "EW", "loops": 1},
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["has_integral"] is False
    assert payload["availability_message"]
