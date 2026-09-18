import json

import pytest

from martini_mapper import frozen_mapping
from martini_mapper.frozen_mapping import (
    FrozenMappingError,
    _adapter_tokens,
    run_local_mapping,
)
from martini_mapper.main import main


BENZENE_MAPPED = "[cH:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1"
PREVIOUS = [
    {"atoms": [1, 2], "type": "TC5"},
    {"atoms": [3, 4], "type": "TC5"},
    {"atoms": [5, 6], "type": "TC5"},
]


def signature(group):
    return frozenset(group["atoms"]), group["type"]


def test_adapter_keeps_large_ring_ids_inside_token_space(monkeypatch):
    tokens = ["C", "1", "C", "1", "C", "1", "C", "1", "C", "99", "C", "99"]
    monkeypatch.setattr(frozen_mapping, "parse_smiles", lambda _: tokens.copy())

    adapted = _adapter_tokens("valid-smiles-is-parsed-before-this-test")

    assert adapted == [
        "C", "1", "C", "1", "C", "100", "C", "100", "C", "99", "C", "99"
    ]
    assert "%100" not in adapted


def test_adapter_rejects_unpaired_ring_labels(monkeypatch):
    monkeypatch.setattr(frozen_mapping, "parse_smiles", lambda _: ["C", "1", "C"])

    with pytest.raises(FrozenMappingError, match="unpaired ring label"):
        _adapter_tokens("invalid-token-pairing")


def test_strict_local_mapping_preserves_every_frozen_bead():
    result = run_local_mapping(
        BENZENE_MAPPED,
        PREVIOUS,
        [1, 2],
        context_layers=1,
    )
    output = {signature(group) for group in result["groups"]}
    assert (frozenset({3, 4}), "TC5") in output
    assert (frozenset({5, 6}), "TC5") in output
    assert result["frozen_violations"] == 0
    assert sorted(atom for group in result["groups"] for atom in group["atoms"]) == list(range(1, 7))
    assert set(result["editable_atom_ids"]) == {1, 2}


def test_boundary_candidate_is_trimmed_to_allowed_singleton_beads():
    # Full benzene mapping pairs 1 with 2 and 6 with 5.  Only the old bead
    # [1, 6] is editable here, so both candidate pairs cross a frozen boundary.
    # The strict local path must keep the frozen pairs unchanged and may emit
    # the allowed one-atom beads [1] and [6].
    previous = [
        {"atoms": [1, 6], "type": "TC5"},
        {"atoms": [2, 3], "type": "TC5", "bead_id": "frozen-A"},
        {"atoms": [4, 5], "type": "TC5"},
    ]
    result = run_local_mapping(BENZENE_MAPPED, previous, [1], context_layers=1)
    output = {signature(group) for group in result["groups"]}
    assert (frozenset({2, 3}), "TC5") in output
    assert (frozenset({4, 5}), "TC5") in output
    assert (frozenset({1}), "TC5") in output
    assert (frozenset({6}), "TC5") in output
    assert result["boundary_trimmed_candidates"] == 2
    assert result["singleton_beads"] == 2
    assert result["frozen_violations"] == 0
    frozen_with_id = next(group for group in result["groups"] if group["atoms"] == [2, 3])
    assert frozen_with_id["bead_id"] == "frozen-A"


def test_strict_local_mapping_requires_persistent_atom_maps():
    with pytest.raises(FrozenMappingError, match="atom-map"):
        run_local_mapping("c1ccccc1", PREVIOUS, [1, 2])


def test_cli_local_mapper_writes_json(tmp_path, capsys):
    previous_path = tmp_path / "previous.json"
    output_path = tmp_path / "local.json"
    previous_path.write_text(json.dumps(PREVIOUS), encoding="utf-8")
    rc = main([
        "benzene-local",
        BENZENE_MAPPED,
        "--local-mapper",
        "--previous-mapping",
        str(previous_path),
        "--reaction-atoms",
        "1,2",
        "--local-output",
        str(output_path),
        "--no-xtb",
        "--no-files",
    ])
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert captured == saved
    assert saved["mode"] == "strict_local_mapper"
    assert saved["frozen_violations"] == 0


def test_cli_local_mapper_requires_inputs():
    with pytest.raises(SystemExit) as exc:
        main(["benzene-local", BENZENE_MAPPED, "--local-mapper"])
    assert exc.value.code == 2
