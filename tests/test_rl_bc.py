from __future__ import annotations

import json
from pathlib import Path

from typesafe_galaga.cli import _demo_ram
from typesafe_galaga.rl.bc import load_demonstrations, record_to_snapshot, train_bc
from typesafe_galaga.rl.observations import BASE_OBSERVATION_DIM, OBSERVATION_DIM
from typesafe_galaga.state import GalagaStateParser


def test_load_demonstrations_and_train_bc(tmp_path: Path) -> None:
    snap = GalagaStateParser().parse(
        {"stage": 1, "lives": 3, "score": 260},
        _demo_ram(),
        audio_events=("laser_shot", "dive_swoop"),
    )
    log_file = tmp_path / "galaga_demo.jsonl"
    records = []
    for idx, action in enumerate(("fire", "left", "right", "noop") * 6):
        records.append(
            json.dumps(
                {
                    "decision": idx,
                    "state": snap.to_state(),
                    "debug_state": snap.to_debug_state(),
                    "action": action,
                    "confidence": 0.9,
                    "probabilities": {action: 0.9},
                }
            )
        )
    log_file.write_text("\n".join(records) + "\n", encoding="utf-8")

    reconstructed = record_to_snapshot(json.loads(records[0]))
    assert reconstructed.x == snap.x
    assert reconstructed.score == 260
    assert reconstructed.audio_events == ("laser_shot", "dive_swoop")

    dataset = load_demonstrations([log_file], min_confidence=0.5)
    assert len(dataset) == 24
    assert dataset.observations.shape == (24, OBSERVATION_DIM)

    legacy_dataset = load_demonstrations([log_file], min_confidence=0.5, include_audio_obs=False)
    assert legacy_dataset.observations.shape == (24, BASE_OBSERVATION_DIM)

    stacked_dataset = load_demonstrations([log_file], min_confidence=0.5, frame_stack=4)
    assert stacked_dataset.observations.shape == (24, OBSERVATION_DIM * 4)

    model_path = tmp_path / "galaga_bc.zip"
    result = train_bc(dataset, model_path=model_path, epochs=3, batch_size=8)
    assert result.model_path.exists()
    assert result.samples == 24
