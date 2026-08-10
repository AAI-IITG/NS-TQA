import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.pronostia import acceleration_files, bearing_feature_trajectory, load_pronostia


def _write_acc(path: Path, values: list[tuple[float, float]]) -> None:
    with open(path, "w") as f:
        for i, (h, v) in enumerate(values):
            f.write(f"0,0,0,{i},{h},{v}\n")


def test_pronostia_loader_builds_feature_trajectories(tmp_path):
    bearing = tmp_path / "Learning_set" / "Bearing1_1"
    bearing.mkdir(parents=True)
    _write_acc(bearing / "acc_00002.csv", [(1.0, -1.0), (2.0, -2.0), (3.0, -3.0)])
    _write_acc(bearing / "acc_00001.csv", [(0.5, 0.5), (1.5, 1.5), (2.5, 2.5)])

    ordered = acceleration_files(bearing)
    assert [p.name for p in ordered] == ["acc_00001.csv", "acc_00002.csv"]

    traj, names = bearing_feature_trajectory(bearing, axis="both", n_bands=1)
    assert traj.shape == (2, 10)
    assert names == [
        "h_rms", "h_kurtosis", "h_peak", "h_crest", "h_band0",
        "v_rms", "v_kurtosis", "v_peak", "v_crest", "v_band0",
    ]

    ds = load_pronostia(tmp_path, splits=["Learning_set"], axis="both", n_bands=1)
    assert ds.bearing_ids == ["Learning_set/Bearing1_1"]
    assert ds.trajectories[0].shape[0] == 2
    assert ds.rul[0].tolist() == [1.0, 0.0]
