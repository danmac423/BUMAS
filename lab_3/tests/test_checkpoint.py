from lab_3.checkpoint import Checkpoint

META = {"model": "m", "dtype": "fp32"}


def test_saved_results_are_resumed(tmp_path):
    path = tmp_path / "partial.pkl"
    first = Checkpoint(path, META, interval=0)
    first.update({"a": 1.0})
    first.update({"b": 2.0})
    assert Checkpoint(path, META).results == {"a": 1.0, "b": 2.0}
    assert list(tmp_path.iterdir()) == [path]


def test_nothing_is_written_before_the_interval_unless_saved(tmp_path):
    path = tmp_path / "partial.pkl"
    ckpt = Checkpoint(path, META, interval=3600)
    ckpt.update({"a": 1.0})
    assert not path.exists()
    ckpt.save()
    assert Checkpoint(path, META).results == {"a": 1.0}


def test_results_from_other_settings_are_ignored(tmp_path):
    path = tmp_path / "partial.pkl"
    Checkpoint(path, META, interval=0).update({"a": 1.0})
    assert Checkpoint(path, {**META, "dtype": "bf16"}).results == {}
