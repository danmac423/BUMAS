import pytest

from lab_3.submission import validate_submission, write_submission

TEST_IDS = ["a", "b", "c"]


def test_valid_submission_is_written(tmp_path):
    path = tmp_path / "submission.csv"
    write_submission(path, ["c", "a", "b"], [1.0, -2.5, 0.0], TEST_IDS)
    assert path.read_text() == "id,score\nc,1.0\na,-2.5\nb,0.0\n"


@pytest.mark.parametrize(
    ("ids", "scores"),
    [
        (["a", "b"], [1.0, 2.0]),
        (["a", "b", "b"], [1.0, 2.0, 3.0]),
        (["a", "b", "c", "c"], [1.0, 2.0, 3.0, 4.0]),
        (["a", "b", "x"], [1.0, 2.0, 3.0]),
        (["a", "b", "c"], [1.0, float("nan"), 3.0]),
        (["a", "b", "c"], [1.0, float("inf"), 3.0]),
        (["a", "b", "c"], [1.0, 2.0]),
    ],
)
def test_invalid_submission_is_rejected(ids, scores):
    with pytest.raises(ValueError):
        validate_submission(ids, scores, TEST_IDS)
