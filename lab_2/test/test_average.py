import torch

from lab_2.average import FC_WEIGHT, average_state_dicts
from lab_2.model import Net


def _save(tmp_path, name, seed):
    torch.manual_seed(seed)
    sd = Net().state_dict()
    p = tmp_path / f"{name}.pt"
    torch.save(sd, p)
    return p, sd


def test_average_normalises_fc_rows_and_averages_others(tmp_path):
    p1, s1 = _save(tmp_path, "a", 0)
    p2, s2 = _save(tmp_path, "b", 1)
    out = average_state_dicts([p1, p2])
    assert torch.allclose(out[FC_WEIGHT].norm(dim=1), torch.ones(20), atol=1e-5)
    k = "r.conv1.weight"
    assert torch.allclose(out[k], (s1[k] + s2[k]) / 2, atol=1e-6)
    Net().load_state_dict(out)  # loads cleanly, incl. integer buffers
