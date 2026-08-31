"""
Tests for src/train/checkpointing.py -- the per-epoch, per-fold resume
mechanism for the Colab training scripts.
"""
import torch
import torch.nn as nn

from src.train.checkpointing import FoldCheckpointer


def _tiny_model_and_opt():
    model = nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt


def test_fresh_checkpointer_has_no_state(tmp_path):
    ck = FoldCheckpointer(tmp_path)
    assert not ck.is_fold_done("2022")
    assert ck.resume_epoch_for("2022") == 0
    assert ck.completed_results() == []


def test_save_and_resume_across_a_new_process(tmp_path):
    model, opt = _tiny_model_and_opt()
    with torch.no_grad():
        model.weight.fill_(1.23)
    ck = FoldCheckpointer(tmp_path)
    ck.save_epoch("2022", epoch=3, model=model, optimizer=opt)

    # simulate a fresh process: new Checkpointer instance, new model/optimizer
    ck2 = FoldCheckpointer(tmp_path)
    assert ck2.resume_epoch_for("2022") == 3
    model2, opt2 = _tiny_model_and_opt()
    loaded = ck2.load_model_state("2022", model2, opt2, device=torch.device("cpu"))
    assert loaded is True
    assert torch.allclose(model2.weight, model.weight)


def test_mark_fold_complete_clears_in_progress_state(tmp_path):
    model, opt = _tiny_model_and_opt()
    ck = FoldCheckpointer(tmp_path)
    ck.save_epoch("2022", epoch=2, model=model, optimizer=opt)
    assert (tmp_path / "checkpoints" / "2022.pt").exists()

    ck.mark_fold_complete("2022", {"fold": "2022", "mean_ic": 0.05})

    assert ck.is_fold_done("2022")
    assert ck.completed_results() == [{"fold": "2022", "mean_ic": 0.05}]
    assert ck.resume_epoch_for("2022") == 0  # current_fold cleared
    assert not (tmp_path / "checkpoints" / "2022.pt").exists()  # cleaned up


def test_completed_folds_persist_across_new_instances(tmp_path):
    ck = FoldCheckpointer(tmp_path)
    ck.mark_fold_complete("2022", {"fold": "2022", "mean_ic": 0.05})

    ck2 = FoldCheckpointer(tmp_path)
    assert ck2.is_fold_done("2022")
    assert ck2.completed_results() == [{"fold": "2022", "mean_ic": 0.05}]


def test_fresh_flag_wipes_existing_state(tmp_path):
    model, opt = _tiny_model_and_opt()
    ck = FoldCheckpointer(tmp_path)
    ck.save_epoch("2022", epoch=2, model=model, optimizer=opt)
    ck.mark_fold_complete("2023", {"fold": "2023"})

    ck2 = FoldCheckpointer(tmp_path, fresh=True)
    assert not ck2.is_fold_done("2023")
    assert ck2.resume_epoch_for("2022") == 0
    assert not (tmp_path / "checkpoints" / "2022.pt").exists()


def test_load_model_state_returns_false_without_crashing_on_shape_mismatch(tmp_path):
    model, opt = _tiny_model_and_opt()
    ck = FoldCheckpointer(tmp_path)
    ck.save_epoch("2022", epoch=1, model=model, optimizer=opt)

    wrong_shape_model = nn.Linear(8, 1)  # different in_features -> state_dict mismatch
    wrong_opt = torch.optim.Adam(wrong_shape_model.parameters(), lr=1e-3)
    ck2 = FoldCheckpointer(tmp_path)
    loaded = ck2.load_model_state("2022", wrong_shape_model, wrong_opt, device=torch.device("cpu"))
    assert loaded is False  # must not raise


def test_config_mismatch_warns_but_does_not_raise(tmp_path, capsys):
    model, opt = _tiny_model_and_opt()
    ck = FoldCheckpointer(tmp_path, run_args={"hidden_size": 128})
    ck.save_epoch("2022", 1, model, opt)

    ck2 = FoldCheckpointer(tmp_path, run_args={"hidden_size": 64})  # different -> should warn, not raise
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert ck2 is not None  # constructed successfully despite the mismatch


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    model, opt = _tiny_model_and_opt()
    ck = FoldCheckpointer(tmp_path)
    ck.save_epoch("2022", epoch=1, model=model, optimizer=opt)
    assert not (tmp_path / "checkpoints" / "2022.pt.tmp").exists()
    assert (tmp_path / "checkpoints" / "2022.pt").exists()
