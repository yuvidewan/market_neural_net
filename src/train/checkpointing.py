"""
Checkpoint/resume for the walk-forward training scripts (M2/M3/v2). This is
the gap flagged repeatedly in the README's Colab notebook: Colab sessions
disconnect, and a multi-hour run with no way to resume loses everything.

Scope, stated rather than assumed: checkpointing is PER-EPOCH, PER-FOLD, not
mid-epoch. On a disconnect, the most you lose is the epochs in progress on
whichever fold was running when it happened -- not the whole run, and not
any already-completed fold's results. Finer-grained (mid-epoch) checkpointing
would need to also persist the shuffled sample order and dataloader position;
not worth the complexity for what this actually needs to solve.

Everything lives under `{out_dir}/checkpoints/` and `{out_dir}/run_state.json`
-- both plain files, safe to delete to force a completely fresh run (or pass
`fresh=True`), and safe to sync to Drive alongside the rest of `experiments/`
since none of it is large (one fold's model+optimizer state, not a dataset).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class FoldCheckpointer:
    def __init__(self, out_dir: Path, run_args: dict | None = None, fresh: bool = False):
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.state_path = self.out_dir / "run_state.json"
        self.config_path = self.out_dir / "run_config.json"

        if fresh:
            if self.state_path.exists():
                self.state_path.unlink()
            if self.ckpt_dir.exists():
                for f in self.ckpt_dir.glob("*.pt"):
                    f.unlink()

        self.state = self._load_state()
        self._check_config_matches(run_args)

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"completed_folds": {}, "current_fold": None, "current_epoch": 0}

    def _save_state(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    def _check_config_matches(self, run_args: dict | None):
        """Not a hard failure -- a changed architecture arg makes an existing
        checkpoint's state_dict incompatible (caught separately, at load
        time), but the user should at least SEE that something changed."""
        if run_args is None:
            return
        run_args = {k: str(v) for k, v in run_args.items()}
        if self.config_path.exists():
            prior = json.loads(self.config_path.read_text())
            diffs = {k: (prior.get(k), v) for k, v in run_args.items() if prior.get(k) != v}
            if diffs and (self.state["completed_folds"] or self.state["current_fold"]):
                print(f"  [checkpoint] WARNING: resuming with different args than the run that "
                      f"started this out-dir: {diffs}", flush=True)
        else:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(run_args, indent=2))

    def is_fold_done(self, fold_label: str) -> bool:
        return fold_label in self.state["completed_folds"]

    def completed_results(self) -> list[dict]:
        return list(self.state["completed_folds"].values())

    def resume_epoch_for(self, fold_label: str) -> int:
        """Next epoch to run for this fold: 0 if there's no in-progress
        checkpoint for it (fresh start), else the epoch count already done."""
        if self.state.get("current_fold") == fold_label:
            return self.state.get("current_epoch", 0)
        return 0

    def load_model_state(self, fold_label: str, model, optimizer, device) -> bool:
        """Returns True if state was actually loaded. False (with a printed
        reason) on any mismatch -- caller should then just train from epoch 0
        rather than crash the whole run over a stale/incompatible checkpoint."""
        path = self.ckpt_dir / f"{fold_label}.pt"
        if self.state.get("current_fold") != fold_label or not path.exists():
            return False
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            return True
        except Exception as e:  # noqa: BLE001 - a bad checkpoint should never crash the run
            print(f"  [checkpoint] could not load {path} ({e}); starting fold {fold_label} from epoch 0", flush=True)
            return False

    def save_epoch(self, fold_label: str, epoch: int, model, optimizer):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"{fold_label}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, tmp_path)
        tmp_path.replace(path)  # atomic on POSIX and NTFS -- never leaves a half-written checkpoint
        self.state["current_fold"] = fold_label
        self.state["current_epoch"] = epoch
        self._save_state()

    def mark_fold_complete(self, fold_label: str, result: dict):
        self.state["completed_folds"][fold_label] = result
        self.state["current_fold"] = None
        self.state["current_epoch"] = 0
        self._save_state()
        ckpt_path = self.ckpt_dir / f"{fold_label}.pt"
        if ckpt_path.exists():
            ckpt_path.unlink()  # fold is done, its own result dict is now the durable record
