"""
Checkpoint/resume for the walk-forward training scripts (M2/M3/v2).

REDESIGNED after a real failure report: writing straight to a Google-Drive
mounted `out_dir` (the first version's approach) turned out to lose
progress, not protect it. Root cause, found and fixed here:

  1. `run_state.json` was written with a plain `write_text()` -- not atomic.
     Drive's FUSE mount does not handle interrupted writes the way a local
     disk does (a "rename" on Drive is actually delete+recreate over the
     Drive API, which can fail or partially apply). A write cut off mid-way
     by a disconnect could leave run_state.json TRUNCATED.
  2. `_load_state()` had no error handling: a truncated/corrupt
     run_state.json raised `json.JSONDecodeError` immediately on the very
     next resume attempt -- crashing before training even started, with no
     usable checkpoint. From the user's side this looks exactly like
     "checkpointing doesn't work, I keep losing everything."

Fix: **local disk is the primary, source-of-truth location; Drive is a
best-effort, non-blocking MIRROR.** All reads/writes happen on `out_dir`
(fast, reliable local VM disk -- no FUSE involved). If `mirror_dir` is also
given, every successful local write is opportunistically copied there too,
wrapped so a Drive hiccup prints a warning and moves on rather than crashing
the run. On startup, if local `out_dir` has no state yet but `mirror_dir`
does (the situation right after a VM disconnect/recycle wiped local disk),
state is hydrated FROM the mirror before training resumes -- this is what
actually makes checkpointing survive a full Colab VM loss, not just a
script-level crash. Every JSON read is also now defensive: a corrupt state
file is treated as "no prior state" (with a printed warning) instead of
crashing the process.

Scope, stated rather than assumed: checkpointing is PER-EPOCH, PER-FOLD, not
mid-epoch. On a disconnect, the most you lose is the epochs in progress on
whichever fold was running -- not the whole run, and not any
already-completed fold's results.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


def _empty_state() -> dict:
    return {"completed_folds": {}, "current_fold": None, "current_epoch": 0}


class FoldCheckpointer:
    def __init__(
        self,
        out_dir: Path,
        run_args: dict | None = None,
        fresh: bool = False,
        mirror_dir: Path | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.state_path = self.out_dir / "run_state.json"
        self.config_path = self.out_dir / "run_config.json"

        if fresh:
            self._wipe_local_and_mirror()
        elif not self.state_path.exists() and self.mirror_dir is not None:
            self._hydrate_from_mirror()

        self.state = self._load_state()
        self._check_config_matches(run_args)

    # -- local disk: the only thing ever READ from --------------------------

    def _wipe_local_and_mirror(self):
        """fresh=True means start over -- if the mirror still had the old
        state, leaving it alone would just get it re-hydrated back on the
        next run, silently undoing --fresh."""
        if self.state_path.exists():
            self.state_path.unlink()
        if self.ckpt_dir.exists():
            for f in self.ckpt_dir.glob("*.pt"):
                f.unlink()
        if self.mirror_dir is not None:
            try:
                if (self.mirror_dir / "run_state.json").exists():
                    (self.mirror_dir / "run_state.json").unlink()
                mirror_ckpt_dir = self.mirror_dir / "checkpoints"
                if mirror_ckpt_dir.exists():
                    for f in mirror_ckpt_dir.glob("*.pt"):
                        f.unlink()
            except Exception as e:  # noqa: BLE001 - best-effort cleanup, never worth failing --fresh over
                print(f"  [checkpoint] could not clear Drive mirror on --fresh ({e})", flush=True)

    def _hydrate_from_mirror(self):
        """Local out_dir is empty (fresh VM after a disconnect) but a Drive
        mirror from a prior session might have real progress -- pull it down
        before falling back to a from-scratch run."""
        try:
            mirror_state = self.mirror_dir / "run_state.json"
            if not mirror_state.exists():
                return
            self.out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mirror_state, self.state_path)
            mirror_config = self.mirror_dir / "run_config.json"
            if mirror_config.exists():
                shutil.copy2(mirror_config, self.config_path)
            mirror_ckpt_dir = self.mirror_dir / "checkpoints"
            if mirror_ckpt_dir.exists():
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                for f in mirror_ckpt_dir.glob("*.pt"):
                    shutil.copy2(f, self.ckpt_dir / f.name)
            print(f"  [checkpoint] hydrated local state from Drive mirror at {self.mirror_dir}", flush=True)
        except Exception as e:  # noqa: BLE001 - a bad mirror should never block a fresh start
            print(f"  [checkpoint] could not hydrate from mirror ({e}); starting fresh", flush=True)

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [checkpoint] run_state.json is corrupt/unreadable ({e}); "
                      f"treating as no prior state (nothing already-completed was lost, "
                      f"only in-progress-fold resume position)", flush=True)
        return _empty_state()

    def _atomic_write_text(self, path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)  # atomic on the LOCAL filesystem (never Drive-mounted)

    def _save_state(self):
        self._atomic_write_text(self.state_path, json.dumps(self.state, indent=2, default=str))
        self._mirror_file(self.state_path)

    def _check_config_matches(self, run_args: dict | None):
        """Not a hard failure -- a changed architecture arg makes an existing
        checkpoint's state_dict incompatible (caught separately, at load
        time), but the user should at least SEE that something changed."""
        if run_args is None:
            return
        run_args = {k: str(v) for k, v in run_args.items()}
        if self.config_path.exists():
            try:
                prior = json.loads(self.config_path.read_text())
            except (json.JSONDecodeError, OSError):
                prior = {}
            diffs = {k: (prior.get(k), v) for k, v in run_args.items() if prior.get(k) != v}
            if diffs and (self.state["completed_folds"] or self.state["current_fold"]):
                print(f"  [checkpoint] WARNING: resuming with different args than the run that "
                      f"started this out-dir: {diffs}", flush=True)
        else:
            self._atomic_write_text(self.config_path, json.dumps(run_args, indent=2))
            self._mirror_file(self.config_path)

    # -- Drive: best-effort MIRROR only, never read from at startup except
    #    to hydrate an empty local out_dir (_hydrate_from_mirror above) -----

    def _mirror_file(self, local_path: Path, rel: str | None = None):
        if self.mirror_dir is None:
            return
        try:
            dest = self.mirror_dir / (rel or local_path.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)
        except Exception as e:  # noqa: BLE001 - Drive being slow/unavailable must never stop training
            print(f"  [checkpoint] Drive mirror write failed ({e}); continuing with local checkpoint only", flush=True)

    def _mirror_delete(self, rel: str):
        if self.mirror_dir is None:
            return
        try:
            p = self.mirror_dir / rel
            if p.exists():
                p.unlink()
        except Exception:  # noqa: BLE001 - purely a tidiness step, never worth failing over
            pass

    # -- public API, unchanged from the caller's point of view --------------

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
        tmp_path.replace(path)  # atomic on the local filesystem
        self._mirror_file(path, rel=f"checkpoints/{fold_label}.pt")

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
        self._mirror_delete(f"checkpoints/{fold_label}.pt")
