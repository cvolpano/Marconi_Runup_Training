# HPC / headless changes

This copy keeps the original segmentation_gym workflow but removes Tk GUI requirements from the canonical scripts and GUI-using utilities. Replaced code is retained immediately nearby as `# PREVIOUS:` comments.

## What changed

- Tk directory/file dialogs were replaced with terminal `input()` path prompts. Paths may be pasted with or without surrounding single/double quotes.
- Tk yes/no message boxes were replaced with terminal `[y/N]` prompts.
- Multi-file weight dialogs in `utils/gen_saved_model.py` and `utils/gen_fullmodel_from_h5.py` now accept comma-separated paths.
- Plot-producing scripts set `MPLBACKEND=Agg`, a non-interactive Matplotlib backend. Existing `savefig()` diagnostics and training-history outputs are retained; no display server is required.
- Existing `*_no_tkinter.py` files were left untouched because they contain broader code changes beyond GUI removal and are not drop-in minimal patches of the canonical scripts.

## Canonical scripts changed

`make_dataset.py`, `train_model.py`, `batch_train_models.py`, and `seg_images_in_folder.py`.

## Utilities changed

`utils/pred2map.py`, `utils/batch_pred2map.py`, `utils/print_pred_labels.py`, `utils/gen_saved_model.py`, `utils/gen_fullmodel_from_h5.py`, `utils/make_class_balanced_subset.py`, and `utils/make_overlays.py`.

## HPC usage

Run the same scripts as before. Instead of a browser/dialog, the script prints a prompt such as `Path to config JSON file:`. Paste an absolute HPC path and press Enter. For batch questions, enter `y`/`yes` to add another item; Enter alone means no.

For non-interactive scheduler jobs (e.g. SLURM), these terminal prompts still require stdin. A later optional enhancement would be command-line arguments, but that was intentionally not introduced here to keep interfaces and internal interactions minimally changed.
