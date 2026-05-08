# AGENTS

## Workflow
- There is no checked-in build, test, lint, CI, or task-runner config. The repo is driven directly from notebooks plus `lib/`; do not assume `pytest`, `make`, or `npm` workflows exist.
- Primary entrypoints are `notebooks/gowalla/gowalla.ipynb` (multi-tree `Trm4Rec`) and `notebooks/gowalla/gowalla_DIN.ipynb` (`DINTrain` baseline).
- `gowalla.ipynb` builds `tree_num` separate `Trm4Rec` models, shares the encoder from tree 0 across all trees, concatenates per-tree predictions, and reranks them with `compute_scores`.
- `DINTrain.update_DIN()` performs `backward()/step()/zero_grad()` internally. `Trm4Rec.update_model()` only returns a loss; the notebook owns the shared optimizer and backprop across trees.
- `Train_instance.training_batches()`, `test_batches()`, `validation_batches()`, and `generate_training_records()` are infinite generators. Training stops via notebook batch counters, not by exhausting an epoch.

## Paths And Data
- Notebook cells assume the working directory is `notebooks/gowalla`: they use `sys.path.append('../..')` and `../../data/gowalla/...` paths.
- Both notebooks are wired for preprocessed data (`have_processed_data=True`). The repo includes `data/gowalla/train_instances_0..9`, `test_instances`, `validation_instances`, `item_node_num.txt`, `DIN_MODEL.pt`, `model/`, and `tree/`, but not the raw `data/gowalla/gowalla.txt`.
- Training data is sharded by prefix. Loaders expect `train_instances_0` through `train_instances_{N-1}` plus an explicit shard count; `_split_train_sample()` deletes the unsuffixed `train_instances` file after sharding.
- Saved artifact names encode `init_way`, `feature_ratio`, `tree_id`, and `k` (example: `embkm1.0_*_tree_id_0_k18.*`). Changing those notebook params invalidates the checked-in saved files.

## Verified Gotchas
- `lib/__init__.py` imports `.JTM_variant`, but `lib/JTM_variant.py` is gitignored and absent in this checkout. On a clean clone, `import lib` fails unless that private file is restored or `__init__.py` is adjusted.
- Treat `lib/generate_train_and_test_data.py` as the source of truth for preprocessing APIs. Some notebook cells are stale: `_read()` currently takes `(raw_file, test_record_num)` and returns 6 values.
- The transformer/tree path is effectively CUDA-only: `lib/Tree_Model.py` hardcodes `.to('cuda')` in both `init_tree()` and `read_tree()`. Switching notebook `device` to CPU is not enough.
- `gowalla.ipynb` sets `tree_has_generated=True`, but `data/gowalla/tree/` currently contains only `*_item_to_code_tree_id_*.npy`; the matching `*_code_to_item_tree_id_*.npy` files required by `Tree.read_tree()` are not committed.
- Dependencies are implicit, not declared. The code imports at least `torch`, `transformers`, `numpy`, `joblib`, `tqdm`, `pandas`, `matplotlib`, and Jupyter/IPython utilities.
