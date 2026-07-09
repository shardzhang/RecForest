from pathlib import Path

import numpy as np


def build_code_to_item(item_to_code: np.ndarray, k: int) -> np.ndarray:
    tree_height = item_to_code.shape[1]
    num_leaves = k ** tree_height
    card = np.array([k ** (tree_height - i - 1) for i in range(tree_height)], dtype=np.int64)
    leaf_codes = (item_to_code * card).sum(axis=1)

    if len(np.unique(leaf_codes)) != item_to_code.shape[0]:
        raise ValueError("item_to_code does not map items to unique leaf codes")

    code_to_item = np.zeros((num_leaves,), dtype=np.int64)
    code_to_item[leaf_codes] = np.arange(item_to_code.shape[0], dtype=np.int64)
    return code_to_item


def main():
    tree_dir = Path("data/gowalla/tree")
    k = 18
    for item_to_code_path in sorted(tree_dir.glob("embkm1.0_item_to_code_tree_id_*_k18.npy")):
        item_to_code = np.load(item_to_code_path)
        code_to_item = build_code_to_item(item_to_code, k)
        code_to_item_path = Path(str(item_to_code_path).replace("item_to_code", "code_to_item"))
        np.save(code_to_item_path, code_to_item)
        print(f"wrote {code_to_item_path} shape={code_to_item.shape}")


if __name__ == "__main__":
    main()
