import collections
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "gowalla"


def parse_train_line(line: str):
    user, history, label = line.strip().split("|")
    history_items = [int(x) for x in history.split(",")]
    return int(user), history_items, int(label)


def parse_eval_line(line: str):
    user, history, labels = line.strip().split("|")
    history_items = [int(x) for x in history.split(",")]
    label_items = [int(x) for x in labels.split(",") if x]
    return int(user), history_items, label_items


def summarize_histories(histories):
    non_pad_lengths = []
    unique_ratios = []
    revisit_rates = []
    item_counter = collections.Counter()

    for history in histories:
        non_pad = [x for x in history if x >= 0]
        non_pad_lengths.append(len(non_pad))
        if non_pad:
            unique_count = len(set(non_pad))
            unique_ratios.append(unique_count / len(non_pad))
            revisit_rates.append(1.0 - unique_count / len(non_pad))
            item_counter.update(non_pad)
        else:
            unique_ratios.append(0.0)
            revisit_rates.append(0.0)

    return {
        "count": len(histories),
        "avg_non_pad_len": sum(non_pad_lengths) / len(non_pad_lengths),
        "avg_unique_ratio": sum(unique_ratios) / len(unique_ratios),
        "avg_revisit_rate": sum(revisit_rates) / len(revisit_rates),
        "top_items": item_counter.most_common(10),
    }


def summarize_train():
    histories = []
    labels = []
    users = set()

    for path in sorted(DATA_DIR.glob("train_instances_*")):
        with path.open() as f:
            for line in f:
                user, history, label = parse_train_line(line)
                users.add(user)
                histories.append(history)
                labels.append(label)

    history_summary = summarize_histories(histories)
    label_counter = collections.Counter(labels)

    return {
        "users": len(users),
        "samples": len(histories),
        **history_summary,
        "top_labels": label_counter.most_common(10),
    }


def summarize_eval(file_name: str):
    histories = []
    label_lists = []
    users = set()
    future_counter = collections.Counter()
    overlap_rates = []

    with (DATA_DIR / file_name).open() as f:
        for line in f:
            user, history, labels = parse_eval_line(line)
            users.add(user)
            histories.append(history)
            label_lists.append(labels)
            future_counter.update(labels)

            non_pad = {x for x in history if x >= 0}
            if labels:
                overlap = len(non_pad.intersection(labels)) / len(labels)
            else:
                overlap = 0.0
            overlap_rates.append(overlap)

    history_summary = summarize_histories(histories)
    future_lengths = [len(x) for x in label_lists]

    return {
        "users": len(users),
        "samples": len(histories),
        **history_summary,
        "avg_future_len": sum(future_lengths) / len(future_lengths),
        "avg_history_future_overlap": sum(overlap_rates) / len(overlap_rates),
        "top_future_items": future_counter.most_common(10),
    }


def print_summary(title: str, summary: dict):
    print(f"\n[{title}]")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main():
    print_summary("train", summarize_train())
    print_summary("validation", summarize_eval("validation_instances"))
    print_summary("test", summarize_eval("test_instances"))


if __name__ == "__main__":
    main()
