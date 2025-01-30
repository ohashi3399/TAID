import string
import evaluate

exact_match_fn = evaluate.load("exact_match", seed=0)
rouge_fn = evaluate.load("rouge", seed=0)


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_text(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(preds, target) -> float:
    return exact_match_fn.compute(
        predictions=[normalize_text(s) for s in preds],
        references=[normalize_text(s) for s in target],
    )["exact_match"]


def rouge(preds, target) -> float:
    scores = rouge_fn.compute(
        predictions=preds, references=target, use_stemmer=True, rouge_types=["rougeL"]
    )
    return scores["rougeL"]


def compute_metrics(preds, target):
    assert len(preds) == len(target)
    metrics = {}
    metrics["exact_match"] = 100.0 * exact_match(preds, target)
    metrics["rougeL"] = 100.0 * rouge(preds, target)
    # metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics
