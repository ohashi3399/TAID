import re
import string
import numpy as np


class ExactMatchMetric:
    def __init__(self, seed=0):
        self.seed = seed
    
    def compute(self, predictions, references, regexes_to_ignore=None, 
                ignore_case=False, ignore_punctuation=False, ignore_numbers=False):
        
        if regexes_to_ignore is not None:
            for s in regexes_to_ignore:
                predictions = np.array([re.sub(s, "", x) for x in predictions])
                references = np.array([re.sub(s, "", x) for x in references])
        else:
            predictions = np.asarray(predictions)
            references = np.asarray(references)
        
        if ignore_case:
            predictions = np.char.lower(predictions)
            references = np.char.lower(references)
        
        if ignore_punctuation:
            repl_table = str.maketrans("", "", string.punctuation)
            predictions = np.array([x.translate(repl_table) for x in predictions])
            references = np.array([x.translate(repl_table) for x in references])
        
        if ignore_numbers:
            repl_table = str.maketrans("", "", string.digits)
            predictions = np.array([x.translate(repl_table) for x in predictions])
            references = np.array([x.translate(repl_table) for x in references])
        
        score_list = predictions == references
        
        return {"exact_match": float(np.mean(score_list))}


class RougeMetric:
    def __init__(self, seed=0):
        self.seed = seed
    
    def compute(self, predictions, references, use_stemmer=True, rouge_types=["rougeL"]):
        """
        ROUGE-L計算の実装
        """
        scores = {}
        
        # すべての予測と参照に対してROUGE-Lスコアを計算
        rouge_l_scores = []
        for pred, ref in zip(predictions, references):
            score = self._rouge_l_sentence_level(self._normalize_text(pred), self._normalize_text(ref))
            rouge_l_scores.append(score)
        
        # 平均スコアを計算
        scores["rougeL"] = np.mean(rouge_l_scores)
        
        return scores
    
    def _normalize_text(self, text):
        """テキストを正規化する"""
        # 小文字に変換
        text = text.lower()
        # 句読点を削除
        text = re.sub(r'[^\w\s]', '', text)
        # 複数の空白を1つに置換
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _lcs(self, a, b):
        """最長共通部分列の長さを計算"""
        lengths = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
        
        # LCSの動的計画法の実装
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                if a[i-1] == b[j-1]:
                    lengths[i][j] = lengths[i-1][j-1] + 1
                else:
                    lengths[i][j] = max(lengths[i-1][j], lengths[i][j-1])
        
        return lengths[len(a)][len(b)]
    
    def _rouge_l_sentence_level(self, pred, ref):
        """
        文レベルのROUGE-Lスコア計算
        pred: 予測テキスト
        ref: 参照テキスト
        """
        pred_words = pred.split()
        ref_words = ref.split()
        
        if len(pred_words) == 0 or len(ref_words) == 0:
            return 0.0
        
        lcs_length = self._lcs(pred_words, ref_words)
        
        # 精度(Precision)と再現率(Recall)の計算
        if len(pred_words) == 0:
            precision = 0.0
        else:
            precision = lcs_length / len(pred_words)
            
        if len(ref_words) == 0:
            recall = 0.0
        else:
            recall = lcs_length / len(ref_words)
        
        # F1スコアの計算（ROUGE-L）
        if precision + recall == 0:
            return 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
            return f1

exact_match_fn = ExactMatchMetric(seed=0)
rouge_fn = RougeMetric(seed=0)


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
