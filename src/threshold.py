from sklearn.metrics import precision_recall_curve


def find_best_threshold(y_true, y_score, min_recall: float = 0.70):
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, y_score=y_score)

    best_precision = 0
    best_recall = 0
    best_threshold = None

    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= min_recall and p > best_precision:
            best_recall = r
            best_precision = p
            best_threshold = t

    return best_precision, best_recall, best_threshold
