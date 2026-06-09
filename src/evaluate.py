from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix


def evaluate_model(y_true, y_score, y_pred, threshold):
    report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    report["roc_auc"] = roc_auc_score(y_score=y_score, y_true=y_true)
    report["average_precision"] = average_precision_score(y_score=y_score, y_true=y_true)
    report["threshold"] = threshold

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    report["TN"] = int(cm[0, 0])
    report["FP"] = int(cm[0, 1])
    report["FN"] = int(cm[1, 0])
    report["TP"] = int(cm[1, 1])

    return report
