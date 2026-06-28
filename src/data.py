import pandas as pd


def load_datasets(p2p_path, trans_path):
    p2_log = pd.read_csv(p2p_path, index_col=0)
    trans_log = pd.read_csv(trans_path)

    return p2_log, trans_log


def prepare_p2p_log(p2p_log):
    p2p_log = p2p_log.copy()

    if p2p_log.isna().any().any():
        raise ValueError("P2P dataset contains missing values")

    p2p_log["EventTime"] = pd.to_datetime(p2p_log["EventTime"])

    return p2p_log


def prepare_trans_log(trans_log):
    trans_log = trans_log.copy()

    if trans_log.isna().any().any():
        raise ValueError("Transactions dataset contains missing values")

    trans_log["EventTime"] = pd.to_datetime(trans_log["EventTime"])

    return trans_log


def make_train_test_split(p2p_log, trans_log, train_size=0.8):
    p2p_log = p2p_log.sort_values("EventTime")
    trans_log = trans_log.sort_values("EventTime")

    p2p_train_end = int(len(p2p_log) * train_size)
    p2p_val_end = int(len(p2p_log) * (train_size + (1 - train_size) / 2))

    validation_start = p2p_log.iloc[p2p_train_end]["EventTime"]
    test_start = p2p_log.iloc[p2p_val_end]["EventTime"]

    p2p_log_train = p2p_log[p2p_log["EventTime"] < validation_start].copy()
    p2p_log_val = p2p_log[
        (p2p_log["EventTime"] >= validation_start)
        & (p2p_log["EventTime"] < test_start)
    ].copy()
    p2p_log_test = p2p_log[p2p_log["EventTime"] >= test_start].copy()

    trans_log_train = trans_log[trans_log["EventTime"] < validation_start].copy()
    trans_log_val = trans_log[
        (trans_log["EventTime"] >= validation_start)
        & (trans_log["EventTime"] < test_start)
    ].copy()
    trans_log_test = trans_log[trans_log["EventTime"] >= test_start].copy()

    return (
        p2p_log_train,
        p2p_log_val,
        p2p_log_test,
        trans_log_train,
        trans_log_val,
        trans_log_test,
    )
