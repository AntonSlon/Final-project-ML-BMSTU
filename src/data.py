import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE


def load_datasets(p2p_path, trans_path):
    p2_log = pd.read_csv(p2p_path, index_col=0)
    trans_log = pd.read_csv(trans_path, index_col=0)

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

    p2p_log_train = p2p_log.iloc[:p2p_train_end].copy()
    p2p_log_val = p2p_log.iloc[p2p_train_end:p2p_val_end].copy()
    p2p_log_test = p2p_log.iloc[p2p_val_end:].copy()

    trans_train_end = int(len(trans_log) * train_size)
    trans_val_end = int(len(trans_log) * (train_size + (1 - train_size) / 2))

    trans_log_train = trans_log.iloc[:trans_train_end].copy()
    trans_log_val = trans_log.iloc[trans_train_end:trans_val_end].copy()
    trans_log_test = trans_log.iloc[trans_val_end:].copy()

    return (
        p2p_log_train,
        p2p_log_val,
        p2p_log_test,
        trans_log_train,
        trans_log_val,
        trans_log_test,
    )
