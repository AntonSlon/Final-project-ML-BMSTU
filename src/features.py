import numpy as np
import pandas as pd


def build_features(
    p2p_log: pd.DataFrame,
    p2p_history: pd.DataFrame,
    trans_history: pd.DataFrame,
):
    p2p_log = p2p_log.copy()
    p2p_history = p2p_history.copy()
    trans_history = trans_history.copy()

    print(len(p2p_log))
    print(len(p2p_history))
    print(len(trans_history))

    if p2p_history["EventTime"].max() >= p2p_log["EventTime"].min():
        raise ValueError("P2P history overlaps the current period")

    if trans_history["EventTime"].max() >= p2p_log["EventTime"].min():
        raise ValueError("Transaction history overlaps the current period")

    target = None
    if "IsFraud" in p2p_log.columns:
        target = p2p_log["IsFraud"].reset_index(drop=True)

    p2p_log = p2p_log.drop(columns=["IsFraud"], errors="ignore")
    p2p_history = p2p_history.drop(columns=["IsFraud"], errors="ignore")

    sender_p2p_feat = (p2p_history
        .groupby(["UserID"])
        .agg(
            sender_p2p_count=("Amount", "count"),
            sender_p2p_mean=("Amount", "mean"),
            sender_p2p_sum=("Amount", "sum"),
            sender_p2p_std=("Amount", "std"),
            sender_unique_recipients=("RecipientID", "nunique")
    ).reset_index())

    p2p_log = p2p_log.merge(
        sender_p2p_feat,
        on="UserID",
        how="left",
    )

    eps = 1e-6
    p2p_log["sender_p2p_count"] = p2p_log["sender_p2p_count"].fillna(0)
    p2p_log["sender_p2p_mean"] = p2p_log["sender_p2p_mean"].fillna(0)
    p2p_log["sender_p2p_sum"] = p2p_log["sender_p2p_sum"].fillna(0)
    p2p_log["sender_p2p_std"] = p2p_log["sender_p2p_std"].fillna(0)
    p2p_log["sender_unique_recipients"] = p2p_log["sender_unique_recipients"].fillna(0)
    p2p_log["sender_p2p_z_score"] = np.where(
        p2p_log["sender_p2p_count"] > 0,
        (p2p_log["Amount"] - p2p_log["sender_p2p_mean"])
        / (p2p_log["sender_p2p_std"] + eps),
        0,
    )

    recipient_p2p_feat = (
        p2p_history
            .groupby(["RecipientID"])
            .agg(
                recipient_p2p_count=("Amount", "count"),
                recipient_p2p_mean=("Amount", "mean"),
                recipient_p2p_sum=("Amount", "sum"),
                recipient_p2p_std=("Amount", "std"),
                recipient_unique_senders=("UserID", "nunique")
    ).reset_index())

    p2p_log = p2p_log.merge(
        recipient_p2p_feat,
        on="RecipientID",
        how="left",
    )

    p2p_log["recipient_p2p_count"] = p2p_log["recipient_p2p_count"].fillna(0)
    p2p_log["recipient_p2p_mean"] = p2p_log["recipient_p2p_mean"].fillna(0)
    p2p_log["recipient_p2p_sum"] = p2p_log["recipient_p2p_sum"].fillna(0)
    p2p_log["recipient_p2p_std"] = p2p_log["recipient_p2p_std"].fillna(0)
    p2p_log["recipient_unique_senders"] = p2p_log["recipient_unique_senders"].fillna(0)
    p2p_log["recipient_p2p_z_score"] = np.where(
        p2p_log["recipient_p2p_count"] > 0,
        (p2p_log["Amount"] - p2p_log["recipient_p2p_mean"])
        / (p2p_log["recipient_p2p_std"] + eps),
        0,
    )

    p2p_log["sender_p2p_amount_ratio"] = np.where(
        p2p_log["sender_p2p_count"] > 0,
        p2p_log["Amount"] / (p2p_log["sender_p2p_mean"] + eps),
        0,
    )

    p2p_log["recipient_p2p_amount_ratio"] = np.where(
        p2p_log["recipient_p2p_count"] > 0,
        p2p_log["Amount"] / (p2p_log["recipient_p2p_mean"] + eps),
        0,
    )

    """фичи для trans"""

    print(trans_history.columns)
    sender_trans_feat = (
        trans_history
            .groupby("UserID")
            .agg(
                sender_trans_count=("Amount", "count"),
                sender_trans_mean=("Amount", "mean"),
                sender_trans_sum=("Amount", "sum"),
                sender_trans_std=("Amount", "std"),
                sender_unique_merchant=("MerchantID", "nunique"),
                trans_success_rate=("TransactionSuccessful", "mean"),
                trans_unique_countries=("Country", "nunique"),
        ).reset_index())

    p2p_log = p2p_log.merge(
        sender_trans_feat,
        on="UserID",
        how="left",
    )

    trans_columns = [
        "sender_trans_count",
        "sender_trans_mean",
        "sender_trans_sum",
        "sender_trans_std",
        "sender_unique_merchant",
        "trans_success_rate",
        "trans_unique_countries",
    ]
    p2p_log[trans_columns] = p2p_log[trans_columns].fillna(0)

    pairs_p2p_feat = (
        p2p_history
            .groupby(["UserID", "RecipientID"])
            .size()
            .rename("pairs_p2p_count")
            .reset_index()
    )

    p2p_log = p2p_log.merge(
        pairs_p2p_feat,
        on=["UserID", "RecipientID"],
        how="left",
    )
    p2p_log["pairs_p2p_count"] = p2p_log["pairs_p2p_count"].fillna(0)

    p2p_log["hour"] = p2p_log["EventTime"].dt.hour
    p2p_log["dayofweek"] = p2p_log["EventTime"].dt.dayofweek
    p2p_log["month"] = p2p_log["EventTime"].dt.month
    p2p_log["is_weekend"] = p2p_log["dayofweek"].isin([5, 6]).astype(int)

    p2p_log = p2p_log.drop(columns=["EventTime"])

    cat_features = [
        "UserID",
        "RecipientID",
        "Currency"
    ]

    return p2p_log, target, cat_features
