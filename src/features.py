import pandas as pd


def build_features(p2p_log: pd.DataFrame, trans_log: pd.DataFrame):
    p2p_log = p2p_log.copy()

    print(len(p2p_log))
    print(len(trans_log))

    target = p2p_log["IsFraud"]

    p2p_log = p2p_log.drop(columns=["IsFraud"])

    sender_p2p_feat = (p2p_log
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
    p2p_log["sender_p2p_std"] = p2p_log["sender_p2p_std"].fillna(0)
    p2p_log["sender_p2p_z_score"] = (p2p_log["Amount"] - p2p_log["sender_p2p_mean"]) / (p2p_log["sender_p2p_std"] + eps)

    recipient_p2p_feat = (
        p2p_log
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

    p2p_log["recipient_p2p_std"] = p2p_log["recipient_p2p_std"].fillna(0)
    p2p_log["recipient_p2p_z_score"] = ((p2p_log["Amount"] - p2p_log["recipient_p2p_mean"])
                                        / (p2p_log["recipient_p2p_std"] + eps))

    p2p_log["sender_p2p_amount_ratio"] = (
            p2p_log["Amount"] / (p2p_log["sender_p2p_mean"] + eps)
    )

    p2p_log["recipient_p2p_amount_ratio"] = (
            p2p_log["Amount"] / (p2p_log["recipient_p2p_mean"] + eps)
    )

    """фичи для trans"""

    trans_log = trans_log.copy()

    print(trans_log.columns)
    sender_trans_feat = (
        trans_log
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

    p2p_log["pairs_p2p_count"] = (p2p_log
                                  .groupby(["UserID", "RecipientID"])["Amount"].transform("count"))

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
