# predictor.py
"""
人数预测模块
------------
首次 import 时会自动检查并加载训练好的模型；若模型不存在，将
利用 “附件1：不同人数的STR图谱数据.xlsx” 重新训练并保存。

外部接口:
    predict(allele_list: list[str]) -> dict
        返回示例: {"num": 3}
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from lightgbm import LGBMClassifier

# ------------------- 常量 -------------------
MODEL_PATH = Path(__file__).with_name("num_predictor.pkl")
DATA_PATH  = Path(__file__).with_name("附件1：不同人数的STR图谱数据.xlsx")
MAX_MARKERS = 100      # 与原始脚本保持一致

# ---------------- 特征工程函数 ---------------
def extract_features(row: pd.Series) -> pd.Series:
    heights = [row.get(f"Height {i}") for i in range(1, MAX_MARKERS + 1)
               if pd.notna(row.get(f"Height {i}"))]
    alleles = [str(row.get(f"Allele {i}")).upper() for i in range(1, MAX_MARKERS + 1)
               if pd.notna(row.get(f"Allele {i}"))]
    sizes   = [row.get(f"Size {i}")   for i in range(1, MAX_MARKERS + 1)
               if pd.notna(row.get(f"Size {i}"))]

    if len(heights) < 2:
        return pd.Series([0] * 30)

    filtered_heights = [h for h in heights if h >= 9]
    sorted_heights   = sorted(heights, reverse=True)
    mean_h = np.mean(filtered_heights) if filtered_heights else 0
    allele_entropy = -sum(
        (alleles.count(a)/len(alleles)) *
        np.log2(alleles.count(a)/len(alleles)) for a in set(alleles)
    )

    return pd.Series([
        len(heights), np.max(heights), np.min(heights), mean_h, np.std(heights),
        np.std(heights)/mean_h if mean_h > 0 else 0,
        len(set(alleles)), alleles.count('OL'), sum(h < 50 for h in heights),
        np.max(heights)/np.min(heights) if np.min(heights) > 0 else 0,
        np.mean(sorted_heights[:3]), skew(heights) if len(set(heights)) > 1 else 0,
        (max(sizes)-min(sizes)) if sizes else 0, len(set(sizes)),
        sum(h > 70 for h in heights)/len(heights),
        sum(h > 100 for h in heights)/len(heights),
        np.median(heights), np.std(sizes) if sizes else 0,
        allele_entropy, sum(h < 10 for h in heights)/sum(heights) if sum(heights) > 0 else 0,
        sum(1 for h in heights if h < 50)/len(heights),
        sorted_heights[0] - sorted_heights[2] if len(sorted_heights) >= 3 else 0,
        sorted_heights[0] / (sorted_heights[1] + 1) if len(sorted_heights) >= 2 else 0,
        sum(a == 'OL' for a in alleles)/len(heights),
        np.median(sizes) if sizes else 0,
        len(set(sizes))/len(sizes) if sizes else 0,
        len(set(heights))/len(heights),
        np.var(heights), np.var(sizes) if sizes else 0,
        np.mean(sizes) if sizes else 0,
        len(alleles) / len(set(alleles)) if len(set(alleles)) > 0 else 0
    ])

def extract_advanced_features(row: pd.Series) -> pd.Series:
    heights = [row.get(f"Height {i}") for i in range(1, MAX_MARKERS + 1)
               if pd.notna(row.get(f"Height {i}"))]
    sizes   = [row.get(f"Size {i}")   for i in range(1, MAX_MARKERS + 1)
               if pd.notna(row.get(f"Size {i}"))]
    alleles = [str(row.get(f"Allele {i}")).upper() for i in range(1, MAX_MARKERS + 1)
               if pd.notna(row.get(f"Allele {i}"))]

    if len(heights) < 2 or len(sizes) < 2:
        return pd.Series([0] * 8)

    total_h  = sum(heights)
    top5_sum = sum(sorted(heights, reverse=True)[:5])
    height_size_ratios = [h/s for h, s in zip(heights, sizes) if s > 0]

    return pd.Series([
        top5_sum / total_h if total_h > 0 else 0,
        np.mean(height_size_ratios) if height_size_ratios else 0,
        np.std(height_size_ratios)  if len(height_size_ratios) > 1 else 0,
        np.var(sizes), np.std(sizes),
        skew(sizes) if len(set(sizes)) > 1 else 0,
        sum(h > 100 for h in heights) / len(alleles) if len(alleles) > 0 else 0,
        len([a for a in alleles if a != 'OL']) / len(set(alleles)) if len(set(alleles)) > 0 else 0
    ])

# ---------------- 训练流程 -------------------
def _train_and_save():
    print("🔄 正在训练人数预测模型 ...")
    df = pd.read_excel(DATA_PATH, sheet_name="不同人数的数据集")

    # 提取真实人数标签
    df["Contributor_Count"] = df["Sample File"].apply(
        lambda x: len(re.search(r'RD14-0003-([\d_]+)-', str(x)).group(1).split('_'))
        if re.search(r'RD14-0003-([\d_]+)-', str(x)) else np.nan
    )

    # ➡️ 如需合成 5 人样本可按原脚本拼合；此处保持简单，直接用原始数据
    df = df.dropna(subset=["Contributor_Count"])

    # 基础 + 高级特征
    base_feat      = df.apply(extract_features, axis=1)
    advanced_feat  = df.apply(extract_advanced_features, axis=1)
    features_df    = pd.concat([base_feat, advanced_feat], axis=1)
    features_df.columns = [f"f{i}" for i in range(features_df.shape[1])]
    # 填补缺失值，防止 NaN 传入 SMOTE
    features_df = features_df.fillna(0)

    X = features_df
    y = df["Contributor_Count"].astype(int)

    # 标签编码
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    # 过采样平衡
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y_enc)

    # -------- 模型集合 ----------
    rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=150, max_depth=6,
                                     learning_rate=0.08, random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=6,
                                  learning_rate=0.08, use_label_encoder=False,
                                  eval_metric='mlogloss', random_state=42)
    lgb_model = LGBMClassifier(n_estimators=120, max_depth=7,
                               learning_rate=0.07, random_state=42)
    lr  = LogisticRegression(max_iter=2000)

    stacking = StackingClassifier(
        estimators=[('rf', rf), ('gb', gb),
                    ('xgb', xgb_model), ('lgb', lgb_model)],
        final_estimator=lr, passthrough=True, cv=5, n_jobs=-1
    )
    voting = VotingClassifier(
        estimators=[('stacking', stacking), ('rf', rf), ('gb', gb),
                    ('xgb', xgb_model), ('lgb', lgb_model)],
        voting='soft', n_jobs=-1
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )
    voting.fit(X_tr, y_tr)

    # 训练集简单评估
    y_pred = voting.predict(X_te)
    y_pred_dec = label_encoder.inverse_transform(y_pred)
    y_true_dec = label_encoder.inverse_transform(y_te)
    print(f"✅ Accuracy: {accuracy_score(y_true_dec, y_pred_dec):.4f}")
    print(classification_report(y_true_dec, y_pred_dec))

    # 持久化
    joblib.dump({
        "model": voting,
        "label_encoder": label_encoder,
        "feature_cols": list(X.columns)
    }, MODEL_PATH)
    print(f"🎉 模型已保存至 {MODEL_PATH}")

# -------------- 加载模型 ---------------------
def _load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到训练数据 {DATA_PATH}，无法训练模型！"
        )
    _train_and_save()
    return joblib.load(MODEL_PATH)

_model_bundle = _load_model()           # 全局加载
_model        = _model_bundle["model"]
_label_enc    = _model_bundle["label_encoder"]
_feat_cols    = _model_bundle["feature_cols"]

# -------------- 预测接口 ---------------------
def _build_row_series(flat_list):
    """
    将 [allele1,size1,height1, allele2,size2,height2, ...]
    转换为与训练阶段一致的“行记录” (pandas.Series)
    """
    if len(flat_list) % 3 != 0:
        raise ValueError("输入列表长度必须是 3 的倍数 (allele, size, height 三元素一组)")

    row_dict = {}
    triplets = [flat_list[i:i+3] for i in range(0, len(flat_list), 3)]
    for idx, (allele, size, height) in enumerate(triplets, start=1):
        if idx > MAX_MARKERS:
            break
        # Allele：保留原始字符串 (可能含 'OL')
        row_dict[f"Allele {idx}"] = str(allele).strip()
        # Size / Height：转 float，非法值设 NaN
        try:
            row_dict[f"Size {idx}"] = float(size)
        except ValueError:
            row_dict[f"Size {idx}"] = np.nan
        try:
            row_dict[f"Height {idx}"] = float(height)
        except ValueError:
            row_dict[f"Height {idx}"] = np.nan

    # 确保所有键齐全（缺失部分填 NaN）
    for i in range(1, MAX_MARKERS + 1):
        for col in ("Allele", "Size", "Height"):
            row_dict.setdefault(f"{col} {i}", np.nan)

    return pd.Series(row_dict)

def predict(allele_list):
    """
    预测贡献者人数

    参数
    ----
    allele_list : list[str]
        形如 ["allele1","size1","height1","allele2",...]
        (长度必须是 3 的倍数)

    返回
    ----
    dict : {"num": int}
    """
    series_row = _build_row_series(allele_list)

    base_feat     = extract_features(series_row)
    advanced_feat = extract_advanced_features(series_row)
    feat_row      = pd.concat([base_feat, advanced_feat])
    feat_row.index = _feat_cols            # 保证列顺序一致
    feat_df       = pd.DataFrame([feat_row])

    pred_enc  = _model.predict(feat_df)[0]
    pred_num  = int(_label_enc.inverse_transform([pred_enc])[0])
    return {"num": pred_num}

# ---------------- CLI 方便调试 ---------------
if __name__ == "__main__":
    # 示例：12,104,1320,14,108,980,13,110,780
    test = input("请输入 allele,size,height 逗号分隔列表：\n")
    flat = [s.strip() for s in test.split(",") if s.strip()]
    print("预测结果:", predict(flat))