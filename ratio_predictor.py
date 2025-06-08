# ratio_predictor.py
"""
比例向量预测模块
---------------
首次 import 会检查并训练模型（若 ratio_predictor.pkl 不存在）。

外部接口:
    predict_ratio(allele_size_height_list, marker="UNKNOWN") -> dict
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from warnings import filterwarnings
filterwarnings("ignore")

# ------------------------ 常量 ------------------------
MAX_MARKERS = 100
MODEL_PATH  = Path(__file__).with_name("ratio_predictor.pkl")
DATA_PATH   = Path(__file__).with_name("附件2：不同混合比例的STR图谱数据.xlsx")

# ---------------------- 工具函数 ----------------------
def _parse_ratio(filename):
    """提取原始整数比例 & 归一化向量"""
    m = re.search(r"(\d+(?:;\d+)+)", str(filename))
    if not m:
        return None, None
    raw  = list(map(int, m.group(1).split(";")))
    norm = [r / sum(raw) for r in raw]
    return norm + [0]*(5-len(norm)), tuple(raw + [0]*(5-len(raw)))

def build_features(row: pd.Series):
    """与原脚本一致，但去掉 'Struct_Label' 泄漏列"""
    allele_cols  = [c for c in row.index if "Allele"  in c]
    height_cols  = [c for c in row.index if "Height"  in c]
    size_cols    = [c for c in row.index if "Size"    in c]

    alleles  = row[allele_cols]
    heights  = row[height_cols].dropna().astype(float).values
    sizes    = row[size_cols].dropna().astype(float).values
    log_h    = np.log1p(heights) if len(heights) else np.array([0])

    top3     = np.sort(heights)[::-1][:3] if len(heights) >= 3 else [0, 0, 0]
    ol_cnt   = sum(alleles == "OL")
    n_all    = alleles.notnull().sum()
    uniq_all = len(set(alleles.dropna()))

    return pd.Series({
        "Num_Alleles"      : n_all,
        "Unique_Alleles"   : uniq_all,
        "OL_Ratio"         : ol_cnt / n_all if n_all else 0,
        "Repeat_Ratio"     : (n_all - uniq_all)/n_all if n_all else 0,
        "Mean_Height"      : np.mean(heights) if len(heights) else 0,
        "Std_Height"       : np.std(heights) if len(heights) else 0,
        "CV_Height"        : np.std(heights)/np.mean(heights) if len(heights) and np.mean(heights) else 0,
        "Top1_Top2_Diff"   : top3[0] - top3[1],
        "Top2_Top3_Diff"   : top3[1] - top3[2],
        "MaxH/TotalH"      : top3[0] / (np.sum(heights) + 1e-5),
        "Sum_Height"       : np.sum(heights),
        "Mean_LogHeight"   : np.mean(log_h),
        "Mean_Size"        : np.mean(sizes) if len(sizes) else 0,
        "Std_Size"         : np.std(sizes) if len(sizes) else 0,
        "Num_People"       : 0,   # 预测阶段占位，可省但保持列数一致
    })

# ------------------- 训练并保存模型 --------------------
def _train_and_save():
    print("🔄 正在训练比例预测模型 ...")
    df = pd.read_excel(DATA_PATH, sheet_name="不同比例的数据集")

    df[["Norm_Ratio", "Ratio_Label"]] = df["Sample File"].apply(
        lambda x: pd.Series(_parse_ratio(x))
    )
    df = df[df["Norm_Ratio"].notnull() & df["Marker"].notna()].copy()

    # ===== 模板库 =====
    template_dict   = {tpl: np.array(tpl)/sum(tpl) for tpl in sorted(set(df["Ratio_Label"]))}
    template_matrix = np.array(list(template_dict.values()))
    template_keys   = list(template_dict.keys())

    # ===== 结构标签 (KMeans) =====
    X_ratio = np.array(df["Norm_Ratio"].tolist())
    kmeans  = KMeans(n_clusters=20, random_state=42, n_init=10).fit(X_ratio)
    df["Struct_Label"] = kmeans.labels_

    # ===== 基础特征 =====
    X_raw = df.apply(build_features, axis=1)

    # ===== Marker One-Hot =====
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    marker_ohe = ohe.fit_transform(df[["Marker"]].astype(str))
    X = pd.concat(
        [X_raw.reset_index(drop=True), pd.DataFrame(marker_ohe)],
        axis=1
    )
    X.columns = X.columns.astype(str)
    y_ratio   = np.array(df["Norm_Ratio"].tolist())          # 5 维向量
    y_struct  = df["Struct_Label"].tolist()

    # ===== 标准化 & 过采样平衡结构标签 =====
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_res, y_struct_res = SMOTE(random_state=42).fit_resample(X_scaled, y_struct)

    # ===== 结构分类器 =====
    struct_clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, random_state=42)
    struct_clf.fit(X_res, y_struct_res)

    # ===== 结构分组回归器 =====
    structure_models = {}
    for s in sorted(set(y_struct)):
        idx = np.array(y_struct) == s
        group_models = []
        for i in range(5):
            model = lgb.LGBMRegressor(
                n_estimators=600, max_depth=7, learning_rate=0.03)
            model.fit(X_scaled[idx], y_ratio[idx, i])
            group_models.append(model)
        structure_models[s] = group_models

    # ===== 持久化 =====
    joblib.dump({
        "scaler"        : scaler,
        "marker_ohe"    : ohe,
        "struct_clf"    : struct_clf,
        "structure_models": structure_models,
        "template_dict" : template_dict,
        "template_matrix": template_matrix,
        "template_keys" : template_keys,
        "feature_cols"  : list(X.columns)
    }, MODEL_PATH)
    print(f"🎉 ratio_predictor.pkl 保存成功 ({MODEL_PATH})")

# -------------------- 加载模型 ------------------------
def _load_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"缺少训练数据 {DATA_PATH}")
    _train_and_save()
    return joblib.load(MODEL_PATH)

_bundle        = _load_bundle()
_scaler        = _bundle["scaler"]
_marker_ohe    = _bundle["marker_ohe"]
_struct_clf    = _bundle["struct_clf"]
_structure_mod = _bundle["structure_models"]
_template_dict = _bundle["template_dict"]
_template_mat  = _bundle["template_matrix"]
_template_keys = _bundle["template_keys"]
_feat_cols     = _bundle["feature_cols"]

# ------------------ 预测辅助函数 ----------------------
def _match_templates(pred_vec, topk=1):
    """Cosine–L2 综合分数匹配模板"""
    vec = np.clip(pred_vec, 0, None)
    vec = vec / vec.sum() if vec.sum() > 0 else vec
    cos_sim = cosine_similarity([vec], _template_mat)[0]
    l2_dist = np.linalg.norm(_template_mat - vec, axis=1)
    scores  = 0.65*cos_sim - 0.35*l2_dist
    return [_template_keys[i] for i in np.argsort(-scores)[:topk]]

def _build_row(flat_list, marker):
    if len(flat_list) % 3 != 0:
        raise ValueError("输入长度必须为 3 的倍数 (allele,size,height)")

    d = {}
    triplets = [flat_list[i:i+3] for i in range(0, len(flat_list), 3)]
    for idx, (allele, size, height) in enumerate(triplets, start=1):
        if idx > MAX_MARKERS: break
        d[f"Allele {idx}"] = str(allele).strip()
        try:   d[f"Size {idx}"]   = float(size)
        except ValueError: d[f"Size {idx}"] = np.nan
        try:   d[f"Height {idx}"] = float(height)
        except ValueError: d[f"Height {idx}"] = np.nan

    # 填满缺失列
    for i in range(1, MAX_MARKERS+1):
        for col in ("Allele", "Size", "Height"):
            d.setdefault(f"{col} {i}", np.nan)
    d["Marker"] = str(marker)
    return pd.Series(d)

# ---------------------- 外部接口 ----------------------
def predict_ratio(flat_list, marker="UNKNOWN"):
    """
    预测贡献者比例向量

    参数
    ----
    flat_list : list[str]
        ["allele1","size1","height1","allele2",...]
    marker    : str, default "UNKNOWN"
        若已知基因位点名称可填，否则保持默认

    返回
    ----
    dict
        {
            "ratio"      : [1,4],                # 最佳整数模板 (去零)
            "ratio_norm" : [0.2,0.8],            # 归一化浮点
            "template"   : (1,4,0,0,0)           # 原始 5 元组
        }
    """
    row          = _build_row(flat_list, marker)
    raw_features = build_features(row)

    # Marker One-Hot
    m_ohe        = _marker_ohe.transform([[marker]])
    feat         = pd.concat([raw_features, pd.Series(m_ohe[0])])
    feat.index   = _feat_cols                       # 对齐顺序
    X_df         = pd.DataFrame([feat])
    X_scaled     = _scaler.transform(X_df)

    # 结构预测 (取概率 Top-2)
    struct_prob  = _struct_clf.predict_proba(X_scaled)[0]
    top2_structs = np.argsort(struct_prob)[-2:]
    vec_sum = np.zeros(5); w_sum = 0
    for s in top2_structs:
        models = _structure_mod.get(s, [None]*5)
        preds  = [m.predict(X_scaled)[0] if m else 0 for m in models]
        vec_sum += np.array(preds) * struct_prob[s]
        w_sum  += struct_prob[s]
    pred_vec = vec_sum / w_sum if w_sum else vec_sum

    # 模板匹配
    best_tpl  = _match_templates(pred_vec, topk=1)[0]   # tuple 长 5
    norm_vec  = np.array(best_tpl)/sum(best_tpl) if sum(best_tpl) else np.zeros(5)
    trim_tpl  = [x for x in best_tpl if x > 0]

    return {
        "ratio"      : trim_tpl,
        "ratio_norm" : norm_vec[norm_vec>0].round(4).tolist(),
        "template"   : best_tpl
    }

# ---------------- CLI 调试 --------------------------
if __name__ == "__main__":
    eg = input("请输入 allele,size,height 列表 (逗号分隔)：\n")
    flat = [s.strip() for s in eg.split(",") if s.strip()]
    print(predict_ratio(flat))