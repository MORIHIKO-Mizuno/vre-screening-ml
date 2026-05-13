# --- Standard Library ---
import re
from collections import Counter

# --- Third-Party Libraries ---
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import statsmodels

# --- Scikit-Learn ---
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    fbeta_score,
    auc,
    precision_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# --- Imbalanced-Learn ---
from imblearn.over_sampling import SMOTE, SMOTENC, SMOTEN, ADASYN

# --- UMAP ---
from umap import UMAP

try:
    from .models import make_model
except ImportError:
    from models import make_model


# ─────────────────────────────────────────────────────────────
# ユーティリティ（モジュールレベルで定義し、関数内に重複させない）
# ─────────────────────────────────────────────────────────────

def _calc_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def _net_benefit(y_true, y_proba, thresholds):
    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    nbs = []
    for pt in thresholds:
        if pt >= 1:
            nbs.append(np.nan)
            continue
        pred = (y_proba >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        nbs.append((tp / n) - (fp / n) * (pt / (1 - pt)))
    return np.array(nbs)


def _treat_all_net_benefit(y_true, thresholds):
    prevalence = np.mean(y_true)
    nbs = []
    for pt in thresholds:
        if pt >= 1:
            nbs.append(np.nan)
            continue
        nbs.append(prevalence - (1 - prevalence) * (pt / (1 - pt)))
    return np.array(nbs)


def _smote_k_neighbors(y):
    if y is None:
        return 5

    min_class_count = pd.Series(y).value_counts().min()
    if min_class_count < 2:
        raise ValueError(
            "SMOTE requires at least 2 samples in the minority class "
            f"within the current training data; got {min_class_count}."
        )
    return min(5, int(min_class_count) - 1)


def _make_smote(smote_type, seed, X_columns=None, numerical_features=None, y=None):
    k_neighbors = _smote_k_neighbors(y)

    if smote_type == "SMOTE":
        return SMOTE(sampling_strategy="auto", random_state=seed, k_neighbors=k_neighbors)
    elif smote_type == "SMOTEN":
        return SMOTEN(sampling_strategy="auto", random_state=seed, k_neighbors=k_neighbors)
    elif smote_type == "SMOTENC":
        categorical_features = [
            j for j, col in enumerate(X_columns)
            if col not in (numerical_features or [])
        ]
        if not categorical_features:
            return SMOTE(sampling_strategy="auto", random_state=seed, k_neighbors=k_neighbors)
        return SMOTENC(
            categorical_features=categorical_features,
            random_state=seed,
            k_neighbors=k_neighbors,
        )
    elif smote_type == "ADASYN":
        return ADASYN(sampling_strategy="auto", random_state=seed, n_neighbors=k_neighbors)
    else:
        raise ValueError(f"Unsupported smote_type: {smote_type}")


# ─────────────────────────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────────────────────────

def plot_roc_curve_cv(
    X, y, title,
    n_tree=1000, seed=42,
    model_type="rf",
    threshold_method="inner_youden",
    smote_type="SMOTE",
    numerical_features=None,
    pcr_unit_cost=171,
    study_start="2023-07-01",
    study_end="2024-11-30",
    plt_show=True,
):
    # ── 初期設定 ──────────────────────────────────────────────
    mean_fpr        = np.linspace(0, 1, 100)
    dca_thresholds  = np.linspace(0.01, 0.99, 99)
    scan_thresholds = np.linspace(0, 1, 1001)  # Sensitivity vs PCR 用
    common_x_calib  = np.linspace(0, 1, 100)        # キャリブレーション用共通グリッド
    common_sens     = np.linspace(0, 1, 100)         # Sensitivity vs PCR 用共通グリッド
    
    # ── コスト換算用の観察期間設定 ──────────────────────────
    obs_days = (pd.Timestamp(study_end) - pd.Timestamp(study_start)).days + 1
    annual_factor = 365.0 / obs_days

    numerical_features = numerical_features or []
    model_base = make_model(model_type, n_tree, seed)
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # フォールドごとの収集リスト
    tprs, aucs                               = [], []
    accuracies, recalls                      = [], []
    precisions, specificities, f1_scores     = [], [], []
    used_thresholds                          = []
    fold_pr_curves, fold_dca_curves          = [], []
    fold_calib_curves                        = []  
    fold_pcr_curves                          = [] 
    fold_cic_curves = []
    fold_npv_curves = [] 

    all_y_true  = []
    all_y_proba = []

    # ── CVループ ─────────────────────────────────────────────
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):

        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx],   y.iloc[val_idx]

        smote = _make_smote(smote_type, seed, X.columns, numerical_features, y_tr)
        X_res, y_res = smote.fit_resample(X_tr, y_tr)

        model = clone(model_base)
        model.fit(X_res, y_res)

        proba = model.predict_proba(X_va)[:, 1]

        all_y_true.extend(y_va.tolist())
        all_y_proba.extend(proba.tolist())

        # ── 閾値をトレインデータ（SMOTE前）から決定 ──────────
        # バリデーションデータを使うとリークになるため、
        # SMOTE前の元のトレインデータの予測確率で閾値を決める
        proba_tr = model.predict_proba(X_tr)[:, 1]

        if threshold_method == "youden_tr":
            fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, proba_tr)
            youden = tpr_tr - fpr_tr
            used_t = thr_tr[np.argmax(youden)]

        elif threshold_method == "youden_res":
            proba_res = model.predict_proba(X_res)[:, 1]
            fpr_res, tpr_res, thr_res = roc_curve(y_res, proba_res)
            youden = tpr_res - fpr_res
            used_t = thr_res[np.argmax(youden)]

        elif threshold_method == "youden_va":
            proba_va = model.predict_proba(X_va)[:, 1]
            fpr_va, tpr_va, thr_va = roc_curve(y_va, proba_va)
            youden = tpr_va - fpr_va
            used_t = thr_va[np.argmax(youden)]

        elif threshold_method == "f1":
            thr = np.linspace(0, 1, 200)
            f1s = [f1_score(y_tr, (proba_tr >= t).astype(int), zero_division=0) for t in thr]
            used_t = thr[np.argmax(f1s)]

        elif threshold_method == "f2":
            thr = np.linspace(0, 1, 200)
            f2s = [fbeta_score(y_tr, (proba_tr >= t).astype(int), beta=2, zero_division=0) for t in thr]
            used_t = thr[np.argmax(f2s)]
            
        elif threshold_method in ["inner_youden", "inner_f1", "inner_f2"]:

            skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            inner_thresholds = []

            for tr_idx, va_idx in skf_inner.split(X_tr, y_tr):
                X_tr_in, X_va_in = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
                y_tr_in, y_va_in = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

                # --- モデル再学習（重要：clone） ---
                model_inner = clone(model)
                model_inner.fit(X_tr_in, y_tr_in)

                proba_va_in = model_inner.predict_proba(X_va_in)[:, 1]

                # --- threshold候補 ---
                thr = np.linspace(0, 1, 200)

                if threshold_method == "inner_youden":
                    fpr, tpr, thr_roc = roc_curve(y_va_in, proba_va_in)
                    youden = tpr - fpr
                    t_best = thr_roc[np.argmax(youden)]

                elif threshold_method == "inner_f1":
                    scores = [
                        f1_score(y_va_in, (proba_va_in >= t).astype(int), zero_division=0)
                        for t in thr
                    ]
                    t_best = thr[np.argmax(scores)]

                elif threshold_method == "inner_f2":
                    scores = [
                        fbeta_score(y_va_in, (proba_va_in >= t).astype(int), beta=2, zero_division=0)
                        for t in thr
                    ]
                    t_best = thr[np.argmax(scores)]

                inner_thresholds.append(t_best)

            # --- 最終threshold ---
            used_t = np.mean(inner_thresholds)
        else:
            used_t = 0.5

        used_thresholds.append(used_t)

        # ── バリデーションデータにトレイン由来の閾値を適用 ───
        fpr, tpr, _ = roc_curve(y_va, proba)
        pred = (proba >= used_t).astype(int)

        accuracies.append(accuracy_score(y_va, pred))
        recalls.append(recall_score(y_va, pred, zero_division=0))
        precisions.append(precision_score(y_va, pred, zero_division=0))
        specificities.append(_calc_specificity(y_va, pred))
        f1_scores.append(f1_score(y_va, pred, zero_division=0))
        aucs.append(roc_auc_score(y_va, proba))

        tpr_interp    = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        # PR（フォールドごと保存）
        precision_curve, recall_curve, _ = precision_recall_curve(y_va, proba)
        mean_recall = np.linspace(0, 1, 100)
        prec_interp = np.interp(mean_recall, recall_curve[::-1], precision_curve[::-1])
        fold_pr_curves.append({
            "precision_interp": prec_interp,
            "ap":               average_precision_score(y_va, proba),
        })

        # DCA（フォールドごと保存）
        fold_dca_curves.append({
            "nb_model": _net_benefit(y_va, proba, dca_thresholds),
            "nb_all":   _treat_all_net_benefit(y_va, dca_thresholds),
        })

        # ── キャリブレーション（フォールドごと保存） ────────────
        frac_pos, mean_pred_val = calibration_curve(
            y_va, proba, n_bins=10, strategy="uniform"
        )
        # 共通グリッドに補間
        frac_interp = np.interp(common_x_calib, mean_pred_val, frac_pos)
        fold_calib_curves.append(frac_interp)
        
        # ── Clinical Impact Curve（フォールドごと保存） ──────────
        high_risk_rate_list = []
        tp_rate_list        = []

        for t in dca_thresholds:
            pred_t = (proba >= t).astype(int)
            tn, fp, fn, tp_t = confusion_matrix(y_va, pred_t, labels=[0, 1]).ravel()

            n_val = len(y_va)
            high_risk_rate_list.append((tp_t + fp) / n_val)  # 高リスク判定割合
            tp_rate_list.append(tp_t / n_val)                # 真陽性割合

        fold_cic_curves.append({
            "high_risk_rate": np.array(high_risk_rate_list),
            "tp_rate":        np.array(tp_rate_list),
        })

        # ── Sensitivity vs PCR削減率（フォールドごと保存） ─────
        n_val     = len(y_va)
        sens_list = []
        pcr_list  = []
        for t in scan_thresholds:
            pred_t = (proba >= t).astype(int)
            tn, fp, fn, tp_t = confusion_matrix(y_va, pred_t, labels=[0, 1]).ravel()
            s   = tp_t / (tp_t + fn) if (tp_t + fn) > 0 else np.nan
            pcr = (tn + fn) / n_val
            sens_list.append(s)
            pcr_list.append(pcr)

        fold_pcr_curves.append({
            "sensitivity":   np.array(sens_list),
            "pcr_reduction": np.array(pcr_list),
        })

        # ── NPV vs 検査削減率（フォールドごと保存） ───────────────
        npv_list  = []
        pcr2_list = []
        for t in scan_thresholds:
            pred_t = (proba >= t).astype(int)
            tn, fp, fn, tp_t = confusion_matrix(y_va, pred_t, labels=[0, 1]).ravel()
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
            pcr = (tn + fn) / n_val
            npv_list.append(npv)
            pcr2_list.append(pcr)

        fold_npv_curves.append({
            "npv":           np.array(npv_list),
            "pcr_reduction": np.array(pcr2_list),
        })

    # ── OOF 配列に変換 ───────────────────────────────────────
    all_y_true  = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)

    # ── 集計 ─────────────────────────────────────────────────
    mean_tpr     = np.mean(tprs, axis=0)
    std_tpr      = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc     = np.mean(aucs)
    std_auc      = np.std(aucs)

    precision_oof, recall_oof, _ = precision_recall_curve(all_y_true, all_y_proba)
    ap_oof     = average_precision_score(all_y_true, all_y_proba)
    prevalence = np.mean(all_y_true)

    mean_recall     = np.linspace(0, 1, 100)
    prec_mat        = np.array([f["precision_interp"] for f in fold_pr_curves])
    mean_prec       = np.mean(prec_mat, axis=0)
    std_prec        = np.std(prec_mat,  axis=0)
    mean_ap         = np.mean([f["ap"] for f in fold_pr_curves])
    std_ap          = np.std([f["ap"]  for f in fold_pr_curves])

    nb_mat          = np.array([f["nb_model"] for f in fold_dca_curves])
    mean_nb         = np.nanmean(nb_mat, axis=0)
    std_nb          = np.nanstd(nb_mat,  axis=0)
    nb_all_mat      = np.array([f["nb_all"] for f in fold_dca_curves])
    mean_nb_all     = np.nanmean(nb_all_mat, axis=0)
    std_nb_all      = np.nanstd(nb_all_mat,  axis=0)
    nb_none         = np.zeros_like(dca_thresholds)

    # キャリブレーション集計
    calib_mat  = np.array(fold_calib_curves)
    mean_calib = np.mean(calib_mat, axis=0)
    std_calib  = np.std(calib_mat,  axis=0)

    # ── Sensitivity vs PCR削減率 集計 ────────────────────────
    """
    PCR削減率: (TN + FN) / 全件数 = 予測陰性の割合

    同一Sensitivityに対してPCR削減率の最大値を採用する。
    理由:
    1. 臨床的観点: 同じSensitivityなら陰性除外数が多い閾値の方が臨床的に優れる。
    2. 補間の安定性: 重複Sensitivity値をmax()で1点に集約し、np.interpが要求する
    単調増加のx系列を保証する。
    """
    pcr_mat = []
    for fold in fold_pcr_curves:
        s   = fold["sensitivity"]
        pcr = fold["pcr_reduction"]

        # nan を除去
        valid = ~np.isnan(s)
        df = pd.DataFrame({"s": s[valid], "pcr": pcr[valid]})

        # ★ 同一Sensitivityに対してPCR削減率の最大値を採用
        #    （同じSensitivityなら、より多くの陰性を除外できる点が臨床的に優れる）
        df = df.groupby("s")["pcr"].max().reset_index().sort_values("s")

        # ★ フォールドのSensitivity範囲内のみで補間（範囲外は端点値でクリップ）
        pcr_interp = np.interp(
            common_sens,
            df["s"].values,
            df["pcr"].values,
            left=df["pcr"].values[0],   # Sensitivity < min の外挿を端点値に固定
            right=df["pcr"].values[-1], # Sensitivity > max の外挿を端点値に固定
        )
        pcr_mat.append(pcr_interp)

    pcr_mat  = np.array(pcr_mat)
    mean_pcr = np.mean(pcr_mat, axis=0)
    std_pcr  = np.std(pcr_mat,  axis=0)

    # ── 共通スタイル設定 ─────────────────────────────────────
    plt.ioff()
    plt.rcParams["font.family"] = "Arial"

    FONT_TITLE  = 60
    FONT_LABEL  = 48
    FONT_TICK   = 40
    FONT_LEGEND = 32
    LW          = 5

    def _apply_style(ax):
        ax.spines["right"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.tick_params(labelsize=FONT_TICK)

    # ── ROC プロット ─────────────────────────────────────────
    from matplotlib.lines import Line2D

    fig_roc, ax_roc = plt.subplots(figsize=(12, 12))

    ax_roc.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color="black", alpha=0.15
    )
    ax_roc.plot(mean_fpr, mean_tpr, color="black", lw=LW)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=LW, color="black", alpha=0.8)

    ax_roc.set_xlim(-0.05, 1.05)
    ax_roc.set_ylim(-0.05, 1.05)
    ax_roc.set_aspect("equal")
    ax_roc.set_title(f"AUC = {mean_auc:.2f} ± {std_auc:.2f}", weight="bold", fontsize=FONT_TITLE)
    ax_roc.set_xlabel("False Positive Rate", fontsize=FONT_LABEL)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=FONT_LABEL)
    _apply_style(ax_roc)
    ax_roc.legend(
        handles=[
            Line2D([0], [0], color="black", lw=LW,           label=f"Mean ROC"),
            Patch(facecolor="black", alpha=0.15, label="±1 std. dev."),
            # Line2D([0], [0], color="black", lw=LW, ls="--",  label="Chance"),
        ],
        loc=(0.5, 0.02), frameon=False,
        prop={"weight": "normal", "size": FONT_LEGEND}
    )
    plt.tight_layout()
    if plt_show:
        plt.show()

    # ── PR プロット ──────────────────────────────────────────
    fig_pr, ax_pr = plt.subplots(figsize=(12, 12))

    ax_pr.fill_between(
        mean_recall,
        np.maximum(mean_prec - std_prec, 0),
        np.minimum(mean_prec + std_prec, 1),
        color="black", alpha=0.15
    )
    ax_pr.plot(mean_recall, mean_prec, color="black", lw=LW)
    ax_pr.axhline(prevalence, linestyle="--", lw=LW, color="black", alpha=0.8)

    ax_pr.set_xlim(-0.05, 1.05)
    ax_pr.set_ylim(-0.05, 1.05)
    ax_pr.set_aspect("equal")
    ax_pr.set_title(f"AP = {mean_ap:.2f} ± {std_ap:.2f}", weight="bold", fontsize=FONT_TITLE)
    ax_pr.set_xlabel("Recall",    fontsize=FONT_LABEL)
    ax_pr.set_ylabel("Precision", fontsize=FONT_LABEL)
    _apply_style(ax_pr)
    ax_pr.legend(
        handles=[
            Line2D([0], [0], color="black", lw=LW,          label="Mean PR"),
            Line2D([0], [0], color="black", lw=LW, ls="--", label=f"Prevalence = {prevalence:.3f}"),
        ],
        loc=(0.5, 0.7), frameon=False,
        prop={"weight": "normal", "size": FONT_LEGEND}
    )
    plt.tight_layout()
    if plt_show:
        plt.show()

    # ── DCA プロット ─────────────────────────────────────────
    def _plot_dca(xlim=None, ylim=None, zoom=False):
        fig_dca, ax_dca = plt.subplots(figsize=(12, 12))

        ax_dca.fill_between(
            dca_thresholds,
            mean_nb - std_nb,
            mean_nb + std_nb,
            color="black", alpha=0.15
        )
        ax_dca.plot(dca_thresholds, mean_nb,     color="black", lw=LW)
        ax_dca.plot(dca_thresholds, mean_nb_all, color="black", lw=LW, ls="--")
        ax_dca.plot(dca_thresholds, nb_none,     color="black", lw=LW, ls=":")

        if xlim:
            ax_dca.set_xlim(xlim)
        if ylim:
            ax_dca.set_ylim(ylim)

        suffix = " (0-0.2)" if zoom else ""
        # ax_dca.set_title(f"DCA{suffix}", weight="bold", fontsize=FONT_TITLE)
        ax_dca.set_xlabel("Threshold Probability", fontsize=FONT_LABEL)
        ax_dca.set_ylabel("Net Benefit",           fontsize=FONT_LABEL)
        _apply_style(ax_dca)
        ax_dca.legend(
            handles=[
                Line2D([0], [0], color="black", lw=LW,          label="Model"),
                Patch(facecolor="black", alpha=0.15, label="±1 std. dev."),
                Line2D([0], [0], color="black", lw=LW, ls="--", label="Treat all"),
                Line2D([0], [0], color="black", lw=LW, ls=":",  label="Treat none"),
            ],
            loc="upper right", frameon=False,
            prop={"weight": "normal", "size": FONT_LEGEND}
        )
        plt.tight_layout()
        if plt_show:
            plt.show()

        return fig_dca, ax_dca

    mask = dca_thresholds <= 0.2
    y_max_zoom = max(np.nanmax(mean_nb[mask]), np.nanmax(mean_nb_all[mask])) * 1.2
    y_min_zoom = -np.nanmax(mean_nb_all[mask]) * 1.5
    fig_dca_zoom, ax_dca_zoom = _plot_dca(xlim=(0, 0.2), ylim=(y_min_zoom, y_max_zoom), zoom=True)

    # ── キャリブレーションプロット ───────────────────────────
    fig_calib, ax_calib = plt.subplots(figsize=(12, 12))

    ax_calib.fill_between(
        common_x_calib,
        np.maximum(mean_calib - std_calib, 0),
        np.minimum(mean_calib + std_calib, 1),
        color="black", alpha=0.15
    )
    ax_calib.plot(common_x_calib, mean_calib, color="black", lw=LW, label="Mean calibration")
    ax_calib.plot([0, 1], [0, 1], linestyle="--", lw=LW, color="black", alpha=0.8, label="Perfect calibration")

    ax_calib.set_xlim(-0.05, 1.05)
    ax_calib.set_ylim(-0.05, 1.05)
    ax_calib.set_aspect("equal")
    ax_calib.set_title("Calibration", weight="bold", fontsize=FONT_TITLE)
    ax_calib.set_xlabel("Mean Predicted Probability", fontsize=FONT_LABEL)
    ax_calib.set_ylabel("Fraction of Positives",      fontsize=FONT_LABEL)
    _apply_style(ax_calib)
    ax_calib.legend(
        handles=[
            Line2D([0], [0], color="black", lw=LW,          label="Mean calibration"),
            Line2D([0], [0], color="black", lw=LW, ls="--", label="Perfect calibration"),
        ],
        loc=(0.02, 0.7), frameon=False,
        prop={"weight": "normal", "size": FONT_LEGEND}
    )
    plt.tight_layout()
    if plt_show:
        plt.show()

    # ── Sensitivity vs PCR削減率プロット ────────────────────
    fig_pcr, ax_pcr = plt.subplots(figsize=(12, 12))

    ax_pcr.fill_between(
        common_sens,
        np.maximum(mean_pcr - std_pcr, 0),
        np.minimum(mean_pcr + std_pcr, 1),
        color="black", alpha=0.15
    )
    ax_pcr.plot(common_sens, mean_pcr, color="black", lw=LW)
    ax_pcr.set_xlim(-0.05, 1.05)
    ax_pcr.set_ylim(-0.05, 1.05)
    ax_pcr.set_aspect("equal")
    ax_pcr.set_title("", weight="bold", fontsize=FONT_TITLE)
    ax_pcr.set_xlabel("Sensitivity",             fontsize=FONT_LABEL)
    ax_pcr.set_ylabel("PCR Reduction Rate", fontsize=FONT_LABEL)
    _apply_style(ax_pcr)
    
    ax_pcr.legend(
    handles=[
        Line2D([0], [0], color="black", lw=LW, label="Mean"),
        Patch(facecolor="black", alpha=0.15, label="±1 std. dev."),
    ],
    loc="upper right",
    frameon=False,
    prop={"weight": "normal", "size": FONT_LEGEND}
    )
    plt.tight_layout()
    if plt_show:
        plt.show()   
    
    # ── Sensitivity vs PCR削減率 + 右軸: 年間コスト削減額 ─────
    fig_pcr_cost_annual, ax_pcr_cost_annual = plt.subplots(figsize=(12, 12))

    ax_pcr_cost_annual.fill_between(
        common_sens,
        np.maximum(mean_pcr - std_pcr, 0),
        np.minimum(mean_pcr + std_pcr, 1),
        color="black", alpha=0.15
    )
    ax_pcr_cost_annual.plot(common_sens, mean_pcr, color="black", lw=LW)

    ax_pcr_cost_annual.set_xlim(-0.05, 1.05)
    ax_pcr_cost_annual.set_ylim(-0.05, 1.05)
    ax_pcr_cost_annual.set_aspect("equal")
    ax_pcr_cost_annual.set_title("", weight="bold", fontsize=FONT_TITLE)
    ax_pcr_cost_annual.set_xlabel("Sensitivity", fontsize=FONT_LABEL)
    ax_pcr_cost_annual.set_ylabel("PCR Reduction Rate", fontsize=FONT_LABEL)
    _apply_style(ax_pcr_cost_annual)

    # 右軸（1年換算）
    ax_pcr_cost_annual_r = ax_pcr_cost_annual.twinx()
    y0, y1 = ax_pcr_cost_annual.get_ylim()
    annual_cost_y0 = y0 * len(y) * pcr_unit_cost * annual_factor
    annual_cost_y1 = y1 * len(y) * pcr_unit_cost * annual_factor
    ax_pcr_cost_annual_r.set_ylim(annual_cost_y0, annual_cost_y1)
    ax_pcr_cost_annual_r.set_ylabel(
        "Estimated Cost Savings (JPY)",
        fontsize=FONT_LABEL
    )
    ax_pcr_cost_annual_r.tick_params(labelsize=FONT_TICK)

    ax_pcr_cost_annual.legend(
        handles=[
            Line2D([0], [0], color="black", lw=LW, label="Mean"),
            Patch(facecolor="black", alpha=0.15, label="±1 std. dev."),
        ],
        loc="upper right",
        frameon=False,
        prop={"weight": "normal", "size": FONT_LEGEND}
    )
    plt.tight_layout()
    if plt_show:
        plt.show()

    # ── Sensitivity vs PCR削減率 + 右軸: 全期間コスト削減額 ─────
    fig_pcr_cost_total, ax_pcr_cost_total = plt.subplots(figsize=(12, 12))

    ax_pcr_cost_total.fill_between(
        common_sens,
        np.maximum(mean_pcr - std_pcr, 0),
        np.minimum(mean_pcr + std_pcr, 1),
        color="black", alpha=0.15
    )
    ax_pcr_cost_total.plot(common_sens, mean_pcr, color="black", lw=LW)

    ax_pcr_cost_total.set_xlim(-0.05, 1.05)
    ax_pcr_cost_total.set_ylim(-0.05, 1.05)
    ax_pcr_cost_total.set_aspect("equal")
    ax_pcr_cost_total.set_title("", weight="bold", fontsize=FONT_TITLE)
    ax_pcr_cost_total.set_xlabel("Sensitivity", fontsize=FONT_LABEL)
    ax_pcr_cost_total.set_ylabel("PCR Reduction Rate", fontsize=FONT_LABEL)
    _apply_style(ax_pcr_cost_total)

    # 右軸（全期間）
    ax_pcr_cost_total_r = ax_pcr_cost_total.twinx()
    y0, y1 = ax_pcr_cost_total.get_ylim()
    total_cost_y0 = y0 * len(y) * pcr_unit_cost
    total_cost_y1 = y1 * len(y) * pcr_unit_cost
    ax_pcr_cost_total_r.set_ylim(total_cost_y0, total_cost_y1)
    ax_pcr_cost_total_r.set_ylabel(
        "Estimated Cost Savings (JPY)",
        fontsize=FONT_LABEL
    )
    ax_pcr_cost_total_r.tick_params(labelsize=FONT_TICK)

    ax_pcr_cost_total.legend(
        handles=[
            Line2D([0], [0], color="black", lw=LW, label="Mean"),
            Patch(facecolor="black", alpha=0.15, label="±1 std. dev."),
        ],
        loc="upper right",
        frameon=False,
        prop={"weight": "normal", "size": FONT_LEGEND}
    )
    plt.tight_layout()
    if plt_show:
        plt.show() 
    
    # ── Clinical Impact Curve 集計 ───────────────────────────
    hr_mat      = np.array([f["high_risk_rate"] for f in fold_cic_curves])
    tp_rate_mat = np.array([f["tp_rate"]        for f in fold_cic_curves])

    mean_hr      = np.mean(hr_mat, axis=0)
    std_hr       = np.std(hr_mat,  axis=0)
    mean_tp_rate = np.mean(tp_rate_mat, axis=0)
    std_tp_rate  = np.std(tp_rate_mat,  axis=0)    

    # ── Clinical Impact Curve プロット ───────────────────────
    fig_cic, ax_cic = plt.subplots(figsize=(12, 12))

    ax_cic.fill_between(
        dca_thresholds,
        np.maximum(mean_hr - std_hr, 0),
        np.minimum(mean_hr + std_hr, 1),
        color="black", alpha=0.10
    )
    ax_cic.plot(dca_thresholds, mean_hr, color="black", lw=LW)

    ax_cic.fill_between(
        dca_thresholds,
        np.maximum(mean_tp_rate - std_tp_rate, 0),
        np.minimum(mean_tp_rate + std_tp_rate, 1),
        color="dimgray", alpha=0.15
    )
    ax_cic.plot(dca_thresholds, mean_tp_rate, color="dimgray", lw=LW, ls="--")

    ax_cic.set_ylabel("Proportion of Patients", fontsize=FONT_LABEL)
    y_max = max(
    np.nanmax(mean_hr + std_hr),
    np.nanmax(mean_tp_rate + std_tp_rate)
    )
    ax_cic.set_ylim(-0.05, y_max * 1.1)
    _apply_style(ax_cic)
    ax_cic.legend(
        handles=[
            Line2D([0], [0], color="black", lw=LW, label="High risk (TP+FP)"),
            Patch(facecolor="dimgray", alpha=0.15, label="±1 std. dev. (TP)"),
            Line2D([0], [0], color="dimgray", lw=LW, ls="--", label="True events (TP)"),
        ],
        loc="upper right", frameon=False,
                prop={"weight": "normal", "size": FONT_LEGEND})
    plt.tight_layout()
    if plt_show:
        plt.show()

    # ── NPV vs 検査削減率 集計 ────────────────────────────────
    npv_lower = 1 - prevalence
    common_npv = np.linspace(npv_lower, 1, 100)

    npv2_mat = []
    for fold in fold_npv_curves:
        npv = fold["npv"]
        pcr = fold["pcr_reduction"]

        valid = ~np.isnan(npv)
        df = pd.DataFrame({"npv": npv[valid], "pcr": pcr[valid]})

        # 同一NPVに対してPCR削減率の最大値を採用
        df = df.groupby("npv")["pcr"].max().reset_index().sort_values("npv")

        npv2_mat.append(np.interp(
            common_npv,
            df["npv"].values,
            df["pcr"].values,
            left=df["pcr"].values[0],
            right=df["pcr"].values[-1],
        ))

    npv2_mat  = np.array(npv2_mat)
    mean_npv2 = np.mean(npv2_mat, axis=0)
    std_npv2  = np.std(npv2_mat,  axis=0)

    # ── NPV vs 検査削減率プロット ────────────────────────────
    fig_npv, ax_npv = plt.subplots(figsize=(12, 12))

    ax_npv.fill_between(
        common_npv,
        np.maximum(mean_npv2 - std_npv2, 0),
        np.minimum(mean_npv2 + std_npv2, 1),
        color="black", alpha=0.15
    )
    ax_npv.plot(common_npv, mean_npv2, color="black", lw=LW)

    ax_npv.set_xlim(npv_lower - 0.01, 1)
    ax_npv.set_ylim(-0.05, 1.05)
    ax_npv.set_title("NPV vs Test Reduction Rate", weight="bold", fontsize=FONT_TITLE)
    ax_npv.set_xlabel("NPV (TN / (TN + FN))", fontsize=FONT_LABEL)
    ax_npv.set_ylabel("Test Reduction Rate", fontsize=FONT_LABEL)
    _apply_style(ax_npv)
    ax_npv.legend(
        handles=[Line2D([0], [0], color="black", lw=LW, label="Mean")],
        loc="upper left", frameon=False,
        prop={"weight": "normal", "size": FONT_LEGEND}
    )
    plt.tight_layout()
    if plt_show:
        plt.show()
    # ── CV メトリクス集計 ────────────────────────────────────
    cv_metrics = {
        "accuracy_mean":    np.mean(accuracies),
        "accuracy_std":     np.std(accuracies),
        "recall_mean":      np.mean(recalls),
        "recall_std":       np.std(recalls),
        "precision_mean":   np.mean(precisions),
        "precision_std":    np.std(precisions),
        "specificity_mean": np.mean(specificities),
        "specificity_std":  np.std(specificities),
        "f1_mean":          np.mean(f1_scores),
        "f1_std":           np.std(f1_scores),
        "auc_mean":         mean_auc,
        "auc_std":          std_auc,
        "ap_oof":           ap_oof,
        "threshold_mean":   np.mean(used_thresholds),
        "threshold_std":    np.std(used_thresholds),
    }

    print("\n=== CV Metrics ===")
    for k, v in cv_metrics.items():
        print(f"{k:20s}: {v:.3f}")

    # ── メトリクス表プロット ─────────────────────────────────
    metrics_display = [
        ("Accuracy",    "accuracy"),
        ("Recall",      "recall"),
        ("Precision",   "precision"),
        ("Specificity", "specificity"),
        ("F1 Score",    "f1"),  
        ("AUC",         "auc"),
    ]
    cell_text  = [[f"{cv_metrics[f'{k}_mean']:.3f} ± {cv_metrics[f'{k}_std']:.3f}"] for _, k in metrics_display]
    row_labels = [label for label, _ in metrics_display]

    fig_table, ax_table = plt.subplots(figsize=(2, 2), dpi=200)
    ax_table.axis("off")
    table = ax_table.table(
        cellText=cell_text, rowLabels=row_labels,
        colLabels=None, loc="center", cellLoc="center"
    )
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.get_text().set_weight("normal")
        cell.get_text().set_color("white")
        cell.set_facecolor("dimgray" if j == 0 else "gray")
    table.auto_set_column_width(col=list(range(len(cell_text[0]))))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)
    plt.tight_layout()
    if plt_show:
        plt.show()

    # ── 全データ再学習 ───────────────────────────────────────
    smote_final  = _make_smote(smote_type, seed, X.columns, numerical_features, y)
    X_all, y_all = smote_final.fit_resample(X, y)
    final_model  = clone(model_base)
    final_model.fit(X_all, y_all)
    final_thr    = cv_metrics["threshold_mean"]

    figures = {
        "roc":              fig_roc,
        "pr":               fig_pr,
        "dca_zoom":         fig_dca_zoom,
        "calib":            fig_calib,
        "pcr":              fig_pcr,
        "pcr_cost_annual":  fig_pcr_cost_annual,
        "pcr_cost_total":   fig_pcr_cost_total,
        "cic":              fig_cic,
        "npv":              fig_npv,
        "table":            fig_table,
    }

    return final_model,cv_metrics, final_thr, all_y_true, all_y_proba, figures, tprs, aucs


def plot_shap_summary(X, y, title, n_tree=1000, seed=42, model_type="rf", plot_label_dict=None):
    import shap

    # =====================================
    # 初期設定
    # =====================================
    # SMOTEによるリサンプリングの設定
    smote = _make_smote("SMOTE", seed, X.columns, y=y)
    X_resampled, y_resampled = smote.fit_resample(X, y)  # リサンプリングの実行
        
    model_base = make_model(model_type, n_tree, seed)
    model = clone(model_base)
        
    clf = model.fit(X_resampled, y_resampled)  # リサンプリング後のデータを使用

    # =====================================
    # SHAPのExplainerを作成し、SHAP値を計算
    # =====================================
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values_for_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif getattr(shap_values, "ndim", None) == 3:
        shap_values_for_plot = shap_values[:, :, 1]
    else:
        shap_values_for_plot = shap_values

    # =====================================
    # SHAPサマリープロットの作成
    # =====================================
    if plot_label_dict is None:
        pass
    else:
        # プロット項目を日本語にする
        matplotlib.rcParams['font.family'] = 'Meiryo'
        X = X.rename(columns={v: k for v, k in plot_label_dict.items() if v in X.columns})

    plt.figure(figsize=(12, 8))
    matplotlib.rcParams['font.family'] = 'Arial'

    # SHAP summary plot（自動スケール、max_display=15）
    shap.summary_plot(
        shap_values=shap_values_for_plot,
        features=X,
        show=False,
        plot_size=(12, 12),
        max_display=X.shape[1]  # 全特徴量を表示
    )

    # 軸ラベル・フォント設定（固定スケールなし）
    plt.xlabel('SHAP value', weight='normal', size=16)
    plt.xticks(fontsize=16, weight='normal')
    plt.yticks(fontsize=32, weight='normal')

    # 下枠線（X 軸）強調
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')

    # 保存は関数外で行う
    plt.tight_layout()
    fig_shap = plt.gcf()   
    plt.show() 
    return shap_values, fig_shap
