import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from group_features import ALL_FEATURES_FOR_DATA_FRAME_EXCLUDE_WASHOUT, morph_features, atten_features, ALL_FEATURES_FOR_DATA_FRAME, PHASES, META_COLS
from models import make_xgb


# -------------------------
# 0) Define feature subsets
# -------------------------
PHASE_TEXTURE = {
    "NC":    ["NC_contrast", "NC_correlation", "NC_energy", "NC_homogeneity", "avg_HU_NC"],
    "AR":    ["AR_contrast", "AR_correlation", "AR_energy", "AR_homogeneity", "avg_HU_AR"],
    "PV":    ["PV_contrast", "PV_correlation", "PV_energy", "PV_homogeneity", "avg_HU_PV"],
    "Delay": ["Delay_contrast", "Delay_correlation", "Delay_energy", "Delay_homogeneity", "avg_HU_Delay"],
}

# Per-phase "token" = morph + that phase's HU/texture (+ optionally attenuation if you want it global)
PHASE_TOKEN_FEATURES = {
    p: morph_features + PHASE_TEXTURE[p] for p in PHASES
}

# Washout/attenuation is typically multi-phase derived; treat as global (shared) features
WASHOUT_FEATURES = atten_features[:]  # can be [] if you want phase-only


# -------------------------
# 1) Utilities
# -------------------------
def eval_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "AUC": auc,
        "ACC": acc,
        "Precision": prec,
        "Recall(Sens)": sens,
        "Specificity": spec,
        "F1": f1,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

def stratified_group_folds(df, n_splits=5, seed=42, group_col="MRN", label_col="malignancy"):
    X_dummy = np.zeros((len(df), 1))
    y = df[label_col].astype(int).values
    groups = df[group_col].astype(str).values
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X_dummy, y, groups))


# -------------------------
# 2) Fusion strategies
# -------------------------

# 2.1 Early fusion: all features (global + phase-specific) concatenated
def prepare_early_fusion(df, with_washout=True):
    if not with_washout:
        X = df[ALL_FEATURES_FOR_DATA_FRAME_EXCLUDE_WASHOUT].copy()
    else:
        X = df[ALL_FEATURES_FOR_DATA_FRAME].copy()
    y = df["malignancy"].astype(int).values
    return X.values, y


# 2.2 Late fusion:
#   Train phase-specific models on samples where that phase is present, then:
#     (A) simple average of available phase probabilities
#     (B) optional stacking meta-model (XGBoost) on phase probs + missing indicators (+ global feats optional)
def prepare_phase_dataset(df, phase, with_washout=True):
    if not with_washout:
        feats = PHASE_TOKEN_FEATURES[phase][:-1] # exclude washout from token, keep avg_HU
    else:
        feats = PHASE_TOKEN_FEATURES[phase] + WASHOUT_FEATURES
    X = df[feats].copy()
    y = df["malignancy"].astype(int).values

    # define phase-present mask: at least the avg_HU_* exists (or available_phases contains phase)
    # Use a robust condition: available_phases string contains phase OR avg_HU_* notna
    avg_hu_col = PHASE_TEXTURE[phase][-1]
    present = df["available_phases"].astype(str).str.contains(phase, na=False) | df[avg_hu_col].notna()
    return X, y, present.values, feats

def late_fusion_simple_average(phase_probs_dict, phase_present_dict):
    """
    phase_probs_dict: {phase: probs for all samples (np.array len N; NaN for missing allowed)}
    phase_present_dict: {phase: boolean mask len N}
    """
    N = len(next(iter(phase_probs_dict.values())))
    probs = np.zeros(N, dtype=float)
    counts = np.zeros(N, dtype=float)

    for p in PHASES:
        present = phase_present_dict[p].astype(bool)
        pp = phase_probs_dict[p]
        probs[present] += pp[present]
        counts[present] += 1.0

    # if a sample has zero phases present (shouldn't happen), set NaN then later fill 0.5
    out = np.full(N, np.nan, dtype=float)
    ok = counts > 0
    out[ok] = probs[ok] / counts[ok]
    out[~ok] = 0.5
    return out

def build_late_fusion_stacking_features(df, phase_probs_dict, phase_present_dict, use_global_feats=True):
    """
    Returns X_meta DataFrame for stacking:
      - phase probability columns: prob_NC, prob_AR, ...
      - missing indicators: miss_NC, ...
      - optional global features (atten_features)
    """
    X_meta = pd.DataFrame(index=df.index)

    for p in PHASES:
        X_meta[f"prob_{p}"] = phase_probs_dict[p]
        X_meta[f"miss_{p}"] = (~phase_present_dict[p].astype(bool)).astype(int)

    # XGBoost can handle NaN; but phase prob is undefined when missing -> keep NaN
    if use_global_feats and len(WASHOUT_FEATURES) > 0:
        for f in WASHOUT_FEATURES:
            X_meta[f] = df[f]

    return X_meta


# 2.3 “Transformer-style” fusion (mask-based tokens) with XGBoost:
#   - Treat each phase as a token vector (same dims across phases)
#   - Add missingness mask per token feature (or per phase)
#   - Flatten [tokens] into one long vector -> XGBoost
#
# Practical way:
#   - choose a fixed token feature set per phase (here: morph + that phase texture/HU)
#   - build columns like: NC__area, NC__perimeter, ..., AR__area, ...
#   - also add mask columns: NC__mask_area,... or simply NC__present
def prepare_mask_token_fusion(df, add_featurewise_mask=True, add_phase_id_onehot=False):
    out = pd.DataFrame(index=df.index)

    # Phase presence flags
    phase_present = {}
    for p in PHASES:
        avg_hu_col = PHASE_TEXTURE[p][-1]
        present = df["available_phases"].astype(str).str.contains(p, na=False) | df[avg_hu_col].notna()
        phase_present[p] = present.values
        out[f"{p}__present"] = present.astype(int)

    # Token features (same semantic list per phase)
    for p in PHASES:
        feats = PHASE_TOKEN_FEATURES[p]
        for f in feats:
            # store under phase-specific column name
            out[f"{p}__{f}"] = df[f]  # same base column name exists (e.g., area) for morph; phase texture names are unique already

            if add_featurewise_mask:
                out[f"{p}__{f}__miss"] = df[f].isna().astype(int)

    # Global features appended once
    for f in WASHOUT_FEATURES:
        out[f"WASHOUT__{f}"] = df[f]
        if add_featurewise_mask:
            out[f"WASHOUT__{f}__miss"] = df[f].isna().astype(int)

    # Optional: phase id one-hot doesn’t add much once we prefix names
    if add_phase_id_onehot:
        for p in PHASES:
            for q in PHASES:
                out[f"{p}__phaseid_{q}"] = 1 if p == q else 0

    y = df["malignancy"].astype(int).values
    return out, y


# -------------------------
# 3) Cross-validated comparison
# -------------------------
def compare_fusion_strategies(df, n_splits=5, seed=42, group_col="MRN", with_washout=True):
    splits = stratified_group_folds(df, n_splits=n_splits, seed=seed, group_col=group_col)

    results = {
        "early": [],
        "late_avg": [],
        "late_stack": [],
        "mask_token": [],
    }

    for fold_i, (tr_idx, te_idx) in enumerate(splits, start=1):
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)

        y_tr = df_tr["malignancy"].astype(int).values
        y_te = df_te["malignancy"].astype(int).values

        # ---- Early fusion ----
        X_tr_early, _ = prepare_early_fusion(df_tr, with_washout=with_washout)
        X_te_early, _ = prepare_early_fusion(df_te, with_washout=with_washout)

        m_early = make_xgb(seed)
        m_early.fit(X_tr_early, y_tr)
        p_te_early = m_early.predict_proba(X_te_early)[:, 1]
        results["early"].append(eval_binary(y_te, p_te_early))

        # ---- Late fusion (phase models) ----
        phase_probs_te = {}
        phase_present_te = {}
        phase_probs_tr = {}
        phase_present_tr = {}

        for p in PHASES:
            X_tr_p, y_tr_p, present_tr, feats_tr = prepare_phase_dataset(df_tr, p, with_washout=with_washout)
            X_te_p, y_te_p, present_te, feats_te = prepare_phase_dataset(df_te, p, with_washout=with_washout)

            # Train only on samples where phase is present
            tr_mask = present_tr.astype(bool)
            te_mask = present_te.astype(bool)

            phase_present_tr[p] = present_tr
            phase_present_te[p] = present_te

            # If a phase is extremely rare in a fold, skip safely
            if tr_mask.sum() < 10 or len(np.unique(y_tr_p[tr_mask])) < 2:
                # fallback: predict 0.5 for present, NaN for missing
                phase_probs_te[p] = np.where(te_mask, 0.5, np.nan).astype(float)
                phase_probs_tr[p] = np.where(tr_mask, 0.5, np.nan).astype(float)
                continue

            m_p = make_xgb(seed)
            m_p.fit(X_tr_p.loc[tr_mask, feats_tr].values, y_tr_p[tr_mask])

            # Predict for all samples; keep NaN where missing
            te_prob = np.full(len(df_te), np.nan, dtype=float)
            tr_prob = np.full(len(df_tr), np.nan, dtype=float)

            te_prob[te_mask] = m_p.predict_proba(X_te_p.loc[te_mask, feats_te].values)[:, 1]
            tr_prob[tr_mask] = m_p.predict_proba(X_tr_p.loc[tr_mask, feats_tr].values)[:, 1]

            phase_probs_te[p] = te_prob
            phase_probs_tr[p] = tr_prob

        # (A) simple average of available phase probs
        p_te_late_avg = late_fusion_simple_average(phase_probs_te, phase_present_te)
        results["late_avg"].append(eval_binary(y_te, p_te_late_avg))

        # (B) stacking: meta model on phase probs + missing indicators (+ global feats)
        X_tr_meta = build_late_fusion_stacking_features(df_tr, phase_probs_tr, phase_present_tr, use_global_feats=True)
        X_te_meta = build_late_fusion_stacking_features(df_te, phase_probs_te, phase_present_te, use_global_feats=True)

        m_meta = make_xgb(seed)
        m_meta.fit(X_tr_meta.values, y_tr)
        p_te_late_stack = m_meta.predict_proba(X_te_meta.values)[:, 1]
        results["late_stack"].append(eval_binary(y_te, p_te_late_stack))

        # ---- Mask-token fusion (“transformer-style” with missingness mask) ----
        X_tr_tok, _ = prepare_mask_token_fusion(df_tr, add_featurewise_mask=True, add_phase_id_onehot=False)
        X_te_tok, _ = prepare_mask_token_fusion(df_te, add_featurewise_mask=True, add_phase_id_onehot=False)

        m_tok = make_xgb(seed)
        m_tok.fit(X_tr_tok.values, y_tr)
        p_te_tok = m_tok.predict_proba(X_te_tok.values)[:, 1]
        results["mask_token"].append(eval_binary(y_te, p_te_tok))

        #


        print(f"Fold {fold_i}/{n_splits} done.")

    # Aggregate
    def summarize(res_list):
        dfm = pd.DataFrame(res_list)
        mean = dfm.mean(numeric_only=True)
        std = dfm.std(numeric_only=True)
        out = pd.DataFrame({"mean": mean, "std": std})
        return out

    summary = {k: summarize(v) for k, v in results.items()}
    return results, summary


