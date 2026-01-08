"""
=============================================================================
PARAMETER GRID SEARCH: Random Forest vs XGBoost
=============================================================================

Testing n_estimators from 10 to 500 (increments of 50 after 10)
Fixed max_depth = 10 for both models

Goal: Find the optimal number of estimators for flight risk detection
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available")

print("=" * 80)
print("PARAMETER GRID SEARCH: n_estimators Optimization")
print("Grid: [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]")
print("Fixed: max_depth = 10 for both models")
print("=" * 80)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

print("\n[LOAD] Loading and preprocessing data...")
df = pd.read_csv('data/employee_attrition.csv')
df['Attrition_Flag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

FEATURE_COLUMNS = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

df_encoded = pd.get_dummies(df[FEATURE_COLUMNS], drop_first=True)
df_model = pd.concat([df_encoded, df['Attrition_Flag']], axis=1)

X = df_model.drop('Attrition_Flag', axis=1)
y = df_model['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for XGBoost
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"✓ Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")

# ============================================================================
# GRID SEARCH
# ============================================================================

estimator_grid = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
MAX_DEPTH = 10

results = []

print("\n" + "=" * 80)
print("RUNNING GRID SEARCH...")
print("=" * 80)

for n_est in estimator_grid:
    print(f"\nTesting n_estimators = {n_est}...", end=" ")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=MAX_DEPTH,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    cm_rf = confusion_matrix(y_test, y_pred_rf).ravel()
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf

    results.append({
        'Model': 'Random Forest',
        'n_estimators': n_est,
        'max_depth': MAX_DEPTH,
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0,
        'Recall': tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0,
        'F1': 2 * (tp_rf / (tp_rf + fp_rf) * tp_rf / (tp_rf + fn_rf)) / (tp_rf / (tp_rf + fp_rf) + tp_rf / (tp_rf + fn_rf)) if (tp_rf + fp_rf) > 0 and (tp_rf + fn_rf) > 0 else 0,
        'AUC': roc_auc_score(y_test, y_prob_rf),
        'True_Positives': tp_rf,
        'False_Positives': fp_rf,
        'False_Negatives': fn_rf
    })

    # XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=n_est,
            max_depth=MAX_DEPTH,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        xgb.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb.predict(X_test_scaled)
        y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]

        cm_xgb = confusion_matrix(y_test, y_pred_xgb).ravel()
        tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb

        results.append({
            'Model': 'XGBoost',
            'n_estimators': n_est,
            'max_depth': MAX_DEPTH,
            'Accuracy': accuracy_score(y_test, y_pred_xgb),
            'Precision': tp_xgb / (tp_xgb + fp_xgb) if (tp_xgb + fp_xgb) > 0 else 0,
            'Recall': tp_xgb / (tp_xgb + fn_xgb) if (tp_xgb + fn_xgb) > 0 else 0,
            'F1': 2 * (tp_xgb / (tp_xgb + fp_xgb) * tp_xgb / (tp_xgb + fn_xgb)) / (tp_xgb / (tp_xgb + fp_xgb) + tp_xgb / (tp_xgb + fn_xgb)) if (tp_xgb + fp_xgb) > 0 and (tp_xgb + fn_xgb) > 0 else 0,
            'AUC': roc_auc_score(y_test, y_prob_xgb),
            'True_Positives': tp_xgb,
            'False_Positives': fp_xgb,
            'False_Negatives': fn_xgb
        })

    print(f"RF Recall: {results[-2]['Recall']*100:.1f}%", end="")
    if XGBOOST_AVAILABLE:
        print(f" | XGB Recall: {results[-1]['Recall']*100:.1f}%")
    else:
        print()

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("GRID SEARCH RESULTS (max_depth = 10)")
print("=" * 80)

# Pivot for comparison
rf_results = results_df[results_df['Model'] == 'Random Forest'].sort_values('n_estimators')
xgb_results = results_df[results_df['Model'] == 'XGBoost'].sort_values('n_estimators')

print("\n┌─────────────────────────────────────────────────────────────────────────────────────────")
print("│                          RANDOM FOREST RESULTS")
print("├──────────┬───────────┬────────────┬───────────┬──────────┬──────────┬───────────────")
print("│   Est    │  Accuracy │  Precision │   Recall  │    F1    │   AUC    │  TP / FN / FP")
print("├──────────┼───────────┼────────────┼───────────┼──────────┼──────────┼───────────────")

for _, row in rf_results.iterrows():
    print(f"│  {int(row['n_estimators']):>4}    │   {row['Accuracy']*100:>5.1f}%  │    {row['Precision']*100:>5.1f}%   │   {row['Recall']*100:>5.1f}%  │  {row['F1']*100:>5.1f}%  │  {row['AUC']:>5.3f}  │ "
          f"{int(row['True_Positives']):>2} / {int(row['False_Negatives']):>2} / {int(row['False_Positives']):>2}")

print("└──────────┴───────────┴────────────┴───────────┴──────────┴──────────┴───────────────")

if XGBOOST_AVAILABLE:
    print("\n┌─────────────────────────────────────────────────────────────────────────────────────────")
    print("│                          XGBOOST RESULTS")
    print("├──────────┬───────────┬────────────┬───────────┬──────────┬──────────┬───────────────")
    print("│   Est    │  Accuracy │  Precision │   Recall  │    F1    │   AUC    │  TP / FN / FP")
    print("├──────────┼───────────┼────────────┼───────────┼──────────┼──────────┼───────────────")

    for _, row in xgb_results.iterrows():
        print(f"│  {int(row['n_estimators']):>4}    │   {row['Accuracy']*100:>5.1f}%  │    {row['Precision']*100:>5.1f}%   │   {row['Recall']*100:>5.1f}%  │  {row['F1']*100:>5.1f}%  │  {row['AUC']:>5.3f}  │ "
              f"{int(row['True_Positives']):>2} / {int(row['False_Negatives']):>2} / {int(row['False_Positives']):>2}")

    print("└──────────┴───────────┴────────────┴───────────┴──────────┴──────────┴───────────────")

# ============================================================================
# BEST CONFIGURATION BY RECALL (Flight Risk Catch Rate)
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMAL CONFIGURATIONS (max_depth = 10)")
print("=" * 80)

best_rf = rf_results.loc[rf_results['Recall'].idxmax()]
print(f"\n🌲 RANDOM FOREST BEST (by Recall):")
print(f"   n_estimators = {int(best_rf['n_estimators'])}")
print(f"   Recall = {best_rf['Recall']*100:.1f}% (catches {int(best_rf['True_Positives'])} flight risks)")
print(f"   Precision = {best_rf['Precision']*100:.1f}%")
print(f"   AUC = {best_rf['AUC']:.3f}")

if XGBOOST_AVAILABLE:
    best_xgb = xgb_results.loc[xgb_results['Recall'].idxmax()]
    print(f"\n🚀 XGBOOST BEST (by Recall):")
    print(f"   n_estimators = {int(best_xgb['n_estimators'])}")
    print(f"   Recall = {best_xgb['Recall']*100:.1f}% (catches {int(best_xgb['True_Positives'])} flight risks)")
    print(f"   Precision = {best_xgb['Precision']*100:.1f}%")
    print(f"   AUC = {best_xgb['AUC']:.3f}")

# ============================================================================
# HEAD-TO-HEAD AT OPTIMAL POINTS
# ============================================================================

if XGBOOST_AVAILABLE:
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD: Best of Each Model")
    print("=" * 80)

    print(f"\n{'Metric':<15} {'Random Forest':<20} {'XGBoost':<20} {'Winner'}")
    print("─" * 70)

    rf_winner = "RF" if best_rf['Recall'] > best_xgb['Recall'] else "XGB"
    print(f"{'Recall':<15} {best_rf['Recall']*100:>6.1f}%              {best_xgb['Recall']*100:>6.1f}%              {rf_winner}")

    rf_winner = "RF" if best_rf['Precision'] > best_xgb['Precision'] else "XGB"
    print(f"{'Precision':<15} {best_rf['Precision']*100:>6.1f}%              {best_xgb['Precision']*100:>6.1f}%              {rf_winner}")

    rf_winner = "RF" if best_rf['AUC'] > best_xgb['AUC'] else "XGB"
    print(f"{'AUC-ROC':<15} {best_rf['AUC']:>6.3f}               {best_xgb['AUC']:>6.3f}               {rf_winner}")

    rf_winner = "RF" if best_rf['True_Positives'] > best_xgb['True_Positives'] else "XGB"
    print(f"{'Caught Risks':<15} {int(best_rf['True_Positives']):>6}                {int(best_xgb['True_Positives']):>6}                {rf_winner}")

# Save results
results_df.to_csv('output/parameter_grid_search_results.csv', index=False)
print(f"\n✓ Results saved to: output/parameter_grid_search_results.csv")

print("\n" + "=" * 80)
print("GRID SEARCH COMPLETE")
print("=" * 80)
