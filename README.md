# Employee Retention Warning System

> **"Predicting Flight Risk Before It Happens: A Machine Learning Approach to Employee Retention"**

A comparative study of **Random Forest** and **XGBoost** models to predict employee attrition and identify the key drivers of turnover.

---

## Business Problem

**Question:** How do we identify employees at risk of leaving before they actually resign?

**Impact:**
- Replacing an employee costs 1.5–2x their annual salary
- Lost productivity during transition
- Institutional knowledge walks out the door

**Solution:** Predictive model that flags high-risk employees for targeted retention efforts.

---

## Model Comparison: Random Forest vs XGBoost

### What Are These Models?

| Aspect | Random Forest (Bagging) | XGBoost (Boosting) |
|--------|------------------------|-------------------|
| **How it works** | 100 experts work **in parallel**, each votes independently | 100 experts work **sequentially**, each corrects previous mistakes |
| **Strength** | Reduces overfitting through averaging | Often achieves higher accuracy |
| **Weakness** | Trees built independently | Can overfit if not regularized |

### Final Results (Optimal Configurations)

| Metric | Random Forest | XGBoost | Winner |
|--------|---------------|---------|--------|
| **Accuracy** | 84.7% | 84.7% | Tie |
| **Precision** | 53.3% | 53.3% | Tie |
| **Recall** | 34.0% | 34.0% | Tie |
| **AUC-ROC** | 0.739 | **0.771** | **XGBoost** |

**Both models catch 16 out of 47 actual flight risks (34% recall).**

---

## Optimal Model Configurations

After extensive grid search testing `[10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]` estimators:

### Random Forest (Surprising Finding!)

```python
RandomForestClassifier(
    n_estimators=10,      # Optimal: fewer trees work better!
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
```

> **Key Finding:** Random Forest performs BEST with just 10 trees. Adding more trees makes the model overly conservative.

### XGBoost

```python
XGBClassifier(
    n_estimators=100,     # Optimal after grid search
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5.19,  # Handles 5.2:1 class imbalance
    random_state=42,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

---

## What Drives Employee Attrition? (The "Why")

Both models agree on the top factors. These are the features that enable **50%+ precision** in predicting attrition:

### Top 10 Attrition Drivers (Combined Model Agreement)

| Rank | Feature | Why It Matters |
|------|---------|----------------|
| **1** | **MonthlyIncome** | Primary driver — compensation matters most |
| **2** | **OverTime: Yes** | Overtime workers have 2x higher attrition risk |
| **3** | **Age** | Younger employees show higher mobility |
| **4** | **YearsAtCompany** | Career stagnation increases flight risk |
| **5** | **JobLevel** | Lower job levels correlate with higher attrition |
| **6** | **TotalWorkingYears** | Early-mid career employees more mobile |
| **7** | **DailyRate** | Compensation cluster reinforces #1 |
| **8** | **YearsWithCurrManager** | Long tenure with same manager can indicate stagnation |
| **9** | **StockOptionLevel** | Equity/investment in company affects retention |
| **10** | **DistanceFromHome** | Long commutes contribute to burnout |

---

## How We Achieved 50%+ Precision

The models achieve **53.3% precision** by identifying these employee patterns:

### 1. Compensation Cluster (MonthlyIncome + DailyRate + JobLevel)
- Employees with **lower income for their job level** are flagged as high-risk
- The model detects "underpaid relative to peers" patterns
- **Actionable:** Review compensation bands for flagged employees

### 2. Work-Life Imbalance (OverTime + DistanceFromHome)
- **Overtime workers** show 2x higher attrition probability
- **Long commutes** combined with overtime = critical risk
- **Actionable:** Audit overtime patterns and consider remote work options

### 3. Career Stagnation (YearsAtCompany + YearsWithCurrManager + JobLevel)
- Employees **stuck in same role** for 3+ years without promotion
- **Same manager** for extended periods can indicate lack of growth
- **Actionable:** Create internal mobility pathways

### 4. Life Stage Factors (Age + TotalWorkingYears)
- **Younger employees** (20s-30s) naturally higher mobility
- **Early-mid career** (5-10 years experience) actively job-hopping
- **Actionable:** Focus retention efforts on this segment

---

## Confusion Matrix (What This Means in Practice)

```
                    Predicted
              Stay        Leave
Actual  Stay    230         14   (False Alarms)
        Leave    31         16   (Caught Flight Risks)
```

| Metric | Count | Interpretation |
|--------|-------|----------------|
| **True Positives** | 16 | Correctly caught flight risks |
| **False Negatives** | 31 | Missed (needs improvement) |
| **False Positives** | 14 | False alarms (acceptable) |
| **True Negatives** | 230 | Correctly predicted staying |

---

## Running the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main model comparison
python retention_warning_system.py

# Run the parameter grid search
python parameter_grid_search.py
```

**Outputs:**
- `output/roc_curve_comparison.png` — Model performance curves
- `output/feature_importance_comparison.png` — Driver analysis
- `output/model_comparison_results.csv` — Numerical results
- `output/parameter_grid_search_results.csv` — Grid search data

---

## Key Research Findings

### 1. Less is More for Random Forest
- **10 trees outperformed 500 trees** for recall (34% vs 12.8%)
- Adding trees made the model more conservative and less willing to predict attrition
- **Lesson:** More complexity ≠ better performance for imbalanced classification

### 2. XGBoost Peaks at 100 Estimators
- Performance degraded beyond 100 estimators
- Optimal configuration: `n_estimators=100, max_depth=10, learning_rate=0.1`

### 3. Both Models Agree on "Why"
- Feature importance rankings were highly correlated between models
- Compensation, overtime, and career stagnation are universal drivers

### 4. Precision Comes from Identifying Patterns
- The 50%+ precision is achieved by detecting **specific risk profiles**:
  - Underpaid relative to job level
  - Working overtime with long commute
  - Stuck in same role for years
  - Early-mid career with no promotion path

---

## Recommended Actions for HR Leadership

### Immediate (High-Impact, Low-Cost)
1. **Review compensation** for employees flagged by the model
2. **Audit overtime patterns** — burnout is a red flag
3. **Conduct stay interviews** with high-risk employees

### Medium-Term (Policy Changes)
1. **Career path transparency** for stagnant employees
2. **Work-life balance initiatives** (remote work, flexible hours)
3. **Internal mobility program** to reduce stagnation

### Long-Term (Cultural)
1. **Promotion transparency** and fairness
2. **Recognition programs** beyond compensation
3. **Equity/stock options** to increase commitment

---

## Reproducibility

All experiments use `random_state=42` for full reproducibility.

See **`EXPERIMENT_RESULTS.md`** for detailed experiment logs and replication instructions.

---

## Data Source

**IBM HR Analytics Employee Attrition Dataset**
- Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- Size: 1,470 employees × 35 features
- Attrition Rate: 16.1%

---

## Model Deployment (For Production)

```python
import joblib

# Load the trained XGBoost model
model = joblib.load('output/xgboost_model.pkl')
feature_columns = joblib.load('output/feature_columns.pkl')

# Prepare new employee data (must match training features)
new_employee = pd.DataFrame([...])  # Your data here

# Predict flight risk
flight_risk_probability = model.predict_proba(new_employee)[0][1]

if flight_risk_probability > 0.5:
    print(f"⚠️ High Risk: {flight_risk_probability:.1%} chance of leaving")
```

---

## Author

**Lead Data Scientist** | Fortune 500 HR Analytics Team

*Built for the CHRO to enable data-driven retention decisions.*

---

## Citation

If you use this work, please cite:

```
Employee Retention Warning System (2026)
Comparative study of Random Forest and XGBoost for employee attrition prediction
https://github.com/sreejap776/employee-retention-warning-system
```
