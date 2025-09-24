import pandas as pd      # data handling
import numpy as np       # numerical operations
import matplotlib.pyplot as plt  # plotting
import seaborn as sns    # advanced plotting
from scipy import stats  
from statsmodels.stats.proportion import proportions_ztest 
from scipy.stats import ttest_ind 
from itertools import combinations
df=pd.read_csv("final_shop_analysis.csv")

to_numeric = ["Cost_per_conversiom", "RPCon", "ROAS", "Profit_Margin", "Cost % of Revenue"]

for col in to_numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")
#print(df.info())

#print(df.describe())
df["Device"] = df["Device"].str.strip()
print(df["Device"].unique())




def run_ab_tests(data, group_col, groupA, groupB):
    # Convert columns safely to numeric
    for col in ["CTR", "CVR", "ROAS", "Cost_per_conversiom",
                "Profit_Margin", "Sale Amount", "CPC", "P&L"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    metrics = [
        "CTR", "CVR", "ROAS", "Cost_per_conversiom",
        "Profit_Margin", "Sale Amount", "CPC", "P&L"
    ]
    
    results = []
    A = data[data[group_col] == groupA]
    B = data[data[group_col] == groupB]

    for metric in metrics:
        if metric in data.columns:
            A_metric = A[metric].dropna()
            B_metric = B[metric].dropna()

            if len(A_metric) > 1 and len(B_metric) > 1:
                # ✅ Check variance to avoid NaNs
                if A_metric.var() == 0 or B_metric.var() == 0:
                    stat, pval = np.nan, np.nan
                else:
                    stat, pval = ttest_ind(A_metric, B_metric, equal_var=False)
            else:
                stat, pval = np.nan, np.nan

            results.append({
                "Metric": metric,
                "Test": "T-test (Mean)",
                "Group A": groupA,
                "Group B": groupB,
                "Statistic": stat,
                "p-value": pval,
                "Mean A": A_metric.mean() if len(A_metric) > 0 else np.nan,
                "Mean B": B_metric.mean() if len(B_metric) > 0 else np.nan
            })

    return pd.DataFrame(results)


# ✅ Run test
results = run_ab_tests(df, group_col="Device", groupA="Mob", groupB="Desk")
print(results)

import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load your dataset
df = pd.read_csv("final_shop_analysis.csv")

# Ensure numeric columns are correctly typed
for col in ["CVR", "CPC", "Profit_Margin"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Function to run 2-way ANOVA
def run_two_way_anova(data, metric):
    formula = f"{metric} ~ C(Device) + C(Offer_Type) + C(Device):C(Offer_Type)"
    model = ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

# Run for selected metrics
results = {}
for metric in ["CVR", "CPC", "Profit_Margin"]:
    results[metric] = run_two_way_anova(df, metric)

# Display results
for metric, table in results.items():
    print(f"\n📊 Two-way ANOVA for {metric}")
    print(table)

# Ensure numeric columns
metrics = ["ROAS", "Profit_Margin", "CPC"]
for col in metrics:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Clean offer column
df["Offer_Type"] = df["Offer_Type"].str.strip()

# List of offers
offers = ['[shop coupon code]', '[shop coupon]', '[shop discount code]',
          '[shop promo code]', '[shop promo]', 'Competitor', 'Coupon Code',
          'Discount Code', 'Free Shipping', 'Offer', 'Promo Code', 'Sale',
          'Black Friday/Cyber Monday']

# Bonferroni correction for multiple comparisons
def bonferroni_correct(p, n_tests):
    return min(p * n_tests, 1.0)

# Store results
all_results = []

for metric in metrics:
    # 1️⃣ One-way ANOVA
    formula = f"{metric} ~ C(Offer_Type)"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_anova = anova_table["PR(>F)"][0]
    print(f"\nOne-way ANOVA for {metric} by Offer_Type: p = {p_anova:.4f}")
    
    if p_anova < 0.05:
        print(f"Significant differences found in {metric}. Running pairwise T-tests...")
        # 2️⃣ Pairwise T-tests
        offer_pairs = list(combinations(offers, 2))
        for (A_offer, B_offer) in offer_pairs:
            A_data = df[df["Offer_Type"] == A_offer][metric].dropna()
            B_data = df[df["Offer_Type"] == B_offer][metric].dropna()
            if len(A_data) > 1 and len(B_data) > 1:
                stat, pval = ttest_ind(A_data, B_data, equal_var=False)
                # Apply Bonferroni correction
                pval_corr = bonferroni_correct(pval, len(offer_pairs))
                all_results.append({
                    "Metric": metric,
                    "Offer A": A_offer,
                    "Offer B": B_offer,
                    "Mean A": A_data.mean(),
                    "Mean B": B_data.mean(),
                    "t-stat": stat,
                    "p-value": pval,
                    "p-value corrected": pval_corr
                })

# Convert to DataFrame for a clean portfolio table
results_df = pd.DataFrame(all_results)
print("\n📌 Pairwise T-test results (with Bonferroni correction):")
print(results_df)