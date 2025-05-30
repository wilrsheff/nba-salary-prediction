import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the dataset
df = pd.read_csv("NBA_Raw_Data.csv")

# Convert salary to numeric (removing "$" and ",")
df["Salary"] = df["Salary"].replace(r"[\$,]", "", regex=True).astype(float)

# Define independent variables (match what you did in Excel)
X = df[["PPG", "APG", "RPG", "BSPG", "TS%", "PER", "WS/48", "Age", "Seasons",
        "Draft Pos", "All Stars", "Games Played", "Guard", "Forward",
        "Big Market", "Rookie Deal"]]

# Define dependent variable
y = df["Salary"]

# Add a constant (for intercept)
X = sm.add_constant(X)

# Run the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Save regression results to a text file
with open("regression_results.txt", "w") as f:
    f.write(model.summary().as_text())

# ---- RESIDUAL ANALYSIS ----
# Calculate residuals
df["Residuals"] = model.resid

# Plot Residuals vs. Predicted Salaries
plt.figure(figsize=(8, 5))
sns.scatterplot(x=model.fittedvalues, y=model.resid, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Salary")
plt.savefig("residuals_plot.png")
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 5))
sns.histplot(model.resid, bins=20, kde=True)
plt.xlabel("Residuals")
plt.title("Residual Distribution")
plt.savefig("residual_distribution.png")
plt.show()

# ---- MULTICOLLINEARITY CHECK ----
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ensure X is a Pandas DataFrame before accessing .columns
if isinstance(X, np.ndarray):
    X_df = pd.DataFrame(X, columns=[f'Var_{i}' for i in range(X.shape[1])])  # Generic column names
else:
    X_df = X.copy()  # If X is already a DataFrame, just copy it

# Calculate VIF for each independent variable
vif_data = pd.DataFrame()
vif_data["Feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

# Save VIF results
vif_data.to_csv("multicollinearity_vif.csv", index=False)
print("\nMulticollinearity Check (Variance Inflation Factor):")
print(vif_data)
