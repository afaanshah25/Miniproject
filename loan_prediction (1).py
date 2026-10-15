
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix)


df = pd.read_csv("train.csv")
print(df.shape)
print(df.isnull().sum())

for col in ["Gender", "Married", "Dependents", "Self_Employed",
            "Credit_History", "Loan_Amount_Term"]:
    df[col] = df[col].fillna(df[col].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)
df = df.drop(columns=["Loan_ID"])

df_enc = pd.get_dummies(df, drop_first=True)

lX = df_enc.drop(columns=["LoanAmount", "Loan_Status_Y"])
y = df_enc["LoanAmount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

lin = LinearRegression()
lin.fit(X_train, y_train)
pred = lin.predict(X_test)

print("\n--- Linear Regression (LoanAmount) ---")
print("R2 score:", round(r2_score(y_test, pred), 3))
print("RMSE    :", round(np.sqrt(mean_squared_error(y_test, pred)), 3))

coef = pd.Series(lin.coef_, index=X.columns).sort_values()
print("\nFeature coefficients:\n", coef)

plt.scatter(y_test, pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual LoanAmount")
plt.ylabel("Predicted LoanAmount")
plt.title("Linear Regression: Actual vs Predicted")
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

Xc = df_enc.drop(columns=["Loan_Status_Y"])
yc = df_enc["Loan_Status_Y"].astype(int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc)

lin_c = LinearRegression().fit(Xc_tr, yc_tr)
lin_pred = (lin_c.predict(Xc_te) >= 0.5).astype(int)
print("\n--- Linear Regression as classifier ---")
print("Accuracy:", round(accuracy_score(yc_te, lin_pred), 3))

log = LogisticRegression(max_iter=2000).fit(Xc_tr, yc_tr)
log_pred = log.predict(Xc_te)
print("\n--- Logistic Regression ---")
print("Accuracy:", round(accuracy_score(yc_te, log_pred), 3))
print("Confusion matrix:\n", confusion_matrix(yc_te, log_pred))
