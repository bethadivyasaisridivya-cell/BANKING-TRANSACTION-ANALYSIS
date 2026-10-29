# ============================================================
# PROJECT 13: BANKING TRANSACTION ANALYSIS SYSTEM
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. IMPORT DATASET
# ============================================================

df = pd.read_csv("sample_dataset.csv")

print("========== ORIGINAL DATASET ==========")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

print("\n========== DATA PREPROCESSING ==========")

# Check duplicate records
duplicate_count = df.duplicated().sum()

print("Number of duplicate records:", duplicate_count)


# Remove duplicate records
df = df.drop_duplicates()

print("Dataset shape after removing duplicates:", df.shape)


# Check missing values
print("\nMissing Values:")

print(df.isnull().sum())


# Convert Date column into datetime format
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# Convert Transaction Amount into numeric format
df["Transaction Amount"] = pd.to_numeric(
    df["Transaction Amount"],
    errors="coerce"
)


# Remove rows with missing important values
df = df.dropna(
    subset=[
        "Customer ID",
        "Transaction Amount",
        "Date",
        "Category"
    ]
)


print("\nDataset after preprocessing:")

print(df.head())


# ============================================================
# 3. TRANSACTION CATEGORIZATION
# ============================================================

print("\n========== TRANSACTION CATEGORIES ==========")

print(df["Category"].value_counts())


# ============================================================
# 4. BASIC TRANSACTION ANALYSIS
# ============================================================

print("\n========== BASIC ANALYSIS ==========")

total_transactions = len(df)

total_amount = df["Transaction Amount"].sum()

average_amount = df["Transaction Amount"].mean()

maximum_amount = df["Transaction Amount"].max()

minimum_amount = df["Transaction Amount"].min()


print(
    "Total Number of Transactions:",
    total_transactions
)

print(
    "Total Transaction Amount:",
    round(total_amount, 2)
)

print(
    "Average Transaction Amount:",
    round(average_amount, 2)
)

print(
    "Maximum Transaction Amount:",
    round(maximum_amount, 2)
)

print(
    "Minimum Transaction Amount:",
    round(minimum_amount, 2)
)


# ============================================================
# 5. MONTHLY EXPENDITURE ANALYSIS
# ============================================================

print("\n========== MONTHLY EXPENDITURE ==========")


# Create Month column
df["Month"] = df["Date"].dt.to_period("M")


# Calculate monthly expenditure
monthly_expenditure = df.groupby("Month")[
    "Transaction Amount"
].sum()


print(monthly_expenditure)


# ============================================================
# 6. CUSTOMER-WISE TRANSACTION SUMMARY
# ============================================================

print("\n========== CUSTOMER-WISE SUMMARY ==========")


customer_summary = df.groupby("Customer ID").agg(
    Total_Transactions=("Transaction Amount", "count"),
    Total_Amount=("Transaction Amount", "sum"),
    Average_Amount=("Transaction Amount", "mean")
)


print(customer_summary.head(10))


# ============================================================
# 7. CATEGORY-WISE EXPENSE ANALYSIS
# ============================================================

print("\n========== CATEGORY-WISE EXPENSES ==========")


category_expenses = df.groupby("Category")[
    "Transaction Amount"
].sum().sort_values(ascending=False)


print(category_expenses)


# ============================================================
# 8. LINE CHART - MONTHLY TRANSACTION TREND
# ============================================================

plt.figure(figsize=(6, 3.5))


plt.plot(
    monthly_expenditure.index.astype(str),
    monthly_expenditure.values,
    marker="o"
)


plt.title("Monthly Transaction Trend")

plt.xlabel("Month")

plt.ylabel("Total Transaction Amount")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# 9. BAR GRAPH - CATEGORY-WISE EXPENSES
# ============================================================

plt.figure(figsize=(6, 3.5))


plt.bar(
    category_expenses.index,
    category_expenses.values
)


plt.title("Category-wise Expenses")

plt.xlabel("Category")

plt.ylabel("Total Transaction Amount")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# 10. PIE CHART - CATEGORY-WISE EXPENSES
# ============================================================

plt.figure(figsize=(5, 5))


plt.pie(
    category_expenses.values,
    labels=category_expenses.index,
    autopct="%1.1f%%",
    startangle=90
)


plt.title("Category-wise Expense Distribution")

plt.tight_layout()

plt.show()


# ============================================================
# 11. HISTOGRAM - TRANSACTION AMOUNT
# ============================================================

plt.figure(figsize=(6, 3.5))


plt.hist(
    df["Transaction Amount"],
    bins=20
)


plt.title("Distribution of Transaction Amount")

plt.xlabel("Transaction Amount")

plt.ylabel("Frequency")

plt.tight_layout()

plt.show()


# ============================================================
# 12. BASIC FRAUD DETECTION
# ============================================================

print("\n========== BASIC FRAUD DETECTION ==========")


# Calculate 95th percentile
fraud_threshold = df[
    "Transaction Amount"
].quantile(0.95)


print(
    "Suspicious Transaction Threshold:",
    round(fraud_threshold, 2)
)


# Identify unusually high transactions
suspicious_transactions = df[
    df["Transaction Amount"] > fraud_threshold
]


print("\nPotentially Suspicious Transactions:")


print(
    suspicious_transactions[
        [
            "Customer ID",
            "Name",
            "Transaction Amount",
            "Date",
            "Merchant Name",
            "Category"
        ]
    ].head(20)
)


print(
    "\nNumber of Potentially Suspicious Transactions:",
    len(suspicious_transactions)
)


# ============================================================
# 13. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "cleaned_transactions.csv",
    index=False
)


print("\nCleaned dataset saved as:")

print("cleaned_transactions.csv")


# ============================================================
# 14. SAVE CUSTOMER SUMMARY
# ============================================================

customer_summary.to_csv(
    "customer_transaction_summary.csv"
)


print("Customer summary saved as:")

print("customer_transaction_summary.csv")


# ============================================================
# 15. SAVE CATEGORY SUMMARY
# ============================================================

category_expenses.to_csv(
    "category_expense_summary.csv"
)


print("Category summary saved as:")

print("category_expense_summary.csv")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n========================================")

print("BANKING TRANSACTION ANALYSIS COMPLETED")

print("========================================")
