# %% [markdown]
# # Student Performance Analysis & Prediction
#
# ## Task 1 — Data Exploration with Pandas

# %%
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("students.csv")

# Print the first 5 rows
print("--- First 5 rows ---")
print(df.head())

# Print the shape and the data type of each column
print("\n--- Shape ---")
print(df.shape)
print("\n--- Data Types ---")
print(df.dtypes)

# Print summary statistics (mean, min, max, std) for all numeric columns
print("\n--- Summary Statistics ---")
print(df.describe())

# Print the count of students who passed and who failed
print("\n--- Pass/Fail Counts ---")
print(df['passed'].value_counts())

# Compute and print the average score per subject separately for passing and failing students
subject_cols = ['math', 'science', 'english', 'history', 'pe']
print("\n--- Average Score per Subject (Pass) ---")
print(df[df['passed'] == 1][subject_cols].mean())
print("\n--- Average Score per Subject (Fail) ---")
print(df[df['passed'] == 0][subject_cols].mean())

# Find and print the student with the highest overall average across all 5 subjects
temp_avg = df[subject_cols].mean(axis=1)
best_student = df.loc[temp_avg.idxmax(), 'name']
print("\n--- Student with Highest Overall Average ---")
print(best_student)

# %% [markdown]
# ## Task 2 — Data Visualization with Matplotlib

# %%
# Add new column avg_score
df['avg_score'] = df[subject_cols].mean(axis=1)

# Plot 1: Bar Chart — Average score per subject across all students
plt.figure(figsize=(8, 5))
avg_scores = df[subject_cols].mean()
plt.bar(avg_scores.index, avg_scores.values, color='skyblue')
plt.title("Average Score per Subject")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig('plot1_bar.png')
# # plt.show() # Used internally but omitted to avoid blocking terminal

# Plot 2: Histogram — Distribution of math scores
plt.figure(figsize=(8, 5))
plt.hist(df['math'], bins=5, edgecolor='black', color='salmon')
mean_math = df['math'].mean()
plt.axvline(mean_math, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mean_math:.2f}')
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig('plot2_hist.png')

# Plot 3: Scatter Plot — study_hours_per_day vs avg_score, coloured by passed
plt.figure(figsize=(8, 5))
pass_data = df[df['passed'] == 1]
fail_data = df[df['passed'] == 0]
plt.scatter(pass_data['study_hours_per_day'], pass_data['avg_score'], color='green', label='Pass', alpha=0.7)
plt.scatter(fail_data['study_hours_per_day'], fail_data['avg_score'], color='red', label='Fail', alpha=0.7)
plt.title("Study Hours vs Average Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Average Score")
plt.legend()
plt.tight_layout()
plt.savefig('plot3_scatter.png')

# Plot 4: Box Plot — distribution of attendance_pct for passing vs failing
pass_attendance = df[df['passed']==1]['attendance_pct'].tolist()
fail_attendance = df[df['passed']==0]['attendance_pct'].tolist()
plt.figure(figsize=(8, 5))
plt.boxplot([pass_attendance, fail_attendance], labels=['Pass', 'Fail'])
plt.title("Attendance Percentage: Pass vs Fail")
plt.ylabel("Attendance Percentage")
plt.tight_layout()
plt.savefig('plot4_box.png')

# Plot 5: Line Plot — math score and science score per student
plt.figure(figsize=(10, 5))
plt.plot(df['name'], df['math'], marker='o', label='Math', linestyle='-')
plt.plot(df['name'], df['science'], marker='s', label='Science', linestyle='--')
plt.title("Math and Science Scores per Student")
plt.xlabel("Student Name")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('plot5_line.png')

# %% [markdown]
# ## Task 3 — Data Visualization with Seaborn

# %%
# Plot 6: Seaborn Bar plot showing average math and science score split by pass/fail
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

sns.barplot(data=df, x='passed', y='math', ax=ax1)
ax1.set_title("Average Math Score by Pass/Fail")

sns.barplot(data=df, x='passed', y='science', ax=ax2)
ax2.set_title("Average Science Score by Pass/Fail")

plt.tight_layout()
plt.savefig('plot6_seaborn_bar.png')

# Plot 7: Seaborn Scatter plot of attendance_pct vs avg_score with regression lines
plt.figure(figsize=(8, 5))
sns.regplot(data=df[df['passed']==1], x='attendance_pct', y='avg_score', label='Pass', scatter_kws={'alpha':0.6})
sns.regplot(data=df[df['passed']==0], x='attendance_pct', y='avg_score', label='Fail', scatter_kws={'alpha':0.6})
plt.title("Attendance Percentage vs Average Score")
plt.legend()
plt.tight_layout()
plt.savefig('plot7_seaborn_scatter.png')

# Comment comparing Seaborn vs Matplotlib
# Seaborn requires much less code to produce aesthetically pleasing plots, especially with built-in statistical routines like regression lines (regplot).
# However, Matplotlib provides finer, lower-level control over individual elements like customized line styles and explicit boxplot groupings.

# %% [markdown]
# ## Task 4 — Machine Learning with scikit-learn

# %%
# Step 1 — Prepare Data
X = df[['math', 'science', 'english', 'history', 'pe', 'attendance_pct', 'study_hours_per_day']]
y = df['passed']

# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Step 2 — Train a Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Print training accuracy
print("\n--- Training Accuracy ---")
train_acc = model.score(X_train_scaled, y_train)
print(f"{train_acc * 100:.2f}%")

# Step 3 — Evaluate the Model
# Predict on test set
predictions = model.predict(X_test_scaled)

# Print test accuracy
print("\n--- Test Accuracy ---")
test_acc = model.score(X_test_scaled, y_test)
print(f"{test_acc * 100:.2f}%")

print("\n--- Test Set Predictions ---")
for idx, pred in zip(X_test.index, predictions):
    actual = y_test.loc[idx]
    name = df.loc[idx, 'name']
    correct = "✅ correct" if pred == actual else "❌ wrong"
    print(f"Student: {name} | Actual: {actual} | Predicted: {pred} | {correct}")

# Step 4 — Feature Importance
coefs = model.coef_[0]
features = X.columns
importance = list(zip(features, coefs))
# Sort by absolute value (largest first)
importance.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n--- Feature Importances ---")
for feat, coef in importance:
    print(f"{feat}: {coef:.4f}")

# Horizontal bar chart for feature coefficients
names = [x[0] for x in importance]
vals = [x[1] for x in importance]
colors = ['green' if x > 0 else 'red' for x in vals]

plt.figure(figsize=(8, 5))
plt.barh(names, vals, color=colors)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  # largest at top
plt.tight_layout()
plt.savefig('plot8_feature_importance.png')

# Step 5 — Predict for a New Student (Bonus)
new_student = [[75, 70, 68, 65, 80, 82, 3.2]]  # match feature columns
new_student_scaled = scaler.transform(new_student)
new_pred = model.predict(new_student_scaled)[0]
new_prob = model.predict_proba(new_student_scaled)[0]

print("\n--- New Student Prediction ---")
print(f"Prediction: {'Pass' if new_pred == 1 else 'Fail'}")
print(f"Probabilities: Fail={new_prob[0]:.2f}, Pass={new_prob[1]:.2f}")
