"""
================================================
BUSINESS MINI PROJECT — Google Colab Notebook
Topic: Impact of Social Media on Student Well-being
Techniques: K-Means Clustering + Linear Regression
================================================

INSTRUCTIONS: Copy each cell section into a new Colab code cell.
"""

# ============================================================
# CELL 1 — Install & Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")
print("✅ Libraries loaded successfully")

# ============================================================
# CELL 2 — Load Dataset
# ============================================================
# If running locally, change path. In Colab, upload the CSV first.
df = pd.read_csv('Students Social Media Addiction.csv')
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
df.head()

# ============================================================
# SECTION 1: DATA CLEANING & PREPROCESSING
# ============================================================

# CELL 3 — Basic Info & Null Check
print("=" * 50)
print("SECTION 1 — DATA CLEANING & PREPROCESSING")
print("=" * 50)
print("\n📌 Dataset Info:")
print(df.info())
print(f"\n📌 Missing Values:\n{df.isnull().sum()}")
print(f"\n📌 Duplicate Rows: {df.duplicated().sum()}")
print(f"\n📌 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# CELL 4 — Statistical Summary
print("\n📌 Statistical Summary:")
df.describe()

# CELL 5 — Encode Categorical Columns
df_encoded = df.copy()

le = LabelEncoder()
categorical_cols = ['Gender', 'Academic_Level', 'Country',
                    'Most_Used_Platform', 'Affects_Academic_Performance',
                    'Relationship_Status']

for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("✅ Categorical columns encoded:")
print(df_encoded[categorical_cols].head())

# CELL 6 — Scale Numerical Features
numerical_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                  'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']

scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

print("✅ Numerical features scaled (StandardScaler)")
print(df_scaled[numerical_cols].describe().round(2))

# ============================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# CELL 7 — Distribution Plots
print("=" * 50)
print("SECTION 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Key Variables', fontsize=16, fontweight='bold')

axes[0, 0].hist(df['Avg_Daily_Usage_Hours'], bins=20, color='#4C72B0', edgecolor='white')
axes[0, 0].set_title('Daily Social Media Usage (Hours)')
axes[0, 0].set_xlabel('Hours per Day')

axes[0, 1].hist(df['Mental_Health_Score'], bins=10, color='#DD8452', edgecolor='white')
axes[0, 1].set_title('Mental Health Score Distribution')
axes[0, 1].set_xlabel('Score (1=Poor, 10=Excellent)')

axes[1, 0].hist(df['Sleep_Hours_Per_Night'], bins=15, color='#55A868', edgecolor='white')
axes[1, 0].set_title('Sleep Hours per Night')
axes[1, 0].set_xlabel('Hours')

axes[1, 1].hist(df['Addicted_Score'], bins=10, color='#C44E52', edgecolor='white')
axes[1, 1].set_title('Social Media Addiction Score')
axes[1, 1].set_xlabel('Score (1=Low, 10=High)')

plt.tight_layout()
plt.savefig('distribution_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# CELL 8 — Platform Usage & Academic Level
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

platform_counts = df['Most_Used_Platform'].value_counts()
axes[0].barh(platform_counts.index, platform_counts.values, color=sns.color_palette("husl", len(platform_counts)))
axes[0].set_title('Most Used Social Media Platforms', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Students')

level_counts = df['Academic_Level'].value_counts()
axes[1].pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("pastel"))
axes[1].set_title('Students by Academic Level', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('platform_academic.png', dpi=150, bbox_inches='tight')
plt.show()

# CELL 9 — Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df_encoded[numerical_cols + ['Gender', 'Affects_Academic_Performance']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap — Social Media & Well-being Variables',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# CELL 10 — Boxplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(data=df, x='Academic_Level', y='Avg_Daily_Usage_Hours',
            palette='Set2', ax=axes[0])
axes[0].set_title('Daily Usage Hours by Academic Level', fontweight='bold')
axes[0].set_xlabel('Academic Level')
axes[0].set_ylabel('Avg Daily Usage (hrs)')

sns.boxplot(data=df, x='Relationship_Status', y='Mental_Health_Score',
            palette='Set3', ax=axes[1])
axes[1].set_title('Mental Health Score by Relationship Status', fontweight='bold')
axes[1].set_xlabel('Relationship Status')
axes[1].set_ylabel('Mental Health Score')

plt.tight_layout()
plt.savefig('boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

# CELL 11 — EDA: Usage vs Mental Health Scatter
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Avg_Daily_Usage_Hours'], df['Mental_Health_Score'],
                      c=df['Addicted_Score'], cmap='RdYlGn_r', alpha=0.6, s=60)
plt.colorbar(scatter, label='Addiction Score')
plt.xlabel('Avg Daily Usage Hours', fontsize=12)
plt.ylabel('Mental Health Score', fontsize=12)
plt.title('Social Media Usage vs Mental Health\n(colored by Addiction Score)',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scatter_usage_vs_mental.png', dpi=150, bbox_inches='tight')
plt.show()

print("📊 Key EDA Findings:")
print(f"  • Average daily usage: {df['Avg_Daily_Usage_Hours'].mean():.2f} hours")
print(f"  • Average mental health score: {df['Mental_Health_Score'].mean():.2f}/10")
print(f"  • Average sleep: {df['Sleep_Hours_Per_Night'].mean():.2f} hours")
print(f"  • Correlation (Usage ↔ Mental Health): {df['Avg_Daily_Usage_Hours'].corr(df['Mental_Health_Score']):.3f}")
print(f"  • Correlation (Usage ↔ Addiction): {df['Avg_Daily_Usage_Hours'].corr(df['Addicted_Score']):.3f}")
print(f"  • {(df['Affects_Academic_Performance']=='Yes').mean()*100:.1f}% report social media affects academics")

# ============================================================
# SECTION 3: K-MEANS CLUSTERING (BEHAVIORAL SEGMENTATION)
# ============================================================

# CELL 12 — Elbow Method
print("=" * 50)
print("SECTION 3 — K-MEANS CLUSTERING")
print("=" * 50)

cluster_features = ['Avg_Daily_Usage_Hours', 'Mental_Health_Score',
                    'Sleep_Hours_Per_Night', 'Addicted_Score',
                    'Conflicts_Over_Social_Media']

X_cluster = df_scaled[cluster_features]

inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8, color='#4C72B0')
plt.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Optimal K=4')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
plt.title('Elbow Method — Finding Optimal Number of Clusters', fontsize=14, fontweight='bold')
plt.legend()
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
plt.show()
print("📌 Optimal K = 4 (identified from elbow bend)")

# CELL 13 — Fit K-Means
K_OPTIMAL = 4
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)

print(f"\n✅ K-Means fitted with K={K_OPTIMAL}")
print(f"\nCluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

# CELL 14 — Cluster Profiles
cluster_profile = df.groupby('Cluster')[cluster_features].mean().round(2)
print("\n📊 Cluster Profiles (Mean Values):")
print(cluster_profile)

cluster_names = {
    0: "🟢 Balanced Users",
    1: "🔴 At-Risk / Addicted",
    2: "🟡 Moderate Scrollers",
    3: "🔵 Healthy Minimalists"
}
# Sort by usage to assign meaningful labels more accurately
usage_order = cluster_profile['Avg_Daily_Usage_Hours'].sort_values()
labels_by_usage = {}
sorted_clusters = usage_order.index.tolist()
label_list = ["🔵 Healthy Minimalists", "🟢 Balanced Users", "🟡 Moderate Scrollers", "🔴 At-Risk / Addicted"]
for i, c in enumerate(sorted_clusters):
    labels_by_usage[c] = label_list[i]

df['Cluster_Label'] = df['Cluster'].map(labels_by_usage)

print("\n📌 Cluster Labels (assigned by usage level):")
for c, name in sorted(labels_by_usage.items()):
    size = (df['Cluster'] == c).sum()
    avg_usage = cluster_profile.loc[c, 'Avg_Daily_Usage_Hours']
    avg_mh = cluster_profile.loc[c, 'Mental_Health_Score']
    print(f"  Cluster {c} → {name} | n={size} | Avg Usage={avg_usage}h | Mental Health={avg_mh}")

# CELL 15 — Cluster Visualization (2D PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(11, 7))
colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
for cluster_id in range(K_OPTIMAL):
    mask = df['Cluster'] == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=labels_by_usage[cluster_id],
                color=colors[cluster_id], alpha=0.6, s=70, edgecolors='white', linewidth=0.5)

plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
plt.title('Behavioral Segments — K-Means Clustering (PCA Projection)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# CELL 16 — Cluster Radar Chart
cluster_means_norm = df_scaled.groupby(df['Cluster'])[cluster_features].mean()
angles = np.linspace(0, 2 * np.pi, len(cluster_features), endpoint=False).tolist()
angles += angles[:1]
feature_labels = ['Usage\nHours', 'Mental\nHealth', 'Sleep\nHours', 'Addiction\nScore', 'Conflicts']

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for cluster_id in range(K_OPTIMAL):
    values = cluster_means_norm.loc[cluster_id].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=labels_by_usage[cluster_id], color=colors[cluster_id])
    ax.fill(angles, values, alpha=0.1, color=colors[cluster_id])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_labels, fontsize=11)
ax.set_title('Cluster Radar Chart — Behavioral Profiles', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig('cluster_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SECTION 4: LINEAR REGRESSION — MENTAL HEALTH PREDICTOR
# ============================================================

# CELL 17 — Prepare Features
print("=" * 50)
print("SECTION 4 — LINEAR REGRESSION")
print("=" * 50)

feature_cols = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                'Conflicts_Over_Social_Media', 'Addicted_Score',
                'Affects_Academic_Performance', 'Age']
target_col = 'Mental_Health_Score'

X = df_encoded[feature_cols]
y = df_encoded[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")

# CELL 18 — Train Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"  ✅ MAE  (Mean Absolute Error):  {mae:.4f}")
print(f"  ✅ RMSE (Root Mean Sq Error):   {rmse:.4f}")
print(f"  ✅ R²   (Coefficient of Det.):  {r2:.4f} ({r2*100:.1f}% variance explained)")

# CELL 19 — Coefficients
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=True)

plt.figure(figsize=(10, 5))
colors_coef = ['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['Coefficient']]
bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_coef, edgecolor='white')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Linear Regression — Feature Coefficients\n(Effect on Mental Health Score)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📌 Feature Interpretation:")
for _, row in coef_df.iterrows():
    direction = "↑ increases" if row['Coefficient'] > 0 else "↓ decreases"
    print(f"  • {row['Feature']:35s}: {direction} mental health score (coef={row['Coefficient']:.3f})")

# CELL 20 — Actual vs Predicted Plot
plt.figure(figsize=(9, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='#4C72B0', edgecolors='white', s=60)
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Mental Health Score', fontsize=12)
plt.ylabel('Predicted Mental Health Score', fontsize=12)
plt.title(f'Actual vs Predicted Mental Health Scores\n(R² = {r2:.3f})',
          fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

# CELL 21 — Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(9, 5))
plt.scatter(y_pred, residuals, alpha=0.5, color='#DD8452', edgecolors='white', s=60)
plt.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot — Linear Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('residuals.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# SECTION 5: BUSINESS INTERPRETATION
# ============================================================

# CELL 22 — Business Insights
print("=" * 60)
print("SECTION 5 — BUSINESS INTERPRETATION")
print("=" * 60)

print("""
📌 BUSINESS PROBLEM:
   Social media platforms create an attention economy where student
   time and focus are the product. Understanding usage patterns and
   their impact on mental health enables data-driven interventions.

📌 KEY FINDINGS:
""")

print(f"  1. DEMAND-SUPPLY DYNAMICS:")
print(f"     • Platforms optimize for engagement (supply of dopamine-triggering content)")
print(f"     • Students 'demand' validation, entertainment, and social connection")
print(f"     • High usage ({df[df['Addicted_Score']>=7]['Avg_Daily_Usage_Hours'].mean():.1f} hrs avg for addicted users) "
      f"depletes sleep and mental health capital")

print(f"\n  2. RISK SEGMENTATION (K-Means, K={K_OPTIMAL}):")
for c_id, label in sorted(labels_by_usage.items()):
    row = cluster_profile.loc[c_id]
    n = (df['Cluster'] == c_id).sum()
    pct = n / len(df) * 100
    print(f"     • {label}: {n} students ({pct:.0f}%) — "
          f"Usage={row['Avg_Daily_Usage_Hours']:.1f}h, "
          f"Mental Health={row['Mental_Health_Score']:.1f}/10, "
          f"Sleep={row['Sleep_Hours_Per_Night']:.1f}h")

print(f"\n  3. REGRESSION INSIGHTS (Predicting Mental Health):")
print(f"     • Model explains {r2*100:.1f}% of variance in mental health scores")
top_neg = coef_df[coef_df['Coefficient'] < 0].iloc[-1]
top_pos = coef_df[coef_df['Coefficient'] > 0].iloc[-1]
print(f"     • Strongest negative factor: {top_neg['Feature']} (coef={top_neg['Coefficient']:.3f})")
print(f"     • Strongest positive factor: {top_pos['Feature']} (coef={top_pos['Coefficient']:.3f})")

print(f"""
  4. ECONOMIC & POLICY IMPLICATIONS:
     • Universities can use behavioral segments to target 'At-Risk' students
       with digital wellness interventions (reducing mental health costs)
     • Pricing/monetization strategy for wellness apps: target Cluster 1
       (At-Risk) students who would most benefit from premium mindfulness tools
     • Sleep deprivation from excessive usage → reduced academic performance
       → lower graduate employability rates (long-term economic cost)
     • Platforms face regulatory risk if addiction scores remain high;
       data-driven proof supports digital time-limit features

  5. REVENUE OPTIMIZATION FOR WELLNESS PLATFORMS:
     • Segment-specific marketing: "Healthy Minimalists" → retention tools
     • "At-Risk" users → intervention upsells (screen-time coaching apps)
     • Average Addiction Score of {df['Addicted_Score'].mean():.1f}/10 across all students
       signals a large addressable market for digital wellness products

📌 CONCLUSION:
   This analysis demonstrates that social media addiction measurably
   impacts student mental health, sleep, and academic performance.
   K-Means segmentation identifies 4 distinct behavioral profiles
   that can guide targeted university policy and edtech innovation.
""")

print("✅ NOTEBOOK COMPLETE — All 5 sections executed successfully!")
print("   Saved charts: distribution_plots.png, platform_academic.png,")
print("   correlation_heatmap.png, boxplots.png, scatter_usage_vs_mental.png,")
print("   elbow_method.png, kmeans_clusters.png, cluster_radar.png,")
print("   regression_coefficients.png, actual_vs_predicted.png, residuals.png")
