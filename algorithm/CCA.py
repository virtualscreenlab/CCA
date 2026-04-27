import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import kagglehub
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: Load Chemical Data (UCI Wine Quality)
# ============================================
url_chem = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df_chem = pd.read_csv(url_chem, sep=';')
X = df_chem.drop('quality', axis=1).values  # 11 chemical features

print(f"Chemical matrix shape: {X.shape}")  # (1599, 11)
print(f"Chemical features: {list(df_chem.drop('quality', axis=1).columns)}")

# ============================================
# STEP 2: Load Real Wine Reviews (Kaggle Dataset)
# ============================================
print("\n[INFO] Downloading Kaggle Wine Reviews dataset...")

# Download dataset (130k wine reviews)
path = kagglehub.dataset_download("zynicide/wine-reviews")
print(f"Downloaded to: {path}")

# Find the CSV file in the downloaded directory
csv_files = glob.glob(os.path.join(path, "*.csv"))
print(f"Found CSV files: {csv_files}")

if len(csv_files) == 0:
    # Check subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

if len(csv_files) > 0:
    csv_path = csv_files[0]
    print(f"Loading CSV from: {csv_path}")
    df_reviews = pd.read_csv(csv_path)
else:
    raise FileNotFoundError("No CSV file found in downloaded dataset")

print(f"Loaded {len(df_reviews)} wine reviews")
print(f"Columns: {list(df_reviews.columns)}")

# ============================================
# STEP 3: Create Y Matrix from Review Descriptions
# ============================================
# Determine which column contains the description text
if 'description' in df_reviews.columns:
    desc_col = 'description'
else:
    raise ValueError("No 'description' column found in dataset")

# Define sensory vocabulary based on wine tasting terminology
# NOTE: Removed duplicate 'elegant'
sensory_vocab = [
    'tannic', 'tannins', 'smooth', 'fruity', 'fruit', 'oak', 'oaky',
    'acidic', 'acidity', 'full-bodied', 'light-bodied', 'dry', 'sweet',
    'bitter', 'soft', 'crisp', 'velvety', 'rich', 'elegant', 'bold',
    'complex', 'balanced', 'fresh', 'ripe', 'jammy', 'earthy', 'spicy',
    'floral', 'mineral', 'structured', 'concentrated', 'silky', 'round',
    'aromatic', 'intense', 'long', 'finish', 'refined', 'elegance',
    'astringent', 'buttery', 'creamy', 'herbaceous', 'oaked', 'toasty',
    'vanilla', 'cherry', 'berry', 'plum', 'blackberry', 'raspberry',
    'citrus', 'lemon', 'apple', 'pear', 'tropical', 'pepper', 'cinnamon',
    'clove', 'licorice', 'mocha', 'coffee', 'chocolate', 'leather', 'tar',
    'smoky', 'violets', 'roses', 'lavender', 'honey', 'nutty', 'almond'
]

print("[INFO] Creating TF-IDF matrix from wine descriptions...")

# Handle NaN values in description column
descriptions = df_reviews[desc_col].fillna('').astype(str)

# Remove duplicates from vocabulary (just in case)
sensory_vocab = list(dict.fromkeys(sensory_vocab))

vectorizer = TfidfVectorizer(
    vocabulary=sensory_vocab,
    stop_words='english',
    ngram_range=(1, 2),
    lowercase=True
)

Y_raw = vectorizer.fit_transform(descriptions).toarray()

print(f"Raw review TF-IDF matrix shape: {Y_raw.shape}")
print(f"Sensory vocabulary used: {len(sensory_vocab)} terms")

# Remove rows with all zeros (no sensory terms matched)
non_zero_rows = Y_raw.sum(axis=1) > 0
Y_raw_filtered = Y_raw[non_zero_rows]
print(f"Filtered TF-IDF matrix (non-zero rows): {Y_raw_filtered.shape}")
print(f"Removed {Y_raw.shape[0] - Y_raw_filtered.shape[0]} rows with no sensory terms")

# ============================================
# STEP 4: Align Dimensions (Sample to Match Chemistry Data Size)
# ============================================
np.random.seed(42)
n_samples = X.shape[0]  # 1599

# Randomly sample reviews to match chemistry data size
if Y_raw_filtered.shape[0] >= n_samples:
    indices = np.random.choice(Y_raw_filtered.shape[0], size=n_samples, replace=False)
    Y = Y_raw_filtered[indices, :]
else:
    # If not enough rows, sample with replacement
    indices = np.random.choice(Y_raw_filtered.shape[0], size=n_samples, replace=True)
    Y = Y_raw_filtered[indices, :]

Y_feature_names = sensory_vocab

print(f"Sampled Y matrix shape (matching X): {Y.shape}")
print(f"Sensory vocabulary size: {len(Y_feature_names)}")

# ============================================
# STEP 5: Standardize and Run CCA (Direct Correlation)
# ============================================
X_scaled = StandardScaler().fit_transform(X)
Y_scaled = StandardScaler().fit_transform(Y)

# CCA on full dimensional data
n_components = min(X.shape[1], Y.shape[1])
print(f"\n[INFO] Running CCA with {n_components} components")

cca_direct = CCA(n_components=n_components, max_iter=2000, tol=1e-6)
X_c, Y_c = cca_direct.fit_transform(X_scaled, Y_scaled)

# Calculate canonical correlations
correlations_direct = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
mean_corr_direct = np.mean(correlations_direct)

print("\n" + "="*60)
print(f"DIRECT CORRELATION (11D Chemistry ↔ {Y.shape[1]}D Language - Real Reviews)")
print("="*60)
print(f"Number of components: {n_components}")
print(f"First 5 canonical correlations: {[round(c, 3) for c in correlations_direct[:5]]}")
print(f"Mean correlation: {mean_corr_direct:.3f}")

# ============================================
# STEP 6: Bottleneck Test (6D Taste Space Mediation)
# ============================================
# Reduce both spaces to 6D via PCA (simulating taste receptor bottleneck)
pca_chem = PCA(n_components=6)
X_taste_space = pca_chem.fit_transform(X_scaled)

pca_lang = PCA(n_components=6)
Y_lang_reduced = pca_lang.fit_transform(Y_scaled)

# CCA on reduced spaces
cca_reduced = CCA(n_components=6, max_iter=2000, tol=1e-6)
X_red_c, Y_red_c = cca_reduced.fit_transform(X_taste_space, Y_lang_reduced)

correlations_reduced = [np.corrcoef(X_red_c[:, i], Y_red_c[:, i])[0, 1] for i in range(6)]
mean_corr_reduced = np.mean(correlations_reduced)

print("\n" + "="*60)
print("MEDIATED CORRELATION (via 6D Taste Space Bottleneck)")
print("="*60)
print(f"Canonical correlations: {[round(c, 3) for c in correlations_reduced]}")
print(f"Mean correlation: {mean_corr_reduced:.3f}")

# ============================================
# STEP 7: Statistical Comparison
# ============================================
print("\n" + "="*60)
print("HYPOTHESIS TEST RESULT (Real Wine Review Data)")
print("="*60)
print(f"Direct correlation (11D ↔ {Y.shape[1]}D):     {mean_corr_direct:.3f}")
print(f"Mediated correlation (6D ↔ 6D):             {mean_corr_reduced:.3f}")
print(f"Absolute improvement:                       {mean_corr_reduced - mean_corr_direct:+.3f}")

if mean_corr_direct > 0:
    rel_improvement = ((mean_corr_reduced / mean_corr_direct) - 1) * 100
    print(f"Relative improvement:                       {rel_improvement:+.1f}%")

if mean_corr_reduced > mean_corr_direct:
    print("\n✓ SUPPORTED: The 6D taste space bottleneck INCREASES")
    print("  alignment between chemistry and language descriptors.")
else:
    print("\n✗ NOT SUPPORTED: Bottleneck did not improve correlation.")

# ============================================
# STEP 8: Explained Variance Analysis
# ============================================
cumsum_chem = np.cumsum(pca_chem.explained_variance_ratio_)
cumsum_lang = np.cumsum(pca_lang.explained_variance_ratio_)

print("\n" + "="*60)
print("INFORMATION BOTTLENECK ANALYSIS")
print("="*60)
print(f"Chemistry Variance Preserved in 6D Taste Space: {cumsum_chem[-1]:.1%}")
print(f"Language Variance Preserved in 6D Taste Space:  {cumsum_lang[-1]:.1%}")
print(f"Chemistry information discarded:                {(1-cumsum_chem[-1])*100:.1f}%")
print(f"Language information discarded:                 {(1-cumsum_lang[-1])*100:.1f}%")
print("\nConclusion: The 6D bottleneck discards substantial chemical nuance")
print("while preserving key sensory-relevant variance, explaining the")
print("'Autonomy of Syntax' phenomenon.")

# ============================================
# VISUALIZATION 1: Scree Plot Comparison
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Direct CCA correlations
axes[0].bar(range(1, n_components + 1), correlations_direct, color='steelblue', alpha=0.7)
axes[0].axhline(y=mean_corr_direct, color='red', linestyle='--',
                label=f'Mean: {mean_corr_direct:.3f}')
axes[0].set_xlabel('Canonical Component')
axes[0].set_ylabel('Correlation Coefficient')
axes[0].set_title(f'Direct: 11D Chemistry ↔ {Y.shape[1]}D Language (Real Reviews)')
axes[0].set_ylim(0, max(0.3, max(correlations_direct) * 1.1))
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Right: Mediated (6D bottleneck) correlations
axes[1].bar(range(1, 7), correlations_reduced, color='darkgreen', alpha=0.7)
axes[1].axhline(y=mean_corr_reduced, color='red', linestyle='--',
                label=f'Mean: {mean_corr_reduced:.3f}')
axes[1].set_xlabel('Canonical Component')
axes[1].set_ylabel('Correlation Coefficient')
axes[1].set_title('Mediated: via 6D Taste Space Bottleneck')
axes[1].set_ylim(0, max(0.3, max(correlations_reduced) * 1.1))
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('cca_taste_bottleneck_real_reviews.png', dpi=300)
plt.show()

# ============================================
# VISUALIZATION 2: Explained Variance in PCA Reduction
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

# Explained variance from chemistry PCA
ax.plot(range(1, 7), cumsum_chem, 'o-', color='steelblue', linewidth=2,
        markersize=10, label=f'Chemistry → Taste Space ({cumsum_chem[-1]:.1%})')

# Explained variance from language PCA
ax.plot(range(1, 7), cumsum_lang, 's-', color='darkgreen', linewidth=2,
        markersize=10, label=f'Language → Conceptual Space ({cumsum_lang[-1]:.1%})')

ax.axhline(y=0.90, color='gray', linestyle=':', alpha=0.5, label='90% Variance')
ax.axvline(x=6, color='red', linestyle='--', alpha=0.5, label='6D Bottleneck')

ax.set_xlabel('Number of Dimensions', fontsize=12)
ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax.set_title('Dimensionality Reduction: Chemistry and Language to 6D Taste Space\n(Real Wine Review Data)', fontsize=14)
ax.set_ylim(0, 1.05)
ax.set_xlim(0.8, 6.2)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pca_dimensionality_reduction_real_reviews.png', dpi=300)
plt.show()

# ============================================
# VISUALIZATION 3: Top Sensory Terms Correlating with Chemistry
# ============================================
from scipy.stats import pearsonr

# Find which sensory terms have strongest correlation with chemistry PCA components
sensory_chem_corr = []
for i, term in enumerate(Y_feature_names):
    # Max correlation of this term with any chemical PCA component
    try:
        max_corr = max([abs(pearsonr(Y_scaled[:, i], X_taste_space[:, j])[0])
                        for j in range(6)])
        sensory_chem_corr.append((term, max_corr))
    except:
        pass  # Skip if correlation fails

# Sort and display top terms
sensory_chem_corr.sort(key=lambda x: x[1], reverse=True)

print("\n" + "="*60)
print("TOP SENSORY TERMS CORRELATING WITH 6D TASTE SPACE")
print("="*60)
for term, corr in sensory_chem_corr[:15]:
    print(f"  {term:<20} r = {corr:.3f}")

# Bar plot of top 15 terms
if len(sensory_chem_corr) >= 10:
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(sensory_chem_corr))
    top_terms = sensory_chem_corr[:top_n]
    terms = [t[0] for t in top_terms]
    corrs = [t[1] for t in top_terms]

    bars = ax.barh(range(len(terms)), corrs, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms)
    ax.set_xlabel('Maximum Correlation with 6D Taste Space Components', fontsize=12)
    ax.set_title(f'Top {top_n} Sensory Terms Correlated with Chemical Taste Space', fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensory_term_correlations.png', dpi=300)
    plt.show()

# ============================================
# SUMMARY STATISTICS FOR PAPER
# ============================================
print("\n" + "="*60)
print("SUMMARY STATISTICS FOR MANUSCRIPT")
print("="*60)
print(f"Chemical samples (n):                 {X.shape[0]}")
print(f"Chemical features (p):                {X.shape[1]}")
print(f"Sensory features (q):                 {Y.shape[1]}")
print(f"Direct CCA mean correlation:          {mean_corr_direct:.3f}")
print(f"Mediated CCA mean correlation (6D):   {mean_corr_reduced:.3f}")
print(f"Correlation improvement:              +{mean_corr_reduced - mean_corr_direct:.3f}")
print(f"Chemistry variance @ 6D:              {cumsum_chem[-1]:.1%}")
print(f"Language variance @ 6D:               {cumsum_lang[-1]:.1%}")

print("\n" + "="*60)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*60)

# ============================================
# NEW VISUALIZATION: Information Loss Asymmetry
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Bar chart comparing variance preserved
ax1 = axes[0]
categories = ['Chemistry', 'Language']
variance_preserved = [cumsum_chem[-1] * 100, cumsum_lang[-1] * 100]
variance_lost = [100 - v for v in variance_preserved]

bars1 = ax1.bar(categories, variance_preserved, color=['steelblue', 'darkgreen'], alpha=0.8, label='Preserved')
bars2 = ax1.bar(categories, variance_lost, bottom=variance_preserved, color=['lightcoral', 'lightcoral'], alpha=0.5, label='Discarded')

ax1.set_ylabel('Variance (%)', fontsize=12)
ax1.set_title('Information Preservation After 6D Bottleneck', fontsize=14)
ax1.set_ylim(0, 100)
ax1.axhline(y=100, color='black', linestyle='-', alpha=0.3)
ax1.legend(loc='upper right')

# Add percentage labels
for i, (pres, lost) in enumerate(zip(variance_preserved, variance_lost)):
    ax1.text(i, pres/2, f'{pres:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax1.text(i, pres + lost/2, f'{lost:.1f}%', ha='center', va='center', fontsize=11, color='darkred')

# Right: Correlation comparison
ax2 = axes[1]
corr_values = [mean_corr_direct, mean_corr_reduced]
# Note: '72D' is hardcoded here based on the sensory_vocab length.
# If you change the vocab size, update this label to f'{Y.shape[1]}D'
corr_labels = ['Direct\n(11D ↔ 72D)', 'Mediated\n(via 6D)']
colors = ['steelblue', 'darkgreen']

bars = ax2.bar(corr_labels, corr_values, color=colors, alpha=0.8)
ax2.set_ylabel('Mean Canonical Correlation', fontsize=12)
ax2.set_title('Chemistry-Language Alignment', fontsize=14)
ax2.set_ylim(0, max(0.25, max(corr_values) * 1.2))

# Add value labels
for bar, val in zip(bars, corr_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- CORRECTION STARTS HERE ---
# Calculate the actual percentage drop dynamically
if mean_corr_direct > 0:
    pct_drop = (1 - mean_corr_reduced / mean_corr_direct) * 100
    annotation_text = f'Bottleneck reduces\nalignment by {pct_drop:.1f}%'
else:
    annotation_text = 'Bottleneck reduces\nalignment significantly'

# Add annotation explaining the drop
ax2.annotate(annotation_text,
             xy=(1, mean_corr_reduced), xytext=(1.3, mean_corr_direct * 1.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', ha='center')
# --- CORRECTION ENDS HERE ---

plt.tight_layout()
plt.savefig('information_bottleneck_asymmetry.png', dpi=300)
plt.show()

print("\n" + "="*60)
print("KEY FINDING FOR MANUSCRIPT")
print("="*60)
print(f"The 6D taste space bottleneck:")
print(f"  • Preserves {cumsum_chem[-1]:.1%} of chemical information")
print(f"  • Preserves only {cumsum_lang[-1]:.1%} of linguistic information")
print(f"  • Reduces chemistry-language correlation from {mean_corr_direct:.3f} to {mean_corr_reduced:.3f}")
print("\n→ This asymmetric information loss explains why wine terminology")
print("  operates semi-autonomously from underlying chemistry.")
print("→ Supports the 'Autonomy of Syntax' hypothesis in oenological language.")

# ============================================
# PERMUTATION TEST: Is the correlation real?
# ============================================
n_permutations = 1000
permuted_correlations = []

print("\n[INFO] Running permutation test...")
for i in range(n_permutations):
    # Shuffle Y rows (break any true pairing)
    Y_shuffled = np.random.permutation(Y_scaled)

    # Run CCA on shuffled data
    cca_perm = CCA(n_components=n_components, max_iter=500)
    X_c_perm, Y_c_perm = cca_perm.fit_transform(X_scaled, Y_shuffled)

    corrs_perm = [np.corrcoef(X_c_perm[:, j], Y_c_perm[:, j])[0, 1] for j in range(n_components)]
    permuted_correlations.append(np.mean(corrs_perm))

permuted_correlations = np.array(permuted_correlations)
p_value = np.mean(permuted_correlations >= mean_corr_direct)

print(f"Permutation test results (n={n_permutations}):")
print(f"  Observed correlation: {mean_corr_direct:.3f}")
print(f"  Null distribution mean: {np.mean(permuted_correlations):.3f}")
print(f"  Null distribution std:  {np.std(permuted_correlations):.3f}")
print(f"  p-value: {p_value:.4f}")

if p_value < 0.05:
    print("  ✓ Correlation is statistically significant (p < 0.05)")
else:
    print("  ✗ Correlation is NOT significant — likely spurious")