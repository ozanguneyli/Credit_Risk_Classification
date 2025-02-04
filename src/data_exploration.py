import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import NoReturn

# Set visualization style
sns.set_style("darkgrid")

# Ensure 'images' directory exists
os.makedirs("images", exist_ok=True)

def plot_histograms(df: pd.DataFrame) -> NoReturn:
    """Generates and saves histograms for numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 8))
    
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col], bins=30, kde=True, color="steelblue")
        plt.title(f"Histogram of {col}")
    
    plt.tight_layout()
    plt.savefig("images/histograms.png")
    plt.close()

def plot_boxplots(df: pd.DataFrame) -> NoReturn:
    """Generates and saves boxplots for numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=df[col], color="salmon")
        plt.title(f"Boxplot of {col}")
    
    plt.tight_layout()
    plt.savefig("images/boxplots.png")
    plt.close()

def plot_categorical_counts(df: pd.DataFrame) -> NoReturn:
    """Generates and saves bar plots for categorical features."""
    categorical_cols = df.select_dtypes(include=['object']).columns
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(3, 3, i)
        sns.countplot(x=df[col], hue=df[col], palette="viridis", legend=False)
        plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("images/categorical_counts.png")
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame) -> NoReturn:
    """Generates and saves a heatmap showing correlation between numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.savefig("images/correlation_heatmap.png")
    plt.close()

def plot_risk_distribution(df: pd.DataFrame) -> NoReturn:
    """Generates and saves a bar plot showing the distribution of the target variable 'Risk'."""
    plt.figure(figsize=(6, 6))
    sns.countplot(x=df['Risk'], hue=df['Risk'], palette=["#1f77b4", "#ff7f0e"], legend=False)
    plt.title("Risk Distribution")
    plt.xticks([0, 1], ['Bad (0)', 'Good (1)'])
    plt.savefig("images/risk_distribution.png")
    plt.close()

def plot_risk_vs_features(df: pd.DataFrame) -> NoReturn:
    """Generates and saves boxplots showing the relationship between 'Risk' and numerical features."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df['Risk'], y=df[col], hue=df['Risk'], palette=["#1f77b4", "#ff7f0e"], legend=False)
        plt.title(f"Risk vs {col}")
        plt.savefig(f"images/risk_vs_{col}.png")
        plt.close()

def plot_pairplot(df: pd.DataFrame) -> NoReturn:
    """Generates and saves a pairplot for selected numerical features, considering 'Risk'."""
    sample_df = df.sample(500) if len(df) > 500 else df  # Sample 500 rows if dataset is too large
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns[:5]  # Use first 5 numerical features
    
    plt.figure(figsize=(12, 12))
    sns.pairplot(sample_df, hue="Risk", vars=numerical_cols, palette=["#1f77b4", "#ff7f0e"])
    plt.savefig("images/pairplot.png")
    plt.close()
