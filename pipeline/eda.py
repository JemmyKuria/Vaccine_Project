import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid")

class Visualizer:
    def __init__(self, data):
        """
        Initialize with a preprocessed DataFrame
        Args:
            data: Cleaned Pandas DataFrame
        """
        self.data = data

    def plot_vaccination_distribution(self, column='h1n1_vaccine'):
        """Bar plot for vaccine uptake"""
        if column not in self.data.columns:
            raise ValueError(f"'{column}' not found in data.")
        
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.data, x=column, palette='Set2')
        plt.title("Vaccination Distribution")
        plt.xlabel("Vaccinated (1 = Yes, 0 = No)")
        plt.ylabel("Count")
        plt.tight_layout()
        return plt

    def plot_correlation_heatmap(self):
        """Heatmap for numerical correlations"""
        num_data = self.data.select_dtypes(include=['int64', 'float64'])
        corr = num_data.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        return plt

    def plot_feature_vs_target(self, feature: str, target: str = 'h1n1_vaccine'):
        """Plot the distribution of a feature grouped by vaccination status"""
        if feature not in self.data.columns:
            raise ValueError(f"'{feature}' not found in data.")
        if target not in self.data.columns:
            raise ValueError(f"'{target}' not found in data.")

        plt.figure(figsize=(8, 5))
        if self.data[feature].dtype == 'object':
            sns.countplot(data=self.data, x=feature, hue=target, palette='Set1')
            plt.xticks(rotation=45)
        else:
            sns.kdeplot(data=self.data, x=feature, hue=target, common_norm=False, fill=True)
        
        plt.title(f"{feature} Distribution by {target}")
        plt.tight_layout()
        return plt

    def plot_risk_score_distribution(self):
        """Distribution plot of calculated risk score (if exists)"""
        if 'risk_score' not in self.data.columns:
            raise ValueError("'risk_score' not found in data. Make sure calculate_risk_scores() was run.")

        plt.figure(figsize=(8, 5))
        sns.histplot(self.data['risk_score'], bins=30, kde=True, color='orange')
        plt.title("Risk Score Distribution")
        plt.xlabel("Risk Score (0–100)")
        plt.tight_layout()
        return plt
