"""
Visualization Module
Creates various chart types for data analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional


class Visualizer:
    """Handles creation of various visualization types"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Visualizer with a pandas DataFrame
        
        Args:
            data: Input pandas DataFrame
        """
        self.data = data
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def create_time_series(self, date_column: str, value_column: str):
        """
        Create a time series plot
        
        Args:
            date_column: Column containing date/time values
            value_column: Column containing values to plot
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        try:
            # Try to convert to datetime
            dates = pd.to_datetime(self.data[date_column], errors='coerce')
            values = self.data[value_column]
            
            # Remove NaT values
            valid_mask = ~dates.isna()
            dates = dates[valid_mask]
            values = values[valid_mask]
            
            # Sort by date
            sorted_indices = dates.argsort()
            dates = dates.iloc[sorted_indices]
            values = values.iloc[sorted_indices]
            
            ax.plot(dates, values, linewidth=2, marker='o', markersize=4, alpha=0.7)
            ax.set_xlabel(date_column, fontsize=12)
            ax.set_ylabel(value_column, fontsize=12)
            ax.set_title(f'{value_column} Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating time series:\n{str(e)}', 
                   ha='center', va='center', fontsize=12)
            return fig
    
    def create_distribution(self, column: str):
        """
        Create a distribution plot (histogram with KDE)
        
        Args:
            column: Column to plot distribution for
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        try:
            data = self.data[column].dropna()
            
            # Histogram with KDE
            ax1.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            ax1.set_xlabel(column, fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            ax2.set_ylabel(column, fontsize=12)
            ax2.set_title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text
            stats_text = f'Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}'
            ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error creating distribution:\n{str(e)}',
                    ha='center', va='center', fontsize=12)
            return fig
    
    def create_bar_chart(self, category_column: str, value_column: str, agg_method: str = 'sum'):
        """
        Create a bar chart with aggregation
        
        Args:
            category_column: Column for categories (x-axis)
            value_column: Column for values (y-axis)
            agg_method: Aggregation method ('sum', 'mean', 'count', 'median')
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        try:
            # Aggregate data
            if agg_method == 'sum':
                aggregated = self.data.groupby(category_column)[value_column].sum()
            elif agg_method == 'mean':
                aggregated = self.data.groupby(category_column)[value_column].mean()
            elif agg_method == 'count':
                aggregated = self.data.groupby(category_column)[value_column].count()
            elif agg_method == 'median':
                aggregated = self.data.groupby(category_column)[value_column].median()
            else:
                aggregated = self.data.groupby(category_column)[value_column].sum()
            
            # Sort and limit to top 20 for readability
            aggregated = aggregated.sort_values(ascending=False).head(20)
            
            # Create bar chart
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(aggregated)))
            bars = ax.bar(range(len(aggregated)), aggregated.values, color=colors, edgecolor='black', alpha=0.8)
            
            ax.set_xticks(range(len(aggregated)))
            ax.set_xticklabels(aggregated.index, rotation=45, ha='right')
            ax.set_xlabel(category_column, fontsize=12)
            ax.set_ylabel(f'{agg_method.capitalize()} of {value_column}', fontsize=12)
            ax.set_title(f'{agg_method.capitalize()} of {value_column} by {category_column}',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, aggregated.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.0f}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating bar chart:\n{str(e)}',
                   ha='center', va='center', fontsize=12)
            return fig
    
    def create_scatter_plot(self, x_column: str, y_column: str):
        """
        Create a scatter plot
        
        Args:
            x_column: Column for x-axis
            y_column: Column for y-axis
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            x_data = self.data[x_column].dropna()
            y_data = self.data[y_column].dropna()
            
            # Get common indices
            common_idx = x_data.index.intersection(y_data.index)
            x_data = x_data.loc[common_idx]
            y_data = y_data.loc[common_idx]
            
            # Create scatter plot
            scatter = ax.scatter(x_data, y_data, alpha=0.6, s=50, 
                               c=range(len(x_data)), cmap='viridis', edgecolors='black', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data.sort_values(), p(x_data.sort_values()), 
                   "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            
            # Calculate correlation
            correlation = x_data.corr(y_data)
            
            ax.set_xlabel(x_column, fontsize=12)
            ax.set_ylabel(y_column, fontsize=12)
            ax.set_title(f'{y_column} vs {x_column}\nCorrelation: {correlation:.3f}',
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating scatter plot:\n{str(e)}',
                   ha='center', va='center', fontsize=12)
            return fig
    
    def create_pie_chart(self, category_column: str):
        """
        Create a pie chart for categorical data
        
        Args:
            category_column: Column for categories
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        try:
            # Get value counts and limit to top 10
            value_counts = self.data[category_column].value_counts().head(10)
            
            # Create pie chart
            colors = plt.cm.Set3(range(len(value_counts)))
            wedges, texts, autotexts = ax.pie(
                value_counts.values, 
                labels=value_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 10}
            )
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            ax.set_title(f'Distribution of {category_column} (Top 10)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating pie chart:\n{str(e)}',
                   ha='center', va='center', fontsize=12)
            return fig
    
    def create_correlation_heatmap(self):
        """
        Create a correlation heatmap for numerical columns
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        try:
            # Get numerical columns
            numerical_data = self.data.select_dtypes(include=['number'])
            
            if len(numerical_data.columns) < 2:
                ax.text(0.5, 0.5, 'Need at least 2 numerical columns for correlation analysis',
                       ha='center', va='center', fontsize=12)
                return fig
            
            # Calculate correlation matrix
            corr_matrix = numerical_data.corr()
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(corr_matrix.columns)))
            ax.set_yticks(range(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values as text
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                                 fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
            
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating correlation heatmap:\n{str(e)}',
                   ha='center', va='center', fontsize=12)
            return fig