"""
Insights Generation Module
Automatically generates insights from data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class InsightGenerator:
    """Generates automated insights from data"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the InsightGenerator with a pandas DataFrame
        
        Args:
            data: Input pandas DataFrame
        """
        self.data = data
    
    def find_top_bottom_performers(self) -> List[Dict[str, Any]]:
        """
        Identify top and bottom performers for numerical columns
        
        Returns:
            List of dictionaries containing top/bottom performer information
        """
        performers = []
        
        numerical_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        for col in numerical_cols[:5]:  # Limit to first 5 numerical columns
            try:
                clean_data = self.data[col].dropna()
                if len(clean_data) > 0 and clean_data.nunique() > 1:
                    top_value = clean_data.max()
                    bottom_value = clean_data.min()
                    mean_value = clean_data.mean()
                    
                    performers.append({
                        'column': col,
                        'top_value': f"{top_value:,.2f}",
                        'bottom_value': f"{bottom_value:,.2f}",
                        'mean_value': f"{mean_value:,.2f}",
                        'range': f"{top_value - bottom_value:,.2f}"
                    })
            except:
                continue
        
        return performers
    
    def detect_trends(self) -> List[str]:
        """
        Detect basic trends and patterns in the data
        
        Returns:
            List of trend descriptions
        """
        trends = []
        
        # Check for missing values
        missing_cols = self.data.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        
        if len(missing_cols) > 0:
            total_missing = missing_cols.sum()
            pct_missing = (total_missing / (len(self.data) * len(self.data.columns))) * 100
            trends.append(f"Dataset has {total_missing:,} missing values ({pct_missing:.1f}% of total)")
        
        # Check for high cardinality categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_ratio = self.data[col].nunique() / len(self.data)
            if unique_ratio > 0.9:
                trends.append(f"Column '{col}' has very high cardinality ({self.data[col].nunique()} unique values)")
            elif unique_ratio < 0.05:
                trends.append(f"Column '{col}' has low diversity ({self.data[col].nunique()} unique values)")
        
        # Check for numerical outliers
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        for col in numerical_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = self.data[(self.data[col] < Q1 - 1.5 * IQR) | 
                                (self.data[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                pct_outliers = (len(outliers) / len(self.data)) * 100
                trends.append(f"Column '{col}' contains {len(outliers)} outliers ({pct_outliers:.1f}%)")
        
        # Check for skewness in numerical columns
        for col in numerical_cols:
            skewness = self.data[col].skew()
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                trends.append(f"Column '{col}' is highly skewed to the {direction} (skewness: {skewness:.2f})")
        
        return trends[:8]  # Return top 8 trends
    
    def analyze_data_quality(self) -> List[str]:
        """
        Analyze data quality issues
        
        Returns:
            List of data quality observations
        """
        quality_issues = []
        
        # Check for duplicate rows
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            pct_duplicates = (duplicates / len(self.data)) * 100
            quality_issues.append(f"Found {duplicates:,} duplicate rows ({pct_duplicates:.1f}% of data)")
        
        # Check for constant columns
        for col in self.data.columns:
            if self.data[col].nunique() == 1:
                quality_issues.append(f"Column '{col}' has only one unique value (constant)")
        
        # Check for high correlation
        numerical_data = self.data.select_dtypes(include=['number'])
        if len(numerical_data.columns) > 1:
            corr_matrix = numerical_data.corr()
            high_corr = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr.append((corr_matrix.columns[i], 
                                        corr_matrix.columns[j], 
                                        corr_matrix.iloc[i, j]))
            
            if high_corr:
                for col1, col2, corr in high_corr[:3]:  # Show top 3
                    quality_issues.append(f"High correlation between '{col1}' and '{col2}' ({corr:.2f})")
        
        # Data completeness
        completeness = (1 - self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        quality_issues.append(f"Overall data completeness: {completeness:.1f}%")
        
        return quality_issues
    
    def find_categorical_insights(self) -> List[str]:
        """
        Find insights about categorical columns
        
        Returns:
            List of categorical insights
        """
        insights = []
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols[:3]:  # Analyze top 3 categorical columns
            value_counts = self.data[col].value_counts()
            
            if len(value_counts) > 0:
                top_category = value_counts.index[0]
                top_count = value_counts.iloc[0]
                top_percentage = (top_count / len(self.data)) * 100
                
                insights.append(
                    f"In '{col}', '{top_category}' is most frequent "
                    f"({top_count:,} occurrences, {top_percentage:.1f}%)"
                )
        
        return insights
    
    def generate_all_insights(self) -> Dict[str, List]:
        """
        Generate all types of insights
        
        Returns:
            Dictionary containing all insight categories
        """
        return {
            'top_bottom': self.find_top_bottom_performers(),
            'trends': self.detect_trends(),
            'quality': self.analyze_data_quality(),
            'categorical': self.find_categorical_insights()
        }