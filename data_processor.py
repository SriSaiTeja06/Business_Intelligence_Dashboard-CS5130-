"""
Data Processing Module
Handles data loading, cleaning, filtering, and transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class DataProcessor:
    """Handles all data processing operations"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataProcessor with a pandas DataFrame
        
        Args:
            data: Input pandas DataFrame
        """
        self.original_data = data.copy()
        self.filtered_data = data.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for the dataset
        
        Returns:
            Dictionary containing numerical stats, categorical stats, and missing values
        """
        stats = {
            'numerical': None,
            'categorical': {},
            'missing': None
        }
        
        # Numerical statistics
        numerical_cols = self.original_data.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) > 0:
            stats['numerical'] = self.original_data[numerical_cols].describe().T
            stats['numerical']['missing'] = self.original_data[numerical_cols].isnull().sum().values
        
        # Categorical statistics
        categorical_cols = self.original_data.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            try:
                value_counts = self.original_data[col].value_counts()
                if len(value_counts) > 0:
                    stats['categorical'][col] = {
                        'unique_count': self.original_data[col].nunique(),
                        'mode': str(value_counts.index[0]),
                        'mode_count': int(value_counts.iloc[0])
                    }
            except:
                # Skip columns that cause issues
                continue
        
        # Missing values
        stats['missing'] = self.original_data.isnull().sum()
        
        return stats
    
    def apply_filters(self,
                     numerical_column: Optional[str] = None,
                     num_min: Optional[float] = None,
                     num_max: Optional[float] = None,
                     categorical_column: Optional[str] = None,
                     categorical_values: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply filters to the dataset
        
        Args:
            numerical_column: Column name for numerical filtering
            num_min: Minimum value for numerical filter
            num_max: Maximum value for numerical filter
            categorical_column: Column name for categorical filtering
            categorical_values: List of values to filter by
        
        Returns:
            Filtered DataFrame
        """
        self.filtered_data = self.original_data.copy()
        
        # Apply numerical filter
        if numerical_column and num_min is not None and num_max is not None:
            self.filtered_data = self.filtered_data[
                (self.filtered_data[numerical_column] >= num_min) &
                (self.filtered_data[numerical_column] <= num_max)
            ]
        
        # Apply categorical filter
        if categorical_column and categorical_values:
            self.filtered_data = self.filtered_data[
                self.filtered_data[categorical_column].isin(categorical_values)
            ]
        
        return self.filtered_data
    
    def get_filtered_data(self) -> pd.DataFrame:
        """
        Get the currently filtered dataset
        
        Returns:
            Filtered DataFrame
        """
        return self.filtered_data
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for numerical columns
        
        Returns:
            Correlation matrix as DataFrame
        """
        numerical_cols = self.original_data.select_dtypes(include=['number']).columns
        if len(numerical_cols) > 1:
            return self.original_data[numerical_cols].corr()
        return pd.DataFrame()
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in a numerical column
        
        Args:
            column: Column name to check for outliers
            method: Method to use ('iqr' or 'zscore')
        
        Returns:
            Boolean Series indicating outliers
        """
        if method == 'iqr':
            Q1 = self.original_data[column].quantile(0.25)
            Q3 = self.original_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (self.original_data[column] < lower_bound) | (self.original_data[column] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((self.original_data[column] - self.original_data[column].mean()) / 
                             self.original_data[column].std())
            return z_scores > 3
        
        return pd.Series([False] * len(self.original_data))
    
    def aggregate_data(self, 
                       group_by_col: str,
                       value_col: str,
                       agg_func: str = 'sum') -> pd.DataFrame:
        """
        Aggregate data by a column
        
        Args:
            group_by_col: Column to group by
            value_col: Column to aggregate
            agg_func: Aggregation function ('sum', 'mean', 'count', 'median')
        
        Returns:
            Aggregated DataFrame
        """
        if agg_func == 'sum':
            result = self.filtered_data.groupby(group_by_col)[value_col].sum().reset_index()
        elif agg_func == 'mean':
            result = self.filtered_data.groupby(group_by_col)[value_col].mean().reset_index()
        elif agg_func == 'count':
            result = self.filtered_data.groupby(group_by_col)[value_col].count().reset_index()
        elif agg_func == 'median':
            result = self.filtered_data.groupby(group_by_col)[value_col].median().reset_index()
        else:
            result = self.filtered_data.groupby(group_by_col)[value_col].sum().reset_index()
        
        return result.sort_values(value_col, ascending=False)