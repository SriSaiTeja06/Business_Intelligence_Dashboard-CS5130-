# Business Intelligence Dashboard

A professional, interactive Business Intelligence dashboard built with Gradio that enables non-technical stakeholders to explore and analyze business data through an intuitive web interface.

## 🎯 Features

### Data Upload & Validation
- Support for CSV and Excel files (.csv, .xlsx, .xls)
- Automatic data type detection
- Dataset preview with first and last rows
- Comprehensive dataset information display

### Data Exploration & Statistics
- Automated data profiling for numerical and categorical columns
- Descriptive statistics (mean, median, std, min, max, quartiles)
- Categorical analysis (unique values, mode, frequency)
- Missing value reports
- Correlation matrix visualization

### Interactive Filtering
- Dynamic filtering based on column types
- Numerical filters with min/max range selection
- Categorical filters with multi-select dropdowns
- Real-time row count updates
- Filtered data preview

### Visualizations
Five powerful visualization types:
1. **Time Series Plot** - Track trends over time
2. **Distribution Plot** - Histogram and box plot for numerical data
3. **Bar Chart** - Category analysis with aggregation (sum, mean, count, median)
4. **Scatter Plot** - Explore relationships between variables
5. **Correlation Heatmap** - Visualize correlations between numerical features

### Automated Insights
- Top/Bottom performer identification
- Trend and pattern detection
- Data quality analysis
- Outlier detection
- Categorical insights

### Export Functionality
- Export filtered data as CSV
- Download visualizations as images

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
```bash
cd business-intelligence-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the dashboard**
Open your web browser and navigate to the local URL displayed in the terminal (typically http://127.0.0.1:7860)

## 📁 Project Structure

```
project/
│
├── app.py                  # Main Gradio application
├── data_processor.py       # Data loading, cleaning, filtering
├── visualizations.py       # Chart creation functions
├── insights.py            # Automated insight generation
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
└── data/                 # datasets
    ├── Online Retail.xlsx
    └── Auto Sales data.csv
```

## 💡 Usage Guide

### 1. Upload Your Data
- Navigate to the "Data Upload" tab
- Click to upload a CSV or Excel file
- View dataset information and preview

### 2. Explore Statistics
- Go to the "Statistics" tab
- Click "Generate Statistics" to see:
  - Numerical column statistics
  - Categorical column summaries
  - Missing value reports

### 3. Filter Your Data
- Open the "Filter & Explore" tab
- Select columns and set filter criteria
- Apply filters to see updated data
- View row counts in real-time

### 4. Create Visualizations
- Navigate to the "Visualizations" tab
- Choose a chart type
- Select appropriate columns
- Choose aggregation method (for bar charts)
- Generate your visualization

### 5. Generate Insights
- Go to the "Insights" tab
- Click "Generate Insights"
- Review automated findings:
  - Top/bottom performers
  - Trends and patterns
  - Data quality observations

### 6. Export Results
- Visit the "Export" tab
- Export filtered data as CSV
- Download visualizations from the Visualizations tab

## 📊 Sample Datasets

The project works with any CSV or Excel file. Recommended datasets:

1. **E-commerce/Retail Data**
   - Columns: Date, Product, Category, Sales, Quantity
   - Use case: Sales analysis, product performance

2. **Sales Operations**
   - Columns: Date, Sales_Rep, Region, Revenue, Deals
   - Use case: Sales performance, regional analysis

3. **Financial Data**
   - Columns: Date, Stock, Price, Volume, Returns
   - Use case: Portfolio analysis, market trends

## 🔧 Technical Details

### Core Technologies
- **pandas**: Data manipulation and analysis
- **Gradio**: Web interface framework
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **numpy**: Numerical computations

### Key Features Implementation

#### Data Processing
- Modular `DataProcessor` class handles all data operations
- Efficient filtering with pandas boolean indexing
- Support for multiple aggregation methods
- Robust error handling for edge cases

#### Visualizations
- Object-oriented `Visualizer` class
- Consistent styling with seaborn themes
- Dynamic chart generation based on data types
- Automatic handling of missing values

#### Insights Generation
- Statistical analysis for trend detection
- Outlier identification using IQR method
- Data quality metrics
- Categorical distribution analysis

## 🎨 Customization

### Adding New Chart Types
Edit `visualizations.py` and add a new method:
```python
def create_new_chart(self, column: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Your chart code here
    return fig
```

### Adding New Insights
Edit `insights.py` and add a new method:
```python
def find_new_insight(self) -> List[str]:
    insights = []
    # Your insight logic here
    return insights
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Module not found error
```bash
# Solution: Install all requirements
pip install -r requirements.txt
```

**Issue**: File upload fails
- Check file format (CSV or Excel)
- Ensure file is not corrupted
- Verify file size is reasonable (<100MB recommended)

**Issue**: Visualization not displaying
- Ensure you've selected appropriate columns
- Check that columns contain valid numerical data
- Verify data is not empty after filtering

## 📝 Best Practices

1. **Data Preparation**
   - Clean your data before uploading
   - Use consistent date formats
   - Remove unnecessary columns

2. **Performance**
   - Limit dataset size to <1M rows for best performance
   - Use filtering to focus on relevant data
   - Export filtered datasets for faster analysis

3. **Visualization**
   - Choose appropriate chart types for your data
   - Use aggregation to simplify complex datasets
   - Export charts for reports and presentations

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional chart types
- More sophisticated insights
- Database connectivity
- Real-time data updates
- Advanced filtering options

## 📄 License

This project is created for educational purposes as part of a data science course.

## 👥 Authors

Created as a Business Intelligence Dashboard project demonstrating:
- Data analysis with pandas
- Interactive web applications with Gradio
- Data visualization best practices
- Clean, modular code design

## 🙏 Acknowledgments

- Gradio for the excellent web framework
- pandas for powerful data manipulation
- matplotlib and seaborn for visualization capabilities

---

**Note**: This dashboard is designed for educational and business analysis purposes. Always validate insights and decisions with domain expertise.