"""
Business Intelligence Dashboard - Main Application
A professional Gradio-based dashboard for interactive data analysis
"""

import gradio as gr
import pandas as pd
import io
import tempfile
import os
from data_processor import DataProcessor
from visualizations import Visualizer
from insights import InsightGenerator
import warnings
warnings.filterwarnings('ignore')

# Global state
current_data = None
processor = None
filtered_data = None
last_figure = None  # Store last generated figure


def upload_and_preview(file):
    """Handle file upload and display preview information"""
    global current_data, processor
    
    print(f"DEBUG: File upload triggered with: {file}")
    
    if file is None:
        return "No file uploaded", "", ""
    
    try:
        # Load data based on file extension
        file_path = file.name if hasattr(file, 'name') else file
        
        if file_path.endswith('.csv'):
            current_data = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            current_data = pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            # Try tab-separated first, then comma-separated
            try:
                current_data = pd.read_csv(file_path, sep='\t')
                # If only one column, try comma separator
                if len(current_data.columns) == 1:
                    current_data = pd.read_csv(file_path, sep=',')
            except:
                current_data = pd.read_csv(file_path, sep=',')
        elif file_path.endswith('.tsv'):
            current_data = pd.read_csv(file_path, sep='\t')
        else:
            return "Error: Unsupported file format. Please upload CSV, Excel, TXT, or TSV files.", "", ""
        
        processor = DataProcessor(current_data)
        
        print(f"DEBUG: Data loaded successfully - {current_data.shape[0]} rows, {current_data.shape[1]} columns")
        
        # Generate preview information
        info = f"""
## Dataset Information

**Shape:** {current_data.shape[0]:,} rows × {current_data.shape[1]} columns

**Columns:** {', '.join(current_data.columns.tolist())}

**Memory Usage:** {current_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        """
        
        # Format data types nicely
        dtypes_df = pd.DataFrame({
            'Column': current_data.dtypes.index,
            'Data Type': current_data.dtypes.values.astype(str)
        })
        info += "\n\n**Data Types:**\n" + dtypes_df.to_html(index=False, border=0)
        
        # First 10 rows with better styling
        preview_head = current_data.head(10).to_html(
            index=False, 
            border=0,
            classes='dataframe',
            max_cols=10,
            max_rows=10
        )
        
        # Last 5 rows
        preview_tail = current_data.tail(5).to_html(
            index=False, 
            border=0,
            classes='dataframe',
            max_cols=10,
            max_rows=5
        )
        
        # Add CSS styling
        style = """
        <style>
        .dataframe { 
            border-collapse: collapse; 
            width: 100%; 
            font-size: 12px;
            overflow-x: auto;
            display: block;
        }
        .dataframe th { 
            background-color: #4CAF50; 
            color: white; 
            padding: 8px; 
            text-align: left;
            position: sticky;
            top: 0;
        }
        .dataframe td { 
            padding: 8px; 
            border-bottom: 1px solid #ddd;
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .dataframe tr:hover {
            background-color: #e8f5e9;
        }
        .dataframe tr:hover td {
            color: #000;
        }
        </style>
        """
        
        preview_head = style + preview_head
        preview_tail = style + preview_tail
        
        return info, preview_head, preview_tail
        
    except Exception as e:
        return f"Error loading file: {str(e)}", "", ""


def generate_statistics():
    """Generate comprehensive statistics for the dataset"""
    global current_data, processor
    
    if current_data is None:
        return "Please upload a dataset first", "", ""
    
    try:
        stats = processor.get_statistics()
        
        # Numerical statistics
        numerical_stats = "## Numerical Columns Statistics\n\n"
        if stats['numerical'] is not None and not stats['numerical'].empty:
            numerical_stats += stats['numerical'].to_html(border=0, classes='dataframe')
        else:
            numerical_stats += "No numerical columns found."
        
        # Categorical statistics
        categorical_stats = "## Categorical Columns Summary\n\n"
        if stats['categorical']:
            cat_summary = []
            for col, info in stats['categorical'].items():
                cat_summary.append({
                    'Column': col,
                    'Unique Values': info['unique_count'],
                    'Most Frequent': str(info['mode'])[:50],  # Truncate long values
                    'Frequency': info['mode_count']
                })
            categorical_stats += pd.DataFrame(cat_summary).to_html(index=False, border=0)
        else:
            categorical_stats += "No categorical columns found."
        
        # Missing values report
        missing_report = "## Missing Values Report\n\n"
        if stats['missing'].sum() > 0:
            missing_df = pd.DataFrame({
                'Column': stats['missing'].index,
                'Missing Count': stats['missing'].values,
                'Percentage': (stats['missing'].values / len(current_data) * 100).round(2)
            })
            missing_report += missing_df[missing_df['Missing Count'] > 0].to_html(index=False, border=0)
        else:
            missing_report += "No missing values detected! ✓"
        
        return numerical_stats, categorical_stats, missing_report
        
    except Exception as e:
        return f"Error generating statistics: {str(e)}", "", ""


def create_correlation_heatmap():
    """Create correlation matrix heatmap"""
    global current_data
    
    if current_data is None:
        return None
    
    try:
        visualizer = Visualizer(current_data)
        return visualizer.create_correlation_heatmap()
    except Exception as e:
        print(f"Correlation error: {str(e)}")
        return None


def update_numerical_range(column):
    """Update numerical filter range based on selected column"""
    global current_data
    
    if current_data is None or not column:
        return gr.update(), gr.update()
    
    try:
        col_data = current_data[column].dropna()
        min_val = float(col_data.min())
        max_val = float(col_data.max())
        
        return (
            gr.update(minimum=min_val, maximum=max_val, value=min_val, visible=True),
            gr.update(minimum=min_val, maximum=max_val, value=max_val, visible=True)
        )
    except:
        return gr.update(), gr.update()


def update_categorical_options(column):
    """Update categorical filter options based on selected column"""
    global current_data
    
    if current_data is None or not column:
        return gr.update(choices=[], visible=False)
    
    try:
        unique_values = current_data[column].dropna().unique().tolist()
        # Limit to top 100 for performance
        unique_values = unique_values[:100]
        return gr.update(choices=unique_values, value=None, visible=True)
    except:
        return gr.update(choices=[], visible=False)


def update_date_range(column):
    """Update date filter range based on selected column"""
    global current_data
    
    if current_data is None or not column:
        return gr.update(visible=False), gr.update(visible=False)
    
    try:
        # Try to convert to datetime
        date_col = pd.to_datetime(current_data[column], errors='coerce')
        date_col = date_col.dropna()
        
        if len(date_col) == 0:
            return gr.update(visible=False), gr.update(visible=False)
        
        min_date = date_col.min().strftime('%Y-%m-%d')
        max_date = date_col.max().strftime('%Y-%m-%d')
        
        return (
            gr.update(value=min_date, visible=True, placeholder=min_date),
            gr.update(value=max_date, visible=True, placeholder=max_date)
        )
    except:
        return gr.update(visible=False), gr.update(visible=False)


def apply_filters(num_column, num_min, num_max, cat_column, cat_values, date_column, date_start, date_end):
    """Apply filters to the dataset and return filtered data preview"""
    global current_data, processor, filtered_data
    
    if current_data is None:
        return "Please upload a dataset first", "0 / 0"
    
    try:
        # Apply numerical and categorical filters using processor
        filtered_data = processor.apply_filters(
            numerical_column=num_column if num_column else None,
            num_min=num_min,
            num_max=num_max,
            categorical_column=cat_column if cat_column else None,
            categorical_values=cat_values if cat_values else None
        )
        
        # Apply date filter manually
        if date_column and date_start and date_end:
            try:
                # Convert column to datetime
                date_col = pd.to_datetime(filtered_data[date_column], errors='coerce')
                start_dt = pd.to_datetime(date_start)
                end_dt = pd.to_datetime(date_end)
                
                # Filter by date range
                date_mask = (date_col >= start_dt) & (date_col <= end_dt)
                filtered_data = filtered_data[date_mask]
            except Exception as e:
                print(f"Date filter error: {str(e)}")
        
        row_count = len(filtered_data)
        
        # Create preview with styling
        preview = filtered_data.head(20).to_html(
            index=False, 
            border=0,
            classes='dataframe',
            max_cols=10
        )
        
        style = """
        <style>
        .dataframe { 
            border-collapse: collapse; 
            width: 100%; 
            font-size: 12px;
        }
        .dataframe th { 
            background-color: #2196F3; 
            color: white; 
            padding: 8px; 
        }
        .dataframe td { 
            padding: 6px; 
            border-bottom: 1px solid #ddd;
        }
        .dataframe tr:hover {
            background-color: #e3f2fd;
        }
        .dataframe tr:hover td {
            color: #000;
        }
        </style>
        """
        
        preview = style + preview
        info = f"**Filtered rows:** {row_count:,} / {len(current_data):,}"
        
        return preview, info
        
    except Exception as e:
        return f"Error applying filters: {str(e)}", "0 / 0"


def update_column_options(chart_type):
    """Update column options based on chart type"""
    global current_data
    
    if current_data is None:
        return gr.update(choices=[]), gr.update(choices=[]), gr.update(visible=False)
    
    num_cols = current_data.select_dtypes(include=['number']).columns.tolist()
    all_cols = current_data.columns.tolist()
    cat_cols = current_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if chart_type == "Time Series":
        return (
            gr.update(choices=all_cols, value=None, label="Date/Time Column", visible=True),
            gr.update(choices=num_cols, value=None, label="Value Column", visible=True),
            gr.update(visible=False)
        )
    elif chart_type == "Distribution":
        return (
            gr.update(choices=num_cols, value=None, label="Column", visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    elif chart_type == "Bar Chart":
        return (
            gr.update(choices=all_cols, value=None, label="Category Column", visible=True),
            gr.update(choices=num_cols, value=None, label="Value Column", visible=True),
            gr.update(visible=True)
        )
    elif chart_type == "Scatter Plot":
        return (
            gr.update(choices=num_cols, value=None, label="X Axis", visible=True),
            gr.update(choices=num_cols, value=None, label="Y Axis", visible=True),
            gr.update(visible=False)
        )
    elif chart_type == "Pie Chart":
        return (
            gr.update(choices=cat_cols if cat_cols else all_cols, value=None, label="Category Column", visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    
    return gr.update(), gr.update(), gr.update()


def create_visualization(chart_type, x_column, y_column, agg_method):
    """Create visualization based on user selections"""
    global current_data, filtered_data, last_figure
    
    # Use filtered data if available, otherwise use full data
    data_to_use = filtered_data if filtered_data is not None else current_data
    
    if data_to_use is None:
        return None
    
    try:
        visualizer = Visualizer(data_to_use)
        
        if chart_type == "Time Series":
            if not x_column or not y_column:
                return None
            last_figure = visualizer.create_time_series(x_column, y_column)
        elif chart_type == "Distribution":
            if not x_column:
                return None
            last_figure = visualizer.create_distribution(x_column)
        elif chart_type == "Bar Chart":
            if not x_column or not y_column:
                return None
            last_figure = visualizer.create_bar_chart(x_column, y_column, agg_method)
        elif chart_type == "Scatter Plot":
            if not x_column or not y_column:
                return None
            last_figure = visualizer.create_scatter_plot(x_column, y_column)
        elif chart_type == "Pie Chart":
            if not x_column:
                return None
            last_figure = visualizer.create_pie_chart(x_column)
        else:
            return None
        
        return last_figure
            
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return None


def generate_insights():
    """Generate automated insights from the data"""
    global current_data
    
    if current_data is None:
        return "Please upload a dataset first"
    
    try:
        insight_gen = InsightGenerator(current_data)
        insights = insight_gen.generate_all_insights()
        
        output = "# Automated Insights\n\n"
        
        # Top/Bottom performers
        if insights['top_bottom']:
            output += "## 🏆 Top & Bottom Performers\n\n"
            for item in insights['top_bottom']:
                output += f"**{item['column']}:**\n"
                output += f"- Maximum: {item['top_value']}\n"
                output += f"- Minimum: {item['bottom_value']}\n"
                output += f"- Average: {item['mean_value']}\n"
                output += f"- Range: {item['range']}\n\n"
        
        # Trends and anomalies
        if insights['trends']:
            output += "## 📈 Trends & Patterns\n\n"
            for trend in insights['trends']:
                output += f"- {trend}\n"
        
        # Data quality insights
        if insights['quality']:
            output += "\n## 🔍 Data Quality Observations\n\n"
            for quality in insights['quality']:
                output += f"- {quality}\n"
        
        return output
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"


def export_filtered_data():
    """Export filtered data as CSV"""
    global filtered_data, current_data
    
    # Use filtered data if available, otherwise use current data
    data_to_export = filtered_data if filtered_data is not None else current_data
    
    if data_to_export is None:
        return None
    
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        data_to_export.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        print(f"Export error: {str(e)}")
        return None


def export_last_visualization():
    """Export the last generated visualization as PNG"""
    global last_figure
    
    if last_figure is None:
        return None
    
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.png')
        last_figure.savefig(temp_file.name, dpi=300, bbox_inches='tight')
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        print(f"Export visualization error: {str(e)}")
        return None


def create_dashboard():
    """Create the main Gradio dashboard"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="BI Dashboard", css="""
        .dataframe { font-size: 11px; }
        .tab-nav button { font-size: 16px; padding: 10px 20px; }
    """) as demo:
        
        gr.Markdown("# 📊 Business Intelligence Dashboard")
        gr.Markdown("Upload your data and explore interactive analytics, visualizations, and insights.")
        
        # Data Upload Tab
        with gr.Tab("📁 Data Upload"):
            with gr.Row():
                file_input = gr.File(
                    label="Upload CSV, Excel, or Text File", 
                    file_types=['.csv', '.xlsx', '.xls', '.txt', '.tsv'],
                    type="filepath"
                )
            
            with gr.Row():
                dataset_info = gr.Markdown("Upload a file to begin")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### First 10 Rows")
                    preview_head = gr.HTML()
                with gr.Column():
                    gr.Markdown("### Last 5 Rows")
                    preview_tail = gr.HTML()
        
        # Statistics Tab
        with gr.Tab("📈 Statistics"):
            stats_button = gr.Button("Generate Statistics", variant="primary", size="lg")
            
            with gr.Row():
                numerical_stats = gr.HTML()
            
            with gr.Row():
                categorical_stats = gr.HTML()
            
            with gr.Row():
                missing_report = gr.HTML()
            
            with gr.Row():
                gr.Markdown("### Correlation Matrix")
                correlation_plot = gr.Plot(label="Correlation Heatmap")
        
        # Filter & Explore Tab
        with gr.Tab("🔍 Filter & Explore"):
            gr.Markdown("### Interactive Filters")
            gr.Markdown("*Apply filters to focus on specific data subsets*")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Numerical Filter**")
                    num_column = gr.Dropdown(label="Select Numerical Column", choices=[], interactive=True)
                    with gr.Row():
                        num_min = gr.Number(label="Minimum Value", visible=False, interactive=True)
                        num_max = gr.Number(label="Maximum Value", visible=False, interactive=True)
                
                with gr.Column():
                    gr.Markdown("**Categorical Filter**")
                    cat_column = gr.Dropdown(label="Select Categorical Column", choices=[], interactive=True)
                    cat_values = gr.Dropdown(
                        label="Select Values", 
                        choices=[], 
                        multiselect=True,
                        visible=False,
                        interactive=True
                    )
                
                with gr.Column():
                    gr.Markdown("**Date Filter**")
                    date_column = gr.Dropdown(label="Select Date Column", choices=[], interactive=True)
                    with gr.Row():
                        date_start = gr.Textbox(label="Start Date (YYYY-MM-DD)", visible=False, interactive=True, placeholder="2010-01-01")
                        date_end = gr.Textbox(label="End Date (YYYY-MM-DD)", visible=False, interactive=True, placeholder="2025-12-31")
            
            with gr.Row():
                apply_button = gr.Button("Apply Filters", variant="primary", size="lg")
                reset_button = gr.Button("Reset All Filters", variant="secondary")
            
            filter_info = gr.Markdown("**Filtered rows:** 0 / 0")
            filtered_preview = gr.HTML()
        
        # Visualizations Tab
        with gr.Tab("📊 Visualizations"):
            gr.Markdown("### Create Interactive Charts")
            gr.Markdown("*Select chart type and columns to visualize your data*")
            
            with gr.Row():
                chart_type = gr.Dropdown(
                    label="Chart Type",
                    choices=["Bar Chart", "Pie Chart", "Distribution", "Scatter Plot", "Time Series"],
                    value="Bar Chart"
                )
            
            with gr.Row():
                x_column = gr.Dropdown(label="X Axis / Column", choices=[], visible=True, interactive=True)
                y_column = gr.Dropdown(label="Y Axis / Value", choices=[], visible=True, interactive=True)
                agg_method = gr.Dropdown(
                    label="Aggregation Method",
                    choices=["sum", "mean", "count", "median"],
                    value="sum",
                    visible=True,
                    interactive=True
                )
            
            viz_button = gr.Button("Generate Visualization", variant="primary", size="lg")
            plot_output = gr.Plot()
        
        # Insights Tab
        with gr.Tab("💡 Insights"):
            gr.Markdown("### Automated Insights")
            gr.Markdown("*AI-powered analysis of your data patterns and anomalies*")
            insights_button = gr.Button("Generate Insights", variant="primary", size="lg")
            insights_output = gr.Markdown()
        
        # Export Tab
        with gr.Tab("💾 Export"):
            gr.Markdown("### Export Your Data and Visualizations")
            gr.Markdown("*Download filtered data or save visualizations*")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Export Data")
                    export_csv_button = gr.Button("📥 Export Filtered Data as CSV", variant="primary", size="lg")
                    csv_output = gr.File(label="Download CSV File")
                
                with gr.Column():
                    gr.Markdown("#### Export Visualization")
                    gr.Markdown("*Save the last generated chart from Visualizations tab*")
                    export_viz_button = gr.Button("🖼️ Export Last Visualization as PNG", variant="primary", size="lg")
                    viz_output = gr.File(label="Download PNG File")
        
        # ==================== EVENT HANDLERS ====================
        # All event handlers defined after all components are created
        
        # File upload handler - updates all dropdowns at once
        def handle_file_upload(file):
            info, head, tail = upload_and_preview(file)
            
            if current_data is None:
                return (
                    info, head, tail,
                    gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]),  # Filter dropdowns
                    gr.update(choices=[]), gr.update(choices=[])   # Viz dropdowns
                )
            
            num_cols = current_data.select_dtypes(include=['number']).columns.tolist()
            cat_cols = current_data.select_dtypes(include=['object', 'category']).columns.tolist()
            all_cols = current_data.columns.tolist()
            
            # Detect potential date columns (datetime or object columns with 'date' in name)
            date_cols = []
            for col in all_cols:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_cols.append(col)
                elif current_data[col].dtype == 'datetime64[ns]':
                    date_cols.append(col)
            
            if not date_cols:
                date_cols = []  # Empty if no date columns
            
            print(f"DEBUG: Updating dropdowns - Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}, Date: {len(date_cols)}")
            
            return (
                info, head, tail,
                gr.update(choices=num_cols, value=None),   # num_column
                gr.update(choices=cat_cols, value=None),   # cat_column
                gr.update(choices=date_cols, value=None),  # date_column
                gr.update(choices=all_cols, value=None),   # x_column
                gr.update(choices=num_cols, value=None)    # y_column
            )
        
        file_input.change(
            handle_file_upload,
            inputs=[file_input],
            outputs=[dataset_info, preview_head, preview_tail, num_column, cat_column, date_column, x_column, y_column]
        )
        
        # Statistics
        def generate_stats_and_correlation():
            num_stats, cat_stats, missing = generate_statistics()
            corr_plot = create_correlation_heatmap()
            return num_stats, cat_stats, missing, corr_plot
        
        stats_button.click(
            generate_stats_and_correlation,
            outputs=[numerical_stats, categorical_stats, missing_report, correlation_plot]
        )
        
        # Filter handlers
        num_column.change(
            update_numerical_range,
            inputs=[num_column],
            outputs=[num_min, num_max]
        )
        
        cat_column.change(
            update_categorical_options,
            inputs=[cat_column],
            outputs=[cat_values]
        )
        
        date_column.change(
            update_date_range,
            inputs=[date_column],
            outputs=[date_start, date_end]
        )
        
        apply_button.click(
            apply_filters,
            inputs=[num_column, num_min, num_max, cat_column, cat_values, date_column, date_start, date_end],
            outputs=[filtered_preview, filter_info]
        )
        
        def reset_filters():
            global filtered_data
            filtered_data = None
            return (
                None, gr.update(visible=False), gr.update(visible=False), 
                None, gr.update(visible=False), 
                None, gr.update(visible=False), gr.update(visible=False),
                "No filters applied", ""
            )
        
        reset_button.click(
            reset_filters,
            outputs=[num_column, num_min, num_max, cat_column, cat_values, date_column, date_start, date_end, filter_info, filtered_preview]
        )
        
        # Visualization handlers
        chart_type.change(
            update_column_options,
            inputs=[chart_type],
            outputs=[x_column, y_column, agg_method]
        )
        
        viz_button.click(
            create_visualization,
            inputs=[chart_type, x_column, y_column, agg_method],
            outputs=[plot_output]
        )
        
        # Insights
        insights_button.click(
            generate_insights,
            outputs=[insights_output]
        )
        
        # Export handlers
        export_csv_button.click(
            export_filtered_data,
            outputs=[csv_output]
        )
        
        export_viz_button.click(
            export_last_visualization,
            outputs=[viz_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(share=False)