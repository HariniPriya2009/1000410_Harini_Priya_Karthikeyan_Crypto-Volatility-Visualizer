import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Page Configuration
st.set_page_config(
    page_title="Crypto Volatility Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Theme with WHITE text
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0a0e14;
        border-right: 1px solid #1a1f2e;
    }
    
    /* Metric cards - WHITE text */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 300;
        color: #00d4ff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #ffffff !important;
    }
    
    /* Headers - WHITE text */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 400;
        color: #ffffff !important;
    }
    
    /* Regular text - WHITE */
    p, div, span, label {
        color: #ffffff !important;
    }
    
    /* Captions - WHITE */
    .stCaption {
        color: #ffffff !important;
    }
    
    /* Info boxes - WHITE text */
    .stAlert {
        background-color: #0a0e14;
        border: 1px solid #1a1f2e;
        border-radius: 4px;
        color: #ffffff !important;
    }
    
    .stAlert p {
        color: #ffffff !important;
    }
    
    /* Sliders - WHITE labels */
    .stSlider label {
        color: #ffffff !important;
    }
    
    /* Selectbox - WHITE text */
    .stSelectbox label {
        color: #ffffff !important;
    }
    
    /* Checkbox - WHITE text */
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* File uploader - WHITE text */
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }
    
    /* Success messages - WHITE */
    .stSuccess {
        color: #ffffff !important;
    }
    
    .stWarning {
        color: #ffffff !important;
    }
    
    .stError {
        color: #ffffff !important;
    }
    
    /* Dataframe - WHITE text */
    .dataframe {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# ============ DATA LOADING FUNCTIONS (NO CACHING TO FIX ERROR) ============

def load_crypto_data(file_path):
    """Load and process cryptocurrency dataset"""
    try:
        # Read CSV file
        df = pd.read_csv(r"C:\Users\Harini Priya\OneDrive\Desktop\AI\AI(B)\math AI\Crypto_data.crdownload")
        
        # Convert Unix timestamp to datetime
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
        
        # Reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Remove rows with zero or negative volume (optional cleaning)
        df = df[df['Volume'] > 0]
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def filter_data_by_days(df, days):
    """Filter data to show only the last N days"""
    if df is None or len(df) == 0:
        return df
    
    # Get the latest date in the dataset
    latest_date = df['Date'].max()
    
    # Calculate the cutoff date
    cutoff_date = latest_date - timedelta(days=days)
    
    # Filter data
    filtered_df = df[df['Date'] >= cutoff_date].copy()
    
    return filtered_df


def aggregate_to_daily(df):
    """Aggregate minute-level data to daily level for better visualization"""
    if df is None or len(df) == 0:
        return df
    
    # Set Date as index and resample to daily
    df_daily = df.set_index('Date').resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Reset index
    df_daily = df_daily.reset_index()
    
    return df_daily


def calculate_volatility(df):
    """Calculate volatility index (standard deviation of returns)"""
    if df is None or len(df) == 0:
        return 0
    returns = df['Close'].pct_change().dropna()
    return returns.std() * 100


def calculate_trend(df):
    """Determine market trend"""
    if df is None or len(df) == 0:
        return "No Data"
    first_price = df['Close'].iloc[0]
    last_price = df['Close'].iloc[-1]
    change = ((last_price - first_price) / first_price) * 100
    
    if change > 2:
        return "Upward ↗"
    elif change < -2:
        return "Downward ↘"
    else:
        return "Sideways →"


def identify_stable_volatile_periods(df, threshold_percent=5):
    """Identify stable and volatile periods based on price volatility"""
    if df is None or len(df) == 0:
        return None
    
    # Calculate daily returns
    df_copy = df.copy()
    df_copy['Returns'] = df_copy['Close'].pct_change() * 100
    df_copy['Abs_Returns'] = df_copy['Returns'].abs()
    
    # Calculate rolling volatility (7-day window)
    df_copy['Rolling_Volatility'] = df_copy['Abs_Returns'].rolling(window=7, min_periods=1).mean()
    
    # Classify periods
    median_volatility = df_copy['Rolling_Volatility'].median()
    df_copy['Period_Type'] = df_copy['Rolling_Volatility'].apply(
        lambda x: 'Volatile' if x > median_volatility else 'Stable'
    )
    
    return df_copy


# ============ VISUALIZATION FUNCTIONS ============

def create_price_chart(df, title, color='#00d4ff'):
    """Create main price movement chart (Line Graph of Price Over Time)"""
    if df is None or len(df) == 0:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color=color, width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#ffffff')),
        plot_bgcolor='#0a0e14',
        paper_bgcolor='#0a0e14',
        font=dict(color='#ffffff', size=12),
        xaxis=dict(
            gridcolor='#1a1f2e',
            showgrid=True,
            zeroline=False,
            title=dict(text='Date', font=dict(color='#ffffff'))
        ),
        yaxis=dict(
            gridcolor='#1a1f2e',
            showgrid=True,
            zeroline=False,
            tickprefix='$',
            tickformat=',.0f',
            title=dict(text='Price (USD)', font=dict(color='#ffffff'))
        ),
        hovermode='x unified',
        margin=dict(l=20, r=20, t=50, b=20),
        height=450
    )
    
    return fig


def create_high_low_chart(df):
    """Create high vs low comparison chart"""
    if df is None or len(df) == 0:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['High'],
        mode='lines',
        name='High',
        line=dict(color='#00d4ff', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>High:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Low'],
        mode='lines',
        name='Low',
        line=dict(color='#ff6b9d', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Low:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Daily High vs Low Comparison', font=dict(size=16, color='#ffffff')),
        plot_bgcolor='#0a0e14',
        paper_bgcolor='#0a0e14',
        font=dict(color='#ffffff', size=12),
        xaxis=dict(
            gridcolor='#1a1f2e',
            showgrid=True,
            title=dict(text='Date', font=dict(color='#ffffff'))
        ),
        yaxis=dict(
            gridcolor='#1a1f2e',
            showgrid=True,
            tickprefix='$',
            tickformat=',.0f',
            title=dict(text='Price (USD)', font=dict(color='#ffffff'))
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color='#ffffff')
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    
    return fig


def create_volume_chart(df):
    """Create volume analysis chart"""
    if df is None or len(df) == 0:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name='Volume',
        marker=dict(color='#7c3aed', opacity=0.7),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Volume:</b> %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Trading Volume Analysis', font=dict(size=16, color='#ffffff')),
        plot_bgcolor='#0a0e14',
        paper_bgcolor='#0a0e14',
        font=dict(color='#ffffff', size=12),
        xaxis=dict(
            gridcolor='#1a1f2e',
            showgrid=True,
            title=dict(text='Date', font=dict(color='#ffffff'))
        ),
        yaxis=dict(
            gridcolor='#1a1f2e',
            showgrid=True,
            title=dict(text='Volume', font=dict(color='#ffffff'))
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    
    return fig


def create_stable_volatile_comparison_chart(df):
    """Create stable vs volatile periods chart with shaded regions"""
    if df is None or len(df) == 0:
        return go.Figure()
    
    # Identify periods
    df_with_periods = identify_stable_volatile_periods(df)
    
    if df_with_periods is None:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df_with_periods['Date'],
        y=df_with_periods['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#00d4ff', width=2),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<br><b>Type:</b> %{text}<extra></extra>',
        text=df_with_periods['Period_Type']
    ))
    
    # Add shaded regions for stable periods (green)
    stable_mask = df_with_periods['Period_Type'] == 'Stable'
    if stable_mask.any():
        stable_df = df_with_periods[stable_mask]
        for i in range(0, len(stable_df), 7):
            subset = stable_df.iloc[i:i+7]
            if len(subset) > 0:
                fig.add_vrect(
                    x0=subset['Date'].min(),
                    x1=subset['Date'].max(),
                    fillcolor="rgba(0, 255, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="Stable" if i == 0 else "",
                    annotation_position="top left"
                )
    
    # Add shaded regions for volatile periods (red)
    volatile_mask = df_with_periods['Period_Type'] == 'Volatile'
    if volatile_mask.any():
        volatile_df = df_with_periods[volatile_mask]
        for i in range(0, len(volatile_df), 7):
            subset = volatile_df.iloc[i:i+7]
            if len(subset) > 0:
                fig.add_vrect(
                    x0=subset['Date'].min(),
                    x1=subset['Date'].max(),
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="Volatile" if i == 0 else "",
                    annotation_position="top left"
                )
    
    fig.update_layout(
        title=dict(text='Stable vs Volatile Periods', font=dict(size=16, color='#ffffff')),
        plot_bgcolor='#0a0e14',
        paper_bgcolor='#0a0e14',
        font=dict(color='#ffffff', size=12),
        xaxis=dict(
            gridcolor='#1a1f2e',
            showgrid=True,
            title=dict(text='Date', font=dict(color='#ffffff'))
        ),
        yaxis=dict(
            gridcolor='#1a1f2e',
            showgrid=True,
            tickprefix='$',
            tickformat=',.0f',
            title=dict(text='Price (USD)', font=dict(color='#ffffff'))
        ),
        legend=dict(
            font=dict(color='#ffffff')
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
        showlegend=True
    )
    
    return fig


def create_volatility_comparison_chart(df):
    """Create comparison chart showing stable vs volatile periods side by side"""
    if df is None or len(df) == 0:
        return go.Figure()
    
    # Identify periods
    df_with_periods = identify_stable_volatile_periods(df)
    
    if df_with_periods is None:
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Stable Period (Low Volatility)', 'Volatile Period (High Volatility)'),
        vertical_spacing=0.15
    )
    
    # Separate stable and volatile data
    stable_df = df_with_periods[df_with_periods['Period_Type'] == 'Stable'].copy()
    volatile_df = df_with_periods[df_with_periods['Period_Type'] == 'Volatile'].copy()
    
    # Add stable period trace
    if len(stable_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=stable_df['Date'],
                y=stable_df['Close'],
                mode='lines',
                name='Stable Price',
                line=dict(color='#00ff88', width=2),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add volatile period trace
    if len(volatile_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=volatile_df['Date'],
                y=volatile_df['Close'],
                mode='lines',
                name='Volatile Price',
                line=dict(color='#ff6b9d', width=2),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_xaxes(
        gridcolor='#1a1f2e',
        showgrid=True,
        title=dict(text='Date', font=dict(color='#ffffff'), size=12)
    )
    
    fig.update_yaxes(
        gridcolor='#1a1f2e',
        showgrid=True,
        tickprefix='$',
        tickformat=',.0f',
        title=dict(text='Price (USD)', font=dict(color='#ffffff'), size=12)
    )
    
    fig.update_layout(
        plot_bgcolor='#0a0e14',
        paper_bgcolor='#0a0e14',
        font=dict(color='#ffffff', size=12),
        height=700,
        showlegend=True,
        legend=dict(font=dict(color='#ffffff'))
    )
    
    return fig


# ============ MAIN APP ============

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# Crypto Volatility Visualizer")
        st.markdown("##### Analyzing cryptocurrency price movements from real market data")
    with col2:
        st.markdown("""
        <div style='text-align: right; padding-top: 20px;'>
            <span style='color: #00d4ff; font-size: 12px;'>● Real Data Analysis</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar - Control Panel
    with st.sidebar:
        st.markdown("## Controls")
        st.markdown("##### Customize data view")
        st.markdown("")
        
        # Data Range Selection
        st.markdown("**TIME RANGE**")
        days_range = st.selectbox(
            "Select time range",
            [7, 15, 30, 60, 90],
            index=2,
            label_visibility="collapsed",
            format_func=lambda x: f"{x} Days"
        )
        
        st.markdown("")
        
        # Data Granularity
        st.markdown("**DATA GRANULARITY**")
        granularity = st.selectbox(
            "Select data granularity",
            ["Daily", "Hourly", "Raw (Minute-level)"],
            label_visibility="collapsed"
        )
        
        st.markdown("")
        
        # File Upload (optional - use default dataset)
        st.markdown("**DATA SOURCE**")
        use_default = st.checkbox("Use default dataset", value=True)
        
        if not use_default:
            uploaded_file = st.file_uploader(
                "Upload your CSV dataset",
                type=['csv'],
                help="Upload a CSV file with columns: Timestamp, Open, High, Low, Close, Volume"
            )
        else:
            uploaded_file = None
        
        st.markdown("---")
        
        # Dataset Info
        st.markdown("""
        **Dataset Information**
        
        - Default: Crypto_data.crdownload
        - Records: 637,114 data points
        - Period: 2011-2013
        - Granularity: Minute-level
        """)
    
    # Load Data
    data_file = "Crypto_data.crdownload"
    
    if not use_default and uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_upload.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_file = "temp_upload.csv"
    elif not use_default:
        st.warning("Please upload a dataset or use the default data.")
        st.stop()
    
    # Load the dataset with progress indicator
    with st.spinner("Loading and processing data..."):
        df = load_crypto_data(data_file)
    
    if df is None or len(df) == 0:
        st.error("Failed to load data. Please check the file format.")
        st.stop()
    
    # Show data loading success
    st.success(f"✅ Loaded {len(df):,} records from the dataset")
    
    # Filter data by selected time range
    df_filtered = filter_data_by_days(df, days_range)
    
    if df_filtered is None or len(df_filtered) == 0:
        st.error(f"No data available for the last {days_range} days.")
        st.stop()
    
    # Apply granularity
    if granularity == "Daily":
        df_display = aggregate_to_daily(df_filtered)
    elif granularity == "Hourly":
        df_display = df_filtered.set_index('Date').resample('H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
    else:  # Raw data
        df_display = df_filtered.copy()
    
    # Display data info
    st.markdown("### Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Records",
            f"{len(df_display):,}",
            help=f"Total data points in current view"
        )
    
    with col2:
        st.metric(
            "Date Range",
            f"{df_display['Date'].min().strftime('%b %d')} - {df_display['Date'].max().strftime('%b %d, %Y')}",
            help="Selected time period"
        )
    
    with col3:
        st.metric(
            "Granularity",
            granularity,
            help="Data aggregation level"
        )
    
    with col4:
        st.metric(
            "Data Points",
            f"{days_range} days",
            help="Time range selected"
        )
    
    st.markdown("")
    
    # Calculate Metrics
    avg_price = df_display['Close'].mean()
    volatility = calculate_volatility(df_display)
    trend = calculate_trend(df_display)
    
    # Metrics Row
    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Price",
            value=f"${avg_price:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Volatility Index",
            value=f"{volatility:.2f}%",
            delta=None,
            help="Standard deviation of returns"
        )
    
    with col3:
        st.metric(
            label="Market Trend",
            value=trend,
            delta=None
        )
    
    st.markdown("")
    
    # Section 1: Line Graph of Price Over Time
    st.markdown("### Price Movement Over Time")
    st.markdown("##### Line graph showing how the price moves up and down")
    st.plotly_chart(
        create_price_chart(df_display, f"Price Movement - Last {days_range} Days ({granularity})"),
        use_container_width=True,
        config={'displayModeBar': False}
    )
    
    st.markdown("")
    
    # Section 2: High vs Low Comparison
    st.markdown("### Daily High vs Low Comparison")
    st.markdown("##### Line graph showing both High and Low prices - helps see daily volatility")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_high_low_chart(df_display),
            use_container_width=True,
            config={'displayModeBar': False}
        )
    
    with col2:
        st.plotly_chart(
            create_volume_chart(df_display),
            use_container_width=True,
            config={'displayModeBar': False}
        )
    
    st.markdown("")
    
    # Section 3: Stable vs Volatile Periods (NEW - FA2 Requirement)
    st.markdown("### Stable vs Volatile Periods")
    st.markdown("##### Marked areas showing stable (flat) vs volatile (sharp ups and downs) periods")
    
    # Chart with shaded regions
    st.plotly_chart(
        create_stable_volatile_comparison_chart(df_display),
        use_container_width=True,
        config={'displayModeBar': False}
    )
    
    st.markdown("")
    
    # Side-by-side comparison
    st.markdown("### Stable vs Volatile Periods Comparison")
    st.markdown("##### Side-by-side view of stable and volatile periods")
    st.plotly_chart(
        create_volatility_comparison_chart(df_display),
        use_container_width=True,
        config={'displayModeBar': False}
    )
    
    st.markdown("")
    
    # Data Preview Section
    st.markdown("### Data Preview")
    with st.expander("View recent data (last 10 records)", expanded=False):
        st.dataframe(
            df_display.tail(10).style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    # Footer Information
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### About This Dashboard")
        st.markdown("""
        This interactive dashboard analyzes real cryptocurrency market data from your dataset.
        The data has been filtered to show the selected time range and aggregated according
        to your chosen granularity level.
        
        **Key Features:**
        - Line Graph of Price Over Time
        - High vs Low Comparison
        - Volume Analysis
        - Stable vs Volatile Periods Analysis
        """)
    
    with col2:
        st.markdown("#### Analysis Components")
        st.markdown("""
        - **Price Movement:** Historical close prices over time
        - **High vs Low:** Daily price range analysis
        - **Trading Volume:** Market activity and liquidity
        - **Volatility Index:** Price fluctuation measurement
        - **Stable Periods:** Low volatility, flat price movement
        - **Volatile Periods:** High volatility, sharp price changes
        """)


if __name__ == "__main__":
    main()
