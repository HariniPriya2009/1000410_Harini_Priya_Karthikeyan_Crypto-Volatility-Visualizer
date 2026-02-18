import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Crypto Volatility Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Theme with proper contrast
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Sidebar styling - DARK background for white text visibility */
    [data-testid="stSidebar"] {
        background-color: #0a0e14 !important;
        border-right: 1px solid #1a1f2e;
    }
    
    /* Sidebar content area */
    [data-testid="stSidebar"] > div {
        background-color: #0a0e14 !important;
    }
    
    /* Metric cards */
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
    
    /* Headers - WHITE text with proper contrast */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 400;
        color: #ffffff !important;
    }
    
    /* Main content text - WHITE */
    .main .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Paragraphs and regular text */
    p, .stMarkdown, div[data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
    
    /* Captions */
    .stCaption {
        color: #e0e0e0 !important;
    }
    
    /* Info boxes - WHITE text on dark background */
    .stAlert {
        background-color: #1a1f2e !important;
        border: 1px solid #2d3748 !important;
        border-radius: 4px;
        color: #ffffff !important;
    }
    
    .stAlert p {
        color: #ffffff !important;
    }
    
    /* Selectbox labels - make them visible */
    .stSelectbox > label {
        color: #00d4ff !important;
        font-weight: 600;
    }
    
    /* Selectbox div */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1f2e !important;
    }
    
    /* Checkbox labels */
    .stCheckbox > label {
        color: #ffffff !important;
    }
    
    /* File uploader labels */
    [data-testid="stFileUploader"] > label {
        color: #ffffff !important;
    }
    
    /* Slider labels */
    .stSlider > label {
        color: #00d4ff !important;
        font-weight: 600;
    }
    
    /* Radio labels */
    .stRadio > label {
        color: #00d4ff !important;
        font-weight: 600;
    }
    
    /* Success messages */
    .stSuccess {
        color: #00ff88 !important;
        font-weight: 600;
    }
    
    .stWarning {
        color: #ffcc00 !important;
        font-weight: 600;
    }
    
    .stError {
        color: #ff4444 !important;
        font-weight: 600;
    }
    
    /* Dataframe */
    .dataframe {
        color: #ffffff !important;
    }
    
    /* Make all selectbox options readable */
    .stSelectbox option {
        background-color: #1a1f2e;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ============ DATA LOADING FUNCTIONS (NO CACHING) ============

def load_crypto_data(file_path, days=30):
    """Load and process cryptocurrency dataset - optimized for last 30 days only"""
    try:
        # Read CSV file with explicit parameters - read in chunks for better performance
        # First, read just the last 100000 rows (approximately 70 days of minute data)
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python', nrows=100000)
        
        # Check if required columns exist
        required_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")
            st.error(f"Available columns: {list(df.columns)}")
            return None
        
        # Convert Unix timestamp to datetime
        try:
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s', errors='coerce')
            # Drop rows with invalid dates
            df = df.dropna(subset=['Date'])
        except Exception as e:
            st.error(f"Error converting timestamp: {e}")
            return None
        
        # Reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Remove rows with zero or negative volume
        df = df[df['Volume'] > 0]
        
        # Sort by date and filter to last 30 days
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Get the latest date
        latest_date = df['Date'].max()
        
        # Calculate the cutoff date (30 days before latest)
        cutoff_date = latest_date - timedelta(days=days)
        
        # Filter to last 30 days only
        df = df[df['Date'] >= cutoff_date].copy()
        
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        st.error("Please ensure the CSV file is in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please check the file format and try again.")
        return None


# ============ SIMULATED DATA GENERATION FUNCTIONS ============

def generate_simulated_data(days=30, pattern='sine', amplitude=50, frequency=0.5, drift=0, volatility_level='stable'):
    """
    Generate simulated cryptocurrency price data based on mathematical models
    
    Parameters:
    - days: Number of days to simulate
    - pattern: 'sine' (sine/cosine waves) or 'random' (random noise)
    - amplitude: Swing size (vertical scale of price movements)
    - frequency: Swing speed (how fast the waves repeat)
    - drift: Long-term slope (positive = upward trend, negative = downward)
    - volatility_level: 'stable' (small swings) or 'volatile' (big swings)
    """
    
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=days*24*60, freq='T')  # Minute data
    
    # Adjust amplitude based on volatility level
    if volatility_level == 'stable':
        amplitude = amplitude * 0.3  # Small swings
        noise_level = 2
    else:  # volatile
        amplitude = amplitude * 1.5  # Big swings
        noise_level = 10
    
    # Generate time series
    t = np.arange(len(dates))
    
    if pattern == 'sine':
        # Sine/cosine waves pattern
        base_price = 1000
        price = base_price + amplitude * np.sin(2 * np.pi * frequency * t / (24*60))
        price += amplitude * 0.5 * np.cos(2 * np.pi * frequency * 0.5 * t / (24*60))
        
        # Add long-term drift
        price += drift * t / (24*60)
        
        # Add some noise
        price += np.random.normal(0, noise_level, len(dates))
        
    elif pattern == 'random':
        # Random walk with drift
        base_price = 1000
        returns = np.random.normal(drift/100, amplitude/500, len(dates))
        price = base_price * (1 + np.cumsum(returns))
        
        # Add additional noise
        price += np.random.normal(0, noise_level, len(dates))
    
    # Ensure price is positive
    price = np.maximum(price, 1)
    
    # Create dataframe
    df = pd.DataFrame({
        'Date': dates,
        'Close': price
    })
    
    # Generate OHLCV data
    df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, amplitude*0.05, len(df))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, amplitude*0.05, len(df))
    df['Volume'] = np.random.uniform(1000, 10000, len(df))
    
    # Ensure Low <= Open/Close <= High
    df['Low'] = np.minimum(df['Low'], df[['Open', 'Close']].min(axis=1))
    df['High'] = np.maximum(df['High'], df[['Open', 'Close']].max(axis=1))
    
    # Clean data
    df = df[df['Volume'] > 0]
    
    return df


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
    """Create comparison chart showing stable vs volatile periods side by side - FIXED VERSION"""
    if df is None or len(df) == 0:
        return go.Figure()
    
    # Identify periods
    df_with_periods = identify_stable_volatile_periods(df)
    
    if df_with_periods is None or len(df_with_periods) == 0:
        return go.Figure()
    
    # Separate stable and volatile data
    stable_df = df_with_periods[df_with_periods['Period_Type'] == 'Stable'].copy()
    volatile_df = df_with_periods[df_with_periods['Period_Type'] == 'Volatile'].copy()
    
    # Check if we have data for both periods
    has_stable = len(stable_df) > 0
    has_volatile = len(volatile_df) > 0
    
    if not has_stable and not has_volatile:
        return go.Figure()
    
    # Create subplots with appropriate number of rows
    if has_stable and has_volatile:
        # Two subplots for stable and volatile
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Stable Period (Low Volatility)', 'Volatile Period (High Volatility)'),
            vertical_spacing=0.15
        )
        
        # Add stable period trace
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
        
        # Update both x-axes and y-axes
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
        
    elif has_stable:
        # Only stable data - single subplot
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Stable Period (Low Volatility)',)
        )
        
        fig.add_trace(
            go.Scatter(
                x=stable_df['Date'],
                y=stable_df['Close'],
                mode='lines',
                name='Stable Price',
                line=dict(color='#00ff88', width=2),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
            )
        )
        
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
        
    else:
        # Only volatile data - single subplot
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Volatile Period (High Volatility)',)
        )
        
        fig.add_trace(
            go.Scatter(
                x=volatile_df['Date'],
                y=volatile_df['Close'],
                mode='lines',
                name='Volatile Price',
                line=dict(color='#ff6b9d', width=2),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
            )
        )
        
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
        height=700 if (has_stable and has_volatile) else 400,
        showlegend=True,
        legend=dict(font=dict(color='#ffffff'))
    )
    
    return fig


def create_stable_volatile_comparison_side_by_side(amplitude=50, frequency=0.5, drift=0, days=30):
    """
    Create side-by-side comparison of stable vs volatile money (small vs big swings)
    This is a key FA2 requirement
    """
    # Generate stable data (small swings)
    df_stable = generate_simulated_data(
        days=days,
        pattern='sine',
        amplitude=amplitude,
        frequency=frequency,
        drift=drift,
        volatility_level='stable'
    )
    
    # Generate volatile data (big swings)
    df_volatile = generate_simulated_data(
        days=days,
        pattern='sine',
        amplitude=amplitude,
        frequency=frequency,
        drift=drift,
        volatility_level='volatile'
    )
    
    # Create subplots for side-by-side comparison
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Stable Money (Small Swings) - Volatility: {calculate_volatility(df_stable):.2f}%',
            f'Volatile Money (Big Swings) - Volatility: {calculate_volatility(df_volatile):.2f}%'
        ),
        vertical_spacing=0.15
    )
    
    # Add stable data trace
    fig.add_trace(
        go.Scatter(
            x=df_stable['Date'],
            y=df_stable['Close'],
            mode='lines',
            name='Stable Price',
            line=dict(color='#00ff88', width=2),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add volatile data trace
    fig.add_trace(
        go.Scatter(
            x=df_volatile['Date'],
            y=df_volatile['Close'],
            mode='lines',
            name='Volatile Price',
            line=dict(color='#ff6b9d', width=2),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update both x-axes and y-axes
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
        title=dict(text='Comparison: Stable Money vs Volatile Money', font=dict(size=18, color='#00d4ff')),
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
        st.markdown("##### Analyzing cryptocurrency price movements from real and simulated data")
    with col2:
        st.markdown("""
        <div style='text-align: right; padding-top: 20px;'>
            <span style='color: #00d4ff; font-size: 12px;'>● Real & Simulated Data</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar - Control Panel
    with st.sidebar:
        st.markdown("## Controls")
        st.markdown("##### Customize data view")
        st.markdown("")
        
        # DATA SOURCE: Real vs Simulated
        st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**DATA SOURCE**</p>', unsafe_allow_html=True)
        data_source = st.radio(
            "Select data source",
            ["Simulated Data", "Real Data"],
            label_visibility="collapsed"
        )
        
        st.markdown("")
        
        if data_source == "Real Data":
            # Warning about CSV file requirement
            st.warning("⚠️ Real Data mode requires the CSV file to be uploaded or present in the app directory.")
            
            # Fixed to 30 days for better performance
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**TIME RANGE**</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: #ffffff; font-size: 12px;">Fixed to last 30 days for optimal performance</p>', unsafe_allow_html=True)
            days_range = 30  # Fixed to 30 days
            
            st.markdown("")
            
            # Data Granularity
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**DATA GRANULARITY**</p>', unsafe_allow_html=True)
            granularity = st.selectbox(
                "Select data granularity",
                ["Daily", "Hourly", "Raw (Minute-level)"],
                label_visibility="collapsed"
            )
            
            st.markdown("")
            
            # File Upload (optional - use default dataset)
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**CUSTOM DATASET**</p>', unsafe_allow_html=True)
            use_default = st.checkbox("Use default dataset", value=True)
            
            if not use_default:
                uploaded_file = st.file_uploader(
                    "Upload your CSV dataset",
                    type=['csv'],
                    help="Upload a CSV file with columns: Timestamp, Open, High, Low, Close, Volume"
                )
            else:
                uploaded_file = None
        
        else:  # Simulated Data
            # Pattern selection
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**PRICE PATTERN**</p>', unsafe_allow_html=True)
            pattern = st.selectbox(
                "Select price swing pattern",
                ["Sine/Cosine Waves", "Random Noise"],
                label_visibility="collapsed"
            )
            pattern_code = 'sine' if pattern == "Sine/Cosine Waves" else 'random'
            
            st.markdown("")
            
            # Amplitude slider
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**SWING SIZE (AMPLITUDE)**</p>', unsafe_allow_html=True)
            amplitude = st.slider(
                "Amplitude - controls how big the price swings are",
                min_value=10,
                max_value=200,
                value=50,
                step=5,
                label_visibility="collapsed"
            )
            
            st.markdown("")
            
            # Frequency slider
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**SWING SPEED (FREQUENCY)**</p>', unsafe_allow_html=True)
            frequency = st.slider(
                "Frequency - controls how fast the price swings repeat",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                label_visibility="collapsed"
            )
            
            st.markdown("")
            
            # Drift slider
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**LONG-TERM SLOPE (DRIFT)**</p>', unsafe_allow_html=True)
            drift = st.slider(
                "Drift - controls the long-term trend (positive=up, negative=down)",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                label_visibility="collapsed"
            )
            
            st.markdown("")
            
            # Data Granularity
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**DATA GRANULARITY**</p>', unsafe_allow_html=True)
            granularity = st.selectbox(
                "Select data granularity",
                ["Daily", "Hourly", "Raw (Minute-level)"],
                label_visibility="collapsed",
                key="sim_granularity"
            )
            
            st.markdown("")
            
            # Time range for simulation
            st.markdown('<p style="color: #00d4ff; font-weight: 600; font-size: 14px;">**TIME RANGE**</p>', unsafe_allow_html=True)
            sim_days = st.slider(
                "Number of days to simulate",
                min_value=7,
                max_value=90,
                value=30,
                step=1,
                label_visibility="collapsed"
            )
        
        st.markdown("---")
        
        # Dataset Info
        if data_source == "Real Data":
            st.markdown("""
            **Dataset Information**
            
            - Default: Crypto_data.crdownload
            - Records: Last 30 days only
            - Period: Most recent 30 days
            - Granularity: Minute-level (aggregated)
            - Performance: Optimized for fast loading
            """)
        else:
            st.markdown(f"""
            **Simulation Parameters**
            
            - Pattern: {pattern}
            - Amplitude: {amplitude}
            - Frequency: {frequency}
            - Drift: {drift}
            - Days: {sim_days}
            - Granularity: {granularity}
            """)
    
    # Load Data based on source
    if data_source == "Real Data":
        data_file = "Crypto_data.crdownload"
        
        if not use_default and uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_upload.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_file = "temp_upload.csv"
        elif not use_default:
            st.warning("Please upload a dataset or use the default data.")
            st.stop()
        
        # Load only the last 30 days of data for better performance
        with st.spinner("Loading and processing last 30 days of data..."):
            df = load_crypto_data(data_file, days=30)
        
        if df is None or len(df) == 0:
            st.error("Failed to load data. The CSV file was not found.")
            st.markdown("""
            **⚠️ Real Data Mode Issue**
            
            To use Real Data mode, you need to:
            1. **Upload your CSV file** using the file uploader in the sidebar
            2. **Or ensure the file is present** in the app directory
            
            **💡 Quick Fix:**
            Try **Simulated Data mode** instead! It works instantly without any file and provides all the same visualizations with customizable parameters.
            
            **CSV File Requirements:**
            - File must contain columns: `Timestamp, Open, High, Low, Close, Volume`
            - Supported formats: `.csv` or `.crdownload`
            - File size should be reasonable (under 100MB recommended)
            """)
            st.stop()
        
        # Show data loading success
        st.success(f"✅ Loaded {len(df):,} records from the last 30 days")
        
        # Apply granularity
        if granularity == "Daily":
            df_display = aggregate_to_daily(df)
        elif granularity == "Hourly":
            df_display = df.set_index('Date').resample('H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna().reset_index()
        else:  # Raw data
            df_display = df.copy()
        
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
                help="Last 30 days of data"
            )
        
        with col3:
            st.metric(
                "Granularity",
                granularity,
                help="Data aggregation level"
            )
        
        with col4:
            st.metric(
                "Time Period",
                "30 Days",
                help="Fixed time range for optimal performance"
            )
    
    else:  # Simulated Data
        # Generate simulated data
        with st.spinner("Generating simulated data..."):
            df = generate_simulated_data(
                days=sim_days,
                pattern=pattern_code,
                amplitude=amplitude,
                frequency=frequency,
                drift=drift,
                volatility_level='mixed'
            )
        
        # Show data generation success
        st.success(f"✅ Generated {len(df):,} records using {pattern} pattern")
        
        # Apply granularity
        if granularity == "Daily":
            df_display = aggregate_to_daily(df)
        elif granularity == "Hourly":
            df_display = df.set_index('Date').resample('H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna().reset_index()
        else:  # Raw data
            df_display = df.copy()
        
        # Display data info
        st.markdown("### Simulation Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records",
                f"{len(df_display):,}",
                help=f"Total data points in simulation"
            )
        
        with col2:
            st.metric(
                "Date Range",
                f"{df_display['Date'].min().strftime('%b %d')} - {df_display['Date'].max().strftime('%b %d, %Y')}",
                help=f"Simulated for {sim_days} days"
            )
        
        with col3:
            st.metric(
                "Pattern",
                pattern,
                help="Mathematical pattern used"
            )
        
        with col4:
            st.metric(
                "Amplitude",
                f"{amplitude}",
                help="Swing size (price movement magnitude)"
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
    if data_source == "Real Data":
        st.markdown("##### Line graph showing how the price moves up and down")
    else:
        st.markdown(f"##### Line graph showing {pattern.lower()} pattern with amplitude={amplitude}, frequency={frequency}, drift={drift}")
    
    st.plotly_chart(
        create_price_chart(df_display, f"Price Movement - {sim_days if data_source == 'Simulated Data' else 30} Days ({granularity})"),
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
    
    # Section 3: Stable vs Volatile Periods (FA2 Requirement)
    st.markdown("### Stable vs Volatile Periods")
    if data_source == "Real Data":
        st.markdown("##### Marked areas showing stable (flat) vs volatile (sharp ups and downs) periods")
    else:
        st.markdown("##### Marked areas showing stable (flat) vs volatile (sharp ups and downs) periods in simulation")
    
    # Chart with shaded regions
    st.plotly_chart(
        create_stable_volatile_comparison_chart(df_display),
        use_container_width=True,
        config={'displayModeBar': False}
    )
    
    st.markdown("")
    
    # Side-by-side comparison
    st.markdown("### Stable vs Volatile Periods Comparison")
    if data_source == "Real Data":
        st.markdown("##### Side-by-side view of stable and volatile periods from real data")
    else:
        st.markdown("##### Side-by-side view of stable and volatile periods from simulation")
    st.plotly_chart(
        create_volatility_comparison_chart(df_display),
        use_container_width=True,
        config={'displayModeBar': False}
    )
    
    st.markdown("")
    
    # NEW FA2 REQUIREMENT: Side-by-side comparison of Stable vs Volatile Money
    if data_source == "Simulated Data":
        st.markdown("### Stable Money vs Volatile Money Comparison")
        st.markdown("##### Compare stable money (small swings) with volatile money (big swings) side by side")
        st.markdown("<p style='color: #00ff88; font-size: 12px;'>🟢 Stable Money (Small Swings) = Low Volatility, Predictable, Safer</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #ff6b9d; font-size: 12px;'>🔴 Volatile Money (Big Swings) = High Volatility, Unpredictable, Riskier</p>", unsafe_allow_html=True)
        st.plotly_chart(
            create_stable_volatile_comparison_side_by_side(
                amplitude=amplitude,
                frequency=frequency,
                drift=drift,
                days=sim_days
            ),
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
        st.markdown(f"""
        This interactive dashboard analyzes {'real cryptocurrency market' if data_source == 'Real Data' else 'simulated'} data.
        
        {'The data has been filtered to show the last 30 days and aggregated' if data_source == 'Real Data' else f'The data has been generated using {pattern} mathematical patterns'}
        according to your chosen parameters.
        
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
