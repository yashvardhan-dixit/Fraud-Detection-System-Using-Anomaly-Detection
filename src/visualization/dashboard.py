"""Streamlit dashboard for fraud detection system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

from src.utils.logger import get_logger, setup_logger
from src.utils.config import get_config

# Setup
setup_logger()
logger = get_logger()
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔍 Fraud Detection Dashboard")
st.markdown("---")


def load_demo_data():
    """Load demo transaction data."""
    try:
        # Try to load existing data
        if os.path.exists('data/processed/demo_data.csv'):
            df = pd.read_csv('data/processed/demo_data.csv', parse_dates=['timestamp'])
        else:
            # Generate demo data
            n_transactions = 1000
            start_date = datetime.now() - timedelta(days=30)
            
            timestamps = [start_date + timedelta(minutes=i*5) for i in range(n_transactions)]
            amounts = np.random.lognormal(4, 1.5, n_transactions)
            is_fraud = np.random.choice([0, 1], n_transactions, p=[0.98, 0.02])
            fraud_scores = np.random.beta(2, 5, n_transactions)
            fraud_scores[is_fraud == 1] = np.random.beta(5, 2, is_fraud.sum())
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'amount': amounts,
                'is_fraud': is_fraud,
                'fraud_score': fraud_scores,
                'transaction_type': np.random.choice(
                    ['purchase', 'withdrawal', 'transfer', 'payment'],
                    n_transactions
                )
            })
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


# Load data
df = load_demo_data()

if not df.empty:
    # Sidebar - Filters
    st.sidebar.header("📊 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )
    
    # Transaction type filter
    transaction_types = ['All'] + list(df['transaction_type'].unique())
    selected_type = st.sidebar.selectbox("Transaction Type", transaction_types)
    
    # Fraud filter
    fraud_filter = st.sidebar.radio(
        "Show",
        ["All Transactions", "Fraud Only", "Normal Only"]
    )
    
    # Apply filters
    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
    
    if selected_type != 'All':
        mask &= df['transaction_type'] == selected_type
    
    if fraud_filter == "Fraud Only":
        mask &= df['is_fraud'] == 1
    elif fraud_filter == "Normal Only":
        mask &= df['is_fraud'] == 0
    
    filtered_df = df[mask]
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Transactions",
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        fraud_count = filtered_df['is_fraud'].sum()
        fraud_rate = (fraud_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric(
            "Fraud Cases",
            f"{fraud_count:,}",
            delta=f"{fraud_rate:.2f}%"
        )
    
    with col3:
        total_amount = filtered_df['amount'].sum()
        st.metric(
            "Total Amount",
            f"${total_amount:,.2f}"
        )
    
    with col4:
        fraud_amount = filtered_df[filtered_df['is_fraud'] == 1]['amount'].sum()
        st.metric(
            "Fraud Amount",
            f"${fraud_amount:,.2f}",
            delta=f"{(fraud_amount/total_amount*100):.1f}% of total" if total_amount > 0 else None
        )
    
    st.markdown("---")
    
    # Row 1: Time series and distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Transactions Over Time")
        
        # Group by date
        time_df = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg({
            'is_fraud': ['sum', 'count']
        }).reset_index()
        time_df.columns = ['date', 'fraud_count', 'total_count']
        time_df['fraud_rate'] = time_df['fraud_count'] / time_df['total_count'] * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_df['date'],
            y=time_df['total_count'],
            name='Total Transactions',
            mode='lines+markers',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=time_df['date'],
            y=time_df['fraud_count'],
            name='Fraud Cases',
            mode='lines+markers',
            line=dict(color='red')
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode='x unified',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Amount Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=filtered_df[filtered_df['is_fraud'] == 0]['amount'],
            name='Normal',
            marker_color='green'
        ))
        fig.add_trace(go.Box(
            y=filtered_df[filtered_df['is_fraud'] == 1]['amount'],
            name='Fraud',
            marker_color='red'
        ))
        fig.update_layout(
            yaxis_title="Amount ($)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 2: Fraud score distribution and transaction types
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Fraud Score Distribution")
        
        fig = px.histogram(
            filtered_df,
            x='fraud_score',
            color='is_fraud',
            nbins=50,
            labels={'is_fraud': 'Fraud', 'fraud_score': 'Fraud Score'},
            color_discrete_map={0: 'green', 1: 'red'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Transaction Types")
        
        type_df = filtered_df.groupby(['transaction_type', 'is_fraud']).size().reset_index(name='count')
        
        fig = px.bar(
            type_df,
            x='transaction_type',
            y='count',
            color='is_fraud',
            barmode='group',
            labels={'is_fraud': 'Fraud', 'count': 'Count'},
            color_discrete_map={0: 'green', 1: 'red'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recent Transactions Table
    st.subheader("🔍 Recent Transactions")
    
    # Add risk level column
    filtered_df['risk_level'] = pd.cut(
        filtered_df['fraud_score'],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # Display recent transactions
    display_df = filtered_df.sort_values('timestamp', ascending=False).head(100)
    
    # Color code by fraud status
    def highlight_fraud(row):
        if row['is_fraud'] == 1:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        display_df[[
            'timestamp', 'amount', 'transaction_type',
            'fraud_score', 'risk_level', 'is_fraud'
        ]].style.apply(highlight_fraud, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Download data
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Data")
    
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name=f"fraud_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    st.error("No data available. Please generate or upload transaction data.")
    st.info("Run the training pipeline to generate demo data.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Fraud Detection System v1.0 | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
