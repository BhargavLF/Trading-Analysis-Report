import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Trading Analysis Dashboard", layout="wide")

st.title("Trading Analysis Dashboard")
st.markdown("Upload your Excel trading blotter to see a comprehensive analysis.")

uploaded_file = st.file_uploader("Choose your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name='Trade Blotter')

    # Data cleaning
    df = df[df['Dealt Amount'].notnull()]
    df['Dealt Amount'] = df['Dealt Amount'].astype(float)
    df['Fee'] = pd.to_numeric(df['Fee'], errors='coerce').fillna(0)
    df['Indicative P/L'] = pd.to_numeric(df['Indicative P/L'], errors='coerce').fillna(0)
    df['Execution Time'] = pd.to_datetime(df['Execution Time'], errors='coerce')
    df['Buy/Sell'] = df['Buy/Sell'].str.strip().str.title()
    df['Dealt Currency'] = df['Dealt Currency'].str.strip().str.upper()
    df['Trade Type'] = df['Trade Type'].str.strip()

    df = df[df['Execution Time'].notnull()]
    df['Exec_Date'] = df['Execution Time'].dt.date
    df['Exec_Year'] = df['Execution Time'].dt.year
    df['Exec_Month'] = df['Execution Time'].dt.to_period('M')
    df['Exec_Week'] = df['Execution Time'].dt.isocalendar().week

    avg_trades_per_day = df.groupby('Exec_Date').size().mean()
    avg_trades_per_week = df.groupby(['Exec_Year', 'Exec_Week']).size().mean()
    avg_trades_per_month = df.groupby('Exec_Month').size().mean()
    avg_trades_per_year = df.groupby('Exec_Year').size().mean()

    total_trades = len(df)
    focus_currency = df['Dealt Currency'].value_counts().idxmax()
    rare_currency = df['Dealt Currency'].value_counts().idxmin()
    currency_counts = df['Dealt Currency'].value_counts(normalize=True) * 100
    avg_trade_size = df['Dealt Amount'].abs().mean()
    trade_size_pref = 'Small' if avg_trade_size < 50000 else 'Moderate' if avg_trade_size < 500000 else 'Large'

    if df['Execution Time'].notnull().any():
        df['Hour'] = df['Execution Time'].dt.hour
        hour_counts = df['Hour'].value_counts().sort_index()
        max_hour = hour_counts.idxmax()
        max_hour_trades = hour_counts.max()
        consistency = "Inconsistent" if hour_counts.std() > 10 else "Consistent"
    else:
        max_hour = None
        max_hour_trades = None
        consistency = "Unknown"
        hour_counts = pd.Series(dtype=int)

    buy_count = (df['Buy/Sell'] == 'Buy').sum()
    sell_count = (df['Buy/Sell'] == 'Sell').sum()
    buy_bias = buy_count / total_trades * 100
    sell_bias = sell_count / total_trades * 100
    direction_bias = "Buy" if buy_bias > sell_bias else "Sell"

    currency_trade_counts = df['Dealt Currency'].value_counts()
    eur_trades = currency_trade_counts.get('EUR', 0)
    gbp_trades = currency_trade_counts.get('GBP', 0)
    aud_trades = currency_trade_counts.get('AUD', 0)

    profits = df[df['Indicative P/L'] > 0]['Indicative P/L']
    losses = df[df['Indicative P/L'] < 0]['Indicative P/L']
    wins_over_losses_ratio = len(profits) / len(losses) if len(losses) > 0 else np.nan
    avg_pl = df['Indicative P/L'].mean()
    biggest_win = df['Indicative P/L'].max()
    largest_loss = df['Indicative P/L'].min()

    total_fee = df['Fee'].sum()
    fee_per_trade = df['Fee'].mean()
    fee_impact_pct = (total_fee / (profits.sum() + abs(losses.sum()))) * 100 if (profits.sum() + abs(losses.sum())) > 0 else np.nan

    df = df.sort_values('Execution Time')
    df['Next Side'] = df['Buy/Sell'].shift(-1)
    df['Next Currency'] = df['Dealt Currency'].shift(-1)
    df['Next Time'] = df['Execution Time'].shift(-1)
    df['Time Diff'] = (df['Next Time'] - df['Execution Time']).dt.total_seconds() / 60
    rapid_reversals = df[(df['Buy/Sell'] != df['Next Side']) &
                         (df['Dealt Currency'] == df['Next Currency']) &
                         (df['Time Diff'] <= 10)]['Time Diff'].count()
    partial_fills = df['Trade Id'].isnull().sum()

    match_col = None
    for col in ['Client Order Id', 'Trade Id']:
        if col in df.columns:
            match_col = col
            break

    if match_col:
        buys = df[df['Buy/Sell'] == 'Buy'].copy()
        sells = df[df['Buy/Sell'] == 'Sell'].copy()
        merged = pd.merge(
            buys,
            sells,
            on=[match_col, 'Dealt Currency', 'Dealt Amount'],
            suffixes=('_buy', '_sell')
        )
        merged['Holding Duration'] = (merged['Execution Time_sell'] - merged['Execution Time_buy']).dt.total_seconds() / 60
        merged = merged[merged['Holding Duration'] >= 0]
        if not merged.empty:
            avg_holding_duration = merged['Holding Duration'].mean()
            avg_holding_duration_fmt = f"{int(avg_holding_duration//60)}h {int(avg_holding_duration%60)}m"

    style_line = (
        f"Aggressive {'high' if avg_trades_per_day > 10 else 'low'}-frequency trader, "
        f"{trade_size_pref} positions, {focus_currency}-centric, "
        f"{'fee-insensitive' if fee_impact_pct < 5 else 'fee-sensitive'}, "
        f"{'buy' if buy_bias > sell_bias else 'sell'} bias, {consistency} trading hours."
    )

    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", total_trades)
    col1.metric("Avg Trades/Day", f"{avg_trades_per_day:.2f}")
    col1.metric("Avg Trades/Week", f"{avg_trades_per_week:.2f}")
    col1.metric("Avg Trades/Month", f"{avg_trades_per_month:.2f}")
    col1.metric("Avg Trades/Year", f"{avg_trades_per_year:.2f}")
    col1.metric("Direction Bias", f"{direction_bias} ({max(buy_bias, sell_bias):.2f}%)")
    col1.metric("EUR Trades", eur_trades)
    col1.metric("GBP Trades", gbp_trades)
    col1.metric("AUD Trades", aud_trades)
    col2.metric("Focus Currency", f"{focus_currency} ({currency_counts[focus_currency]:.1f}%)")
    col2.metric("Rare Currency", f"{rare_currency} ({currency_counts[rare_currency]:.1f}%)")
    col2.metric("Avg Trade Size", f"{avg_trade_size:,.0f} ({trade_size_pref})")
    col2.metric("Max Trades in Hour", f"{max_hour} ({max_hour_trades})")
    col2.metric("Consistency", consistency)
    col2.metric("Wins/Losses Ratio", f"{wins_over_losses_ratio:.2f}")
    col2.metric("Avg P/L", f"{avg_pl:.2f}")
    col2.metric("Biggest Win", f"{biggest_win:.2f}")
    col2.metric("Largest Loss", f"{largest_loss:.2f}")
    col3.metric("Avg Fee/Trade", f"{fee_per_trade:.6f}")
    col3.metric("Fee Impact (%)", f"{fee_impact_pct:.2f}")
    col3.metric("Buy Bias (%)", f"{buy_bias:.2f}")
    col3.metric("Sell Bias (%)", f"{sell_bias:.2f}")

    st.markdown("### Patterns and Execution")
    st.write(f"Rapid Reversals (within 10 min): {rapid_reversals}")
    st.write(f"Partial Fills/Missing Trade IDs: {partial_fills}")

    st.markdown("### Trader Style Summary")
    st.success(style_line)

    st.markdown("## 📊 Visual Insights")

    # 1. Trade Frequency by Hour
    if not hour_counts.empty:
        fig1 = px.bar(
            x=hour_counts.index,
            y=hour_counts.values,
            labels={'x': 'Hour (24h)', 'y': 'Number of Trades'},
            title='Trades by Hour of Day',
            text_auto=True  # Show values on bars
        )
        st.plotly_chart(fig1, use_container_width=True)

    # 2. Currency Distribution
    currency_dist = df['Dealt Currency'].value_counts()
    fig2 = px.bar(
        x=currency_dist.index,
        y=currency_dist.values,
        labels={'x': 'Currency', 'y': 'Number of Trades'},
        title='Currency Distribution',
        text_auto=True  # Show values on bars
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Buy vs Sell Pie
    buy_sell_counts = df['Buy/Sell'].value_counts()
    fig3 = px.pie(
        values=buy_sell_counts.values,
        names=buy_sell_counts.index,
        title="Buy vs Sell",
        hole=0.6
    )
    fig3.update_traces(textinfo='label+value+percent')  # Show label, value, and percent
    st.plotly_chart(fig3, use_container_width=True)

    # 4. P/L Histogram
    fig4 = px.histogram(
        df,
        x='Indicative P/L',
        nbins=30,
        title="P/L Distribution",
        text_auto=True  # Show values on bars
    )
    st.plotly_chart(fig4, use_container_width=True)

    # 5. Trade Size Histogram
    fig5 = px.histogram(
        df,
        x=df['Dealt Amount'].abs(),
        nbins=30,
        title="Trade Size Distribution",
        labels={'x': 'Dealt Amount (abs)'},
        text_auto=True  # Show values on bars
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### Raw Data (first 10 rows)")
    st.dataframe(df.head(10))
else:
    st.info("Please upload an Excel file to begin analysis.")
