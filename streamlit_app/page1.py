import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import cvxpy as cp
from datetime import datetime
import plotly.graph_objects as go

# --- Load data ---
df = pd.read_csv("streamlit_app/Action.csv")
#df = pd.read_csv("Action.csv")

# Convert 'date' column to datetime and set as index
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.set_index('date', inplace=True)

# Drop unused columns if exist
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')

# Sort by ascending date
df = df.sort_index(ascending=True)

# --- Data processing ---
# Drop columns with more than 70% missing values
missing_stats = df.isnull().mean()
df = df[missing_stats[missing_stats <= 0.7].index]

# Impute missing values (forward fill, backward fill)
df = df.ffill().bfill()

# Convert to numeric if necessary
df = df.apply(pd.to_numeric, errors='coerce')




# --- Top 5 returns function ---
def Top_5_Action(df):
    df_months = df[df.index.month <= 6]
    returns = (df_months.iloc[-1] / df_months.iloc[0]) - 1
    sorted_returns = returns.sort_values(ascending=False)
    return sorted_returns[:5]

# --- Volatility and coefficient of variation functions ---
def top_cv_volatility_graph(df, top_n=5):
    df_6months = df[df.index.month <= 6]
    mean_prices = df_6months.mean()
    std_prices = df_6months.std()
    cv = std_prices / mean_prices
    log_returns = np.log(df_6months / df_6months.shift(1))
    volatility = log_returns.std()
    top_cv = cv.sort_values(ascending=False).head(top_n)
    top_vol = volatility.sort_values(ascending=False).head(top_n)
    return top_cv, top_vol




def calcul_volatility_1yr(price_series, freq=252):
    """
    Calculate annualized volatility over the last 12 months.

    Parameters:
    - price_series : pandas Series of prices (index = dates)
    - freq : observation frequency (252 for daily data)

    Returns:
    - annualized volatility over 1 year (float)
    """
    series_1yr = price_series.dropna().iloc[-freq:]
    returns = series_1yr.pct_change().dropna()
    volatility = series_1yr.std() * np.sqrt(freq)
    return volatility


def calcul_return_period(price_series, freq=252):
    """
    Calculate total return over the last period (e.g. 1 year).

    Parameters:
    - price_series : pandas Series of prices (index = dates)
    - freq : number of points in the period (252 for 1 year daily trading)

    Returns:
    - total return over the period (float)
    """
    series_1yr = price_series.dropna().iloc[-freq:]
    ret = (series_1yr.iloc[-1] / series_1yr.iloc[0]) - 1
    return ret





def portfolio_optimization(df, capital, target_return=0.10, max_weight=0.25, max_correlation=0.35):
    # 1. Filter period July 2024 to today
    df = df[(df.index >= pd.Timestamp('2024-07-01')) & (df.index <= pd.Timestamp(datetime.now().date()))]
    df = df.ffill().bfill()

    if df.empty or df.shape[0] < 2:
        st.error("❌ Insufficient data to perform analysis.")
        return

    # 2. Monthly returns
    returns = df.resample('M').last().pct_change().dropna()

    # 3. Total return over the period
    cum_returns = (df.iloc[-1] / df.iloc[0]) - 1
    profitable = cum_returns[cum_returns >= target_return]

    if profitable.empty:
        st.warning("❌ No stocks with return ≥ 10%.")
        return

    # 4. Correlation
    returns_filtered = returns[profitable.index]
    corr_matrix = returns_filtered.corr()
    avg_corr = corr_matrix.apply(lambda x: x.drop(x.name).mean())
    low_correlated = avg_corr[avg_corr <= max_correlation].index.tolist()

    if not low_correlated:
        st.warning("❌ No stocks with average correlation ≤ threshold.")
        return

    # 5. Optimization
    rets = returns[low_correlated]
    mean_returns = rets.mean()
    cov_matrix = rets.cov()
    n = len(low_correlated)

    w = cp.Variable(n)
    risk = cp.quad_form(w, cov_matrix.values)
    expected_return = mean_returns.values @ w

    constraints = [
        cp.sum(w) == 1,
        expected_return >= target_return / 12,  # monthly target
        w >= 0,
        w <= max_weight
    ]
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()

    if w.value is None:
        st.error("❌ Optimization failed. Try adjusting your constraints.")
        return

    # Results
    weights = w.value
    weights = np.round(weights, 4)
    expected_return = mean_returns.values @ weights
    annual_return = (1 + expected_return)**12 - 1
    # Portfolio risk (annualized std deviation)
    portfolio_variance = weights.T @ cov_matrix.values @ weights
    annual_risk = np.sqrt(portfolio_variance) * np.sqrt(12)

    portfolio = pd.Series(weights, index=low_correlated)
    portfolio = portfolio[portfolio > 0]

    st.success("🎯 Optimization successful!")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.metric("Projected Annual Return", f"{annual_return * 100:.4f}%")

    with col1:
        st.metric("Expected Monthly Return", f"{expected_return * 100:.4f}%")
    
    with col3:
        st.metric("📉 Risk (Annualized Volatility)", f"{annual_risk * 100:.2f}%")

    st.write(f"📊 Actual Diversification: max weight = {max_weight * 100:.0f}%, max correlation = {max_correlation}")

    # Allocation summary
    st.subheader("💼 Optimal Portfolio Details")
    allocation = pd.DataFrame({
        "Ticker": portfolio.index,
        "Weight (%)": (portfolio * 100).round(2),
        "Invested Amount (FCFA)": (portfolio * capital).round(0).astype(int)
    }).set_index("Ticker")
    st.dataframe(allocation)
    return {
        "portfolio": portfolio,
        "annual_return": annual_return,
        "annual_risk": annual_risk,
        "capital": capital
    }


#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def simulate_portfolio_monte_carlo(S0, mu, sigma, T=1, steps=252, n_sim=1000, target_return=0.10):
    import numpy as np
    import matplotlib.pyplot as plt

    dt = T / steps
    np.random.seed(42)  # reproductibilité

    # Mouvement brownien
    W = np.random.normal(0, np.sqrt(dt), size=(steps, n_sim))
    W = np.cumsum(W, axis=0)

    time = np.linspace(0, T, steps).reshape(-1, 1)
    S_t = S0 * np.exp((mu - 0.5 * sigma**2) * time + sigma * W)

    # Proba d’atteindre le seuil de 10%
    final_values = S_t[-1]
    prob_above_target = np.mean(final_values >= S0 * (1 + target_return))

    return S_t, prob_above_target


#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


   

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# --- Streamlit Interface ---
st.title("📊 Stock Return & Volatility Analysis (Jan-Jun 2025)")

# 1. Top 5 Returns
top_returns = Top_5_Action(df)

fig_return = px.bar(
    x=top_returns.values,
    y=top_returns.index,
    orientation='h',
    labels={'x': 'Return', 'y': 'Stock'},
    title="Top 5 Stocks by Return",
    color=top_returns.values,
    color_continuous_scale='viridis'
)
fig_return.update_layout(yaxis=dict(autorange="reversed"))


# 2. Top 5 Volatility and Coefficient of Variation
top_cv, top_vol = top_cv_volatility_graph(df)

fig_vol = px.bar(
    x=top_vol.values,
    y=top_vol.index,
    orientation='h',
    labels={'x': 'Volatility', 'y': 'Stock'},
    title="Top 5 Most Volatile Stocks",
    color=top_vol.values,
    color_continuous_scale='viridis'
)
fig_vol.update_layout(yaxis=dict(autorange="reversed"))


fig_cv = px.bar(
    x=top_cv.values,
    y=top_cv.index,
    orientation='h',
    labels={'x': 'Coefficient of Variation', 'y': 'Stock'},
    title="Top 5 Highest Coefficient of Variation",
    color=top_cv.values,
    color_continuous_scale='plasma'
)
fig_cv.update_layout(yaxis=dict(autorange="reversed"))


# Streamlit layout
col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(fig_vol, use_container_width=True)
    # st.plotly_chart(fig_cv, use_container_width=True)

with col2:
    st.plotly_chart(fig_return, use_container_width=True)
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
st.divider()
risk_free_rate = 0.03  # taux sans risque annuel 
st.title('Analyse de Tritre')

Ticker = st.selectbox("Select a stock", df.columns)
data = df[Ticker]
returns = data.pct_change().fillna(0)

fig = px.line(returns, title='Return Evolution')
#st.plotly_chart(fig)

fig_price = px.line(data, title=f"Évolution du prix de {Ticker}", labels={"value": "Prix", "index": "Date"})
#st.plotly_chart(fig_price)

col1, col2 = st.columns([1, 1])

with col1:
    st.plotly_chart(fig_price)
    # st.plotly_chart(fig_cv, use_container_width=True)

with col2:
    st.plotly_chart(fig)

# Calculs sur 1 an
monthly_returns = data.resample("M").last().pct_change().dropna()
if len(monthly_returns) > 12:
    monthly_returns = monthly_returns[-12:]

mean_return = monthly_returns.mean()
volatility = monthly_returns.std()
annual_return = (1 + mean_return)**12 - 1
annual_volatility = volatility * np.sqrt(12)
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan

# Affichage des métriques
st.subheader("📌 Statistiques sur 1 an")
col1, col2, col3 = st.columns(3)
col1.metric("Rendement annuel (%)", f"{annual_return * 100:.2f}")
col2.metric("Volatilité annuelle (%)", f"{annual_volatility * 100:.2f}")
col3.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")

st.divider()
st.title("Portfolio Optimization")

capital = st.number_input("Available Capital (FCFA)", value=10_000_000)
target_return = st.slider("Annual Target Return (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.05) / 100
# max_weight = st.slider("Max weight per stock (%)", min_value=5, max_value=100, value=20, step=5) / 100
# max_correlation = st.slider("Max allowed correlation", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

if st.button("Run Optimization"):
    results = portfolio_optimization(df, capital, target_return, max_weight=0.20, max_correlation=0.35)


    if results:
        S0 = results["capital"]
        mu = results["annual_return"]
        sigma = results["annual_risk"]

        # Simulation
        paths, prob = simulate_portfolio_monte_carlo(S0, mu, sigma)

        st.subheader("🔁 Monte Carlo Simulation")

        # Préparation du graphique Plotly
        fig = go.Figure()

        # On trace les 100 premières trajectoires (pour lisibilité)
        for i in range(min(100, paths.shape[1])):
            fig.add_trace(go.Scatter(
                y=paths[:, i],
                mode='lines',
                line=dict(color='skyblue', width=1),
                opacity=0.3,
                showlegend=False
            ))

        # Ligne cible +10%
        fig.add_trace(go.Scatter(
            x=list(range(paths.shape[0])),
            y=[S0 * 1.10] * paths.shape[0],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Target +10%'
        ))

        fig.update_layout(
            title="Simulated Portfolio Value Trajectories (GBM)",
            xaxis_title="Days",
            yaxis_title="Portfolio Value (FCFA)",
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.metric(f"📊 Probabilité d’atteindre au moins {target_return*100}% de rendement", f"{prob * 100:.2f}%")




 #results = portfolio_optimization(df, capital = 10000000, target_return=0.10, max_weight=0.25, max_correlation=0.35)


