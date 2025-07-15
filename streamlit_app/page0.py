# Streamlit User Interface
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Optimal Portfolio on the BRVM",
    page_icon="💼"
)

# Horizontal alignment container
col1, col2, col3 = st.columns([1, 4, 1])

# Left column: Image
with col1:
    st.image(
        "https://media.licdn.com/dms/image/v2/D4E03AQF7cVN_iger2w/profile-displayphoto-shrink_800_800/B4EZdURvE.HsAc-/0/1749465625806?e=1755734400&v=beta&t=4FXq1wVFGgbqDEOVVw-MHUZt9wkZWEx0kndiMZQqMwo",
        width=80,
        use_container_width=False,
    )

# Center column: Title
with col2:
    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 0;'>Building an Optimal Portfolio on the BRVM</h1>
        """,
        unsafe_allow_html=True,
    )

# Right column: Name and LinkedIn
with col3:
    st.markdown(
        """
        <div style='text-align: right;'>
            <a href="https://www.linkedin.com/in/marcel-an%C3%A9e-2aa3091bb/" target="_blank" style='text-decoration: none; color: #0077b5;'>
                <strong>ANEE MARCEL</strong>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write(" ")
st.write(" ")

# Project Introduction
st.title("Project Overview")

st.markdown(
    """
    This project aims to build an **optimal portfolio of stocks listed on the BRVM**, using a quantitative approach
    and subject to performance and diversification constraints.

    The main objectives are:
    - Maximize diversification
    - Ensure a target annual return
    - Minimize portfolio risk (volatility)
    - Reduce correlation between selected stocks
    - Limit the individual weight of each asset

    The project uses **Pandas, NumPy, Plotly** for data analysis and **cvxpy** to solve the constrained quadratic optimization problem.
    """
)

st.write(" ")
st.write(" ")

# Phase 1 - Data Collection & Preparation
st.markdown("## **Phase 1: Data Collection and Preparation**")
st.markdown(
    """
    - Price data is extracted from historical quotes of BRVM-listed stocks.
    - Columns with more than 70% missing values are removed.
    - Missing data is imputed using forward and backward fill methods.
    - Monthly returns are then calculated.
    - The analysis period starts from **July 2024 to the present day**.
    """
)

# Phase 2 - Stock Analysis
st.markdown("## **Phase 2: Stock Analysis**")
st.markdown(
    """
    - Display of the most profitable stocks over the last 6 months.
    - Identification of the most volatile stocks.
    - Visualization of return trends, with the ability to select a specific stock.
    """
)

# Phase 3 - Asset Selection
st.markdown("## **Phase 3: Asset Filtering for the Portfolio**")
st.markdown(
    """
    - Only stocks with a **cumulative return of 10% or more** during the analysis period are selected.
    - The **average correlation** of each stock with the others is computed.
    - Stocks with an average correlation above **0.35** are excluded to promote diversification and avoid contagion.
    """
)

# Phase 4 - Optimization
st.markdown("## **Phase 4: Portfolio Optimization**")
st.markdown(
    """
    - The portfolio is optimized to **minimize risk (volatility)** under the following constraints:
        - The sum of asset weights must equal 100%
        - The expected return must be ≥ **10% annually**
        - The weight of each asset must not exceed **25%** to avoid concentration risk
    - The optimization problem is solved using the `cvxpy` quadratic programming solver.
    """
)

# Phase 5 - Results
st.markdown("## **Phase 5: Displayed Results**")
st.markdown(
    """
    - 📈 **Expected annual return** of the optimized portfolio
    - 📉 **Minimum risk (annualized volatility)** achieved through optimization
    - 📊 **Detailed portfolio table** showing selected stocks, assigned weights, and investment amounts
    - ✅ Interactive interface to adjust: capital, target return, max weight, and correlation threshold
    """
)

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:14px; color:gray;">
        Application developed as part of an applied research project on asset management for the BRVM.
    </p>
    """,
    unsafe_allow_html=True,
)
