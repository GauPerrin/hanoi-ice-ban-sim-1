# streamlit_app.py (Final Enhanced Version)
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from collections import deque
import pandas as pd

import streamlit as st

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Hanoi ICE Motorcycle Ban Simulation")
st.title("🚦 Hanoi ICE Motorcycle Ban – Monte Carlo Simulation (2026–2035)")

# --- Reset Button ---
if st.sidebar.button("🔁 Reset to Defaults"):
    st.session_state.clear()

# --- Sidebar Inputs ---
# Section 1: 📋 Simulation Settings
st.sidebar.markdown("### 📋 Simulation Settings")
with st.sidebar.expander("Simulation Controls"):
    num_simulations = st.slider("🔄 Number of Simulations", 100, 2000, 1000,
                                 help="Number of Monte Carlo runs to simulate different future scenarios.")
    fleet_growth_rate = st.slider("📈 Motorcycle Fleet Growth (Annual)", 0.0, 0.05, 0.02, step=0.005,
                                  help="Year-on-year percentage increase in total motorcycle fleet.")
    retirement_age = st.slider("🛠️ Motorcycle Retirement Age (Years)", 5, 20, 10, step=1,
                               help="How long motorcycles typically remain in use before being retired.")
    adoption_model = st.selectbox("🔀 EV Adoption Model", ["Linear", "S-curve"],
                                  help="Select a simple linear or logistic adoption model for EVs.")
    seed_value = st.slider("🔢 Random Seed", 0, 10_000, 42,
                       help="Set a seed to make simulation results reproducible. Change for different runs.")

# Section 2: 🧮 Starting Conditions & Input Distributions
st.sidebar.markdown("### 🧮 Starting Conditions & Input Distributions")
with st.sidebar.expander("Initial Values & Random Walks"):
    initial_total_bikes = st.slider("🛵 Initial Total Motorcycles", 1_000_000, 10_000_000, 7_000_000, step=100_000,
                                    help="Motorcycle fleet size at the start of the simulation (2026).")
    initial_ev_share = st.slider("⚡ Starting EV Share (%)", 0.0, 1.0, 0.18, step=0.01,
                                 help="Initial proportion of electric motorcycles in 2026.")
    avg_km_year = st.slider("🛣️ Avg. Distance per Motorcycle (km/year)", 3000, 15000, 6000, step=500,
                            help="Average distance traveled per motorcycle each year.")

    st.markdown("**⛽ Fuel Use per km (L/km)**")
    col1, col2 = st.columns(2)
    with col1:
        fuel_per_km_mean = st.slider("Mean fuel/km", 0.01, 0.05, 0.02, step=0.001,
                                     help="Average fuel consumption per km by ICE motorcycles.")
    with col2:
        fuel_per_km_std = st.slider("Std Dev fuel/km", 0.0, 0.01, 0.002, step=0.0005,
                                    help="Year-to-year variability in fuel consumption.")

    st.markdown("**🔌 Electricity Use per km (kWh/km)**")
    col3, col4 = st.columns(2)
    with col3:
        elec_per_km_mean = st.slider("Mean elec/km", 0.01, 0.05, 0.03, step=0.001,
                                     help="Average electricity consumption per km by EVs.")
    with col4:
        elec_per_km_std = st.slider("Std Dev elec/km", 0.0, 0.01, 0.002, step=0.0005,
                                    help="Year-to-year variability in electricity use.")

    st.markdown("**🌫️ Emission Variability of Electricity (EV Grid Emissions)**")
    col5, col6 = st.columns(2)
    with col5:
        grid_emission_std_mean = st.slider("Mean grid elec emissions", 0.0, 0.05, 0.02, step=0.001,
                                           help="Expected average standard deviation in grid emission intensity.")
    with col6:
        grid_emission_std_std = st.slider("Volatility grid elec emissions", 0.0, 0.02, 0.005, step=0.001,
                                          help="Annual volatility of grid emission variability.")

# Section 3: 💰 Cost Trends & Emissions
st.sidebar.markdown("### 💰 Cost Trends & Emissions")
with st.sidebar.expander("Prices and Emission Intensity"):
    ev_price_mode = st.radio("EV Initial Price Mode", ["Fixed", "Uniform Range"], horizontal=True,
                             help="Choose whether EV price is a fixed value or randomly drawn from a range.")
    if ev_price_mode == "Fixed":
        ev_cost_init = st.slider("💸 Initial EV Price (USD)", 1000, 3000, 1500, step=50,
                                 help="Initial purchase price of electric motorcycles in 2026.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            ev_price_min = st.slider("EV Price Min (USD)", 1000, 3000, 1200, step=50,
                                     help="Minimum value for randomly selected EV prices.")
        with col2:
            ev_price_max = st.slider("EV Price Max (USD)", 1000, 3000, 1800, step=50,
                                     help="Maximum value for randomly selected EV prices.")
        ev_cost_init = np.random.uniform(ev_price_min, ev_price_max)
    salvage_value_pct = st.slider(
    "ICE Salvage Value at End (% of EV price)",
    0.0, 1.0, 0.1, step=0.05,
    help="Assumed resale or scrap value of ICE Motorcycles, expressed as % of EV price."
    )
    # Add to Section 3: 💰 Cost Trends & Emissions
    st.markdown("**⛽ Fuel and Electricity Prices**")
    col1, col2 = st.columns(2)

    with col1:
        fuel_price_init = st.slider(
            "Initial Fuel Price (USD/L)",
            0.5, 2.0, 0.9, step=0.05,
            help="Starting price of gasoline per liter, affecting ICE running costs."
        )
        fuel_price_trend = st.slider(
            "Fuel Price Trend (%/year)",
            -0.2, 0.2, 0.01, step=0.01,
            help="Annual percentage change in fuel prices. Positive = price increases over time."
        )
        fuel_price_vol = st.slider(
            "Fuel Price Volatility",
            0.0, 0.2, 0.05, step=0.01,
            help="Year-to-year randomness in fuel prices, capturing global shocks or fluctuations."
        )

    with col2:
        elec_price_init = st.slider(
            "Initial Elec Price (USD/kWh)",
            0.05, 0.5, 0.1, step=0.01,
            help="Starting electricity price per kWh, used to compute EV running costs."
        )
        elec_price_trend = st.slider(
            "Elec Price Trend (%/year)",
            -0.2, 0.2, 0.005, step=0.005,
            help="Expected annual percentage increase or decrease in electricity price."
        )
        elec_price_vol = st.slider(
            "Elec Price Volatility",
            0.0, 0.2, 0.02, step=0.01,
            help="Random fluctuations in electricity price over time due to market or policy shifts."
        )

    cost_decline_rate = st.slider("📉 Annual EV Cost Decline Rate", 0.0, 0.2, 0.05, step=0.01,
                                  help="Annual percentage drop in EV price due to technology learning and scale.")
    fuel_emission_factor = st.slider("⛽ Fuel Emissions (kg/L)", 1.5, 3.0, 2.3, step=0.05,
                                     help="Carbon emissions per liter of fuel consumed.")
    grid_emission_mean = st.slider("🌍 Grid Emission Intensity (kg/kWh)", 0.05, 0.5, 0.1, step=0.01,
                                   help="Average emissions per kWh of electricity used.")
    grid_emission_decline = st.slider("📉 Grid Emission Decline Rate", 0.0, 0.1, 0.03, step=0.01,
                                      help="Expected annual decrease in grid carbon intensity.")

# Section 4: 🧠 Behavioral Response to Policy & Price
st.sidebar.markdown("### 🧠 Behavioral Response to Policy & Price")
with st.sidebar.expander("Subsidies and Perception of Cost"):
    subsidy_amount = st.slider("🎁 Subsidy (% of EV Price)", 0.0, 1.0, 0.2, step=0.01,
                               help="Government subsidy offered per EV as a percentage of its price.")
    subsidy_end_year = st.slider("📅 Subsidy End Year", 2026, 2035, 2029,
                                 help="Final year subsidies are applied.")
    subsidized_fraction = st.slider("📊 Fraction of Fleet Receiving Subsidy", 0.0, 1.0, 0.5, step=0.01,
                                    help="Percentage of EV buyers who qualify for subsidies.")
    k_subsidy = st.slider("📈 Behavioral Sensitivity to Subsidy", 0.0, 1.0, 0.1, step=0.01,
                          help="Impact of subsidies on EV adoption probability.")
    k_price = st.slider("🧮 Price Sensitivity to EV Cost", 0.0, 10.0, 3.0, step=0.1,
                        help="Responsiveness of adoption rate to EV price changes.")
    price_elasticity_mode = st.selectbox(
        "🧠 Price Elasticity Reference Frame",
        ["Baseline (2026 EV Price)", "YoY EV Price Change", "Deviation from Deterministic Price"],
        help="Frame of reference used to compute price-induced behavioral changes."
    )

# Section 5: 📊 Adoption Curve Calibration
st.sidebar.markdown("### 📊 Adoption Curve Calibration")
with st.sidebar.expander("Curve Dynamics"):
    base_linear_rate = st.slider("🔁 Base Linear Conversion Rate", 0.01, 0.3, 0.1, step=0.01,
                                 help="Base rate of ICE to EV conversion for the linear model.")
    linear_std_dev = st.slider("📉 Volatility in Linear Conversion", 0.0, 0.1, 0.02, step=0.005,
                               help="Random fluctuation in yearly conversion rate.")
    k_s_curve = st.slider("📈 S-Curve Steepness", 0.1, 2.0, 0.6, step=0.1,
                          help="Controls how steep the S-curve adoption model is.")
    s_midpoint = st.slider("⏳ S-Curve Midpoint Year", 2026, 2035, 2029, step=1,
                           help="Year at which S-curve adoption rate reaches 50%.")
    policy_boost_2026 = st.slider("🚀 Policy Conversion Boost (2026)", 1.0, 3.0, 1.5, step=0.1,
                                  help="Temporary boost to conversion rate due to 2026 policies.")
    policy_boost_2028 = st.slider("🚀 Policy Conversion Boost (2028)", 1.0, 3.0, 1.3, step=0.1,
                                  help="Temporary boost to conversion rate due to 2028 policies.")

# Section 6: 🌍 Global Supply Chain Shocks
st.sidebar.markdown("### 🌍 Global Supply Chain Shocks")
with st.sidebar.expander("Cost Volatility"):
    import_dependency = st.slider("🔗 Import Dependency Level", 0.0, 1.0, 0.5, step=0.05,
                                  help="Extent to which EV price depends on volatile imported components.")
    shock_volatility = st.slider("📉 Shock Volatility", 0.0, 0.5, 0.1, step=0.01,
                                 help="Level of volatility from global shocks such as material costs or tariffs.")

# Section: ✅ Show Summary Metrics
show_metrics = st.sidebar.checkbox("📊 Show Summary Metrics", True,
                                   help="Display key simulation metrics after each run.")
# Constants
years = list(range(2026, 2036))
num_years = len(years)
# initial_total_bikes = 7_000_000
# initial_ev_share = 0.05
# avg_km_year = 6000
# fuel_per_km = 0.02
# elec_per_km = 0.03
# grid_emission_std = 0.02

# Output structure
def summarize(x):
    return {
        "mean": x.mean(axis=0),
        "p10": np.percentile(x, 10, axis=0),
        "p90": np.percentile(x, 90, axis=0)
    }

results = {
    "ev_share": np.zeros((num_simulations, num_years)),
    "ev_costs": np.zeros((num_simulations, num_years)),
    "ev_costs_noshock": np.zeros((num_simulations, num_years)),
    "emissions": np.zeros((num_simulations, num_years)),
    "people_cost": np.zeros((num_simulations, num_years)),
    "gov_cost": np.zeros((num_simulations, num_years)),
    "shock_multiplier": np.zeros((num_simulations, num_years))
}

# Prepare deterministic cost walk for price elasticity if needed
ev_cost_deterministic = [ev_cost_init * (1 - cost_decline_rate) ** i for i in range(num_years)]
ev_cost_prev_global = ev_cost_init  # Reset at the beginning of each simulation
np.random.seed(seed_value)

for sim in range(num_simulations):
    if sim == 0:
        results["fuel_price"] = np.zeros((num_simulations, num_years))
        results["elec_price"] = np.zeros((num_simulations, num_years))
    ice = initial_total_bikes * (1 - initial_ev_share)
    ev = initial_total_bikes * initial_ev_share
    ev_history = deque(maxlen=retirement_age)
    ice_history = deque(maxlen=retirement_age)
    ev_history.append(ev)
    ice_history.append(ice)

    ev_cost_prev = ev_cost_prev_global
    external_cost_shocks = np.zeros(num_years)
    fuel_prices = np.zeros(num_years)
    elec_prices = np.zeros(num_years)


    
    fuel_prices[0] = np.random.normal(fuel_price_init, fuel_price_vol)
    elec_prices[0] = np.random.normal(elec_price_init, elec_price_vol)
    external_cost_shocks[0] = np.random.normal(0, shock_volatility)
    for t in range(1, num_years):
        external_cost_shocks[t] = 0.8 * external_cost_shocks[t - 1] + np.random.normal(0, shock_volatility)
        fuel_prices[t] = fuel_prices[t-1] * (1 + fuel_price_trend) + np.random.normal(0, fuel_price_vol)
        elec_prices[t] = elec_prices[t-1] * (1 + elec_price_trend) + np.random.normal(0, elec_price_vol)

        # Prevent negative prices
        fuel_prices[t] = max(0.01, fuel_prices[t])
        elec_prices[t] = max(0.005, elec_prices[t])

    for y_idx, year in enumerate(years):
        # Fleet growth
        total_fleet = (ice + ev) * (1 + fleet_growth_rate)
        new_bikes = total_fleet - (ice + ev)
        ice += new_bikes

        # Retirement
        if y_idx >= retirement_age:
            ice -= ice_history[0]
            ev -= ev_history[0]
        # Instead of fixed:
        # fuel_per_km
        # elec_per_km
        # grid_emission_std

        fuel_per_km = np.random.normal(fuel_per_km_mean, fuel_per_km_std)
        elec_per_km = np.random.normal(elec_per_km_mean, elec_per_km_std)
        grid_emission_std = max(0.001, np.random.normal(grid_emission_std_mean, grid_emission_std_std))

        # Cost and shock

        
        expected_cost = ev_cost_init * ((1 - cost_decline_rate) ** (year - 2025))
        shock_multiplier = 1 + import_dependency * external_cost_shocks[y_idx]
        ev_cost = expected_cost * shock_multiplier

        # Behavior
        year_boost = 1.0
        if year == 2026:
            year_boost = policy_boost_2026
        elif year == 2028:
            year_boost = policy_boost_2028
        # subsidy_effect = 1 - np.exp(-(k_subsidy + 0.5) * (subsidy_amount * subsidized_fraction))
        # 1. Compute EV cost decline effect (more affordable → more likely to convert)
    # --- PRICE EFFECT BASED ON USER CHOICE ---
        if price_elasticity_mode == "Baseline (2026 EV Price)":
            price_effect = np.exp(-k_price * (ev_cost / ev_cost_init))

        elif price_elasticity_mode == "YoY EV Price Change":
            if y_idx > 0:
                price_effect = np.exp(-k_price * ((ev_cost_prev - ev_cost) / ev_cost_prev))
            else:
                price_effect = 1.0
            ev_cost_prev = ev_cost  # Save for next year

        elif price_elasticity_mode == "Deviation from Deterministic Price":
            expected_cost_baseline = ev_cost_deterministic[y_idx]
            price_effect = np.exp(-k_price * (ev_cost / expected_cost_baseline))

        price_effect = np.clip(price_effect, 0.1, 2.0)


        # 2. Keep existing subsidy behavioral boost
        subsidy_effect = 1 - np.exp(-(k_subsidy + 0.5) * (subsidy_amount * subsidized_fraction))

        # 3. Combine both effects
        behavior_multiplier = 1 + subsidy_effect * price_effect


        if adoption_model == "S-curve":
            frac = 1 / (1 + np.exp(-k_s_curve * (year - s_midpoint)))
        else:
            frac = np.random.normal(loc=base_linear_rate, scale=linear_std_dev)
            frac = max(0.0, min(0.5, frac))  # clamp

            if year == 2026:
                frac *= policy_boost_2026
            elif year == 2028:
                frac *= policy_boost_2028

            frac = min(frac, 0.5)  # ensure upper bound after boost

        convert = ice * frac * behavior_multiplier
        ice -= convert
        ev += convert

        # History tracking
        ice_history.append(convert + new_bikes - ice_history[0] if len(ice_history) == retirement_age else convert + new_bikes)
        ev_history.append(convert)

        # Emissions
        grid_emission = max(0.01, np.random.normal(grid_emission_mean * ((1 - grid_emission_decline) ** (year - 2025)), grid_emission_std))
        emissions_baseline = initial_total_bikes * avg_km_year * fuel_per_km * fuel_emission_factor / 1000
        emissions_current = (
            ice * avg_km_year * fuel_per_km * fuel_emission_factor +
            ev * avg_km_year * elec_per_km * grid_emission
        ) / 1000
        avoided = emissions_baseline - emissions_current

        # Cost components
        fuel_price = fuel_prices[y_idx]
        elec_price = elec_prices[y_idx]
        salvage = salvage_value_pct* ev_cost * convert
        subsidy = subsidy_amount * ev_cost * subsidized_fraction if year <= subsidy_end_year else 0.0

        fuel_cost = convert * avg_km_year * fuel_per_km * fuel_price
        elec_cost = convert * avg_km_year * elec_per_km * elec_price
        net_ev_cost = ev_cost * convert - salvage

        people_cost = net_ev_cost - (fuel_cost - elec_cost)
        gov_cost = -subsidy * convert

        # Store results
        results["fuel_price"][sim] = fuel_prices
        results["elec_price"][sim] = elec_prices
        results["ev_share"][sim, y_idx] = ev / (ice + ev)
        results["ev_costs"][sim, y_idx] = ev_cost
        results["ev_costs_noshock"][sim, y_idx] = expected_cost
        results["emissions"][sim, y_idx] = avoided

        results["people_cost"][sim, y_idx] = people_cost
        results["gov_cost"][sim, y_idx] = gov_cost
        results["shock_multiplier"][sim, y_idx] = shock_multiplier
    results["people_cost"][sim] = np.cumsum(results["people_cost"][sim])
    results["gov_cost"][sim] = np.cumsum(results["gov_cost"][sim])
# --- Metrics and Plotting ---
def plot_metric(label, data, xvals):
    y_labels = {
        "EV Share (%)": "Percentage of Total Fleet",
        "EV Cost (USD)": "EV Price (USD)",
        "Avoided Emissions (tons)": "CO₂ Avoided (tons)",
        "People Cost (USD)": "Net Cost to Consumers (USD)",
        "Government Cost (USD)": "Net Cost to Government (USD)"
    }
    hover = "%{x}: %{y:.2f}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xvals, y=data["mean"], name="Mean", line=dict(color="darkblue"),
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=xvals, y=data["p10"], name="P10", line=dict(color="lightblue", dash="dot"),
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=xvals, y=data["p90"], name="P90", line=dict(color="lightblue", dash="dot"),
                             hovertemplate=hover, fill="tonexty", fillcolor="rgba(173,216,230,0.2)"))
    fig.update_layout(title=label, xaxis_title="Year", yaxis_title=y_labels.get(label, "Value"), height=420)
    return fig

st.markdown("## 📊 Simulation Dashboard")

metrics = {
    "EV Share (%)": summarize(results["ev_share"] * 100),
    "EV Cost (USD)": summarize(results["ev_costs"]),
    "Avoided Emissions (tons)": summarize(results["emissions"]),
    "People Cost (USD)": summarize(results["people_cost"]),
    "Government Cost (USD)": summarize(results["gov_cost"])
}

tabs = st.tabs(list(metrics.keys()))
for i, key in enumerate(metrics):
    with tabs[i]:
        st.plotly_chart(plot_metric(key, metrics[key], years), use_container_width=True)
        st.markdown(f"**2035 P50:** {metrics[key]['mean'][-1]:,.2f}")
        st.markdown(f"**2035 P10:** {metrics[key]['p10'][-1]:,.2f}")
        st.markdown(f"**2035 P90:** {metrics[key]['p90'][-1]:,.2f}")

# --- YoY Delta Table (% Change) ---
st.markdown("## 📈 Year-over-Year % Change Table (Mean Values)")

deltas_pct = {}
for key, data in metrics.items():
    prev_years = np.array(data["mean"][:-1])
    curr_years = np.array(data["mean"][1:])
    delta_pct = (curr_years - prev_years) / np.where(prev_years == 0, 1, prev_years) * 100
    deltas_pct[key] = delta_pct

df_deltas_pct = pd.DataFrame(deltas_pct, index=years[1:])

# Format with softer colors and no gmap
st.dataframe(
    df_deltas_pct.style
        .format("{:+.1f}%")
        .background_gradient(axis=0, cmap="coolwarm", low=0.2, high=0.8),
    height=320
)


# --- Shock Comparison Plot ---
with st.expander("🔀 EV Price: With vs Without Shock + Uncertainty Bands"):
    ev_cost_with_shock = summarize(results["ev_costs"])
    ev_cost_noshock = summarize(results["ev_costs_noshock"])
    hover = "%{x}: %{y:.2f}"

    fig = go.Figure()

    # With Shock
    fig.add_trace(go.Scatter(x=years, y=ev_cost_with_shock["mean"], name="With Shock – Mean", line=dict(color="black"), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=years, y=ev_cost_with_shock["p90"], name="With Shock – P90", line=dict(color="lightblue", dash="dot"), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=years, y=ev_cost_with_shock["p10"], name="With Shock – P10", line=dict(color="lightblue", dash="dot"), hovertemplate=hover,
                             fill="tonexty", fillcolor="rgba(173,216,230,0.3)"))

    # No Shock
    fig.add_trace(go.Scatter(x=years, y=ev_cost_noshock["mean"], name="No Shock – Mean", line=dict(color="green", dash="dash"), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=years, y=ev_cost_noshock["p90"], name="No Shock – P90", line=dict(color="lightgreen", dash="dot"), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=years, y=ev_cost_noshock["p10"], name="No Shock – P10", line=dict(color="lightgreen", dash="dot"), hovertemplate=hover,
                             fill="tonexty", fillcolor="rgba(144,238,144,0.3)"))

    fig.update_layout(title="EV Cost: With vs Without Shock + Uncertainty Bands",
                      xaxis_title="Year", yaxis_title="EV Cost (USD)", height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- Fuel & Electricity Price Bands ---
with st.expander("⛽ Fuel & Electricity Price Bands"):
    fuel_price_summary = summarize(results["fuel_price"])
    elec_price_summary = summarize(results["elec_price"])
    hover = "%{x}: %{y:.2f}"

    fig3 = go.Figure()

    # Fuel Price Traces
    fig3.add_trace(go.Scatter(x=years, y=fuel_price_summary["mean"], name="Fuel Price – Mean",
                              line=dict(color="brown"), hovertemplate=hover))
    fig3.add_trace(go.Scatter(x=years, y=fuel_price_summary["p10"], name="Fuel Price – P10",
                              line=dict(color="sienna", dash="dot"), hovertemplate=hover))
    fig3.add_trace(go.Scatter(x=years, y=fuel_price_summary["p90"], name="Fuel Price – P90",
                              line=dict(color="sienna", dash="dot"), hovertemplate=hover,
                              fill="tonexty", fillcolor="rgba(205,133,63,0.2)"))

    # Electricity Price Traces
    fig3.add_trace(go.Scatter(x=years, y=elec_price_summary["mean"], name="Electricity Price – Mean",
                              line=dict(color="green"), hovertemplate=hover))
    fig3.add_trace(go.Scatter(x=years, y=elec_price_summary["p10"], name="Electricity Price – P10",
                              line=dict(color="limegreen", dash="dot"), hovertemplate=hover))
    fig3.add_trace(go.Scatter(x=years, y=elec_price_summary["p90"], name="Electricity Price – P90",
                              line=dict(color="limegreen", dash="dot"), hovertemplate=hover,
                              fill="tonexty", fillcolor="rgba(144,238,144,0.2)"))

    fig3.update_layout(
        title="Fuel & Electricity Prices – P10/P50/P90",
        xaxis_title="Year",
        yaxis_title="Price (USD)",
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)


# --- Shock Multiplier Plot ---
with st.expander("📈 Shock Multiplier Bands"):
    shock_summary = summarize(results["shock_multiplier"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=years, y=shock_summary["mean"], name="Mean", line=dict(color="blue")))
    fig2.add_trace(go.Scatter(x=years, y=shock_summary["p10"], name="P10", line=dict(color="lightblue", dash="dot")))
    fig2.add_trace(go.Scatter(x=years, y=shock_summary["p90"], name="P90", line=dict(color="lightblue", dash="dot"),
                              fill="tonexty", fillcolor="rgba(173,216,230,0.2)"))
    fig2.update_layout(title="Shock Multiplier (AR(1))", xaxis_title="Year", yaxis_title="x Multiplier")
    st.plotly_chart(fig2, use_container_width=True)

# --- State Initialization for Comparison ---
if "previous_metrics" not in st.session_state:
    st.session_state.previous_metrics = None
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = None
if "snapshot_history" not in st.session_state:
    st.session_state.snapshot_history = []

# --- Save Current Metrics After Simulation ---
st.session_state.previous_metrics = st.session_state.current_metrics
st.session_state.current_metrics = metrics

st.session_state.snapshot_history.append({
    "params": {
        "subsidy_amount": subsidy_amount,
        "subsidy_end_year": subsidy_end_year,
        "fleet_growth_rate": fleet_growth_rate,
        "retirement_age": retirement_age,
        "adoption_model": adoption_model
    },
    "metrics": metrics
})
if len(st.session_state.snapshot_history) > 5:
    st.session_state.snapshot_history.pop(0)

# --- Year-by-Year % Change in P50 (Median) from Previous Run (Years as Rows) ---
st.markdown("## 📈 Year-by-Year % Change from Previous Run (Median Only)")

if st.session_state.previous_metrics:
    year_labels = [str(y) for y in years]
    yoy_data = []

    # Track all metric % changes year-by-year
    for year_idx, y in enumerate(years):
        row = {}
        for key in metrics:
            prev_val = st.session_state.previous_metrics[key]["mean"][year_idx]
            curr_val = st.session_state.current_metrics[key]["mean"][year_idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = (curr_val - prev_val) / prev_val * 100 if prev_val != 0 else 0
            row[key] = pct
        yoy_data.append(row)

    df_yoy = pd.DataFrame(yoy_data, index=year_labels)

    # --- Show only changed parameters (compare with previous run) ---
    if len(st.session_state.snapshot_history) >= 2:
        current_params = st.session_state.snapshot_history[-1]["params"]
        previous_params = st.session_state.snapshot_history[-2]["params"]
        changed_params = {k: v for k, v in current_params.items() if previous_params.get(k) != v}

        if changed_params:
            st.markdown("### 🛠️ Parameters Changed in This Run")
            for k, v in changed_params.items():
                st.markdown(f"- **{k.replace('_', ' ').capitalize()}**: `{v}`")
        else:
            st.info("No input parameters changed since last run.")

    # --- Display YoY % Change Table ---
    st.dataframe(
        df_yoy.style.format("{:+.1f}%").background_gradient(axis=0, cmap="RdYlBu_r", low=0.2, high=0.8),
        use_container_width=True
    )
else:
    st.info("Run the simulation at least twice to compare results.")

# --- Snapshot History ---
st.markdown("## 🧾 Snapshot History (last 5 runs)")
for i, snap in enumerate(reversed(st.session_state.snapshot_history)):
    st.markdown(f"### Run #{len(st.session_state.snapshot_history) - i}")
    with st.expander("Parameters & Results", expanded=False):
        st.write("**Parameters:**", snap["params"])
        for key, val in snap["metrics"].items():
            st.write(f"**{key} (P50 2035):** {val['mean'][-1]:.2f}")