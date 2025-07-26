"""
Streamlit application for simulating Hanoi’s internal combustion engine (ICE) motorcycle ban.

This version introduces a few enhancements:

1. **Carbon cost slider** – a user‑configurable price of CO₂ emissions in USD per ton.  This value is
   used to translate the avoided emissions from replacing ICE motorcycles with electric vehicles into
   avoided carbon costs.  A new results key (``co2_cost_avoided``) stores these values by
   simulation and year.
2. **Reorganised dashboard** – the EV cost plot has been moved into its own expander so that the
   primary metrics tabs can instead highlight avoided emissions costs and health benefits.  A new tab
   summarises the health cost avoided due to the reduction in air pollution.  The CO₂ cost
   avoided also appears as a dedicated tab.
3. **Sanity checks** – comments and clean variable names help ensure that the cost calculations and
   units remain consistent throughout the simulation.

The code below is largely the same as the baseline provided by the user but with the above
enhancements applied.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Hanoi ICE Motorcycle Ban Simulation")
st.title("🚦 Hanoi ICE Motorcycle Ban – Monte Carlo Simulation (2026–2035)")

# --- Reset Button ---
if st.sidebar.button("🔁 Reset to Defaults"):
    # Clear the session state so that sliders reset to default values
    st.session_state.clear()

# === Sidebar UI Organization ===

# Section 1: 📋 Simulation Settings
st.sidebar.markdown("### 📋 Simulation Settings")
with st.sidebar.expander("Simulation Controls"):
    num_simulations = st.slider(
        "🔄 Number of Simulations", 100, 3000, 2000,
        help="Number of Monte Carlo runs to simulate different future scenarios."
    )
    fleet_growth_rate = st.slider(
        "📈 Motorcycle Fleet Growth (Annual)", 0.0, 0.05, 0.02, step=0.005,
        help="Annual increase in total motorcycle fleet size, reflecting population or economic growth."
    )
    adoption_model = st.selectbox(
        "🔀 EV Adoption Model", ["Linear", "S-curve"],
        help="Determines how EV adoption progresses over time: constant rate (Linear) or accelerating (S‑curve)."
    )
    seed_value = st.slider(
        "🔢 Random Seed", 0, 10_000, 42,
        help="Fixes the seed to ensure reproducibility of simulation outcomes."
    )

# Section 2: 🧮 Starting Conditions & Input Distributions
st.sidebar.markdown("### 🚗 Fleet & Usage Characteristics")
with st.sidebar.expander("Fleet & Usage Characteristics"):
    initial_total_bikes = st.slider(
        "🛵 Initial Total Motorcycles", 1_000_000, 10_000_000, 7_000_000, step=100_000,
        help="Motorcycle fleet size at the start of the simulation (2026)."
    )
    initial_ev_share = st.slider(
        "⚡ Starting EV Share (%)", 0.0, 1.0, 0.05, step=0.01,
        help="Initial proportion of electric motorcycles in 2026."
    )
    avg_km_year = st.slider(
        "🛣️ Avg. Distance per Motorcycle (km/year)", 3000, 15000, 6000, step=500,
        help="Average number of kilometres travelled annually per motorcycle."
    )

    # Fuel consumption distribution
    st.markdown("**⛽ Fuel Use per km (L/km)**")
    col1, col2 = st.columns(2)
    with col1:
        fuel_per_km_mean = st.slider(
            "Mean fuel/km", 0.01, 0.05, 0.02, step=0.001,
            help="Average litres of gasoline consumed per km by ICE motorcycles."
        )
    with col2:
        fuel_per_km_std = st.slider(
            "Std Dev fuel/km", 0.0, 0.01, 0.002, step=0.0005,
            help="Volatility or uncertainty around average fuel consumption."
        )

    # Electricity consumption distribution
    st.markdown("**🔌 Electricity Use per km (kWh/km)**")
    col3, col4 = st.columns(2)
    with col3:
        elec_per_km_mean = st.slider(
            "Mean elec/km", 0.01, 0.05, 0.03, step=0.001,
            help="Average electricity consumption per km by EV motorcycles."
        )
    with col4:
        elec_per_km_std = st.slider(
            "Std Dev elec/km", 0.0, 0.01, 0.002, step=0.0005,
            help="Volatility or uncertainty around electricity consumption."
        )


# Section 3: 💸 EV Price & Operating Costs
st.sidebar.markdown("### 💸 EV Price & Operating Costs")
with st.sidebar.expander("EV Price & Operating Costs"):
    ev_price_mode = st.radio(
        "EV Initial Price Mode", ["Uniform Range","Fixed"], horizontal=True,
        help="Define whether EV prices are fixed or vary within a range for each run."
    )
    if ev_price_mode == "Fixed":
        ev_cost_init = st.slider(
            "💸 Initial EV Price (USD)", 1000, 3000, 1500, step=50,
            help="Price of an EV at simulation start (2026)."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            ev_price_min = st.slider(
                "EV Price Min (USD)", 1000, 3000, 1200, step=50,
                help="Lower bound for EV prices (used in random draw)."
            )
        with col2:
            ev_price_max = st.slider(
                "EV Price Max (USD)", 1000, 3000, 1800, step=50,
                help="Upper bound for EV prices (used in random draw)."
            )
        # When using a range, sample a random price for this run
        ev_cost_init = np.random.uniform(ev_price_min, ev_price_max)

    salvage_value_pct = st.slider(
        "ICE Salvage Value at End (% of EV price)", 0.0, 1.0, 0.1, step=0.05,
        help="Estimated resale value of ICEs as percent of EV price at end‑of‑life."
    )

    # Fuel and electricity price dynamics
    st.markdown("**⛽ Fuel and Electricity Prices**")
    col1, col2 = st.columns(2)
    with col1:
        fuel_price_init = st.slider(
            "Initial Fuel Price (USD/L)", 0.5, 2.0, 0.9, step=0.05,
            help="Fuel price at simulation start."
        )
        fuel_price_trend = st.slider(
            "Fuel Price Trend (%/year)", -0.2, 0.2, 0.01, step=0.01,
            help="Annual percentage change in fuel prices."
        )
        fuel_price_vol = st.slider(
            "Fuel Price Volatility", 0.0, 0.2, 0.05, step=0.01,
            help="Yearly fuel price volatility due to shocks."
        )
    with col2:
        elec_price_init = st.slider(
            "Initial Elec Price (USD/kWh)", 0.05, 0.5, 0.1, step=0.01,
            help="Electricity price at simulation start (base for EV fuel)."
        )
        elec_price_trend = st.slider(
            "Elec Price Trend (%/year)", -0.2, 0.2, 0.005, step=0.005,
            help="Annual percentage change in electricity prices."
        )
        elec_price_vol = st.slider(
            "Elec Price Volatility", 0.0, 0.2, 0.02, step=0.01,
            help="Yearly electricity price volatility."
        )

    cost_decline_rate = st.slider(
        "📉 Annual EV motorcycle Cost Decline Rate", 0.0, 0.2, 0.05, step=0.01,
        help="Expected yearly cost reduction in EV prices."
    )

# Section: 🌍 Emission & Health Parameters
st.sidebar.markdown("### 🌍 Emission & Health Parameters")
with st.sidebar.expander("Emission & Health Parameters"):
    # Health cost avoided per ICE
    health_cost_per_ice = st.slider(
        "🫁 Health Cost Avoided per ICE (USD/year)", 0, 200, 80, step=5,
        help="Estimated health cost due to one ICE vehicle. Based on OECD and World Bank studies in Southeast Asia."
    )
    # Split of health cost savings between government and individuals
    health_share_gov = st.slider(
        "🏥 Share of Health Savings to Government (%)", 0, 100, 60, step=5,
        help="Portion of avoided public health costs attributed to government expenditure (e.g. public hospitals)."
    ) / 100.0
    health_share_people = 1.0 - health_share_gov

    st.markdown("**🌫️ Emission Variability of Electricity (EV Grid Emissions)**")
    col5, col6 = st.columns(2)
    with col5:
        grid_emission_std_mean = st.slider(
            "Mean grid elec emissions", 0.0, 0.05, 0.02, step=0.001,
            help="Standard deviation of grid carbon intensity across scenarios."
        )
    with col6:
        grid_emission_std_std = st.slider(
            "Volatility grid elec emissions", 0.0, 0.02, 0.005, step=0.001,
            help="Variability in emission uncertainty over time."
        )
    # Emission and health factors
    fuel_emission_factor = st.slider(
        "⛽ Fuel Emissions (kg/L)", 1.5, 3.0, 2.3, step=0.05,
        help="Carbon emissions produced per litre of gasoline."
    )
    grid_emission_mean = st.slider(
        "🌍 Grid Emission Intensity (kg/kWh)", 0.05, 0.5, 0.1, step=0.01,
        help="Emissions per kWh of electricity."
    )
    grid_emission_decline = st.slider(
        "📉 Grid Emission Decline Rate", 0.0, 0.1, 0.03, step=0.01,
        help="Annual decline in grid emissions due to renewable integration."
    )

# Section: 💨 Carbon Pricing
st.sidebar.markdown("### 💨 Carbon Pricing")
with st.sidebar.expander("Carbon Pricing : Internalization of Emssion costs"):
    co2_price_mode = st.radio(
        "Carbon Price Mode",
        ["Random Walk","Fixed"], horizontal=True,
        help="Choose a fixed carbon price over the simulation or allow it to follow a random walk."
    )
    if co2_price_mode == "Fixed":
        co2_cost_per_ton = st.slider(
            "CO₂ Cost (USD/ton)", 0.0, 200.0, 50.0, step=5.0,
            help="Price per tonne of CO₂ emissions, used to value avoided emissions."
        )
        co2_price_start = None
        co2_price_trend = None
        co2_price_volatility = None
    else:
        co2_price_start = st.slider(
            "Starting CO₂ Price (USD/ton)", 0.0, 200.0, 50.0, step=5.0,
            help="Initial carbon price for the random walk."
        )
        co2_price_trend = st.slider(
            "CO₂ Price Trend (%/year)", -0.2, 0.2, 0.0, step=0.01,
            help="Annual percentage change in the carbon price."
        )
        co2_price_volatility = st.slider(
            "CO₂ Price Volatility", co2_price_start*0.01, co2_price_start*0.5, co2_price_start*0.1, step=co2_price_start*0.03,
            help="Year-to-year randomness in the carbon price."
        )
        co2_cost_per_ton = None

# Section 4: 🧠 Behavioral Response to Policy & Price
st.sidebar.markdown("### 🧠 Behavioral Response to Policy & Price")
with st.sidebar.expander("Subsidies and Perception of Cost"):
    subsidy_amount = st.slider(
        "🎁 Subsidy (% of EV Price)", 0.0, 1.0, 0.2, step=0.01,
        help="Government support expressed as a % of EV price."
    )
    subsidy_end_year = st.slider(
        "📅 Subsidy End Year", 2026, 2035, 2029,
        help="The final year in which subsidies are applied."
    )
    subsidized_fraction = st.slider(
        "📊 Fraction of Fleet Receiving Subsidy", 0.0, 1.0, 0.5, step=0.01,
        help="Share of new EV buyers eligible for subsidies."
    )
    k_subsidy = st.slider(
        "📈 Behavioral Sensitivity to Subsidy", 0.0, 1.0, 0.1, step=0.01,
        help="Impact of subsidy on increasing EV adoption likelihood. (High ==> Individual propension to switch due do subsidiary; we assume that at some point the subsidy cannot overcome some obstacles)"
    )
    k_price = st.slider(
        "🧮 Price Sensitivity to EV Cost", 0.0, 10.0, 3.0, step=0.1,
        help="Elasticity of EV adoption with respect to upfront price (Related to external Shocks)."
    )
    price_elasticity_mode = st.selectbox(
        "🧠 Price Elasticity Reference Frame",
        ["Deviation from Deterministic Price","YoY EV Price Change","Baseline (2026 EV Price)"],
        help="Determines the baseline for calculating price responsiveness to external shocks."
    )

# Section 5: 📊 Adoption Curve Calibration
st.sidebar.markdown("### 📊 Adoption Curve Calibration")
with st.sidebar.expander("Curve Dynamics"):
    base_linear_rate = st.slider(
        "🔁 Base Linear Conversion Rate", 0.01, 0.3, 0.05, step=0.01,
        help="Annual fixed % of remaining ICE motorcycles converted to EVs under the Linear model. (Natural asymptotic behavior when close to 100%)"
    )
    linear_std_dev = st.slider(
        "📉 Volatility in Linear Conversion", 0.0, 0.1, 0.02, step=0.005,
        help="Random fluctuation in yearly conversion rate for the Linear model."
    )
    policy_boost_2026 = st.slider(
        "🚀 Policy Conversion Boost Linear Scenario (2026)", 1.0, 3.0, 1.5, step=0.1,
        help="Temporary boost factor to EV adoption rate in 2026 due to launch-year incentives."
    )
    policy_boost_2028 = st.slider(
        "🚀 Policy Conversion Boost Linear Scenario(2028)", 1.0, 3.0, 1.3, step=0.1,
        help="Temporary boost factor to EV adoption rate in 2028 due to expanded policies."
    )
    # Maximum fraction of ICE fleet that can convert to EVs per year under the S‑curve model
    s_curve_max_fraction = st.slider(
        "🚫 Max S-curve Conversion Fraction", 0.05, 0.5, 0.3, step=0.01,
        help="Maximum fraction of the remaining ICE fleet that can convert to EVs in any year under the S-curve model."
    )
    k_s_curve = st.slider(
        "📈 S-Curve Steepness", 0.1, 2.0, 0.6, step=0.1,
        help="Controls steepness of the S-curve: higher values mean faster mid‑phase adoption."
    )
    s_midpoint = st.slider(
        "⏳ S-Curve Midpoint Year", 2026, 2035, 2029, step=1,
        help="Year at which the S‑curve reaches 50% of the total conversion."
    )


# Section 6: 🌍 Global Supply Chain Shocks
st.sidebar.markdown("### 🌍 Global Supply Chain Shocks")
with st.sidebar.expander("Cost Volatility"):
    import_dependency = st.slider(
        "🔗 Import Dependency Level", 0.0, 1.0, 0.5, step=0.05,
        help="Degree to which EV prices depend on imported parts (e.g., batteries, chips)."
    )
    shock_volatility = st.slider(
        "📉 Shock Volatility", 0.0, 0.5, 0.1, step=0.01,
        help="Year-to-year variation in EV price due to global supply chain disruptions or tariff shocks."
    )

# Section: ✅ Show Summary Metrics
show_metrics = st.sidebar.checkbox(
    "📊 Show Summary Metrics", True,
    help="Display key simulation metrics after each run."
)

# --- Simulation Constants ---
years = list(range(2026, 2036))
num_years = len(years)

# Initialize results data structure.  Each key holds an array of shape
# (num_simulations, num_years).  New keys can be added here so that the
# structure is consistent throughout the simulation loop.
results = {
    "ev_share": np.zeros((num_simulations, num_years)),
    "ev_costs": np.zeros((num_simulations, num_years)),
    "ev_costs_noshock": np.zeros((num_simulations, num_years)),
    "emissions": np.zeros((num_simulations, num_years)),
    "people_cost": np.zeros((num_simulations, num_years)),
    "gov_cost": np.zeros((num_simulations, num_years)),
    "shock_multiplier": np.zeros((num_simulations, num_years)),
    "health_cost avoided this year through conversion": np.zeros((num_simulations, num_years)),
    "gov_health": np.zeros((num_simulations, num_years)),
    "people_health": np.zeros((num_simulations, num_years)),
    "co2_cost_avoided": np.zeros((num_simulations, num_years)),
    # Store the carbon price path for each simulation (for diagnostics)
    "co2_price": np.zeros((num_simulations, num_years)),
}

# Pre‑compute the deterministic EV cost path for the price elasticity modes
ev_cost_deterministic = [ev_cost_init * (1 - cost_decline_rate) ** i for i in range(num_years)]
ev_cost_prev_global = ev_cost_init  # Used when computing year‑on‑year price changes
np.random.seed(seed_value)

# --- Capture all current inputs into a dictionary for snapshot history ---
tracked_inputs = {
    # Simulation Controls
    "num_simulations": num_simulations,
    "fleet_growth_rate": fleet_growth_rate,
    "adoption_model": adoption_model,
    "seed_value": seed_value,
    # Starting Conditions
    "initial_total_bikes": initial_total_bikes,
    "initial_ev_share": initial_ev_share,
    "avg_km_year": avg_km_year,
    "fuel_per_km_mean": fuel_per_km_mean,
    "fuel_per_km_std": fuel_per_km_std,
    "elec_per_km_mean": elec_per_km_mean,
    "elec_per_km_std": elec_per_km_std,
    "grid_emission_std_mean": grid_emission_std_mean,
    "grid_emission_std_std": grid_emission_std_std,
    # Cost Trends
    "ev_price_mode": ev_price_mode,
    "ev_cost_init": ev_cost_init,
    "salvage_value_pct": salvage_value_pct,
    "fuel_price_init": fuel_price_init,
    "fuel_price_trend": fuel_price_trend,
    "fuel_price_vol": fuel_price_vol,
    "elec_price_init": elec_price_init,
    "elec_price_trend": elec_price_trend,
    "elec_price_vol": elec_price_vol,
    "cost_decline_rate": cost_decline_rate,
    "fuel_emission_factor": fuel_emission_factor,
    "grid_emission_mean": grid_emission_mean,
    "grid_emission_decline": grid_emission_decline,
    "health_cost_per_ice": health_cost_per_ice,
    "health_share_gov": health_share_gov,
    # Carbon pricing inputs
    "co2_price_mode": co2_price_mode,
    "co2_cost_per_ton": co2_cost_per_ton,
    "co2_price_start": co2_price_start,
    "co2_price_trend": co2_price_trend,
    "co2_price_volatility": co2_price_volatility,
    # Behavioral Response
    "subsidy_amount": subsidy_amount,
    "subsidy_end_year": subsidy_end_year,
    "subsidized_fraction": subsidized_fraction,
    "k_subsidy": k_subsidy,
    "k_price": k_price,
    "price_elasticity_mode": price_elasticity_mode,
    # Adoption Curve
    "base_linear_rate": base_linear_rate,
    "linear_std_dev": linear_std_dev,
    "k_s_curve": k_s_curve,
    "s_midpoint": s_midpoint,
    "policy_boost_2026": policy_boost_2026,
    "policy_boost_2028": policy_boost_2028,
    "s_curve_max_fraction": s_curve_max_fraction,
    # Global Shocks
    "import_dependency": import_dependency,
    "shock_volatility": shock_volatility,
    # Display Flag
    "show_metrics": show_metrics,
}

# --- Run the simulations ---
for sim in range(num_simulations):
    # Initialise arrays for this simulation on first pass
    if sim == 0:
        results["fuel_price"] = np.zeros((num_simulations, num_years))
        results["elec_price"] = np.zeros((num_simulations, num_years))

    # Starting fleet composition
    ice = initial_total_bikes * (1 - initial_ev_share)
    ev = initial_total_bikes * initial_ev_share

    # Set up variables for price and shock random walks
    external_cost_shocks = np.zeros(num_years)
    fuel_prices = np.zeros(num_years)
    elec_prices = np.zeros(num_years)
    # Prepare an array for carbon prices; constant or random walk depending on user choice
    co2_prices = np.zeros(num_years)

    # Initialise year‑0 prices with randomness
    fuel_prices[0] = np.random.normal(fuel_price_init, fuel_price_vol)
    elec_prices[0] = np.random.normal(elec_price_init, elec_price_vol)
    external_cost_shocks[0] = np.random.normal(0, shock_volatility)

    # Generate price and shock series using AR(1) processes
    for t in range(1, num_years):
        external_cost_shocks[t] = 0.8 * external_cost_shocks[t - 1] + np.random.normal(0, shock_volatility)
        fuel_prices[t] = fuel_prices[t - 1] * (1 + fuel_price_trend) + np.random.normal(0, fuel_price_vol)
        elec_prices[t] = elec_prices[t - 1] * (1 + elec_price_trend) + np.random.normal(0, elec_price_vol)
        # Prevent negative prices
        fuel_prices[t] = max(0.01, fuel_prices[t])
        elec_prices[t] = max(0.005, elec_prices[t])

    # Generate carbon price series. If the user selects a fixed price the series is constant,
    # otherwise it follows a random walk with user‑specified trend and volatility.
    if co2_price_mode == "Fixed":
        co2_prices[:] = co2_cost_per_ton if co2_cost_per_ton is not None else 0.0
    else:
        # Start with a random draw around the starting price and ensure non‑negative
        co2_prices[0] = max(0.0, np.random.normal(co2_price_start, co2_price_volatility))
        for t in range(1, num_years):
            co2_prices[t] = co2_prices[t - 1] * (1 + co2_price_trend) + np.random.normal(0, co2_price_volatility)
            co2_prices[t] = max(0.0, co2_prices[t])

    # Iterate over the years
    for y_idx, year in enumerate(years):
        # Fleet growth: assume new bikes added are ICE by default
        total_fleet = (ice + ev) * (1 + fleet_growth_rate)
        new_bikes = total_fleet - (ice + ev)
        ice += new_bikes
        ev_share_year= ev/(ice + ev)

        new_ev = new_bikes * ev_share_year
        new_ice = new_bikes * (1 - ev_share_year)
        ev += new_ev
        ice += new_ice

        # Draw consumption values from their respective distributions
        fuel_per_km = np.random.normal(fuel_per_km_mean, fuel_per_km_std)
        elec_per_km = np.random.normal(elec_per_km_mean, elec_per_km_std)
        grid_emission_std = max(0.001, np.random.normal(grid_emission_std_mean, grid_emission_std_std))

        # Expected EV cost path without shocks
        expected_cost = ev_cost_init * ((1 - cost_decline_rate) ** (year - 2025))
        # Apply AR(1) shock multiplier to EV cost for this year
        shock_multiplier = 1 + import_dependency * external_cost_shocks[y_idx]
        ev_cost = expected_cost * shock_multiplier

        # Compute behavioural multipliers (subsidy and price effects)
        year_boost = 1.0
        if year == 2026:
            year_boost = policy_boost_2026
        elif year == 2028:
            year_boost = policy_boost_2028

        # Price elasticity modes
        if price_elasticity_mode == "Baseline (2026 EV Price)":
            price_effect = np.exp(-k_price * (ev_cost / ev_cost_init))
        elif price_elasticity_mode == "YoY EV Price Change":
            if y_idx > 0:
                price_effect = np.exp(-k_price * ((ev_cost_prev_global - ev_cost) / ev_cost_prev_global))
            else:
                price_effect = 1.0
            ev_cost_prev_global = ev_cost  # Update for next iteration
        elif price_elasticity_mode == "Deviation from Deterministic Price":
            expected_cost_baseline = ev_cost_deterministic[y_idx]
            price_effect = np.exp(-k_price * (ev_cost / expected_cost_baseline))
        else:
            price_effect = 1.0
        price_effect = np.clip(price_effect, 0.1, 2.0)

        # Subsidy effect – combine subsidy rate and behavioural sensitivity
        subsidy_effect = 1 - np.exp(-(k_subsidy + 0.5) * (subsidy_amount * subsidized_fraction))
        # Behavioural multiplier
        behavior_multiplier = 1 + subsidy_effect * price_effect

        # Adoption dynamics
        if adoption_model == "S-curve":
            # Limit the S-curve so that the annual conversion fraction never exceeds s_curve_max_fraction
            frac = s_curve_max_fraction * (1 / (1 + np.exp(-k_s_curve * (year - s_midpoint))))
        else:
            frac = np.random.normal(loc=base_linear_rate, scale=linear_std_dev)
            frac = max(0.0, min(0.5, frac))
            if year == 2026:
                frac *= policy_boost_2026
            elif year == 2028:
                frac *= policy_boost_2028
            frac = min(frac, 0.5)

        # Conversion from ICE to EV this year
        convert = ice * frac * behavior_multiplier
        ice -= convert
        ev += convert
        # EV share of fleet
        ev_share_year = ev / (ice + ev)

        # Health cost avoided is proportional to the number of EVs in the fleet
        health_cost_avoided = (ice + ev) * ev_share_year * health_cost_per_ice
        gov_health_saving = health_cost_avoided * health_share_gov
        people_health_saving = health_cost_avoided * (1 - health_share_gov)

        # Emissions: compute baseline and current and avoidances (in tonnes)
        grid_emission = max(
            0.01,
            np.random.normal(
                grid_emission_mean * ((1 - grid_emission_decline) ** (year - 2025)), grid_emission_std
            )
        )
        emissions_baseline = total_fleet * avg_km_year * fuel_per_km * fuel_emission_factor / 1000.0
        emissions_current = (
            ice * avg_km_year * fuel_per_km * fuel_emission_factor +
            ev * avg_km_year * elec_per_km * grid_emission
        ) / 1000.0
        avoided = emissions_baseline - emissions_current

        # Cost components
        fuel_price = fuel_prices[y_idx]
        elec_price = elec_prices[y_idx]
        salvage = salvage_value_pct * ev_cost * convert
        subsidy = subsidy_amount * ev_cost * subsidized_fraction if year <= subsidy_end_year else 0.0
        fuel_cost = convert * avg_km_year * fuel_per_km * fuel_price
        elec_cost = convert * avg_km_year * elec_per_km * elec_price
        net_ev_cost = ev_cost * convert - salvage
        people_cost = net_ev_cost - (fuel_cost - elec_cost)
        gov_cost = subsidy * convert

        # Carbon cost avoided from emissions reduction
        # Use the appropriate carbon price for the year (fixed or random walk)
        co2_cost_avoided = avoided * co2_prices[y_idx]

        # Store results for this simulation/year
        results["fuel_price"][sim] = fuel_prices
        results["elec_price"][sim] = elec_prices
        results["ev_share"][sim, y_idx] = ev_share_year
        results["ev_costs"][sim, y_idx] = ev_cost
        results["ev_costs_noshock"][sim, y_idx] = expected_cost
        results["emissions"][sim, y_idx] = avoided
        results["people_cost"][sim, y_idx] = people_cost
        results["gov_cost"][sim, y_idx] = gov_cost
        results["shock_multiplier"][sim, y_idx] = shock_multiplier
        results["health_cost avoided this year through conversion"][sim, y_idx] = health_cost_avoided
        results["gov_health"][sim, y_idx] = gov_health_saving
        results["people_health"][sim, y_idx] = people_health_saving
        results["co2_cost_avoided"][sim, y_idx] = co2_cost_avoided

    # Store price series for diagnostic plots
    results["co2_price"][sim] = co2_prices

    # Accumulate people and government costs over years
    results["people_cost"][sim] = np.cumsum(results["people_cost"][sim])
    results["gov_cost"][sim] = np.cumsum(results["gov_cost"][sim])

# --- Metrics and Plotting ---
def plot_metric(label, data, xvals):

    y_labels = {
        "EV Share (%)": "Percentage of Total Fleet",
        "EV Cost (USD)": "EV Price (USD)",
        "EV Enabled Avoided Emissions (tons)": "CO₂ Avoided (tons)",
        "CO₂ Cost Avoided (USD)": "Avoided Carbon Cost (USD)",
        "Health Cost Avoided (USD)": "Avoided Health Cost (USD)",
        "People Direct Cost (USD)": "Net Cost to Consumers (USD)",
        "Government Direct Cost (USD)": "Net Cost to Government (USD)"
    }
    hover = "%{x}: %{y:.2f}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xvals, y=data["mean"], name="Mean", line=dict(color="darkblue"), hovertemplate=hover
    ))
    fig.add_trace(go.Scatter(
        x=xvals, y=data["p10"], name="P10", line=dict(color="lightblue", dash="dot"), hovertemplate=hover
    ))
    fig.add_trace(go.Scatter(
        x=xvals, y=data["p90"], name="P90", line=dict(color="lightblue", dash="dot"), hovertemplate=hover,
        fill="tonexty", fillcolor="rgba(173,216,230,0.2)"
    ))
    fig.update_layout(
        title=label,
        xaxis_title="Year",
        yaxis_title=y_labels.get(label, "Value"),
        height=420
    )
    return fig

# Summarisation helper
def summarise(x: np.ndarray) -> dict:
    """Compute mean and 10th/90th percentiles along axis=0."""
    return {
        "mean": x.mean(axis=0),
        "p10": np.percentile(x, 10, axis=0),
        "p90": np.percentile(x, 90, axis=0)
    }

st.markdown("## 📊 Simulation Dashboard")

# Compute summary statistics for each metric.  Remove the EV cost from the tab list and
# instead store it separately for display in an expander.  Also add carbon cost and health cost
# avoided to the tab metrics.
ev_share_summary = summarise(results["ev_share"] * 100)
ev_cost_summary = summarise(results["ev_costs"])
emissions_summary = summarise(results["emissions"])
co2_cost_summary = summarise(results["co2_cost_avoided"])
health_cost_summary = summarise(results["health_cost avoided this year through conversion"])
people_cost_summary = summarise(results["people_cost"])
gov_cost_summary = summarise(results["gov_cost"])

# Assemble metrics for the tabs (excluding EV cost)
metrics = {
    "EV Share (%)": ev_share_summary,
    "EV Enabled Avoided Emissions (tons)": emissions_summary,
    "CO₂ Cost Avoided (USD)": co2_cost_summary,
    "Health Cost Avoided (USD)": health_cost_summary,
    "People Direct Cost (USD)": people_cost_summary,
    "Government Direct Cost (USD)": gov_cost_summary,
}

# Provide explanatory text for each tab so users understand how each metric is computed.
tab_descriptions = {
    "EV Share (%)": "This metric shows the fraction of the total motorcycle fleet that is electric each year. It is calculated as the number of EVs divided by the total number of bikes (EVs + ICEs) after accounting for conversions and fleet growth.",
    "EV Enabled Avoided Emissions (tons)": "Estimated tonnes of CO₂ avoided each year as ICE motorcycles are replaced by EVs. Baseline emissions assume the entire fleet remains ICE, while current emissions account for EV adoption and the carbon intensity of the electricity grid.",
    "CO₂ Cost Avoided (USD)": "The monetary value of avoided emissions each year, computed by multiplying the avoided tonnes of CO₂ by the carbon price (either fixed or following a random walk).", 
    "Health Cost Avoided (USD)": "Estimated healthcare costs avoided due to fewer pollution-related illnesses. It multiplies the health cost per ICE motorcycle by the number of EVs in the fleet each year.",
    "People Direct Cost (USD)": "Cumulative cost to consumers, combining EV purchase costs (net of salvage) and operational savings from using electricity instead of fuel.",
    "Government Direct Cost (USD)": "Cumulative cost to the government in the form of EV purchase subsidies, applied to the fraction of new EV buyers eligible for subsidy (doesn't account for tax loss on fuel though).",
}

# Create tabs for each metric
tab_names = list(metrics.keys())
tabs = st.tabs(tab_names)
for i, key in enumerate(tab_names):
    with tabs[i]:
        st.plotly_chart(plot_metric(key, metrics[key], years), use_container_width=True)
        st.markdown(f"**2035 Mean:** {metrics[key]['mean'][-1]:,.2f}")
        st.markdown(f"**2035 P10:** {metrics[key]['p10'][-1]:,.2f}")
        st.markdown(f"**2035 P90:** {metrics[key]['p90'][-1]:,.2f}")
        # Explanatory note for this metric
        st.markdown(tab_descriptions.get(key, ""))


# --- Year-over-year % Change Table (Mean Values) ---
st.markdown("## 📈 Year-over-Year % Change Table (Mean Values)")
deltas_pct = {}
for key, data in metrics.items():
    prev_years = np.array(data["mean"][:-1])
    curr_years = np.array(data["mean"][1:])
    delta_pct = (curr_years - prev_years) / np.where(prev_years == 0, 1, prev_years) * 100
    deltas_pct[key] = delta_pct
df_deltas_pct = pd.DataFrame(deltas_pct, index=years[1:])
st.dataframe(
    df_deltas_pct.style
        .format("{:+.1f}%")
        .background_gradient(axis=0, cmap="coolwarm", low=0.2, high=0.8),
    height=320
)

# Present the EV cost evolution in its own expander below the year‑over‑year comparison
with st.expander("💸 EV Cost Over Time", expanded=False):
    st.plotly_chart(plot_metric("EV Cost (USD)", ev_cost_summary, years), use_container_width=True)
    st.markdown(f"**2035 Mean EV Cost:** {ev_cost_summary['mean'][-1]:,.2f}")
    st.markdown(f"**2035 P10 EV Cost:** {ev_cost_summary['p10'][-1]:,.2f}")
    st.markdown(f"**2035 P90 EV Cost:** {ev_cost_summary['p90'][-1]:,.2f}")

# --- Additional diagnostic plots ---
# EV price with vs without shock
with st.expander("🔀 EV Price: With vs Without Shock + Uncertainty Bands"):
    ev_cost_with_shock = summarise(results["ev_costs"])
    ev_cost_noshock = summarise(results["ev_costs_noshock"])
    hover = "%{x}: %{y:.2f}"
    fig = go.Figure()
    # With shock
    fig.add_trace(go.Scatter(
        x=years, y=ev_cost_with_shock["mean"], name="With Shock – Mean", line=dict(color="black"), hovertemplate=hover
    ))
    fig.add_trace(go.Scatter(
        x=years, y=ev_cost_with_shock["p90"], name="With Shock – P90", line=dict(color="lightblue", dash="dot"), hovertemplate=hover
    ))
    fig.add_trace(go.Scatter(
        x=years, y=ev_cost_with_shock["p10"], name="With Shock – P10", line=dict(color="lightblue", dash="dot"), hovertemplate=hover,
        fill="tonexty", fillcolor="rgba(173,216,230,0.3)"
    ))
    # Without shock
    fig.add_trace(go.Scatter(
        x=years, y=ev_cost_noshock["mean"], name="No Shock – Mean", line=dict(color="green", dash="dash"), hovertemplate=hover
    ))
    fig.add_trace(go.Scatter(
        x=years, y=ev_cost_noshock["p90"], name="No Shock – P90", line=dict(color="lightgreen", dash="dot"), hovertemplate=hover
    ))
    fig.add_trace(go.Scatter(
        x=years, y=ev_cost_noshock["p10"], name="No Shock – P10", line=dict(color="lightgreen", dash="dot"), hovertemplate=hover,
        fill="tonexty", fillcolor="rgba(144,238,144,0.3)"
    ))
    fig.update_layout(
        title="EV Cost: With vs Without Shock + Uncertainty Bands",
        xaxis_title="Year",
        yaxis_title="EV Cost (USD)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Fuel & Electricity Price Bands
with st.expander("⛽ Fuel & Electricity Price Bands"):
    fuel_price_summary = summarise(results["fuel_price"])
    elec_price_summary = summarise(results["elec_price"])
    hover = "%{x}: %{y:.2f}"
    fig3 = go.Figure()
    # Fuel price traces
    fig3.add_trace(go.Scatter(
        x=years, y=fuel_price_summary["mean"], name="Fuel Price – Mean", line=dict(color="brown"), hovertemplate=hover
    ))
    fig3.add_trace(go.Scatter(
        x=years, y=fuel_price_summary["p10"], name="Fuel Price – P10", line=dict(color="sienna", dash="dot"), hovertemplate=hover
    ))
    fig3.add_trace(go.Scatter(
        x=years, y=fuel_price_summary["p90"], name="Fuel Price – P90", line=dict(color="sienna", dash="dot"), hovertemplate=hover,
        fill="tonexty", fillcolor="rgba(205,133,63,0.2)"
    ))
    # Electricity price traces
    fig3.add_trace(go.Scatter(
        x=years, y=elec_price_summary["mean"], name="Electricity Price – Mean", line=dict(color="green"), hovertemplate=hover
    ))
    fig3.add_trace(go.Scatter(
        x=years, y=elec_price_summary["p10"], name="Electricity Price – P10", line=dict(color="limegreen", dash="dot"), hovertemplate=hover
    ))
    fig3.add_trace(go.Scatter(
        x=years, y=elec_price_summary["p90"], name="Electricity Price – P90", line=dict(color="limegreen", dash="dot"), hovertemplate=hover,
        fill="tonexty", fillcolor="rgba(144,238,144,0.2)"
    ))
    fig3.update_layout(
        title="Fuel & Electricity Prices – P10/P50/P90",
        xaxis_title="Year",
        yaxis_title="Price (USD)",
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)

# Shock multiplier bands
with st.expander("📈 Shock Multiplier Bands"):
    shock_summary = summarise(results["shock_multiplier"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=years, y=shock_summary["mean"], name="Mean", line=dict(color="blue")
    ))
    fig2.add_trace(go.Scatter(
        x=years, y=shock_summary["p10"], name="P10", line=dict(color="lightblue", dash="dot")
    ))
    fig2.add_trace(go.Scatter(
        x=years, y=shock_summary["p90"], name="P90", line=dict(color="lightblue", dash="dot"),
        fill="tonexty", fillcolor="rgba(173,216,230,0.2)"
    ))
    fig2.update_layout(
        title="Shock Multiplier (AR(1))",
        xaxis_title="Year",
        yaxis_title="× Multiplier"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Public health benefits display remains in its own expander for further detail
with st.expander("🫁 Public Health Benefits (USD, 2026–2035)"):
    # Summarise health cost avoided values
    def summarise_array(array):
        return {
            "mean": np.mean(array, axis=0),
            "p10": np.percentile(array, 10, axis=0),
            "p90": np.percentile(array, 90, axis=0),
        }
    health_summary = summarise_array(results["health_cost avoided this year through conversion"])
    fig_health = go.Figure()
    fig_health.add_trace(go.Scatter(
        x=years, y=health_summary["mean"], name="Mean", line=dict(color="green", width=2)
    ))
    fig_health.add_trace(go.Scatter(
        x=years, y=health_summary["p10"], name="P10", line=dict(color="lightgreen", dash="dot")
    ))
    fig_health.add_trace(go.Scatter(
        x=years, y=health_summary["p90"], name="P90", line=dict(color="lightgreen", dash="dot"),
        fill="tonexty", fillcolor="rgba(144,238,144,0.2)"
    ))
    fig_health.update_layout(
        title="Estimated Public Health Cost Avoided (USD/year)",
        xaxis_title="Year",
        yaxis_title="Avoided Cost (USD)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_health, use_container_width=True)
    st.markdown(
        "This chart visualises the **healthcare cost avoided each year** due to fewer respiratory and pollution‑related illnesses, "
        "as internal combustion engine (ICE) bikes are replaced by electric vehicles. "
        "Values reflect a constant health burden avoided per ICE, applied to the number of EVs in the fleet each year."
    )
    st.markdown("---")
    # Stacked health cost savings (gov vs people)
    gov_summary_stack = np.mean(results["gov_health"], axis=0)
    people_summary_stack = np.mean(results["people_health"], axis=0)
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(
        x=years, y=gov_summary_stack, name="Gov Savings", mode="lines", stackgroup="one",
        line=dict(width=0.5, color="blue"), hoverinfo="x+y"
    ))
    fig_split.add_trace(go.Scatter(
        x=years, y=people_summary_stack, name="People Savings", mode="lines", stackgroup="one",
        line=dict(width=0.5, color="orange"), hoverinfo="x+y"
    ))
    fig_split.update_layout(
        title="Stacked Health Cost Avoided – Gov vs People (Mean)",
        xaxis_title="Year",
        yaxis_title="USD/year",
        hovermode="x unified",
        showlegend=True
    )
    st.plotly_chart(fig_split, use_container_width=True)
    st.markdown(f"**Split Assumption:** {int(health_share_gov*100)}% Gov – {int((1-health_share_gov)*100)}% People")

# --- State Initialisation for Comparison and Snapshot History ---
if "previous_metrics" not in st.session_state:
    st.session_state.previous_metrics = None
if "current_metrics" not in st.session_state:
    st.session_state.current_metrics = None
if "snapshot_history" not in st.session_state:
    st.session_state.snapshot_history = []

# Update session state with tracked inputs and metrics
st.session_state.previous_metrics = st.session_state.get("current_metrics", None)
st.session_state.previous_inputs = st.session_state.get("current_inputs", None)
st.session_state.current_metrics = metrics
st.session_state.current_inputs = tracked_inputs
st.session_state.snapshot_history.append({
    "params": tracked_inputs,
    "metrics": metrics
})
# Keep only the last 5 snapshots
if len(st.session_state.snapshot_history) > 5:
    st.session_state.snapshot_history.pop(0)

# Year-by-Year % Change in P50 from Previous Run
st.markdown("## 📈 Year-by-Year % Change from Previous Run (Median Only)")
if st.session_state.previous_metrics:
    year_labels = [str(y) for y in years]
    yoy_data = []
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
    st.dataframe(
        df_yoy.style.format("{:+.1f}%").background_gradient(axis=0, cmap="RdYlBu_r", low=0.2, high=0.8),
        use_container_width=True
    )
else:
    st.info("Run the simulation at least twice to compare results.")

# Snapshot History (last 5 runs)
st.markdown("## 🧾 Snapshot History (last 5 runs)")
for i, snap in enumerate(reversed(st.session_state.snapshot_history)):
    st.markdown(f"### Run #{len(st.session_state.snapshot_history) - i}")
    with st.expander("Parameters & Results", expanded=False):
        st.write("**Parameters:**", snap["params"])
        for key, val in snap["metrics"].items():
            st.write(f"**{key} (Mean 2035):** {val['mean'][-1]:.2f}")