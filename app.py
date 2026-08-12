# app.py -- Streamlit dashboard for the German gas demand project.
#

#   pip install streamlit plotly
#   streamlit run app.py
#
# Deploy: push the repo to GitHub, then go to share.streamlit.io and point
# it at app.py -- Streamlit Community Cloud installs requirements.txt and
# hosts it on a public URL, no server management needed.

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="German Gas Demand Model", layout="wide")


@st.cache_data
def load_data():
    merged = pd.read_csv("data/merged_daily.csv", parse_dates=["date"])
    model_dataset = pd.read_csv("data/model_dataset.csv", parse_dates=["date"])
    shap_importance = pd.read_csv("data/shap_feature_importance.csv")
    validation = pd.read_csv("data/validation_results.csv")
    monte_carlo = pd.read_csv("data/monte_carlo_forecast.csv", parse_dates=["date"])
    return merged, model_dataset, shap_importance, validation, monte_carlo


try:
    merged, model_dataset, shap_importance, validation, monte_carlo = load_data()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}. Run the pipeline scripts (clean_*.py, "
             f"merge_and_features.py, train_model.py, train_quantile_models.py, "
             f"monte_carlo_weather.py) before launching the dashboard.")
    st.stop()

# ============================================================
# Glossaries -- reused across tabs
# ============================================================
SERIES_DEFINITIONS = {
    "demand_gwh": (
        "Final consumer demand",
        "Gas delivered to end users -- households, industry, businesses -- via "
        "ENTSOG's 'Final Consumers'/'Letztverbraucher' points only. This is "
        "the model's target: winter-peaked, as real heating demand should be."
    ),
    "total_system_offtake_gwh": (
        "Total system offtake",
        "Everything leaving the German transmission network summed together: "
        "final consumer demand + storage injection + cross-border exports. "
        "Summer-peaked, because storage refill and transit exports outweigh "
        "the (smaller) heating-demand component in summer months."
    ),
    "storage_exit_gwh": (
        "Storage injection/withdrawal flow",
        "Gas flowing into underground storage. Booked as an 'exit' from the "
        "transmission grid, so it's counted separately here -- it isn't "
        "consumption, it's gas being saved for winter. Heavily April-October "
        "(injection season); near zero in winter (withdrawal season, when "
        "gas flows the other way and isn't captured by exit-side data)."
    ),
    "interconnection_exit_gwh": (
        "Cross-border interconnection (transit) flow",
        "Gas leaving Germany to neighbouring countries (Netherlands, Belgium, "
        "Czech Republic, Poland, etc.) via interconnection points. The "
        "largest single category by volume (~70% of total offtake) -- driven "
        "by price arbitrage and transit routing, not German demand."
    ),
}

FEATURE_GLOSSARY = {
    "day_of_week": "Day of week (0 = Monday ... 6 = Sunday)",
    "month": "Calendar month (1-12)",
    "is_weekend": "1 if Saturday/Sunday, else 0",
    "day_of_year": "Day number within the year (1-365/366)",
    "hdd": "Heating degree days, today -- max(0, 18°C − avg temp). Higher = colder = more heating demand expected",
    "hdd_lag1": "Heating degree days, yesterday",
    "hdd_ma7": "7-day trailing average heating degree days",
    "temperature_2m_mean": "Mean air temperature, today (°C)",
    "temperature_2m_min": "Minimum air temperature, today (°C)",
    "temperature_2m_max": "Maximum air temperature, today (°C)",
    "storage_pct_full_lag1": "Gas storage fill level, yesterday (% of capacity)",
    "storage_twh_lag1": "Gas storage level, yesterday (TWh)",
    "withdrawal_gwh_lag1": "Gas withdrawn from storage, yesterday (GWh)",
    "injection_gwh_lag1": "Gas injected into storage, yesterday (GWh)",
    "agsi_consumption_gwh_lag1": "AGSI-reported implied consumption, yesterday (GWh)",
    "resid_lag1": "Demand 'surprise' from yesterday -- how far actual demand was above/below the expected trend+seasonal level, 1 day ago",
    "resid_lag7": "Demand surprise from 7 days ago (same weekday last week)",
    "resid_ma7": "7-day trailing average demand surprise -- recent momentum above/below normal",
    "resid_ma30": "30-day trailing average demand surprise -- medium-term drift above/below normal",
    "stl_trend": "Long-run trend level of demand (STL decomposition)",
    "stl_seasonal": "Expected seasonal (time-of-year) demand pattern (STL decomposition)",
    "stl_resid": "The 'surprise' component -- actual demand minus trend minus seasonal. This is what the model predicts",
}


def human_label(feature_name):
    desc = FEATURE_GLOSSARY.get(feature_name, "")
    short = desc.split(" -- ")[0].split(",")[0]
    return f"{feature_name} ({short})" if short else feature_name


st.title("German Natural Gas Demand: Drivers & Forecast")
st.caption("Country-level daily demand model built on free public data "
           "(ENTSOG, AGSI+, Open-Meteo) -- XGBoost driver analysis + "
           "quantile-regression Monte Carlo forecast.")

tab_insights, tab_overview, tab_forecast, tab_drivers, tab_validation, tab_next, tab_methodology = st.tabs(
    ["Key Insights", "Overview", "Monte Carlo Forecast", "Drivers",
     "Model Validation", "Next Steps", "Methodology"]
)

# ============================================================
# TAB: OVERVIEW -- toggle between the different flow series
# ============================================================
with tab_overview:
    st.subheader("Daily gas flow series")
    st.markdown(
        "ENTSOG's 'exit flow' data isn't one homogeneous number -- it splits "
        "into genuinely different things. Toggle between them below."
    )

    choice_col = st.selectbox(
        "Series to display",
        list(SERIES_DEFINITIONS.keys()),
        format_func=lambda c: SERIES_DEFINITIONS[c][0],
    )
    title, description = SERIES_DEFINITIONS[choice_col]
    st.info(f"**{title}**  \n{description}")

    show_decomposition = st.checkbox(
        "Overlay STL trend + seasonal (final consumer demand only)",
        value=(choice_col == "demand_gwh"),
        disabled=(choice_col != "demand_gwh"),
    )
    if show_decomposition and choice_col == "demand_gwh":
        st.caption(
            "**Final consumer demand** (green) is the real, noisy day-to-day series. "
            "**STL trend + seasonal** (blue dashed) is the *expected* value for each "
            "day -- the long-run trend plus the typical pattern for that time of "
            "year -- with short-term noise deliberately smoothed out. The gap "
            "between the two lines on any given day is the 'residual': how far "
            "actual demand came in above or below what was expected. That gap is "
            "what the model in the Drivers/Validation tabs is actually trying to "
            "predict, not the green line directly -- see the Key Insights tab for why."
        )

    # Date range picker -- calendar dropdowns, day-level precision. Lets you
    # zoom past a one-off spike or focus on a single winter/summer instead
    # of the full 5-year series dwarfing itself.
    data_min = merged["date"].min().date()
    data_max = merged["date"].max().date()
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        range_start = st.date_input("Start date", value=data_min,
                                     min_value=data_min, max_value=data_max)
    with date_col2:
        range_end = st.date_input("End date", value=data_max,
                                   min_value=data_min, max_value=data_max)
    if range_start > range_end:
        st.error("Start date must be before end date.")
        st.stop()
    view = merged[(merged["date"] >= pd.Timestamp(range_start)) & (merged["date"] <= pd.Timestamp(range_end))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["date"], y=view[choice_col],
        mode="lines", name=title, line=dict(width=2.5, color="#2ea043"),
    ))
    if show_decomposition and choice_col == "demand_gwh":
        fig.add_trace(go.Scatter(
            x=view["date"], y=view["stl_trend"] + view["stl_seasonal"],
            mode="lines", name="STL trend + seasonal",
            line=dict(width=2, dash="dash", color="#1f6feb"),
        ))
    fig.update_layout(height=750, yaxis_title="GWh/day", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{view[choice_col].mean():,.0f} GWh/day")
    c2.metric("Min", f"{view[choice_col].min():,.0f} GWh/day")
    c3.metric("Max", f"{view[choice_col].max():,.0f} GWh/day")
    c4.metric("Latest (in range)", f"{view[choice_col].iloc[-1]:,.0f} GWh/day")

# ============================================================
# TAB: KEY INSIGHTS -- computed live from the current pipeline output,
# so it stays accurate after every rerun rather than hardcoding numbers
# that go stale the moment the target or model changes.
# ============================================================
with tab_insights:
    st.subheader("Key insights")

    st.markdown(
        "**0. It is more useful for the model to predict the residual from the seasonal-trend decomposition, "
        "instead of raw demand.** The rationale: the calendar alone already "
        "explains most of demand (e.g., cold in January means higher "
        "demand, warm in July means lower demand). Predicting the leftover "
        "residual instead answers the question: what's *left* to explain "
        "once the obvious seasonal pattern is removed"
    )


    # Insight 1: demand definition
    consumer_share = (merged["demand_gwh"].mean() / merged["total_system_offtake_gwh"].mean()) * 100
    st.markdown(
        f"**1. 'Demand' isn't what it first looks like.** Final consumer "
        f"demand is only ~{consumer_share:.0f}% of total ENTSOG exit flow -- "
        f"the rest is cross-border transit + storage injection, which is "
        f"summer-heavy and was making the naive 'total flow' series look "
        f"summer-peaked instead of winter-peaked."
    )

    # Insight 2: does XGBoost beat the naive seasonal baseline?
    if {"model", "fold", "mae"}.issubset(validation.columns):
        pivot = validation.pivot_table(index="fold", columns="model", values="mae")
        xgb_col = [c for c in pivot.columns if "XGBoost" in c]
        seasonal_col = [c for c in pivot.columns if "seasonal baseline" in c]
        if xgb_col and seasonal_col:
            wins = (pivot[xgb_col[0]] < pivot[seasonal_col[0]]).sum()
            total_folds = len(pivot)
            st.markdown(
                f"**2. XGBoost beat the naive 'assume no surprise' baseline "
                f"in {wins} of {total_folds} folds.** This could mean there's genuinely "
                f"little predictable structure left once trend, "
                f"seasonality, and weather are removed - or it could "
                f"reflect limitations of this specific setup: demand is "
                f"built from only 12 reporting points nationally (thin and "
                f"noise-prone), the feature set may be missing relevant "
                f"drivers (industrial activity, gas prices), or the model "
                f"may be overfitting a small dataset (~1,800 daily rows). "
                f"See Next Steps for what would help distinguish between "
                f"these."
            )

    # Insight 3: top SHAP drivers
    if {"feature", "mean_abs_shap"}.issubset(shap_importance.columns):
        top3 = shap_importance.sort_values("mean_abs_shap", ascending=False).head(3)
        top3_labels = [human_label(f) for f in top3["feature"]]
        st.markdown(
            f"**3. Top drivers of demand surprises:** {', '.join(top3_labels)}. "
            f"Full ranking in the Drivers tab."
        )

    st.markdown(
        "**4. The forecast shouldn't be allowed to feed on its own output.** "
        "An earlier version recursively re-predicted demand day-by-day, "
        "compounding small biases into a forecast that declined into "
        "winter. Fixed by predicting the residual and, for simulation, "
        "sourcing each day from real historical analogs instead of the "
        "model's own prior predictions."
    )

# ============================================================
# TAB: DRIVERS -- SHAP feature importance, with a glossary
# ============================================================
with tab_drivers:
    st.subheader("What drives the residual (day-to-day surprise vs. seasonal baseline)")
    st.markdown(
        "The model predicts the STL residual -- the deviation from the "
        "expected trend + seasonal level -- not raw demand. These are the "
        "features driving that deviation, ranked by mean absolute SHAP value."
    )

    shap_sorted = shap_importance.sort_values("mean_abs_shap", ascending=True).copy()
    fig = px.bar(
        shap_sorted, x="mean_abs_shap", y="feature",
        orientation="h", height=550,
    )
    fig.update_layout(xaxis_title="Mean |SHAP value|", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Feature glossary"):
        glossary_df = pd.DataFrame([
            {"Feature": f, "Meaning": FEATURE_GLOSSARY.get(f, "")}
            for f in shap_importance["feature"]
        ])
        st.dataframe(glossary_df, use_container_width=True, hide_index=True)

    st.markdown("#### Residual vs. temperature")
    fig2 = px.scatter(
        model_dataset, x="temperature_2m_mean", y="stl_resid",
        color="month", opacity=0.6, height=400,
        labels={"temperature_2m_mean": "Mean temperature (°C)",
                "stl_resid": "STL residual (GWh/day)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TAB: MODEL VALIDATION -- honest naive-baseline comparison
# ============================================================
with tab_validation:
    st.subheader("Walk-forward validation: XGBoost vs. naive baselines")
    st.markdown(
        "5-fold expanding-window validation, 60-day test windows. Reported "
        "honestly: the naive seasonal-only baseline (trend + seasonal, "
        "assume zero residual) is competitive with, and often beats, XGBoost."
    )

    metric = st.radio("Metric", ["mae", "rmse", "mape"], horizontal=True,
                       format_func=lambda m: m.upper())
    fig = px.bar(
        validation, x="fold", y=metric, color="model",
        barmode="group", height=450,
    )
    fig.update_layout(yaxis_title=metric.upper())
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(validation, use_container_width=True)

# ============================================================
# TAB: MONTE CARLO FORECAST
# ============================================================
with tab_forecast:
    st.subheader("Probabilistic demand forecast")
    st.caption(
        "Method: XGBoost quantile regression (5th/25th/50th/75th/95th "
        "percentile models). Each simulated day's inputs are sourced from a "
        "real historical analog day, not from a prior simulated day -- so "
        "nothing compounds. 1,000 simulations. Full detail in the "
        "Methodology tab."
    )

    mc_min = monte_carlo["date"].min().date()
    mc_max = monte_carlo["date"].max().date()
    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        mc_start = st.date_input("Start date", value=mc_min,
                                  min_value=mc_min, max_value=mc_max, key="mc_start")
    with mc_col2:
        mc_end = st.date_input("End date", value=mc_max,
                                min_value=mc_min, max_value=mc_max, key="mc_end")
    if mc_start > mc_end:
        st.error("Start date must be before end date.")
        st.stop()
    mc_view = monte_carlo[
        (monte_carlo["date"] >= pd.Timestamp(mc_start)) & (monte_carlo["date"] <= pd.Timestamp(mc_end))
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([mc_view["date"], mc_view["date"][::-1]]),
        y=pd.concat([mc_view["p95"], mc_view["p5"][::-1]]),
        fill="toself", fillcolor="rgba(31,111,235,0.15)",
        line=dict(width=0), name="5th-95th percentile", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([mc_view["date"], mc_view["date"][::-1]]),
        y=pd.concat([mc_view["p75"], mc_view["p25"][::-1]]),
        fill="toself", fillcolor="rgba(31,111,235,0.3)",
        line=dict(width=0), name="25th-75th percentile", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=mc_view["date"], y=mc_view["p50"],
        mode="lines", name="Median", line=dict(width=2, color="#1f6feb"),
    ))
    recent_actual = merged[merged["date"] >= merged["date"].max() - pd.Timedelta(days=180)]
    fig.add_trace(go.Scatter(
        x=recent_actual["date"], y=recent_actual["demand_gwh"],
        mode="lines", name="Recent actual", line=dict(width=1, color="#2ea043"),
    ))
    fig.update_layout(height=500, yaxis_title="GWh/day", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Forecast table"):
        st.dataframe(mc_view, use_container_width=True, hide_index=True)

# ============================================================
# TAB: NEXT STEPS
# ============================================================
with tab_next:
    st.subheader("Where this could go next")
    st.markdown(
        "**Link demand to price.** The natural extension: does a demand "
        "'surprise' (actual above/below the seasonal-adjusted expectation) "
        "lead TTF price moves, or does price already anticipate it? Also "
        "worth testing whether the model's residual predictions -- weak as "
        "they are for demand itself -- carry more signal when reframed as "
        "an input to a price or volatility model rather than judged as a "
        "demand forecast in isolation.\n\n"
        "**Supply-side balance.** Add ENTSOG entry-point (import) data "
        "alongside exit flow to build a net supply-demand balance, rather "
        "than looking at demand alone.\n\n"
        "**More countries.** Extend beyond Germany -- France, Netherlands, "
        "Italy -- to compare demand drivers and eventually model cross-border "
        "flow dynamics, not just single-country levels.\n\n"
        "**Sector-level splits.** Eurostat publishes gas consumption by "
        "sector (residential, industrial, power generation) at lower "
        "frequency -- could sharpen the 'final consumer' definition further, "
        "or explain some of the SHAP residual structure.\n\n"
        "**Statistical baseline comparison.** Compare XGBoost against "
        "SARIMA/Prophet on the same residual target, since the naive "
        "seasonal baseline already competes with XGBoost here -- worth "
        "checking whether a classical time-series model does any better "
        "before concluding the residual genuinely lacks structure.\n\n"
        "**Renewable forecast-error modeling for intraday power.** A "
        "parallel project: wind/solar generation forecast errors as a "
        "driver of intraday price volatility and imbalance pricing -- more "
        "directly relevant to short-term power trading than day-ahead gas "
        "demand."
    )

# ============================================================
# TAB: METHODOLOGY
# ============================================================
with tab_methodology:
    st.subheader("Data sources")
    st.markdown(
        "- **ENTSOG Transparency Platform** -- daily physical flow at every "
        "German network exit point, split into final-consumer, storage, and "
        "cross-border interconnection categories\n"
        "- **AGSI+ (GIE)** -- daily gas storage levels, injection/withdrawal\n"
        "- **Open-Meteo** -- historical weather, converted to heating degree "
        "days (HDD = max(0, 18°C − avg temp))"
    )

    st.subheader("Key methodology decisions")
    st.markdown(
        "**Demand definition.** ENTSOG 'exit flow', naively summed across "
        "every point, is dominated (~70%) by cross-border transit, plus "
        "storage injection (April-October). Both are summer-heavy, so the "
        "naive total looks summer-peaked -- the opposite of real heating "
        "demand. This model targets only ENTSOG's 'Final Consumers' points "
        "instead, which restores the expected winter-peaked shape.\n\n"
        "**STL decomposition.** Demand is split into trend + annual "
        "seasonal + residual (Cleveland, Cleveland, McRae & Terpenning, "
        "1990). The model predicts the residual, not raw demand -- this "
        "avoids an autoregressive-anchoring bug where a Monte Carlo "
        "simulation seeded from a low starting point kept declining "
        "instead of rising into winter.\n\n"
        "**Naive baseline comparison.** Every model result is checked "
        "against 'assume no surprise' (trend + seasonal + zero residual), "
        "yesterday's value, and a 7-day trailing average. This is what "
        "surfaced the finding that XGBoost doesn't clearly add value over "
        "the seasonal baseline in most folds.\n\n"
        "**Walk-forward validation.** 5-fold expanding-window validation "
        "with 60-day test windows -- never trains on future data relative "
        "to its own test fold.\n\n"
        "**Monte Carlo via quantile regression, not recursion.** Each "
        "simulated day's features (weather, storage, calendar, lagged "
        "residual) are sourced from a real historical analog day, never "
        "from a prior *simulated* day -- so nothing compounds. Five "
        "XGBoost quantile models (5th/25th/50th/75th/95th percentile) "
        "predict a distribution for that day, and one random value is "
        "sampled from it per simulation."
    )