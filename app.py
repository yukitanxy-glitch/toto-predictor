"""
Singapore TOTO Predictor -- Streamlit Web Application

Tech stack choice: Streamlit
Justification: Fastest to build, excellent for data dashboards with Plotly charts,
free deployment on Streamlit Community Cloud, built-in dark mode support,
and native DataFrame/chart integration. Perfect for this ML prediction app.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from src.scraper import load_data
from src import analysis

# -- Page Config ----------------------------------------------------------

st.set_page_config(
    page_title="TOTO Predictor SG",
    page_icon="🎱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS -----------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #FF6B6B, #FFE66D, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .board-card {
        background: linear-gradient(135deg, #1A1F2E, #2D3548);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #3D4663;
    }
    .board-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFE66D;
        margin-bottom: 0.5rem;
    }
    .number-ball {
        display: inline-block;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        text-align: center;
        line-height: 48px;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 4px;
        color: white;
    }
    .ball-main { background: linear-gradient(135deg, #FF6B6B, #EE5A24); }
    .ball-additional { background: linear-gradient(135deg, #4ECDC4, #2ECC71); }
    .stat-box {
        background: #1A1F2E;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
        border: 1px solid #3D4663;
    }
    .disclaimer {
        background: #2D1B1B;
        border: 1px solid #FF6B6B;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# -- Data Loading (Cached) ------------------------------------------------

@st.cache_data(ttl=3600)
def get_data():
    return load_data()


@st.cache_data(ttl=3600)
def get_analysis(_df):
    return analysis.get_full_analysis(_df)


@st.cache_data(ttl=3600)
def get_predictions(_df):
    from src.models.weighted_scoring import predict as ws_predict
    from src.models.monte_carlo import predict as mc_predict
    from src.models.markov_chain import predict as mk_predict
    from src.models.ensemble import predict as ensemble_predict
    from src.predictor import generate_all_boards

    ws = ws_predict(_df)
    mc = mc_predict(_df)
    mk = mk_predict(_df)

    model_results = {
        "weighted_scoring": ws,
        "monte_carlo": mc,
        "markov_chain": mk,
    }
    ens = ensemble_predict(_df, model_results)
    boards = generate_all_boards(ens["rankings"], _df)
    return boards, ens, model_results


# -- Sidebar --------------------------------------------------------------

st.sidebar.markdown("## TOTO Predictor SG")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Quant Engine v4.0", "Predictions", "Model Performance",
     "Historical Results", "Multi-Draw Strategy", "Odds & Methodology"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disclaimer:** This is for educational purposes only. "
    "TOTO is a random lottery -- no model can guarantee wins."
)


# -- Load Data ------------------------------------------------------------

df = get_data()
ana = get_analysis(df)


# ==========================================================================
# PAGE 1: DASHBOARD
# ==========================================================================

if page == "Dashboard":
    st.markdown('<div class="main-header">Statistical Analysis Dashboard</div>', unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Draws", len(df))
    with col2:
        st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} -> {df['date'].max().strftime('%Y-%m-%d')}")
    with col3:
        real = len(df[~df["is_synthetic"]]) if "is_synthetic" in df.columns else len(df)
        st.metric("Real Data", f"{real} draws")
    with col4:
        synth = len(df[df["is_synthetic"]]) if "is_synthetic" in df.columns else 0
        st.metric("Synthetic Data", f"{synth} draws")

    st.markdown("---")

    # -- 1. Frequency Bar Chart -------------------------------------------
    st.subheader("Number Frequency Analysis")

    freq = ana.get("frequency", {})
    main_freq = freq.get("main_counts", {})
    if main_freq:
        freq_df = pd.DataFrame([
            {"Number": num, "Count": count}
            for num, count in sorted(main_freq.items())
        ])

        # Color code: hot (green), cold (red), normal (blue)
        hot_cold = ana.get("hot_cold", {})
        hot_3m = set(n for n, _ in hot_cold.get("hot_3m", []))
        cold_3m = set(n for n, _ in hot_cold.get("cold_3m", []))

        colors = []
        for _, row in freq_df.iterrows():
            n = row["Number"]
            if n in hot_3m:
                colors.append("#2ECC71")
            elif n in cold_3m:
                colors.append("#E74C3C")
            else:
                colors.append("#3498DB")

        fig = go.Figure(go.Bar(
            x=freq_df["Number"],
            y=freq_df["Count"],
            marker_color=colors,
            hovertemplate="Number %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig.update_layout(
            title="Main Number Frequency (All Draws)",
            xaxis_title="Number",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400,
        )
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                           text="Green=Hot | Blue=Normal | Red=Cold",
                           showarrow=False, font=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

    # -- 2. Pair Frequency Heatmap ----------------------------------------
    st.subheader("Top Pair Frequencies")

    pairs = ana.get("pairs_triplets", {})
    top_pairs = pairs.get("top_pairs", [])
    if top_pairs:
        # Build a small heatmap from top pairs
        pair_data = []
        for p in top_pairs[:30]:
            pair_data.append({"Pair": f"{p['pair'][0]}-{p['pair'][1]}", "Frequency": p["count"]})
        pair_df = pd.DataFrame(pair_data)

        fig = px.bar(pair_df, x="Pair", y="Frequency",
                     title="Top 30 Most Frequent Number Pairs",
                     template="plotly_dark", color="Frequency",
                     color_continuous_scale="YlOrRd")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # -- 3. Odd/Even and High/Low -----------------------------------------
    col_oe, col_hl = st.columns(2)

    with col_oe:
        st.subheader("Odd/Even Distribution")
        oe = ana.get("odd_even", {})
        oe_dist = oe.get("distribution", {})
        if oe_dist:
            oe_df = pd.DataFrame([
                {"Split": f"{k[0]}O/{k[1]}E", "Count": v}
                for k, v in oe_dist.items()
            ])
            fig = px.pie(oe_df, names="Split", values="Count",
                         title="Odd/Even Split Distribution",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    with col_hl:
        st.subheader("High/Low Distribution")
        hl = ana.get("high_low", {})
        hl_dist = hl.get("distribution", {})
        if hl_dist:
            hl_df = pd.DataFrame([
                {"Split": f"{k[0]}L/{k[1]}H", "Count": v}
                for k, v in hl_dist.items()
            ])
            fig = px.pie(hl_df, names="Split", values="Count",
                         title="High/Low Split Distribution",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

    # -- 4. Sum Range Histogram -------------------------------------------
    st.subheader("Sum Range Analysis")

    sum_data = ana.get("sum_range", {})
    if sum_data:
        sums_series = sum_data.get("sums_series", pd.Series(dtype=float))
        sums = sums_series.tolist() if hasattr(sums_series, 'tolist') else list(sums_series)
        if sums:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=sums, nbinsx=40, name="Draw Sums",
                                        marker_color="#3498DB"))
            zone_70 = sum_data.get("zone_70", (0, 0))
            fig.add_vrect(x0=zone_70[0], x1=zone_70[1],
                          fillcolor="green", opacity=0.15,
                          annotation_text="70% Zone", annotation_position="top")
            stats_d = sum_data.get("stats", {})
            fig.update_layout(
                title=f"Sum of 6 Numbers Distribution (Mean={stats_d.get('mean', 0):.0f}, "
                      f"Std={stats_d.get('std', 0):.1f})",
                xaxis_title="Sum", yaxis_title="Count",
                template="plotly_dark", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Optimal Zone (70%):** {zone_70[0]} – {zone_70[1]}")
            zone_50 = sum_data.get("zone_50", (0, 0))
            st.markdown(f"**Tight Zone (50%):** {zone_50[0]} – {zone_50[1]}")

    # -- 5. Number Gap Tracker --------------------------------------------
    st.subheader("Number Gap Analysis")

    gaps_result = ana.get("gaps", {})
    gap_stats = gaps_result.get("gap_stats", {})
    overdue_set = set(gaps_result.get("overdue", []))
    if gap_stats:
        gap_df = pd.DataFrame([
            {"Number": num, "Current Gap": d["current_gap"],
             "Avg Gap": d["avg_gap"], "Overdue": num in overdue_set}
            for num, d in sorted(gap_stats.items())
        ])
        colors_gap = ["#E74C3C" if r["Overdue"] else "#3498DB" for _, r in gap_df.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=gap_df["Number"], y=gap_df["Current Gap"],
                              name="Current Gap", marker_color=colors_gap))
        fig.add_trace(go.Scatter(x=gap_df["Number"], y=gap_df["Avg Gap"],
                                  mode="lines", name="Average Gap",
                                  line=dict(color="#FFE66D", width=2)))
        fig.update_layout(title="Current Gap vs Average Gap per Number",
                          xaxis_title="Number", yaxis_title="Draws",
                          template="plotly_dark", height=400,
                          barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)

    # -- 6. Group Spread --------------------------------------------------
    st.subheader("Decade Group Spread")

    group_data = ana.get("decade_spread", {})
    pattern_counts = group_data.get("pattern_counts", {})
    if pattern_counts:
        spread_df = pd.DataFrame([
            {"Pattern": str(k), "Count": v}
            for k, v in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        ])
        fig = px.bar(spread_df, x="Pattern", y="Count",
                     title="Top 15 Group Spread Patterns",
                     template="plotly_dark", color="Count",
                     color_continuous_scale="Viridis")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # -- 7. Day of Week Comparison ----------------------------------------
    st.subheader("Monday vs Thursday")

    dow = ana.get("day_of_week", {})
    if dow:
        col_m, col_t = st.columns(2)
        mon_freq = dow.get("monday_freq", {})
        thu_freq = dow.get("thursday_freq", {})

        with col_m:
            if mon_freq:
                st.markdown("**Monday Top 10:**")
                top_mon = sorted(mon_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                for n, c in top_mon:
                    st.write(f"  Number {n}: {c}")
        with col_t:
            if thu_freq:
                st.markdown("**Thursday Top 10:**")
                top_thu = sorted(thu_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                for n, c in top_thu:
                    st.write(f"  Number {n}: {c}")

        sig = dow.get("significant", None)
        if sig is not None:
            p_val = dow.get("chi2_pvalue", 0)
            st.info(f"Chi-squared test: {'Significant' if sig else 'Not significant'} "
                    f"difference between Monday and Thursday (p={p_val:.4f})")

    # -- 8. Burst/Dormancy Status -----------------------------------------
    st.subheader("Burst & Dormancy Status")

    bd = ana.get("burst_dormancy", {})
    classifications = bd.get("classifications", {})
    if classifications:
        status_rows = []
        for num in range(1, 50):
            s = classifications.get(num, {})
            status_rows.append({
                "Number": num,
                "Status": s.get("state", "UNKNOWN"),
                "Recent Count (10 draws)": s.get("recent_count_last_10", 0),
                "Current Gap": s.get("current_gap", 0),
                "Avg Gap": s.get("avg_gap", 0),
            })
        status_df = pd.DataFrame(status_rows)

        def color_status(val):
            colors = {
                "BURST": "background-color: #27ae60; color: white",
                "ACTIVE": "background-color: #2ecc71; color: white",
                "NORMAL": "background-color: #3498db; color: white",
                "COOLING": "background-color: #f39c12; color: white",
                "DORMANT": "background-color: #e74c3c; color: white",
            }
            return colors.get(val, "")

        styled = status_df.style.map(color_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, height=400)


# ==========================================================================
# PAGE 2: QUANT ENGINE v4.0
# ==========================================================================

elif page == "Quant Engine v4.0":
    st.markdown('<div class="main-header">Quant Engine v4.0</div>', unsafe_allow_html=True)

    st.markdown("""
    > **Philosophy:** Two-stage approach: (1) Maximize **prediction accuracy** using 7 independent
    > signals (Bayesian, momentum, time-decayed pairs, sequence analysis, mean reversion, entropy,
    > triplets), then (2) optimize **Expected Prize Value** via anti-popularity.
    > Deterministic greedy board construction. No gambler's fallacy.
    """)

    # Run Quant Engine v4.0
    @st.cache_data(ttl=3600)
    def run_quant_engine_v4(_df):
        from src.models.quant_engine_v4 import QuantEngineV4
        engine = QuantEngineV4(_df)
        report = engine.analyze()
        boards = engine.generate_all_boards()
        return report, boards, engine

    with st.spinner("Running Quant Engine v4.0 (7-factor composite + greedy optimization)..."):
        report, qboards, engine = run_quant_engine_v4(df)

    # -- Phase 1: Regime Detection --
    st.subheader("Phase 1: Market Regime Detection")

    regime = report['regime']
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Current Regime</div>
            <div class="stat-value" style="font-size:1.2rem;">{regime['regime']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Sum Trend</div>
            <div class="stat-value">{regime['sum_trend']:+.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">High/Low Ratio</div>
            <div class="stat-value">{regime['high_ratio']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Odd Ratio</div>
            <div class="stat-value">{regime['odd_ratio']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    recs = regime.get('recommendations', {})
    prefer = recs.get('prefer_range', 'balanced')
    if prefer == 'high':
        st.success("Regime favors HIGH numbers (25-49). Boards adjusted accordingly.")
    elif prefer == 'low':
        st.success("Regime favors LOW numbers (1-24). Boards adjusted accordingly.")
    else:
        st.info("Regime is balanced. No directional bias applied.")

    # Mutual information
    mi = report.get('mutual_info', {})
    if mi:
        st.caption(f"Draw-to-draw overlap: {mi.get('observed_overlap', 0):.3f} "
                   f"(expected: {mi.get('expected_overlap', 0):.3f}, "
                   f"excess: {mi.get('excess_overlap', 0):+.3f})")

    # -- Phase 2: Bayesian Edge Detection --
    st.subheader("Phase 2: Bayesian Edge Detection")

    edge_numbers = report['edge_numbers']
    hot_nums = [(n, p) for n, p in edge_numbers if p['p_hot'] > 0.7]
    cold_nums = [(n, p) for n, p in edge_numbers if p['p_hot'] < 0.3]

    col_h, col_c = st.columns(2)
    with col_h:
        st.markdown("**HOT Numbers** (Bayesian P(hot) > 0.7)")
        if hot_nums:
            for num, post in sorted(hot_nums, key=lambda x: -x[1]['p_hot'])[:10]:
                st.write(f"  Number **{num}**: P(hot) = {post['p_hot']:.3f}, "
                         f"Edge = {post['edge_over_fair']:+.4f}")
        else:
            st.info("No statistically hot numbers detected")

    with col_c:
        st.markdown("**COLD Numbers** (Bayesian P(hot) < 0.3)")
        if cold_nums:
            for num, post in sorted(cold_nums, key=lambda x: x[1]['p_hot'])[:10]:
                st.write(f"  Number **{num}**: P(hot) = {post['p_hot']:.3f}, "
                         f"Edge = {post['edge_over_fair']:+.4f}")
        else:
            st.info("No statistically cold numbers detected")

    st.markdown(f"**Total numbers with Bayesian edge:** {len(edge_numbers)} out of 49")

    # -- Phase 3: Composite Rankings --
    st.subheader("Phase 3: 7-Factor Composite Rankings")

    top_composite = report['top_composite']
    comp_df = pd.DataFrame([
        {"Rank": i+1, "Number": n, "Composite": s['composite'],
         "Prediction": s.get('prediction', 0),
         "Bayesian": s['bayesian'], "Momentum": s.get('momentum', 0),
         "Sequence": s.get('sequence', 0), "Reversion": s.get('reversion', 0),
         "Entropy": s.get('entropy', 0), "Anti-Pop": s.get('anti_pop', 0)}
        for i, (n, s) in enumerate(top_composite)
    ])
    st.dataframe(comp_df, use_container_width=True, height=400)

    st.caption("Prediction: Bay 20% + Mom 15% + Pairs 15% + Seq 20% + Rev 15% + Ent 10% + Trip 5% | "
               "Composite: 70% Prediction + 30% Anti-Pop")

    # -- Phase 4: Optimized Boards --
    st.subheader("Phase 4: Predicted Boards (Greedy Optimized)")

    for b in qboards['boards']:
        ev = b['expected_value']
        pop_label = ('UNPOPULAR [+EV]' if ev['popularity_ratio'] < 0.8
                     else 'Average' if ev['popularity_ratio'] < 1.2
                     else 'Popular [-EV]')
        pop_color = ('green' if ev['popularity_ratio'] < 0.8
                     else 'orange' if ev['popularity_ratio'] < 1.2
                     else 'red')

        nums_html = ' '.join(
            f'<span class="number-ball ball-main">{n}</span>' for n in b['numbers']
        )

        st.markdown(f"""
        <div class="board-card">
            <h3>Board {b['board_number']}: {b['strategy']}</h3>
            <div style="margin: 1rem 0;">{nums_html}</div>
            <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
                <div>Sum: <b>{b['validation']['sum']}</b></div>
                <div>Odd/Even: <b>{b['validation']['odd_count']}/{6-b['validation']['odd_count']}</b></div>
                <div>Decades: <b>{b['validation']['decades']}</b></div>
                <div>Popularity: <b style="color: {pop_color};">{ev['popularity_ratio']:.3f} ({pop_label})</b></div>
            </div>
            <div style="margin-top: 0.5rem;">
                Expected prize if win: <b>${ev['expected_prize_if_win']:,.0f}</b>
                | Confidence: <b>{b.get('confidence', 0):.1%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-number breakdown
        with st.expander(f"Number Details - Board {b['board_number']}"):
            det_df = pd.DataFrame(b['number_details'])
            st.dataframe(det_df, use_container_width=True)

    # Additional number
    st.markdown(f"**Additional Number Prediction:** {qboards['additional_number']}")

    # Coverage analysis
    st.subheader("Coverage Analysis")
    cov = qboards['coverage']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Numbers", f"{cov['unique_numbers']}/49")
    with col2:
        st.metric("Coverage", f"{cov['coverage_pct']}%")
    with col3:
        st.metric("Overlap Penalty", f"{cov['overlap_penalty']:.3f}")

    # Anti-popularity heatmap (using v4.0's built-in scores)
    st.subheader("Number Popularity Map (Lower = Higher Expected Value)")

    anti_pop = engine._anti_popularity
    if anti_pop:
        # Create 7x7 grid
        grid = np.zeros((7, 7))
        labels = np.full((7, 7), '', dtype=object)
        for num in range(1, 50):
            row = (num - 1) // 7
            col_idx = (num - 1) % 7
            grid[row][col_idx] = anti_pop[num]
            labels[row][col_idx] = str(num)

        fig = go.Figure(data=go.Heatmap(
            z=grid,
            text=labels,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Anti-Pop Score"),
        ))
        fig.update_layout(
            title="Number Anti-Popularity Scores (Green = High EV, Red = Low EV)",
            height=400,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange='reversed'),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ---
    **How Quant Engine v4.0 works:**
    - **Bayesian Scoring** (20%): Beta-Binomial conjugate prior with time-decay weighting
    - **Momentum** (15%): Recent posterior captures short-term frequency trends
    - **Pair Network** (15%): Time-decayed eigenvector centrality (half-life=200 draws)
    - **Sequence/Lag** (20%): Autoregressive features - lag appearance, rolling freq, repeat probability
    - **Mean Reversion** (15%): Z-score of short vs long window frequency detects reversals
    - **Entropy** (10%): Information-theoretic surprise scoring via -log2(P)
    - **Triplet** (5%): Co-occurrence of number triplets beyond pair analysis
    - **Anti-Popularity** (30% of final score): EV optimization applied AFTER prediction scoring
    - **Board Generation**: Deterministic greedy selection (no stochastic sampling)
    """)


# ==========================================================================
# PAGE 3: PREDICTIONS (LEGACY)
# ==========================================================================

elif page == "Predictions":
    st.markdown('<div class="main-header">Next Draw Predictions</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
    <strong>IMPORTANT DISCLAIMER:</strong> TOTO is a truly random lottery.
    No prediction model can guarantee wins. Actual odds of Group 1 jackpot: <strong>1 in 13,983,816</strong>.
    This model identifies historically common patterns but cannot predict random outcomes with certainty.
    Play responsibly.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running prediction models..."):
        boards, ens, model_results = get_predictions(df)

    # Next draw info
    next_draw = boards["next_draw"]
    st.markdown(f"### Target Draw: {next_draw['date']} ({next_draw['day']})")

    # Display boards
    for b in boards["boards"]:
        st.markdown("---")

        # Board card
        nums_html = " ".join(
            f'<span class="number-ball ball-main">{n}</span>' for n in b["numbers"]
        )
        add_num = b.get("additional_number", "?")
        add_html = f'<span class="number-ball ball-additional">{add_num}</span>'

        st.markdown(f"""
        <div class="board-card">
            <div class="board-title">{b['name']}</div>
            <div style="margin: 1rem 0;">
                {nums_html}
                <span style="margin: 0 8px; color: #888;">+</span>
                {add_html}
            </div>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem;">
                <div class="stat-box">Confidence: <strong>{b['filter_result']['confidence']}</strong></div>
                <div class="stat-box">Sum: <strong>{b['stats']['sum']}</strong></div>
                <div class="stat-box">Odd/Even: <strong>{b['stats']['odd_even']}</strong></div>
                <div class="stat-box">High/Low: <strong>{b['stats']['high_low']}</strong></div>
                <div class="stat-box">Groups: <strong>{b['stats']['group_count']}</strong></div>
            </div>
            <div style="margin-top: 0.75rem; color: #aaa; font-size: 0.85rem;">
                Strategy: {b.get('strategy', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable filter details
        with st.expander(f"Filter Details -- {b['name']}"):
            for f in b["filter_result"]["results"]:
                icon = "✅" if f["passed"] else "❌"
                st.write(f"{icon} **{f['name']}:** {f['detail']}")

    # Ensemble ranking table
    st.markdown("---")
    st.subheader("Full Ensemble Ranking")

    rank_df = pd.DataFrame([
        {"Rank": i + 1, "Number": num, "Ensemble Score": round(score, 4)}
        for i, (num, score) in enumerate(ens["rankings"])
    ])
    st.dataframe(rank_df, use_container_width=True, height=500)


# ==========================================================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================================================

elif page == "Model Performance":
    st.markdown('<div class="main-header">Model Performance & Backtesting</div>', unsafe_allow_html=True)

    # Check for backtest results
    bt_path = os.path.join("data", "backtest_results.csv")
    if os.path.exists(bt_path):
        bt_df = pd.read_csv(bt_path)

        # Summary stats
        st.subheader("Backtest Summary")
        for board in ["board1", "board2", "board3"]:
            board_data = bt_df[bt_df["board"] == board]
            if len(board_data) > 0:
                avg = board_data["matches"].mean()
                best = board_data["matches"].max()
                st.metric(f"{board.upper()} Avg Matches", f"{avg:.3f} / 6",
                          delta=f"Best: {best}")

        # Distribution chart
        st.subheader("Match Distribution")
        fig = px.histogram(bt_df, x="matches", color="board",
                           barmode="group", nbins=7,
                           title="Number of Matches per Draw",
                           template="plotly_dark",
                           labels={"matches": "Numbers Matched", "count": "Frequency"})
        st.plotly_chart(fig, use_container_width=True)

        # Prediction log
        log_path = os.path.join("data", "predictions_log.csv")
        if os.path.exists(log_path):
            st.subheader("Prediction Log")
            log_df = pd.read_csv(log_path)
            st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No backtest results available yet. Run the backtester first.")
        st.markdown("""
        ```bash
        python -c "from src.backtester import run_backtest; from src.scraper import load_data; run_backtest(load_data())"
        ```
        """)

        # Show model descriptions
        st.subheader("Model Descriptions")
        models_info = {
            "Weighted Scoring": "Baseline model scoring all 49 numbers across 9 weighted factors including frequency, recency, pair correlation, and structural balance.",
            "Random Forest": "ML classifier using engineered features (gap analysis, rolling frequencies, burst states) with walk-forward validation.",
            "LSTM / Neural Net": "Sequential model processing windows of past draws to capture temporal patterns.",
            "Monte Carlo": "1M+ simulated draws using historical frequency as probability weights.",
            "Markov Chain": "49x49 transition matrix modeling P(number Y next | number X current).",
            "Cluster Analysis": "K-means clustering of historical winning combinations to identify common profiles.",
            "Ensemble": "Weighted consensus of all models, with model weights based on backtest performance.",
        }
        for name, desc in models_info.items():
            st.markdown(f"**{name}:** {desc}")


# ==========================================================================
# PAGE 4: HISTORICAL RESULTS
# ==========================================================================

elif page == "Historical Results":
    st.markdown('<div class="main-header">Historical Results Browser</div>', unsafe_allow_html=True)

    # Search and filter
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        search_num = st.number_input("Search by number (1-49)", min_value=0, max_value=49, value=0)
    with col_s2:
        day_filter = st.selectbox("Day of Week", ["All", "Monday", "Thursday"])
    with col_s3:
        date_range = st.date_input("Date Range",
                                    value=(df["date"].min().date(), df["date"].max().date()),
                                    min_value=df["date"].min().date(),
                                    max_value=df["date"].max().date())

    filtered = df.copy()

    if len(date_range) == 2:
        filtered = filtered[
            (filtered["date"].dt.date >= date_range[0]) &
            (filtered["date"].dt.date <= date_range[1])
        ]

    if day_filter != "All":
        filtered = filtered[filtered["day_of_week"] == day_filter]

    if search_num > 0:
        mask = False
        for i in range(1, 7):
            mask = mask | (filtered[f"num{i}"] == search_num)
        mask = mask | (filtered["additional_number"] == search_num)
        filtered = filtered[mask]

    st.markdown(f"**Showing {len(filtered)} draws**")

    # Display table
    display_cols = ["draw_number", "date", "day_of_week",
                    "num1", "num2", "num3", "num4", "num5", "num6",
                    "additional_number"]
    if "group1_prize" in filtered.columns:
        display_cols.append("group1_prize")
    if "is_synthetic" in filtered.columns:
        display_cols.append("is_synthetic")

    st.dataframe(
        filtered[display_cols].sort_values("date", ascending=False),
        use_container_width=True,
        height=600,
    )


# ==========================================================================
# PAGE 5: MULTI-DRAW STRATEGY
# ==========================================================================

elif page == "Multi-Draw Strategy":
    st.markdown('<div class="main-header">Multi-Draw Strategy</div>', unsafe_allow_html=True)

    with st.spinner("Generating multi-draw strategy..."):
        boards, ens, _ = get_predictions(df)
        from src.predictor import generate_multi_draw_strategy
        strategy = generate_multi_draw_strategy(ens["rankings"], df)

    st.subheader("4-Draw Rolling Strategy (2 Weeks)")

    for draw in strategy["draws"]:
        st.markdown(f"### {draw['label']}")
        cols = st.columns(len(draw["boards"]))
        for i, b in enumerate(draw["boards"]):
            with cols[i]:
                nums = b.get("numbers", [])
                st.markdown(f"**{b.get('name', 'Board')}**")
                st.markdown(f"Numbers: **{', '.join(str(n) for n in nums)}**")
                stats = b.get("stats", {})
                st.caption(f"Sum: {stats.get('sum', '?')} | O/E: {stats.get('odd_even', '?')}")

    # Coverage stats
    st.markdown("---")
    st.subheader("Coverage Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Numbers Covered", strategy["unique_numbers_covered"])
    with col2:
        st.metric("Total Boards (4 draws)", strategy["total_boards_played"])
    with col3:
        st.metric("Expected Matches/Board", strategy["expected_match_per_board"])

    # Projections
    st.subheader("Projections")
    proj = strategy["projections"]
    for period, data in proj.items():
        st.write(f"**{period.replace('_', ' ').title()}:** {data}")

    # Comparison
    st.subheader("Strategy Comparison")
    comp = strategy["comparison"]
    for strategy_name, desc in comp.items():
        st.write(f"- **{strategy_name.replace('_', ' ').title()}:** {desc}")


# ==========================================================================
# PAGE 6: ODDS & METHODOLOGY
# ==========================================================================

elif page == "Odds & Methodology":
    st.markdown('<div class="main-header">Odds & Methodology</div>', unsafe_allow_html=True)

    # Odds
    st.subheader("TOTO Prize Structure & Odds")

    odds_data = pd.DataFrame({
        "Group": [1, 2, 3, 4, 5, 6, 7],
        "Match": [
            "6 numbers", "5 + additional", "5 numbers",
            "4 + additional", "4 numbers", "3 + additional", "3 numbers"
        ],
        "Prize": [
            "38% pool (min $1M)", "8% pool", "5.5% pool",
            "3% pool", "$50", "$25", "$10"
        ],
        "Odds": [
            "1 in 13,983,816", "1 in 2,330,636", "1 in 55,491",
            "1 in 22,197", "1 in 1,083", "1 in 812", "1 in 61"
        ],
    })
    st.dataframe(odds_data, use_container_width=True, hide_index=True)
    st.caption("Overall odds of winning any prize: approximately 1 in 54.")

    # Methodology
    st.subheader("Prediction Methodology")

    st.markdown("""
    ### Models Used

    **1. Weighted Scoring (Baseline)**
    Scores all 49 numbers across 9 weighted factors: overall frequency (10%),
    recent 3-month frequency (15%), 6-month frequency (10%), recency/overdue (20%),
    pair correlation (15%), odd/even balance (8%), high/low balance (8%),
    sum range fit (7%), group spread (7%).

    **2. Random Forest Classifier**
    Machine learning model with engineered features per number: gap since last appearance,
    rolling 10/30-draw frequencies, burst/dormancy state, day of week, and previous draw characteristics.
    Uses walk-forward validation (never trains on future data).

    **3. LSTM / Neural Network**
    Sequential model processing windows of past N draws to capture temporal dependencies.
    Falls back to MLP if LSTM libraries unavailable.

    **4. Monte Carlo Simulation**
    Runs 1,000,000+ simulated TOTO draws using historical frequency as probability weights.
    Identifies numbers that appear disproportionately often under weighted sampling.

    **5. Markov Chain Transition Model**
    Builds a 49x49 transition matrix: P(number Y in next draw | number X in current draw).
    Uses the most recent draw to compute forward probabilities.

    **6. Cluster Analysis**
    K-means clustering of historical winning combinations to identify the most common
    structural profiles. Predicted boards are validated against these clusters.

    **7. Ensemble Model**
    Combines top 15 numbers from each model using weighted consensus voting.
    Model weights can be adjusted based on backtest performance.

    ### Hard Filters (7 Rules)
    Every generated board must pass ALL filters:
    1. **Sum Rule** -- within historical 70% optimal zone
    2. **Odd/Even** -- must be 3/3, 4/2, or 2/4
    3. **High/Low** -- must be 3/3, 4/2, or 2/4
    4. **Group Spread** -- at least 3 decade groups
    5. **Consecutive** -- max 2 consecutive numbers
    6. **No Duplicate** -- not matching any recent win
    7. **Cluster Fit** -- within common cluster profiles
    """)

    # Honest disclaimer
    st.markdown("---")
    st.subheader("Honest Disclaimer")
    st.error("""
    **This model CANNOT predict truly random outcomes.**

    TOTO numbers are drawn using a certified random process. Each draw is independent.
    Past results have NO influence on future draws. The probability of any specific
    6-number combination winning is always 1 in 13,983,816.

    **What this model CAN do:**
    - Identify historically common structural patterns (odd/even balance, sum ranges, etc.)
    - Avoid statistically rare combinations that almost never win
    - Provide a more structured selection than pure random guessing
    - Offer anti-sharing strategies to maximize prize value IF you win

    **What this model CANNOT do:**
    - Predict which specific numbers will be drawn
    - Increase your fundamental odds of winning
    - Guarantee any prize

    **Play responsibly. Never spend more than you can afford to lose.**
    """)
