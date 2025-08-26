# OptiNexus — Advanced Streamlit Demo (EconoSphere+)
# Developed by Sourab. Finoptiv.
# Single-file advanced demo showcasing all features with improved visuals
# Run: streamlit run OptiNexus_Advanced_Streamlit_Demo.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
from datetime import datetime

st.set_page_config(page_title='OptiNexus — EconoSphere+', layout='wide', initial_sidebar_state='expanded')

# -------------------- Utilities & Demo Data --------------------
@st.cache_data
def gen_macro_data(n=120, seed=42):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start='2023-01-01', periods=n, freq='M')
    gdp = 100 + 0.3 * np.arange(n) + 3 * np.sin(np.arange(n) / 6) + rng.normal(0, 1.2, n)
    unemployment = 5 + 0.06 * np.cos(np.arange(n) / 5) + rng.normal(0, 0.25, n)
    wage = 10 + 0.025 * np.arange(n) + rng.normal(0, 0.12, n)
    df = pd.DataFrame({'date': t, 'GDP': gdp, 'Unemployment': unemployment, 'Wage': wage})
    return df

@st.cache_data
def gen_firm_data(n_firms=12, seed=2):
    rng = np.random.default_rng(seed)
    firms = [f'Firm_{i+1}' for i in range(n_firms)]
    size = rng.integers(50, 1200, size=n_firms)
    productivity = np.round(rng.normal(1.0, 0.18, size=n_firms), 3)
    region = rng.choice(['North', 'South', 'East', 'West'], size=n_firms)
    sector = rng.choice(['Agriculture', 'Manufacturing', 'Services'], size=n_firms)
    df = pd.DataFrame({'Firm': firms, 'Size': size, 'Productivity': productivity, 'Region': region, 'Sector': sector})
    return df

macro_df = gen_macro_data()
firms_df = gen_firm_data()

# Feature list (keep in sync with tabs)
FEATURES = [
    'Home (Dashboard)', 'Counterfactual Twin Lab', 'Nash Bargain Simulator', 'Job Matching ABM',
    'Climate–Econ–Ops Coupler', 'SIR↔Econ↔HR Runner', 'Skill Diffusion & OT Planner', 'Shadow Economy Estimator',
    'Fairness-Constrained Optimizers', 'Real Options Valuation', 'Social Welfare Frontier', 'Policy Narrative Generator',
    'Federated-Lite Capsules', 'Synthetic Microdata Forge', 'Robustness Gym & Stress Test'
]

# -------------------- Sidebar --------------------
st.sidebar.image(
    "https://raw.githubusercontent.com/plotly/datasets/master/logo-colors.png",
    use_column_width=False, width=120)
st.sidebar.title('OptiNexus — EconoSphere+')
st.sidebar.caption('Developed by Sourab. Finoptiv.')

st.sidebar.markdown('---')
st.sidebar.subheader('Scenario / Run Controls')
scenario_name = st.sidebar.text_input('Scenario name', value='Base / Coastal Factory')
horizon = st.sidebar.selectbox('Horizon', ['6 months', '12 months (default)', '24 months'])
time_step = st.sidebar.selectbox('Time step', ['Daily', 'Weekly', 'Monthly'], index=2)
upload = st.sidebar.file_uploader('Upload dataset (CSV) — optional', type=['csv','parquet'])

st.sidebar.markdown('---')
st.sidebar.subheader('Features')
# feature navigation: selectbox + checklist for quick toggles
selected_feature = st.sidebar.selectbox('Open feature pane', FEATURES, index=0)
st.sidebar.markdown('Quick toggle (show tiles on home):')
feature_toggles = {f: st.sidebar.checkbox(f, value=True if i < 6 else False) for i, f in enumerate(FEATURES)}

st.sidebar.markdown('---')
st.sidebar.subheader('Run & Export')
run_tournament = st.sidebar.button('▶ Run Tournament (Aggregate)')
export_report = st.sidebar.button('Export full report (demo)')

# -------------------- Header / Top KPIs --------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric('GDP Δ (sim)', '+3.2%')
col2.metric('Unemployment Δ', '-0.5 ppt')
col3.metric('Equity Index', '+3.6')
col4.metric('Robustness Score', '0.89')

# -------------------- Main Area with Dynamic Feature Pane --------------------
st.header(f'Feature: {selected_feature}')

# helper to show feature description
def feature_description(name):
    desc = {
        'Home (Dashboard)': 'Consolidated view: KPI mosaic, active scenarios, recent runs, and quick launch tiles for each module.',
        'Counterfactual Twin Lab': 'Create synthetic counterfactuals (BSTS / synthetic control style) with CI bands and donor diagnostics.',
        'Nash Bargain Simulator': 'Alternating-offers and Nash solution visualizer for union-management bargaining.',
        'Job Matching ABM': 'Mortensen–Pissarides inspired matching with vacancy→hire→turnover visualization and Sankey flows.',
        'Climate–Econ–Ops Coupler': 'Physics-inspired propagation from climate shock to supply delays and sector outputs.',
        'SIR↔Econ↔HR Runner': 'Coupled SIR epidemic model linked to workforce availability and productivity losses.',
        'Skill Diffusion & OT Planner': 'Skill-network diffusion (Laplacian) and entropy-regularized OT allocation for reskilling.',
        'Shadow Economy Estimator': 'Latent informal sector estimation using proxy signals and sensitivity analysis.',
        'Fairness-Constrained Optimizers': 'Optimize hiring/training with demographic parity or equality-of-opportunity constraints.',
        'Real Options Valuation': 'Project investment valuation with waiting option (LSM/binomial approximation).',
        'Social Welfare Frontier': 'Multi-criteria Pareto frontier visualizer (GDP, employment, equity).',
        'Policy Narrative Generator': 'Persona-aware auto narrative generator (Minister, CEO, Union, Academic).',
        'Federated-Lite Capsules': 'SQLite capsule demo for federated summary aggregation with privacy-safe exchanges.',
        'Synthetic Microdata Forge': 'DP-Copula inspired synthetic microdata generator (demo with noise controls).',
        'Robustness Gym & Stress Test': 'Adversarial perturbations, covariate shift, and stability scoring for models.'
    }
    return desc.get(name, '')

st.markdown(f"**Description:** {feature_description(selected_feature)}")

# -------------------- Home Dashboard (if selected) --------------------
if selected_feature == 'Home (Dashboard)':
    st.subheader('Executive Command Center — Snapshot')
    # show tiles based on toggles
    visible_tiles = [f for f in FEATURES if feature_toggles.get(f, False)]
    cols = st.columns(3)
    for i, tile in enumerate(visible_tiles):
        with cols[i % 3]:
            st.markdown(f'### {tile}')
            # show small preview or interactive small chart
            if 'Economic' in tile or 'Home' in tile:
                fig = px.line(macro_df.tail(36), x='date', y='GDP', title='GDP (last 36 months)')
                fig.update_layout(height=220)
                st.plotly_chart(fig, use_container_width=True)
            elif 'Job Matching' in tile:
                sank = go.Figure(data=[
                    go.Sankey(node=dict(label=['Vacancies','Applications','Hires','Separations']),
                              link=dict(source=[0,0,1], target=[1,2,3], value=[220,120,60]))
                ])
                sank.update_layout(height=220)
                st.plotly_chart(sank, use_container_width=True)
            elif 'AI Policy Co-Pilot' in tile:
                st.write('Persona previews: Minister / CEO / Union')
            else:
                st.info('Click full feature for details')
    st.markdown('---')
    st.subheader('Tri-Coupler (Applied Econ ↔ HR ↔ MIS) — Interactive Big Picture')
    shock = st.slider('Systemic shock magnitude (0-1)', 0.0, 1.0, 0.18)
    coupling = st.slider('Coupling intensity (0-1)', 0.0, 1.0, 0.5)
    econ_imp = -shock * (1 + 0.6 * coupling)
    hr_imp = -0.6 * shock * (1 + coupling)
    mis_imp = -0.35 * shock * (1 + 0.75 * coupling)
    df_c = pd.DataFrame({'Module': ['Economy', 'HR', 'MIS'], 'Impact': [econ_imp, hr_imp, mis_imp]})
    fig = px.bar(df_c, x='Module', y='Impact', text='Impact', color='Module')
    fig.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Counterfactual Twin Lab --------------------
if selected_feature == 'Counterfactual Twin Lab':
    st.subheader('Build counterfactual twin — donor diagnostics & uncertainty')
    target = st.selectbox('Target series', ['GDP', 'Unemployment', 'Wage'])
    intervention_date = st.date_input('Intervention date', value=macro_df['date'].iloc[-12])
    effect_val = st.number_input('Simulated additive effect (post intervention)', min_value=-50.0, max_value=50.0, value=3.0)
    donors = st.multiselect('Donor series (auto-selected)', ['Unemployment', 'Wage'], default=['Unemployment', 'Wage'])
    if st.button('Create Twin & Run Diagnostics'):
        # prepare data
        idx = macro_df[macro_df['date'] <= pd.to_datetime(intervention_date)].index.max()
        X = macro_df[donors].values
        y = macro_df[target].values
        model = Ridge(alpha=1.0)
        model.fit(X[:idx], y[:idx])
        pred = model.predict(X)
        observed = y.copy()
        observed[idx + 1:] = observed[idx + 1:] + effect_val
        # plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=macro_df.date, y=observed, name='Observed (with intervention)', mode='lines'))
        fig.add_trace(go.Scatter(x=macro_df.date, y=pred, name='Counterfactual Twin (pred)', mode='lines', line=dict(dash='dash')))
        fig.add_vline(x=macro_df.date.iloc[idx], line=dict(dash='dot'), annotation_text='Intervention')
        # bootstrap CI
        preds = []
        rng = np.random.default_rng(123)
        pre_n = max(10, idx)
        for i in range(200):
            sample_idx = rng.choice(np.arange(pre_n), size=pre_n, replace=True)
            m = Ridge(alpha=1.0)
            m.fit(X[sample_idx], y[sample_idx])
            preds.append(m.predict(X))
        preds = np.array(preds)
        lower = np.percentile(preds, 2.5, axis=0)
        upper = np.percentile(preds, 97.5, axis=0)
        fig.add_traces([go.Scatter(x=macro_df.date, y=upper, name='Upper 95% CI', line=dict(width=0), showlegend=False),
                        go.Scatter(x=macro_df.date, y=lower, name='Lower 95% CI', fill='tonexty', line=dict(width=0), showlegend=False)])
        st.plotly_chart(fig, use_container_width=True)
        # donor weights
        coefs = pd.Series(model.coef_, index=donors)
        st.subheader('Donor coefficients (model)')
        st.bar_chart(coefs)
        st.success('Counterfactual twin created. Check CI band and donor diagnostics above.')

# -------------------- Nash Bargain --------------------
if selected_feature == 'Nash Bargain Simulator':
    st.subheader('Nash Bargaining & Alternating Offers')
    st.markdown('Adjust utilities and bargaining power; visualize surplus split and offer timeline.')
    U1 = st.slider('Union fallback utility', 0.0, 1.0, 0.35)
    U2 = st.slider('Management fallback utility', 0.0, 1.0, 0.45)
    w = st.slider('Union bargaining weight', 0.0, 1.0, 0.55)
    rounds = st.slider('Rounds of offers', 1, 12, 6)
    surplus = max(0, 1 - (U1 + U2))
    util1 = U1 + w * surplus
    util2 = U2 + (1 - w) * surplus
    st.metric('Nash solution - union utility', f'{util1:.3f}')
    st.metric('Nash solution - management utility', f'{util2:.3f}')
    # alternating offers timeline
    offers = []
    for r in range(1, rounds + 1):
        if r % 2 == 1:
            # union offers closer to util1
            proposal = U1 + (util1 - U1) * (r / rounds)
        else:
            proposal = U2 + (util2 - U2) * (r / rounds)
        offers.append({'Round': r, 'ProposalValue': proposal})
    st.line_chart(pd.DataFrame(offers).set_index('Round'))

# -------------------- Job Matching ABM --------------------
if selected_feature == 'Job Matching ABM':
    st.subheader('Job-Matching ABM (time dynamics + Sankey)')
    T = st.slider('Periods (t)', 6, 96, 36)
    alpha = st.slider('Matching elasticity α', 0.1, 0.9, 0.6)
    A = st.slider('Matching efficiency A', 0.1, 3.0, 1.0)
    vac = np.zeros(T)
    unemployed = np.zeros(T)
    hires = np.zeros(T)
    unemployed[0] = 200
    vac[0] = 40
    for t in range(1, T):
        matches = A * (unemployed[t - 1] ** alpha) * (vac[t - 1] ** (1 - alpha))
        hires[t] = matches
        vac[t] = max(5, vac[t - 1] + np.random.normal(0, 3) + 0.05 * (10 - matches))
        unemployed[t] = max(0, unemployed[t - 1] - matches + np.random.poisson(3))
    dfm = pd.DataFrame({'t': np.arange(T), 'Vacancies': vac, 'Hires': hires, 'Unemployed': unemployed})
    fig = px.line(dfm, x='t', y=['Vacancies', 'Hires', 'Unemployed'], labels={'value': 'Count', 't': 'Period'})
    st.plotly_chart(fig, use_container_width=True)
    # Sankey breakdown sample
    sank = go.Figure(data=[
        go.Sankey(node=dict(label=['Vacancies', 'Applications', 'Interviews', 'Hires', 'Separations']),
                  link=dict(source=[0, 0, 1, 2], target=[1, 2, 3, 4], value=[220, 80, 60, 20]))
    ])
    sank.update_layout(height=380)
    st.plotly_chart(sank, use_container_width=True)

# -------------------- Climate Coupler --------------------
if selected_feature == 'Climate–Econ–Ops Coupler':
    st.subheader('Climate Shock propagation & supply-chain lead time heatmap')
    exposure = st.slider('Avg direct exposure % loss (sector mean)', 0.0, 0.6, 0.12)
    lag = st.slider('Avg propagation lag (periods)', 0, 6, 2)
    sectors = ['Agriculture', 'Manufacturing', 'Services']
    M = np.array([[1.0, 0.25, 0.15], [0.12, 1.0, 0.3], [0.07, 0.18, 1.0]])
    shock_vec = np.array([exposure, exposure * 0.65, exposure * 0.45])
    B = M - np.eye(3)
    impact = np.linalg.inv(np.eye(3) - 0.45 * B).dot(shock_vec)
    df_imp = pd.DataFrame({'Sector': sectors, 'Impact': impact})
    fig = px.bar(df_imp, x='Sector', y='Impact', text='Impact')
    fig.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
    # heatmap of lead time change
    lead = np.clip(np.round(1 + 10 * impact + np.random.normal(0, 0.8, 3), 2), 0.2, 50)
    heat = pd.DataFrame({'Sector': sectors, 'LeadDays': lead}).set_index('Sector')
    st.dataframe(heat)

# -------------------- SIR↔Econ↔HR Runner --------------------
if selected_feature == 'SIR↔Econ↔HR Runner':
    st.subheader('Epidemic-driven workforce & demand coupling')
    beta = st.slider('Transmission β', 0.01, 1.0, 0.28)
    gamma = st.slider('Recovery γ', 0.01, 1.0, 0.12)
    pop = st.number_input('Population (workforce)', 100, 200000, 5000)
    I0 = st.number_input('Initial infected', 1, pop // 2, 10)
    days = st.slider('Sim days', 30, 720, 180)

    def sir(y, t, N, beta, gamma):
        S, I, R = y
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return dS, dI, dR

    t = np.linspace(0, days - 1, days)
    y0 = (pop - I0, I0, 0)
    res = odeint(sir, y0, t, args=(pop, beta, gamma))
    S, I, R = res.T
    df_sir = pd.DataFrame({'day': t, 'S': S, 'I': I, 'R': R})
    fig = px.line(df_sir, x='day', y=['S', 'I', 'R'])
    st.plotly_chart(fig, use_container_width=True)
    absentee = I / pop
    prod_loss = absentee * 0.6
    st.line_chart(pd.DataFrame({'AbsenteeRate': absentee, 'ProdLoss': prod_loss}))

# -------------------- Skill Diffusion & OT Planner --------------------
if selected_feature == 'Skill Diffusion & OT Planner':
    st.subheader('Skill network diffusion & OT-based reskilling allocation')
    G = nx.barabasi_albert_graph(14, 2, seed=3)
    pos = nx.spring_layout(G, seed=3)
    fig = plt.figure(figsize=(6, 3))
    nx.draw(G, pos, with_labels=True, node_color='lightsteelblue', node_size=300)
    st.pyplot(fig)
    st.markdown('Simulate diffusion (graph Laplacian)')
    gamma = st.slider('Diffusion rate γ', 0.01, 2.0, 0.6)
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G).astype(float).todense()
    x0 = np.zeros(n); x0[0] = 1.0
    tgrid = np.linspace(0, 4, 40)
    X = [x0]
    for _ in tgrid[1:]:
        X.append(np.array(X[-1]) - gamma * (L @ np.array(X[-1])).squeeze())
    X = np.array(X)
    st.line_chart(pd.DataFrame(X, columns=[f'node{i}' for i in range(n)]))
    # OT approximate allocation: naive proportional allocation
    budget = st.slider('Training Budget (units)', 1, 500, 80)
    deg = np.array([G.degree(i) for i in range(n)]) + 0.1
    target = np.zeros(n); target[np.argsort(-deg)[:3]] = 1
    gap = np.maximum(0, target - X[-1])
    alloc = budget * deg * gap / (deg * gap).sum() if (deg * gap).sum() > 0 else np.zeros(n)
    st.bar_chart(pd.DataFrame({'node': list(range(n)), 'alloc': alloc}).set_index('node'))

# -------------------- Shadow Economy Estimator --------------------
if selected_feature == 'Shadow Economy Estimator':
    st.subheader('Latent informal output estimation using proxies')
    size = st.slider('Sample points', 20, 500, 80)
    electricity = np.linspace(80, 140, size) + np.random.normal(0, 4, size)
    nightlights = np.linspace(8, 25, size) + np.random.normal(0, 1.2, size)
    reported = 0.8 * electricity + 2.0 * nightlights + np.random.normal(0, 5, size)
    implied = 0.95 * electricity + 2.3 * nightlights + np.random.normal(0, 5, size)
    latent = implied - reported
    df = pd.DataFrame({'reported': reported, 'implied': implied, 'latent': latent})
    fig = px.line(df.cumsum(), title='Cumulative reported vs implied (demo)')
    st.plotly_chart(fig, use_container_width=True)
    informal_share = st.slider('Assumed informal share (%)', 0, 60, 22)
    st.write('Estimated tax base effect (demo):', f"{informal_share * 0.55:.1f}%")

# -------------------- Fairness-Constrained Optimizers --------------------
if selected_feature == 'Fairness-Constrained Optimizers':
    st.subheader('Constrained allocation for hiring/training')
    st.markdown('We demonstrate demographic parity constraint via a projection approach.')
    groups = ['Group A', 'Group B', 'Group C']
    applicants = np.array([120, 80, 60])
    slots = st.number_input('Training slots', 1, 1000, 120)
    tol = st.slider('Parity tolerance (abs difference in rates)', 0.0, 0.5, 0.05)
    alloc = (applicants / applicants.sum()) * slots
    rates = alloc / applicants
    if (rates.max() - rates.min()) > tol:
        avg = np.mean(rates)
        alloc = avg * applicants
    df_alloc = pd.DataFrame({'group': groups, 'applicants': applicants, 'alloc_slots': np.round(alloc).astype(int)})
    st.table(df_alloc)

# -------------------- Real Options --------------------
if selected_feature == 'Real Options Valuation':
    st.subheader('Real Options Approx (option to delay)')
    capex = st.number_input('CapEx (M)', 0.1, 500.0, 5.0)
    cash = st.number_input('Annual expected cashflow (M)', 0.1, 500.0, 1.8)
    r = st.slider('Discount rate', 0.0, 0.2, 0.05)
    sigma = st.slider('Volatility (σ)', 0.01, 1.0, 0.3)
    npv = cash / r - capex if r > 0 else cash * 10 - capex
    option_val = 0.2 * sigma * max(0, npv)
    st.metric('NPV (naive)', f'{npv:.2f} M')
    st.metric('Real Option (approx)', f'{option_val:.2f} M')

# -------------------- Welfare Frontier --------------------
if selected_feature == 'Social Welfare Frontier':
    st.subheader('Pareto frontier (GDP vs Equity, bubble = Employment)')
    n = 180
    gdp = np.linspace(80, 130, n) + np.random.normal(0, 2, n)
    equity = 55 - 0.25 * (gdp - 100) + np.random.normal(0, 1.2, n)
    employ = 100 - 0.4 * (gdp - 100) + np.random.normal(0, 3, n)
    df_pf = pd.DataFrame({'gdp': gdp, 'equity': equity, 'employ': employ})
    fig = px.scatter(df_pf, x='gdp', y='equity', size='employ', hover_data=['employ'], title='Policies sample space')
    st.plotly_chart(fig, use_container_width=True)
    # compute and overlay pareto
    pts = df_pf[['gdp', 'equity']].values
    pareto = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if (q[0] >= p[0] and q[1] >= p[1]) and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    st.plotly_chart(px.scatter(df_pf.iloc[pareto], x='gdp', y='equity', size='employ', color_discrete_sequence=['crimson']), use_container_width=True)

# -------------------- Policy Narrative Generator --------------------
if selected_feature == 'Policy Narrative Generator':
    st.subheader('Persona-aware policy narrative & brief generator')
    persona = st.selectbox('Persona', ['Minister', 'CEO', 'Union Rep', 'Academic'])
    findings = st.text_area('Key findings (short)', 'GDP down 1.2%, unemployment up 0.6%, manufacturing disrupted')
    tone = st.selectbox('Tone', ['Concise', 'Technical', 'Persuasive'])
    if st.button('Generate Brief'):
        header = f'Brief for {persona} — {scenario_name} — {datetime.utcnow().date()}'
        if persona == 'Minister':
            lead = f'{findings}. Immediate actions: targeted wage subsidy, temporary labor flex, rapid re-skilling program.'
        elif persona == 'CEO':
            lead = f'{findings}. Business actions: diversify suppliers, temporary overtime plans, demand-smoothing incentives.'
        elif persona == 'Union Rep':
            lead = f'{findings}. Worker actions: retain headcount, negotiate safety nets, training guarantees.'
        else:
            lead = f'{findings}. Methods: coupled simulation, bootstrap CI, sensitivity analyses; see Appendix.'
        st.markdown(f'**{header}**')
        st.write(lead)

# -------------------- Federated Capsules --------------------
if selected_feature == 'Federated-Lite Capsules':
    st.subheader('SQLite Capsules & Federated Synthesis (demo)')
    st.write('Connected capsules (demo): BankA, UnivX, FirmZ')
    if st.button('Simulate capsule sync'):
        st.success('Capsule sync completed. Meta-model updated (demo).')

# -------------------- Synthetic Microdata Forge --------------------
if selected_feature == 'Synthetic Microdata Forge':
    st.subheader('Synthetic microdata (DP-like noise controls)')
    n = st.slider('Rows', 100, 5000, 800)
    eps = st.slider('Privacy epsilon (demo: lower => more noise)', 0.1, 10.0, 1.0)
    base = pd.DataFrame({'age': np.random.randint(18, 62, n), 'wage': np.random.lognormal(2.4, 0.55, n), 'skill': np.random.choice(['low', 'med', 'high'], n)})
    if st.button('Generate Synthetic (demo)'):
        synth = base.copy()
        synth['wage'] = synth['wage'] + np.random.normal(0, eps, n)
        st.dataframe(synth.head(80))
        st.success('Synthetic microdata generated (demo, not certified DP)')

# -------------------- Robustness Gym --------------------
if selected_feature == 'Robustness Gym & Stress Test':
    st.subheader('Robustness & Stress-test harness')
    shift_type = st.selectbox('Shift type', ['Covariate shift', 'Label noise', 'Adversarial perturbation'])
    severity = st.slider('Severity', 0, 100, 25)
    baseline = 0.9
    if st.button('Run robustness test'):
        degraded = baseline - (severity / 100) * 0.55
        st.metric('Baseline metric (AUC)', f'{baseline:.3f}')
        st.metric('Post-shift metric (AUC)', f'{degraded:.3f}')
        st.write('Stability score (post/pre):', f'{max(0, degraded / baseline):.3f}')
        st.info('Run summary: adversarial perturbation applied, see model checklist for recommended mitigations.')

# -------------------- Footer --------------------
st.markdown('---')
st.markdown('**Developed by Sourab. Finoptiv.**')
st.caption('This advanced demo is a feature-rich prototype. For productionization we will add model registries, DP-certified synthetic pipelines, federated secure aggregation, audit logging, and end-to-end tests.')
