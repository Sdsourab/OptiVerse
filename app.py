# EconoSphere+ (OptiNexus) - Streamlit Demo App
# Single-file prototype demonstrating all features for FinOptiv
# Run with: streamlit run this_file.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from math import exp
from scipy.integrate import odeint

st.set_page_config(page_title='OptiNexus - EconoSphere+', layout='wide')

# ----------------------- Helper: demo data -----------------------
@st.cache_data
def gen_macro_data(n=60, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    gdp = 100 + 0.2*t + 2*np.sin(t/6) + rng.normal(0,1,n)
    unemployment = 5 + 0.05*np.cos(t/5) + rng.normal(0,0.2,n)
    wage = 10 + 0.02*t + rng.normal(0,0.1,n)
    df = pd.DataFrame({'t':t,'GDP':gdp,'Unemployment':unemployment,'Wage':wage})
    return df

@st.cache_data
def gen_firm_panel(n_firms=8, seed=1):
    rng = np.random.default_rng(seed)
    firms = [f'Firm_{i+1}' for i in range(n_firms)]
    size = rng.integers(50,1000,size=n_firms)
    productivity = rng.normal(1.0,0.15,size=n_firms)
    region = rng.choice(['North','South','East','West'], size=n_firms)
    df = pd.DataFrame({'Firm':firms,'Size':size,'Productivity':productivity,'Region':region})
    return df

macro_df = gen_macro_data()
firms_df = gen_firm_panel()

# ----------------------- Title & global controls -----------------------
st.title('OptiNexus — EconoSphere+ (FinOptiv)')
st.markdown('**Demo Streamlit frontend** showcasing all platform features. Use the left sidebar to navigate and interact.')

# Sidebar: global scenario controls
with st.sidebar:
    st.header('Scenario Controls')
    scenario_name = st.text_input('Scenario name', value='Base / Coastal Factory')
    horizon = st.selectbox('Horizon', ['6 months','12 months (default)','24 months'])
    time_step = st.selectbox('Time step', ['Daily','Weekly','Monthly'], index=2)
    upload = st.file_uploader('Upload dataset (CSV) — optional', type=['csv','parquet'])
    st.markdown('---')
    st.header('Run & Export')
    run_tournament = st.button('▶ Run Multi-Scenario Tournament')
    export_report = st.button('Export Report (PDF/PowerPoint - demo)')

# Quick KPI bar
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric('GDP Δ (sim)', '3.8%')
kpi2.metric('Unemployment Δ', '-0.4%')
kpi3.metric('Equity Index', '+4')
kpi4.metric('Robustness Score', '0.87')

# ----------------------- Tabs for each feature -----------------------
tabs = st.tabs(['Home','Counterfactual Twin','Nash Bargain','Job Matching ABM','Climate Coupler',
                'SIR↔Econ↔HR','Skill Diffusion & OT','Shadow Economy','Fairness Optimizers',
                'Real Options','Welfare Frontier','Policy Narrative','Federated Capsules','Synthetic Data','Robustness Gym'])

# ----------------------- 0) HOME -----------------------
with tabs[0]:
    st.header('Home — Consolidated Dashboard')
    st.markdown('This homepage aggregates all feature outputs. Use tiles below to open feature demos.')
    # small grid of charts
    col1, col2, col3 = st.columns([1.2,1,1])
    with col1:
        st.subheader('Economic Forecast')
        fig = px.line(macro_df, x='t', y='GDP', labels={'t':'Time','GDP':'GDP proxy'})
        fig.add_traces(px.line(macro_df, x='t', y='GDP').update_traces(line=dict(dash='dash')).data)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader('Job Matching (summ)')
        jm = pd.DataFrame({'Stage':['Vacancies','Applications','Hires','Separations'], 'Value':[120,900,85,60]})
        st.bar_chart(jm.set_index('Stage'))
    with col3:
        st.subheader('Policy Optimization view')
        x = np.random.normal(size=40)
        y = np.random.normal(size=40)
        s = np.abs(np.random.normal(10,6,size=40))
        fig2 = px.scatter(x=x,y=y,size=s, labels={'x':'Welfare','y':'Inequality'})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('---')
    st.subheader('Tri-Coupler (Applied Econ ↔ HR ↔ MIS) — Big Picture')
    st.markdown('Adjust the shock and see how the three modules react. This demo shows simplified coupling logic.')
    shock = st.slider('Shock magnitude (e.g., flood severity % loss)', 0.0, 1.0, 0.15)
    coupling = st.slider('Coupling intensity (how strongly modules influence each other)',0.0,1.0,0.45)
    # simplified coupling: compute impacts
    econ_impact = -shock * (1 + 0.5*coupling)
    hr_impact = -0.5*shock * (1 + coupling)
    mis_impact = -0.3*shock * (1 + 0.8*coupling)
    df_coupler = pd.DataFrame({'Module':['Economy','HR','MIS'],'Impact':[econ_impact,hr_impact,mis_impact]})
    figc = px.bar(df_coupler, x='Module', y='Impact', labels={'Impact':'% change (relative)'})
    st.plotly_chart(figc, use_container_width=True)

# ----------------------- 1) Counterfactual Twin -----------------------
with tabs[1]:
    st.header('Counterfactual Twin Lab')
    st.markdown('Create a counterfactual twin by choosing an intervention and a donor series. Demo uses a simple synthetic-control-like approach (ridge regression).')
    colA, colB = st.columns(2)
    with colA:
        st.subheader('Select target & intervention')
        target = st.selectbox('Target series', ['GDP','Unemployment','Wage'])
        intervention_time = st.slider('Intervention time (index)', int(macro_df.t.min()), int(macro_df.t.max()), int(macro_df.t.max()-12))
        effect = st.slider('Simulated intervention effect (additive)', -10.0, 10.0, 3.0)
        run_cf = st.button('Create Counterfactual Twin')
    with colB:
        st.subheader('Donor series (auto-selected)')
        st.write('Using other macro variables as donors')
        st.write(macro_df.head())
    if run_cf:
        # simple approach: fit linear model on pre-intervention using donors
        from sklearn.linear_model import Ridge
        donors = macro_df[['Unemployment','Wage']].values
        y = macro_df[target].values
        pre = intervention_time
        model = Ridge(alpha=1.0)
        model.fit(donors[:pre], y[:pre])
        pred = model.predict(donors)
        # create observed with intervention
        observed = y.copy()
        observed[intervention_time:] = observed[intervention_time:] + effect
        df_plot = pd.DataFrame({'t':macro_df.t, 'Observed':observed, 'Counterfactual':pred})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot.t, y=df_plot.Observed, name='Observed (with intervention)'))
        fig.add_trace(go.Scatter(x=df_plot.t, y=df_plot.Counterfactual, name='Counterfactual (twin)', line=dict(dash='dash')))
        fig.add_vline(x=intervention_time, line=dict(dash='dot'))
        st.plotly_chart(fig, use_container_width=True)
        # show uncertainty band via bootstrap
        st.caption('Uncertainty band (bootstrap)')
        # quick bootstrap
        preds = []
        rng = np.random.default_rng(0)
        for i in range(100):
            idx = rng.choice(pre, pre, replace=True)
            model = Ridge(alpha=1.0)
            model.fit(donors[idx], y[idx])
            preds.append(model.predict(donors))
        preds = np.array(preds)
        lower = np.percentile(preds,2.5,axis=0)
        upper = np.percentile(preds,97.5,axis=0)
        fig.add_traces([go.Scatter(x=df_plot.t, y=upper, name='Upper 95%'), go.Scatter(x=df_plot.t, y=lower, name='Lower 95%', fill='tonexty')])
        st.plotly_chart(fig, use_container_width=True)

# ----------------------- 2) Nash Bargain Simulator -----------------------
with tabs[2]:
    st.header('Nash Bargain Simulator (Union ↔ Management)')
    st.markdown('Alternating-offers inspired simplified simulator. Adjust utilities, fallback options, and bargaining power.')
    u_min = st.slider('Union utility baseline', 0.0, 1.0, 0.4)
    u_man = st.slider('Management utility baseline', 0.0, 1.0, 0.6)
    bargaining_power = st.slider('Union bargaining weight (0..1)', 0.0, 1.0, 0.5)
    rounds = st.number_input('Rounds of offers', min_value=1, max_value=20, value=6)

    def nash_solution(U1_base, U2_base, w=0.5):
        # simple closed-form Nash bargaining: maximize product of gains -> shares of surplus
        # assume disagreement utilities are U1_base, U2_base; total pie normalized to 1
        surplus = 1 - (U1_base + U2_base)
        if surplus<=0:
            return U1_base, U2_base
        x1 = U1_base + w*surplus
        x2 = U2_base + (1-w)*surplus
        return x1, x2

    x1,x2 = nash_solution(u_min, u_man, bargaining_power)
    st.metric('Union agreement utility', f'{x1:.3f}')
    st.metric('Management agreement utility', f'{x2:.3f}')
    st.markdown('Simulating alternating offers (simplified):')
    offers = []
    for r in range(int(rounds)):
        # alternation: union proposes a split favoring itself by small concession
        if r%2==0:
            prop = (u_min + (x1-u_min)* (r+1)/rounds)
        else:
            prop = (u_man + (x2-u_man)* (r+1)/rounds)
        offers.append({'round':r+1,'proposal':prop})
    st.table(pd.DataFrame(offers))

# ----------------------- 3) Job Matching ABM -----------------------
with tabs[3]:
    st.header('Job-Matching & Vacancy Flow ABM (Mortensen–Pissarides-lite)')
    st.markdown('A simple agent-based approximation of vacancy→hire→turnover dynamics using a matching function.')
    n_periods = st.slider('Sim periods', 6, 48, 24)
    alpha = st.slider('Matching elasticity α', 0.1, 1.0, 0.6)
    A = st.slider('Matching efficiency A', 0.1, 5.0, 1.0)
    vacancies = np.zeros(n_periods)
    hires = np.zeros(n_periods)
    unemployed = np.zeros(n_periods)
    unemployed[0]=50
    vacancies[0]=20
    for t in range(1,n_periods):
        matches = A * (unemployed[t-1]**alpha) * (vacancies[t-1]**(1-alpha))
        hires[t] = matches
        # dynamics: new vacancies respond to recent hires
        vacancies[t] = max(5, vacancies[t-1] + np.random.normal(0,2) + 0.2*(10-matches))
        unemployed[t] = max(0, unemployed[t-1] - matches + np.random.poisson(2))
    dfm = pd.DataFrame({'t':np.arange(n_periods),'Vacancies':vacancies,'Hires':hires,'Unemployed':unemployed})
    figm = px.line(dfm, x='t', y=['Vacancies','Hires','Unemployed'])
    st.plotly_chart(figm, use_container_width=True)
    st.markdown('Sankey-style flow (approx)')
    try:
        # simplified sankey counts
        import plotly
        sankey = dict(
            type='sankey',
            node=dict(label=['Vacancies','Applications','Interviews','Hires','Separations']),
            link=dict(source=[0,0,1,2], target=[1,2,3,4], value=[100,40,30,10])
        )
        fig_s = go.Figure(data=[sankey])
        st.plotly_chart(fig_s, use_container_width=True)
    except Exception:
        st.info('Sankey not available in this environment')

# ----------------------- 4) Climate–Econ–Ops Shock Coupler -----------------------
with tabs[4]:
    st.header('Climate→Econ→Ops Shock Coupler')
    st.markdown('Simple sectoral propagation using a sector-exposure matrix and delay kernel.')
    sectors = ['Agriculture','Manufacturing','Services']
    exposure = st.slider('Direct exposure to shock (avg % loss)', 0.0, 0.8, 0.12)
    lag = st.slider('Average propagation lag (periods)', 0, 6, 2)
    # simple IO propagation matrix
    M = np.array([[1.0,0.2,0.1],[0.1,1.0,0.3],[0.05,0.2,1.0]])
    shock_vec = np.array([exposure, exposure*0.6, exposure*0.4])
    # propagate: (I - B)^-1 * shock approx
    B = M - np.eye(3)
    try:
        impact = np.linalg.inv(np.eye(3)-0.5*B).dot(shock_vec)
    except Exception:
        impact = shock_vec
    df_imp = pd.DataFrame({'Sector':sectors, 'Impact':impact})
    st.bar_chart(df_imp.set_index('Sector'))
    st.markdown('Supply chain heatmap (simulated lead time increases)')
    lead_times = np.clip(np.random.normal(2+5*impact, 0.6), 0.5, 30)
    st.table(pd.DataFrame({'Sector':sectors,'LeadTimeDelta(days)':np.round(lead_times,2)}))

# ----------------------- 5) SIR ↔ Economy ↔ HR Epidemic Runner -----------------------
with tabs[5]:
    st.header('SIR ↔ Economy ↔ HR Epidemic Runner')
    st.markdown('Coupled SIR and workforce availability simulation. Change transmission, hospitalization, and see absenteeism impact.')
    beta = st.slider('Transmission β', 0.0, 1.0, 0.3)
    gamma = st.slider('Recovery γ', 0.0, 1.0, 0.1)
    population = st.number_input('Population (workforce)', min_value=100, max_value=100000, value=1000)
    I0 = st.number_input('Initial Infected', min_value=1, max_value=population//2, value=5)
    days = st.slider('Sim days', 10, 365, 120)

    def sir_model(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, days-1, days)
    y0 = (population - I0, I0, 0)
    ret = odeint(sir_model, y0, t, args=(population,beta,gamma))
    S,I,R = ret.T
    df_sir = pd.DataFrame({'day':t,'S':S,'I':I,'R':R})
    fig_sir = px.line(df_sir, x='day', y=['S','I','R'])
    st.plotly_chart(fig_sir, use_container_width=True)
    st.markdown('Estimate workforce available and absenteeism-driven productivity loss')
    absentee_rate = I / population
    productivity_loss = 0.5 * absentee_rate
    st.line_chart(pd.DataFrame({'AbsenteeRate':absentee_rate,'ProdLoss':productivity_loss}), use_container_width=True)

# ----------------------- 6) Skill Diffusion + Optimal Transport Planner -----------------------
with tabs[6]:
    st.header('Skill Diffusion & Optimal Transport (reskilling planner)')
    st.markdown('Graph diffusion model (heat equation on graph) + a simplified OT-based allocation (entropy-regularized approx).')
    # skill network
    G = nx.erdos_renyi_graph(10, 0.25, seed=2)
    pos = nx.spring_layout(G, seed=2)
    fig_net = plt.figure(figsize=(5,3))
    nx.draw(G, pos, with_labels=True, node_color='skyblue')
    st.pyplot(fig_net)
    st.markdown('Simulate diffusion on skills: initial shock to node 0')
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G).todense()
    x0 = np.zeros(n); x0[0]=1.0
    tgrid = np.linspace(0,5,50)
    gamma = st.slider('Diffusion rate γ', 0.01, 2.0, 0.5)
    X = [x0]
    for _ in tgrid[1:]:
        X.append(X[-1] - gamma * (L @ X[-1]))
    X = np.array(X)
    st.line_chart(pd.DataFrame(X, columns=[f'node{i}' for i in range(n)]))
    st.markdown('OT planner (simplified): allocate training budget to nodes to minimize distance to target profile')
    budget = st.slider('Training budget (units)', 1, 100, 20)
    target = np.zeros(n); target[np.argsort(-np.array([G.degree(i) for i in range(n)]))[:3]] = 1
    # naive allocation: proportional to degree * gap
    deg = np.array([G.degree(i) for i in range(n)])+0.1
    gap = np.maximum(0, target - X[-1])
    alloc = budget * deg * gap / (deg * gap).sum() if (deg*gap).sum()>0 else np.zeros(n)
    df_alloc = pd.DataFrame({'node':list(range(n)),'alloc':alloc})
    st.bar_chart(df_alloc.set_index('node'))

# ----------------------- 7) Shadow Economy Estimator -----------------------
with tabs[7]:
    st.header('Shadow Economy & Informal Sector Estimator')
    st.markdown('Estimate latent informal output using proxies (electricity, nightlights). This demo uses a simple regression / residual approach.')
    proxies = pd.DataFrame({'electricity': np.linspace(80,120,20) + np.random.normal(0,3,20), 'nightlights': np.linspace(10,20,20) + np.random.normal(0,1,20)})
    # assume official GDP reported lower than proxy-implied
    reported = proxies['electricity']*0.8 + proxies['nightlights']*2 + np.random.normal(0,2,20)
    implied = proxies['electricity']*0.95 + proxies['nightlights']*2.2
    latent = implied - reported
    df_shadow = pd.DataFrame({'reported':reported,'implied':implied,'latent':latent})
    st.line_chart(df_shadow)
    st.markdown('Policy sensitivity: if informal share changes, show employment & tax base effects')
    informal_share = st.slider('Assumed informal share (%)', 0, 50, 22)
    st.write('Estimated tax base reduction:', f"{informal_share*0.6:.1f}% (demo calc)")

# ----------------------- 8) Fairness-constrained Optimizers -----------------------
with tabs[8]:
    st.header('Fairness-Constrained Optimizers (hiring allocation demo)')
    st.markdown('We allocate training slots subject to demographic parity constraint (simple projection).')
    groups = ['A','B']
    applicants = np.array([120,80])
    score = np.array([0.6,0.5])
    budget_slots = st.number_input('Training slots budget', min_value=1, max_value=500, value=60)
    parity_tol = st.slider('Demographic parity tolerance (abs difference)', 0.0, 0.5, 0.05)
    # naive fair allocation: proportional to applicants but clamp to parity
    alloc = (applicants / applicants.sum()) * budget_slots
    # enforce parity by smoothing
    diff = alloc[0]/applicants[0] - alloc[1]/applicants[1]
    if abs(diff) > parity_tol:
        avg_rate = (alloc[0]/applicants[0] + alloc[1]/applicants[1]) / 2
        alloc = np.array([avg_rate*applicants[0], avg_rate*applicants[1]])
    st.table(pd.DataFrame({'group':groups,'alloc_slots':alloc.astype(int),'applicants':applicants}))

# ----------------------- 9) Real Options for Investment -----------------------
with tabs[9]:
    st.header('Real Options for Investment (LSM demo)')
    st.markdown('This demo shows option value of waiting vs investing now using a simplified binomial approximation.')
    invest_cost = st.number_input('CapEx cost (M)', min_value=0.1, max_value=100.0, value=5.0)
    vol = st.slider('Project volatility (σ)', 0.01, 1.0, 0.25)
    r = st.slider('Discount rate (annual)', 0.0, 0.2, 0.05)
    # simplified: value ~ max(E[PV future cashflows] - cost, 0) ; option adds value of waiting
    expected_cashflow = st.number_input('Expected annual cashflow (M)', min_value=0.1, max_value=200.0, value=1.2)
    # naive NPV
    npv = expected_cashflow / r - invest_cost if r>0 else expected_cashflow*10 - invest_cost
    st.metric('NPV (naive)', f'{npv:.2f} M')
    # option premium approx proportional to volatility
    option_premium = vol * npv * 0.2
    st.metric('Real option value (approx)', f'{option_premium:.2f} M')

# ----------------------- 10) Social Welfare Frontier (MCDA-Pareto) -----------------------
with tabs[10]:
    st.header('Social Welfare Frontier (Pareto)')
    st.markdown('Show trade-offs between GDP, Employment, Equity. Interactive slider changes stakeholder weights and highlights optimal policy points.')
    n_points = 120
    gdp = np.linspace(80,120,n_points) + np.random.normal(0,2,n_points)
    employ = 100 - 0.5*(gdp-100) + np.random.normal(0,1,n_points)
    equity = 50 - 0.2*(gdp-100) + np.random.normal(0,1,n_points)
    df_pf = pd.DataFrame({'gdp':gdp,'employ':employ,'equity':equity})
    fig_pf = px.scatter(df_pf, x='gdp', y='equity', size='employ', hover_data=['employ'])
    st.plotly_chart(fig_pf, use_container_width=True)
    st.markdown('Highlight Pareto frontier by dominance filter')
    # compute pareto front (maximize gdp & equity)
    pts = df_pf[['gdp','equity']].values
    pareto_idx = []
    for i,p in enumerate(pts):
        dominated = False
        for j,q in enumerate(pts):
            if (q[0]>=p[0] and q[1]>=p[1]) and (q[0]>p[0] or q[1]>p[1]):
                dominated = True; break
        if not dominated: pareto_idx.append(i)
    st.write('Pareto points:', len(pareto_idx))
    st.plotly_chart(px.scatter(df_pf.iloc[pareto_idx], x='gdp', y='equity', size='employ', color_discrete_sequence=['crimson']), use_container_width=True)

# ----------------------- 11) Policy Narrative Generator -----------------------
with tabs[11]:
    st.header('Policy Narrative Generator (Persona-aware)')
    persona = st.selectbox('Persona', ['Minister','CEO','Union Rep','Academic'])
    key_findings = st.text_area('Key findings (brief)', 'GDP down 1.2%, unemployment up 0.6%, major disruptions in manufacturing')
    tone = st.selectbox('Tone', ['Concise','Technical','Persuasive'], index=0)
    if st.button('Generate Narrative'):
        # simple template-based generation
        if persona=='Minister':
            lead = f'Brief for Minister — TL;DR: {key_findings}. Recommend a temporary wage subsidy and targeted re-skilling.'
        elif persona=='CEO':
            lead = f'Executive summary for CEO — {key_findings}. Recommend supply-chain contingency, cost control, and selective hiring.'
        elif persona=='Union Rep':
            lead = f'Note for Union — {key_findings}. Focus on job protection, fair compensation, and rapid re-skilling.'
        else:
            lead = f'Academic brief — {key_findings}. Methods: coupling model, sensitivity analysis, bootstrap CI.'
        st.markdown('**Generated Narrative:**')
        st.write(lead)

# ----------------------- 12) Federated-Lite Collaboration (SQLite Capsules) -----------------------
with tabs[12]:
    st.header('Federated-Lite Collaboration (SQLite Capsules)')
    st.markdown('Demo of capsules: each capsule holds local summary stats and only summaries are shared.')
    st.write('Connected capsules: BankA, UnivX, FirmZ (demo)')
    if st.button('Simulate Federated Update'):
        st.success('Federated summary received. Meta-model updated (demo).')

# ----------------------- 13) Synthetic Microdata Forge (DP-Copula) -----------------------
with tabs[13]:
    st.header('Synthetic Microdata Forge (DP-Copula demo)')
    st.markdown('Create synthetic microdata with privacy noise (demo uses Gaussian noise as placeholder for DP mechanisms).')
    n = st.slider('Rows', 100, 5000, 500)
    base = pd.DataFrame({'age':np.random.randint(18,60,n), 'wage':np.random.lognormal(2.5,0.5,n), 'skill':np.random.choice(['low','med','high'],n)})
    if st.button('Generate Synthetic'):
        eps = st.slider('Privacy epsilon (smaller = more noise)', 0.1, 10.0, 1.0)
        synth = base.copy()
        synth['wage'] = synth['wage'] + np.random.normal(0, eps, n)
        st.dataframe(synth.head(50))
        st.success('Synthetic dataset generated (demo, not true DP)')

# ----------------------- 14) Robustness Gym & Stress-Test Harness -----------------------
with tabs[14]:
    st.header('Robustness Gym & Stress-Test Harness')
    st.markdown('Run distribution-shift / adversarial perturbations on models and compute stability metrics.')
    shift = st.selectbox('Type of shift', ['Covariate shift','Label noise','Adversarial'])
    severity = st.slider('Severity', 0, 100, 20)
    if st.button('Run Stress Test'):
        # simple demo: degrade metric linearly with severity
        baseline_acc = 0.88
        new_acc = baseline_acc - (severity/100)*0.5
        st.metric('Baseline metric (AUC)', f'{baseline_acc:.2f}')
        st.metric('Post-shift metric (AUC)', f'{new_acc:.2f}')
        st.write('Stability score:', max(0, (new_acc/baseline_acc)))

# ----------------------- Footer -----------------------
st.markdown('---')
st.caption('Demo built for FinOptiv — OptiNexus (EconoSphere+). This single-file prototype demonstrates UI/UX and simplified logic for every requested feature. For production, each module requires thorough engineering, provenance, privacy and testing layers.')
