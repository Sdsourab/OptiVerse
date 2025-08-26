# OptiNexus — EconoSphere+ (FinOptiv)
# Developed by Sourab. Finoptiv.
# Full single-file Streamlit app (A → Z demo)
# Run: streamlit run OptiNexus_app.py

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

st.set_page_config(page_title='OptiNexus — EconoSphere+', layout='wide')

# ----------------------- THEME / STYLES -----------------------
PRIMARY = '#2b6ef6'
ACCENT = '#1fb6ff'
CARD_BG = '#ffffff'
BG = '#f6f8fa'

# small helper
def pct(x):
    return f"{x*100:.2f}%"

# ----------------------- Demo data generators -----------------------
@st.cache_data
def gen_macro_data(n=120, seed=42):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start='2022-01-01', periods=n, freq='M')
    gdp = 100 + 0.25 * np.arange(n) + 3 * np.sin(np.arange(n) / 6) + rng.normal(0, 1.2, n)
    unemployment = 5 + 0.06 * np.cos(np.arange(n) / 5) + rng.normal(0, 0.25, n)
    wage = 10 + 0.03 * np.arange(n) + rng.normal(0, 0.12, n)
    df = pd.DataFrame({'date': t, 'GDP': gdp, 'Unemployment': unemployment, 'Wage': wage})
    return df

@st.cache_data
def gen_firms(n=12, seed=2):
    rng = np.random.default_rng(seed)
    names = [f'Firm_{i+1}' for i in range(n)]
    size = rng.integers(50, 1200, size=n)
    prod = np.round(rng.normal(1.0, 0.18, n), 3)
    sector = rng.choice(['Agriculture','Manufacturing','Services'], n)
    region = rng.choice(['North','South','East','West'], n)
    df = pd.DataFrame({'Firm': names, 'Size': size, 'Productivity': prod, 'Sector': sector, 'Region': region})
    return df

macro_df = gen_macro_data()
firms_df = gen_firms()

# ----------------------- Sidebar (features nav + scenario controls) -----------------------
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/8/89/HD_transparent_picture.png', width=80)
st.sidebar.title('OptiNexus — EconoSphere+')
st.sidebar.caption('Developed by Sourab. Finoptiv.')
st.sidebar.markdown('---')

st.sidebar.header('Scenario Controls')
scenario_name = st.sidebar.text_input('Scenario name', value='Base / Coastal Factory')
horizon = st.sidebar.selectbox('Horizon', ['6 months','12 months (default)','24 months'])
time_step = st.sidebar.selectbox('Time step', ['Daily','Weekly','Monthly'], index=2)
uploaded_file = st.sidebar.file_uploader('Upload dataset (CSV) — optional', type=['csv','parquet'])

st.sidebar.markdown('---')

FEATURES = [
    'Home','Counterfactual Twin','Nash Bargain','Job Matching ABM','Climate Coupler',
    'SIR↔Econ↔HR','Skill Diffusion & OT','Shadow Economy','Fairness Optimizers',
    'Real Options','Welfare Frontier','Policy Narrative','Federated Capsules','Synthetic Data','Robustness Gym'
]

st.sidebar.header('Features')
selected_feature = st.sidebar.radio('Open feature', FEATURES, index=0)

st.sidebar.markdown('---')
if st.sidebar.button('Run Tournament (demo)'):
    st.sidebar.success('Tournament run queued (demo).')
if st.sidebar.button('Export Report'):
    st.sidebar.info('Exported PDF/PowerPoint (demo).')

# ----------------------- Top KPIs -----------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric('GDP Δ (sim)', '+3.2%')
col2.metric('Unemployment Δ', '-0.5 ppt')
col3.metric('Equity Index', '+3.6')
col4.metric('Robustness', '0.89')

# ----------------------- Feature pages -----------------------
# Helper for common layout
def show_footer():
    st.markdown('---')
    st.caption('Developed by Sourab. Finoptiv. OptiNexus (EconoSphere+).')

# Home
if selected_feature == 'Home':
    st.header('Executive Command Center — Home')
    st.markdown('Consolidated view of active runs, KPIs, and quick-launch tiles.')

    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader('GDP Forecast (with CI)')
        df = macro_df.tail(48)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.date, y=df.GDP, mode='lines', name='GDP', line=dict(width=3, color=PRIMARY)))
        # simple CI band (synthetic)
        mu = df.GDP
        lower = mu - 2.5
        upper = mu + 2.5
        fig.add_traces([go.Scatter(x=df.date, y=upper, name='Upper CI', line=dict(width=0), showlegend=False),
                        go.Scatter(x=df.date, y=lower, name='Lower CI', fill='tonexty', line=dict(width=0), showlegend=False)])
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Policy Tournament Snapshot')
        policies = pd.DataFrame({'policy':['A','B','C','D'],'welfare':[0.72,0.68,0.59,0.45],'fairness':[0.6,0.71,0.5,0.8]})
        fig2 = px.scatter(policies, x='welfare', y='fairness', text='policy', size='welfare', title='Policy space (demo)')
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader('Latest Run Summary')
        st.metric('Run ID', f'run-{np.random.randint(1000,9999)}')
        st.write('Scenario:', scenario_name)
        st.write('Snapshot:', datetime.utcnow().isoformat())
        st.markdown('**Key metrics**')
        st.write('- GDP Δ: +3.2%')
        st.write('- Unemployment Δ: -0.5 ppt')
        st.write('- Robustness: 0.89')

    st.markdown('---')
    st.subheader('Coupling — Applied Econ ↔ HR ↔ MIS')
    shock = st.slider('Shock magnitude', 0.0, 1.0, 0.15)
    coupling = st.slider('Coupling intensity', 0.0, 1.0, 0.5)
    econ_imp = -shock * (1 + 0.6*coupling)
    hr_imp = -0.6*shock * (1 + coupling)
    mis_imp = -0.35*shock * (1 + 0.75*coupling)
    dfc = pd.DataFrame({'Module':['Economy','HR','MIS'],'Impact':[econ_imp,hr_imp,mis_imp]})
    figc = px.bar(dfc, x='Module', y='Impact', color='Module', title='Module Impacts (demo)')
    st.plotly_chart(figc, use_container_width=True)
    show_footer()

# Counterfactual Twin Lab
elif selected_feature == 'Counterfactual Twin':
    st.header('Counterfactual Twin Lab')
    st.markdown('Create a counterfactual twin (synthetic-control-like) and inspect donor diagnostics + CI bands.')
    target = st.selectbox('Target', ['GDP','Unemployment','Wage'])
    intervention_idx = st.slider('Intervention index (month)', 12, len(macro_df)-1, len(macro_df)-13)
    effect = st.number_input('Additive effect (post intervention)', value=3.0)
    donors = ['Unemployment','Wage']
    if st.button('Run Twin'):
        X = macro_df[donors].values
        y = macro_df[target].values
        pre = intervention_idx
        model = Ridge(alpha=1.0)
        model.fit(X[:pre], y[:pre])
        pred = model.predict(X)
        obs = y.copy(); obs[pre+1:] = obs[pre+1:] + effect
        dfp = pd.DataFrame({'date':macro_df.date, 'Observed':obs, 'Counterfactual':pred})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfp.date, y=dfp.Observed, name='Observed', line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=dfp.date, y=dfp.Counterfactual, name='Counterfactual', line=dict(dash='dash')))
        fig.add_vline(x=dfp.date.iloc[pre], line=dict(dash='dot'), annotation_text='Intervention')
        # bootstrap CI
        preds = []
        rng = np.random.default_rng(0)
        for i in range(200):
            idx = rng.choice(pre, pre, replace=True)
            m = Ridge(alpha=1.0); m.fit(X[idx], y[idx])
            preds.append(m.predict(X))
        preds = np.array(preds)
        lower = np.percentile(preds,2.5,axis=0); upper = np.percentile(preds,97.5,axis=0)
        fig.add_traces([go.Scatter(x=dfp.date, y=upper, name='Upper 95%', line=dict(width=0), showlegend=False), go.Scatter(x=dfp.date, y=lower, name='Lower 95%', fill='tonexty', line=dict(width=0), showlegend=False)])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Donor coefficients')
        coefs = pd.Series(model.coef_, index=donors)
        st.bar_chart(coefs)
    show_footer()

# Nash Bargain
elif selected_feature == 'Nash Bargain':
    st.header('Nash Bargain Simulator')
    st.markdown('Interactive Nash bargaining + alternating-offers timeline.')
    U1 = st.slider('Union fallback', 0.0, 1.0, 0.35)
    U2 = st.slider('Management fallback', 0.0, 1.0, 0.45)
    w = st.slider('Union weight', 0.0, 1.0, 0.5)
    rounds = st.slider('Rounds', 1, 12, 6)
    surplus = max(0, 1 - (U1 + U2))
    sol1 = U1 + w*surplus; sol2 = U2 + (1-w)*surplus
    st.metric('Union utility (Nash)', f'{sol1:.3f}'); st.metric('Management utility (Nash)', f'{sol2:.3f}')
    offers = []
    for r in range(1, rounds+1):
        if r%2==1:
            prop = U1 + (sol1-U1)*(r/rounds)
        else:
            prop = U2 + (sol2-U2)*(r/rounds)
        offers.append({'round':r,'proposal':prop})
    st.line_chart(pd.DataFrame(offers).set_index('round'))
    show_footer()

# Job Matching ABM
elif selected_feature == 'Job Matching ABM':
    st.header('Job-Matching ABM')
    T = st.slider('Periods', 6, 96, 36)
    alpha = st.slider('Elasticity α', 0.1, 0.9, 0.6)
    A = st.slider('Efficiency A', 0.1, 3.0, 1.0)
    vac = np.zeros(T); unemp = np.zeros(T); hires = np.zeros(T)
    unemp[0]=200; vac[0]=40
    for t in range(1,T):
        matches = A * (unemp[t-1]**alpha) * (vac[t-1]**(1-alpha))
        hires[t]=matches
        vac[t]=max(5, vac[t-1] + np.random.normal(0,3) + 0.05*(10-matches))
        unemp[t]=max(0, unemp[t-1]-matches + np.random.poisson(3))
    df = pd.DataFrame({'t':np.arange(T),'Vacancies':vac,'Hires':hires,'Unemployed':unemp})
    fig = px.line(df, x='t', y=['Vacancies','Hires','Unemployed'], labels={'value':'Count','t':'Period'})
    st.plotly_chart(fig, use_container_width=True)
    # Sankey
    sank = go.Figure(data=[go.Sankey(node=dict(label=['Vacancies','Applications','Interviews','Hires','Separations']), link=dict(source=[0,0,1,2], target=[1,2,3,4], value=[220,80,60,20]))])
    sank.update_layout(height=380)
    st.plotly_chart(sank, use_container_width=True)
    show_footer()

# Climate Coupler
elif selected_feature == 'Climate Coupler':
    st.header('Climate → Econ → Ops Coupler')
    exposure = st.slider('Avg direct exposure % loss', 0.0, 0.6, 0.12)
    lag = st.slider('Propagation lag (periods)', 0, 6, 2)
    sectors = ['Agriculture','Manufacturing','Services']
    M = np.array([[1.0,0.25,0.15],[0.12,1.0,0.3],[0.07,0.18,1.0]])
    shock = np.array([exposure, exposure*0.65, exposure*0.45])
    B = M - np.eye(3)
    impact = np.linalg.inv(np.eye(3)-0.45*B).dot(shock)
    df = pd.DataFrame({'Sector':sectors,'Impact':impact})
    fig = px.bar(df, x='Sector', y='Impact', color='Sector', title='Sectoral Impact')
    fig.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
    # lead time heatmap (toy)
    lead = np.clip(np.round(1 + 10*impact + np.random.normal(0,0.8,3),2), 0.2, 50)
    st.table(pd.DataFrame({'Sector':sectors,'LeadTimeDelta(days)':lead}))
    show_footer()

# SIR ↔ Econ ↔ HR Runner
elif selected_feature == 'SIR↔Econ↔HR':
    st.header('SIR ↔ Economy ↔ HR Runner')
    beta = st.slider('β', 0.01, 1.0, 0.28)
    gamma = st.slider('γ', 0.01, 1.0, 0.12)
    N = st.number_input('Population', 100, 200000, 5000)
    I0 = st.number_input('Initial infected', 1, N//2, 10)
    days = st.slider('Days', 30, 720, 180)
    def sir(y,t,N,b,g):
        S,I,R = y
        dS = -b*S*I/N
        dI = b*S*I/N - g*I
        dR = g*I
        return dS,dI,dR
    t = np.linspace(0,days-1,days)
    y0 = (N-I0,I0,0)
    res = odeint(sir,y0,t,args=(N,beta,gamma))
    S,I,R = res.T
    df = pd.DataFrame({'day':t,'S':S,'I':I,'R':R})
    fig = px.line(df, x='day', y=['S','I','R'])
    st.plotly_chart(fig, use_container_width=True)
    absentee = I/N
    prod_loss = absentee * 0.6
    st.line_chart(pd.DataFrame({'AbsenteeRate':absentee,'ProdLoss':prod_loss}))
    show_footer()

# Skill Diffusion & OT
elif selected_feature == 'Skill Diffusion & OT':
    st.header('Skill Diffusion & Optimal Transport Planner')
    G = nx.barabasi_albert_graph(14,2,seed=3)
    pos = nx.spring_layout(G, seed=3)
    fig = plt.figure(figsize=(6,3)); nx.draw(G,pos,with_labels=True,node_color='lightsteelblue',node_size=280)
    st.pyplot(fig)
    gamma = st.slider('Diffusion γ', 0.01, 2.0, 0.6)
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G).astype(float).todense()
    x0 = np.zeros(n); x0[0]=1.0
    tgrid = np.linspace(0,4,40); X=[x0]
    for _ in tgrid[1:]: X.append(np.array(X[-1]) - gamma * (L @ np.array(X[-1])).squeeze())
    X = np.array(X)
    st.line_chart(pd.DataFrame(X, columns=[f'node{i}' for i in range(n)]))
    budget = st.slider('Training budget',1,500,80)
    deg = np.array([G.degree(i) for i in range(n)])+0.1
    target = np.zeros(n); target[np.argsort(-deg)[:3]] = 1
    gap = np.maximum(0, target - X[-1])
    alloc = budget * deg * gap / (deg * gap).sum() if (deg*gap).sum()>0 else np.zeros(n)
    st.bar_chart(pd.DataFrame({'node':list(range(n)),'alloc':alloc}).set_index('node'))
    show_footer()

# Shadow Economy
elif selected_feature == 'Shadow Economy':
    st.header('Shadow Economy & Informal Sector Estimator')
    size = st.slider('Sample points', 20, 500, 80)
    electricity = np.linspace(80,140,size) + np.random.normal(0,4,size)
    nightlights = np.linspace(8,25,size) + np.random.normal(0,1.2,size)
    reported = 0.8*electricity + 2.0*nightlights + np.random.normal(0,5,size)
    implied = 0.95*electricity + 2.3*nightlights + np.random.normal(0,5,size)
    latent = implied - reported
    df = pd.DataFrame({'reported':reported,'implied':implied,'latent':latent})
    st.plotly_chart(px.line(df.cumsum(), title='Cumulative reported vs implied (demo)'), use_container_width=True)
    informal_share = st.slider('Assumed informal share (%)', 0, 60, 22)
    st.write('Estimated tax base effect (demo):', f"{informal_share*0.55:.1f}%")
    show_footer()

# Fairness Optimizers
elif selected_feature == 'Fairness Optimizers':
    st.header('Fairness-Constrained Optimizers')
    groups = ['Group A','Group B','Group C']
    applicants = np.array([120,80,60])
    slots = st.number_input('Slots', 1, 1000, 120)
    tol = st.slider('Parity tolerance', 0.0, 0.5, 0.05)
    alloc = (applicants / applicants.sum()) * slots
    rates = alloc / applicants
    if (rates.max() - rates.min()) > tol:
        avg = np.mean(rates); alloc = avg * applicants
    st.table(pd.DataFrame({'group':groups,'alloc_slots':np.round(alloc).astype(int),'applicants':applicants}))
    show_footer()

# Real Options
elif selected_feature == 'Real Options':
    st.header('Real Options Valuation')
    capex = st.number_input('CapEx (M)', 0.1, 500.0, 5.0)
    cash = st.number_input('Annual cashflow (M)', 0.1, 500.0, 1.8)
    r = st.slider('Discount rate', 0.0, 0.2, 0.05)
    sigma = st.slider('Volatility', 0.01, 1.0, 0.3)
    npv = cash / r - capex if r>0 else cash*10 - capex
    option_val = 0.2 * sigma * max(0,npv)
    st.metric('NPV (naive)', f'{npv:.2f} M')
    st.metric('Option value (approx)', f'{option_val:.2f} M')
    show_footer()

# Welfare Frontier
elif selected_feature == 'Welfare Frontier':
    st.header('Social Welfare Frontier (Pareto)')
    n=180
    gdp = np.linspace(80,130,n) + np.random.normal(0,2,n)
    equity = 55 - 0.25*(gdp-100) + np.random.normal(0,1.2,n)
    employ = 100 - 0.4*(gdp-100) + np.random.normal(0,3,n)
    df = pd.DataFrame({'gdp':gdp,'equity':equity,'employ':employ})
    st.plotly_chart(px.scatter(df,x='gdp',y='equity',size='employ',hover_data=['employ'], title='Policy sample space'), use_container_width=True)
    # pareto
    pts = df[['gdp','equity']].values; pareto=[]
    for i,p in enumerate(pts):
        dom=False
        for j,q in enumerate(pts):
            if (q[0]>=p[0] and q[1]>=p[1]) and (q[0]>p[0] or q[1]>p[1]): dom=True; break
        if not dom: pareto.append(i)
    st.plotly_chart(px.scatter(df.iloc[pareto],x='gdp',y='equity',size='employ', color_discrete_sequence=['crimson']), use_container_width=True)
    show_footer()

# Policy Narrative
elif selected_feature == 'Policy Narrative':
    st.header('Policy Narrative Generator')
    persona = st.selectbox('Persona', ['Minister','CEO','Union Rep','Academic'])
    findings = st.text_area('Key findings', 'GDP down 1.2%, unemployment up 0.6%')
    tone = st.selectbox('Tone', ['Concise','Technical','Persuasive'])
    if st.button('Generate'):
        header = f'Brief for {persona} — {scenario_name} — {datetime.utcnow().date()}'
        if persona=='Minister': lead = f'{findings}. Recommend targeted subsidies and re-skilling.'
        elif persona=='CEO': lead = f'{findings}. Recommend supply-chain contingency and cost-control.'
        elif persona=='Union Rep': lead = f'{findings}. Recommend job protection and training guarantees.'
        else: lead = f'{findings}. Methods: coupled simulation, bootstrap CI.'
        st.markdown(f'**{header}**')
        st.write(lead)
    show_footer()

# Federated Capsules
elif selected_feature == 'Federated Capsules':
    st.header('Federated-Lite Capsules (SQLite)')
    st.write('Demo connected capsules: BankA, UnivX, FirmZ')
    if st.button('Simulate capsule sync'):
        st.success('Capsules synced (demo). Meta-model updated.')
    show_footer()

# Synthetic Data
elif selected_feature == 'Synthetic Data':
    st.header('Synthetic Microdata Forge (DP-Copula demo)')
    n = st.slider('Rows', 100, 5000, 500)
    eps = st.slider('Privacy epsilon (demo)', 0.1, 10.0, 1.0)
    base = pd.DataFrame({'age':np.random.randint(18,62,n),'wage':np.random.lognormal(2.4,0.55,n),'skill':np.random.choice(['low','med','high'],n)})
    if st.button('Generate synthetic'):
        synth = base.copy(); synth['wage'] = synth['wage'] + np.random.normal(0,eps,n)
        st.dataframe(synth.head(80))
        st.success('Synthetic dataset generated (demo, not certified DP).')
    show_footer()

# Robustness Gym
elif selected_feature == 'Robustness Gym':
    st.header('Robustness Gym & Stress-Test Harness')
    shift = st.selectbox('Shift type', ['Covariate shift','Label noise','Adversarial'])
    severity = st.slider('Severity', 0, 100, 25)
    if st.button('Run stress test'):
        baseline = 0.9
        degraded = baseline - (severity/100)*0.55
        st.metric('Baseline (AUC)', f'{baseline:.3f}'); st.metric('Post-shift (AUC)', f'{degraded:.3f}')
        st.write('Stability score:', f'{max(0,degraded/baseline):.3f}')
    show_footer()

# ----------------------- End -----------------------
# Note: This demo is intentionally self-contained and uses synthetic data. Productionization requires
# model registries, privacy-preserving synthesis (DP), federated secure aggregation, audit logs, and tests.
