import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Social Media & Well-being",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background: #0f1117; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d27 100%); }

    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border: 1px solid #2d3250;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7c83fd; }
    .metric-label { font-size: 0.85rem; color: #8b8fa8; margin-top: 4px; }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        border-left: 4px solid #7c83fd;
        padding-left: 14px;
        margin: 24px 0 16px 0;
    }

    .cluster-card {
        border-radius: 14px;
        padding: 18px;
        margin: 8px 0;
        border-left: 5px solid;
    }
    .insight-box {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 12px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd, #5c6bc0);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 131, 253, 0.4);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv('Students Social Media Addiction.csv')

    df_enc = df.copy()
    le = LabelEncoder()
    cat_cols = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform',
                'Affects_Academic_Performance', 'Relationship_Status']
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col])

    num_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']
    scaler = StandardScaler()
    df_scaled = df_enc.copy()
    df_scaled[num_cols] = scaler.fit_transform(df_enc[num_cols])

    # K-Means
    cluster_features = ['Avg_Daily_Usage_Hours', 'Mental_Health_Score',
                        'Sleep_Hours_Per_Night', 'Addicted_Score', 'Conflicts_Over_Social_Media']
    X_cluster = df_scaled[cluster_features]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)

    cluster_profile = df.groupby('Cluster')[cluster_features].mean()
    usage_order = cluster_profile['Avg_Daily_Usage_Hours'].sort_values().index.tolist()
    label_list = ["🔵 Healthy Minimalists", "🟢 Balanced Users",
                  "🟡 Moderate Scrollers", "🔴 At-Risk / Addicted"]
    cluster_labels = {c: label_list[i] for i, c in enumerate(usage_order)}
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

    # Linear Regression
    reg_features = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                    'Conflicts_Over_Social_Media', 'Addicted_Score',
                    'Affects_Academic_Performance', 'Age']
    X = df_enc[reg_features]
    y = df_enc['Mental_Health_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cluster)

    return (df, df_enc, df_scaled, kmeans, cluster_labels, cluster_feature_scaler_params(df, cluster_features),
            lr, X_test, y_test, y_pred, X_pca, cluster_features, reg_features)

def cluster_feature_scaler_params(df, features):
    return {f: (df[f].mean(), df[f].std()) for f in features}

(df, df_enc, df_scaled, kmeans, cluster_labels, cluster_params,
 lr_model, X_test, y_test, y_pred, X_pca, cluster_features, reg_features) = load_and_prepare()

cluster_profile = df.groupby('Cluster')[cluster_features].mean()
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

CLUSTER_COLORS = {
    "🔵 Healthy Minimalists": "#3498db",
    "🟢 Balanced Users": "#2ecc71",
    "🟡 Moderate Scrollers": "#f39c12",
    "🔴 At-Risk / Addicted": "#e74c3c",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📱 Social Media & Well-being")
    st.markdown("*Student Behavioral Analysis*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🤖 Predict Mental Health", "🔍 Find My Cluster"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Dataset Stats**")
    st.markdown(f"- 📋 **{len(df):,}** student records")
    st.markdown(f"- 🌍 **{df['Country'].nunique()}** countries")
    st.markdown(f"- 📱 **{df['Most_Used_Platform'].nunique()}** platforms")
    st.markdown(f"- 🔗 [Dataset on Kaggle](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)")
    st.divider()
    st.caption("Group Project | AI & Business | Feb 2025")

# ═══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# 📊 Student Social Media & Well-being Dashboard")
    st.markdown("*Exploring how social media usage impacts mental health, sleep, and academic performance*")
    st.divider()

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("👥 Total Students", f"{len(df):,}")
    with col2:
        st.metric("⏱ Avg Daily Usage", f"{df['Avg_Daily_Usage_Hours'].mean():.1f} hrs")
    with col3:
        st.metric("🧠 Avg Mental Health", f"{df['Mental_Health_Score'].mean():.1f}/10")
    with col4:
        st.metric("😴 Avg Sleep", f"{df['Sleep_Hours_Per_Night'].mean():.1f} hrs")
    with col5:
        pct_affected = (df['Affects_Academic_Performance'] == 'Yes').mean() * 100
        st.metric("📚 Academic Impact", f"{pct_affected:.0f}%")

    st.divider()

    # Row 1: Usage Distribution + Platform
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Daily Usage Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Avg_Daily_Usage_Hours', nbins=25,
                           color='Academic_Level',
                           barmode='overlay',
                           title='',
                           color_discrete_sequence=['#7c83fd', '#f39c12', '#2ecc71'],
                           template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend_title='Academic Level', height=320,
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Most Used Platforms</div>', unsafe_allow_html=True)
        platform_counts = df['Most_Used_Platform'].value_counts().reset_index()
        platform_counts.columns = ['Platform', 'Count']
        fig = px.bar(platform_counts, x='Count', y='Platform', orientation='h',
                     color='Count', color_continuous_scale='Viridis',
                     template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(t=10, b=10), showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Correlation + Scatter
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">Usage vs Mental Health</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='Avg_Daily_Usage_Hours', y='Mental_Health_Score',
                         color='Addicted_Score', size='Conflicts_Over_Social_Media',
                         color_continuous_scale='RdYlGn_r',
                         hover_data=['Academic_Level', 'Country', 'Sleep_Hours_Per_Night'],
                         labels={'Avg_Daily_Usage_Hours': 'Daily Usage (hrs)',
                                 'Mental_Health_Score': 'Mental Health Score',
                                 'Addicted_Score': 'Addiction Score'},
                         template='plotly_dark')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        num_cols_heat = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
                         'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score', 'Age']
        corr = df[num_cols_heat].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu', zmid=0, text=corr.values.round(2),
            texttemplate='%{text}', showscale=True
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Cluster PCA
    st.markdown('<div class="section-header">Behavioral Segments (K-Means Clusters)</div>', unsafe_allow_html=True)
    col_e, col_f = st.columns([2, 1])
    with col_e:
        pca_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1],
                               'Cluster': df['Cluster_Label'],
                               'Usage': df['Avg_Daily_Usage_Hours'],
                               'MentalHealth': df['Mental_Health_Score']})
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                         color_discrete_map={v: c for v, c in CLUSTER_COLORS.items()},
                         hover_data=['Usage', 'MentalHealth'],
                         template='plotly_dark',
                         labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=380, margin=dict(t=10, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=-0.3)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.markdown("**Cluster Profiles**")
        for cluster_id in cluster_profile.index:
            label = cluster_labels[cluster_id]
            color = list(CLUSTER_COLORS.values())[list(CLUSTER_COLORS.keys()).index(label)]
            n = (df['Cluster'] == cluster_id).sum()
            row = cluster_profile.loc[cluster_id]
            st.markdown(f"""
            <div class="cluster-card" style="border-color:{color}; background:rgba(0,0,0,0.2);">
                <b style="color:{color}">{label}</b><br>
                <small>n={n} students</small><br>
                📱 {row['Avg_Daily_Usage_Hours']:.1f}h usage | 
                🧠 {row['Mental_Health_Score']:.1f}/10 MH<br>
                😴 {row['Sleep_Hours_Per_Night']:.1f}h sleep | 
                🎯 {row['Addicted_Score']:.1f}/10 addiction
            </div>
            """, unsafe_allow_html=True)

    # Business insights
    st.divider()
    st.markdown('<div class="section-header">💼 Business Insights</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="insight-box">
        <b>📉 Demand-Supply Dynamics</b><br><br>
        Platforms <em>supply</em> infinite engagement loops. 
        Students <em>demand</em> social validation, leading to 
        overuse that degrades their mental health "capital."
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <b>⚠️ Risk Analysis</b><br><br>
        At-Risk segment shows >7h daily usage with mental health 
        scores below 5/10 — signalling burnout and academic 
        performance risk requiring early intervention.
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="insight-box">
        <b>💡 Revenue Opportunities</b><br><br>
        Digital wellness apps can target the <em>At-Risk</em> 
        cluster (~25% of students) with premium screen-time 
        coaching and mindfulness subscription products.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 2 — PREDICT MENTAL HEALTH
# ═══════════════════════════════════════════════════════════
elif page == "🤖 Predict Mental Health":
    st.markdown("# 🤖 Mental Health Score Predictor")
    st.markdown("*Linear Regression model trained on 706 student records*")
    st.divider()

    # Model Performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2:.3f}", f"{r2*100:.1f}% variance explained")
    with col2:
        st.metric("MAE", f"{mae:.3f}", "Mean Absolute Error")
    with col3:
        st.metric("RMSE", f"{rmse:.3f}", "Root Mean Squared Error")

    st.divider()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### 🎛 Enter Student Profile")
        usage_hrs = st.slider("📱 Daily Social Media Usage (hours)", 0.5, 12.0, 5.0, 0.1)
        sleep_hrs = st.slider("😴 Sleep Hours per Night", 3.0, 12.0, 6.5, 0.1)
        conflicts = st.slider("💬 Conflicts Over Social Media (count)", 0, 10, 2)
        addicted = st.slider("🎯 Addiction Score (1-10)", 1, 10, 5)
        affects_acad = st.selectbox("📚 Affects Academic Performance?", ["Yes", "No"])
        age = st.slider("🎂 Age", 16, 25, 20)
        affects_enc = 1 if affects_acad == "Yes" else 0

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Mental Health Score")

    with col_right:
        st.markdown("### 📊 Model Insights")
        coef_df = pd.DataFrame({
            'Feature': ['Daily Usage', 'Sleep Hours', 'Conflicts', 'Addiction Score', 'Affects Academics', 'Age'],
            'Coefficient': lr_model.coef_
        }).sort_values('Coefficient', key=abs, ascending=True)

        fig = go.Figure(go.Bar(
            x=coef_df['Coefficient'],
            y=coef_df['Feature'],
            orientation='h',
            marker_color=['#e74c3c' if c < 0 else '#2ecc71' for c in coef_df['Coefficient']],
            text=[f"{c:.3f}" for c in coef_df['Coefficient']],
            textposition='outside'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=280, margin=dict(t=10, b=10, l=10, r=60),
            xaxis_title='Effect on Mental Health Score',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Actual vs Predicted
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode='markers',
            marker=dict(color='#7c83fd', opacity=0.6, size=7),
            name='Predictions'
        ))
        min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig2.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode='lines', line=dict(color='#e74c3c', dash='dash', width=2),
            name='Perfect Prediction'
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=250, margin=dict(t=10, b=10),
            xaxis_title='Actual Score', yaxis_title='Predicted Score',
            font=dict(color='white'), legend=dict(orientation='h')
        )
        st.plotly_chart(fig2, use_container_width=True)

    if predict_btn:
        input_arr = np.array([[usage_hrs, sleep_hrs, conflicts, addicted, affects_enc, age]])
        pred_score = lr_model.predict(input_arr)[0]
        pred_score = np.clip(pred_score, 1, 10)

        st.divider()
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        with col_res2:
            color = "#2ecc71" if pred_score >= 7 else "#f39c12" if pred_score >= 5 else "#e74c3c"
            status = "✅ Good" if pred_score >= 7 else "⚠️ Moderate" if pred_score >= 5 else "🚨 At Risk"
            st.markdown(f"""
            <div style="text-align:center; background:linear-gradient(135deg,#1e2130,#252b40);
                        border-radius:20px; padding:32px; border: 2px solid {color};">
                <div style="font-size:1rem; color:#8b8fa8; margin-bottom:8px;">Predicted Mental Health Score</div>
                <div style="font-size:4.5rem; font-weight:800; color:{color};">{pred_score:.1f}<span style="font-size:1.5rem; color:#8b8fa8">/10</span></div>
                <div style="font-size:1.3rem; margin-top:8px;">{status}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if pred_score < 5:
                st.error("⚠️ **High Risk Zone**: Consider reducing daily usage and improving sleep hygiene. Seek university counselling resources.")
            elif pred_score < 7:
                st.warning("💛 **Moderate Zone**: Your digital habits are manageable, but watch for increasing conflicts and sleep disruption.")
            else:
                st.success("✅ **Healthy Zone**: Your social media usage appears to be well-balanced. Keep it up!")

# ═══════════════════════════════════════════════════════════
# PAGE 3 — FIND MY CLUSTER
# ═══════════════════════════════════════════════════════════
elif page == "🔍 Find My Cluster":
    st.markdown("# 🔍 Find My Behavioral Segment")
    st.markdown("*K-Means clustering model with K=4 behavioral profiles*")
    st.divider()

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("### 📝 Your Social Media Profile")
        c_usage = st.slider("📱 Daily Usage Hours", 0.5, 12.0, 5.0, 0.1, key='c_usage')
        c_mh = st.slider("🧠 Mental Health Score (self-rated, 1-10)", 1, 10, 6, key='c_mh')
        c_sleep = st.slider("😴 Sleep Hours per Night", 3.0, 12.0, 6.5, 0.1, key='c_sleep')
        c_addicted = st.slider("🎯 Addiction Score (1-10)", 1, 10, 5, key='c_addicted')
        c_conflicts = st.slider("💬 Conflicts Over Social Media", 0, 10, 2, key='c_conflicts')

        st.markdown("<br>", unsafe_allow_html=True)
        cluster_btn = st.button("🔮 Find My Segment")

    with col_r:
        st.markdown("### 📊 All Behavioral Segments")
        colors_list = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
        for idx, cluster_id in enumerate(cluster_profile.index):
            label = cluster_labels[cluster_id]
            c = colors_list[idx % len(colors_list)]
            row = cluster_profile.loc[cluster_id]
            n = (df['Cluster'] == cluster_id).sum()
            st.markdown(f"""
            <div class="cluster-card" style="border-color:{c}; background:rgba(0,0,0,0.2); margin:8px 0;">
                <b style="color:{c}; font-size:1.05rem">{label}</b>
                <span style="color:#8b8fa8; font-size:0.8rem; float:right">n={n}</span><br>
                📱 <b>{row['Avg_Daily_Usage_Hours']:.1f}h</b> daily usage &nbsp;|&nbsp;
                🧠 <b>{row['Mental_Health_Score']:.1f}/10</b> mental health<br>
                😴 <b>{row['Sleep_Hours_Per_Night']:.1f}h</b> sleep &nbsp;|&nbsp;
                🎯 <b>{row['Addicted_Score']:.1f}/10</b> addiction score
            </div>
            """, unsafe_allow_html=True)

    if cluster_btn:
        # Normalize user input against the dataset's feature distribution
        user_input = np.array([[c_usage, c_mh, c_sleep, c_addicted, c_conflicts]], dtype=float)
        for i, feat in enumerate(cluster_features):
            mean, std = cluster_params[feat]
            user_input[0, i] = (user_input[0, i] - mean) / (std if std > 0 else 1)

        user_cluster = kmeans.predict(user_input)[0]
        user_label = cluster_labels[user_cluster]
        user_color = CLUSTER_COLORS.get(user_label, "#7c83fd")
        user_profile = cluster_profile.loc[user_cluster]

        st.divider()
        col_res1, col_res2, col_res3 = st.columns([0.5, 2, 0.5])
        with col_res2:
            st.markdown(f"""
            <div style="text-align:center; background:linear-gradient(135deg,#1e2130,#252b40);
                        border-radius:20px; padding:28px; border: 2px solid {user_color};">
                <div style="font-size:0.9rem; color:#8b8fa8; margin-bottom:8px;">You belong to</div>
                <div style="font-size:2.8rem; font-weight:800; color:{user_color};">{user_label}</div>
                <hr style="border-color:#2d3250; margin:16px 0">
                <div style="display:flex; justify-content:space-around; color:#e0e0e0; font-size:0.9rem;">
                    <div>📱 Avg Usage<br><b style="font-size:1.3rem">{user_profile['Avg_Daily_Usage_Hours']:.1f}h</b></div>
                    <div>🧠 Mental Health<br><b style="font-size:1.3rem">{user_profile['Mental_Health_Score']:.1f}/10</b></div>
                    <div>😴 Sleep<br><b style="font-size:1.3rem">{user_profile['Sleep_Hours_Per_Night']:.1f}h</b></div>
                    <div>🎯 Addiction<br><b style="font-size:1.3rem">{user_profile['Addicted_Score']:.1f}/10</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if "At-Risk" in user_label:
            st.error("🚨 **At-Risk / Addicted**: Your usage pattern places you in the highest-risk group. Consider a digital detox, screen time limits, and prioritizing sleep.")
        elif "Moderate" in user_label:
            st.warning("⚠️ **Moderate Scrollers**: You're in the middle — be mindful of usage creep and monitor how social media affects your sleep quality.")
        elif "Balanced" in user_label:
            st.info("💚 **Balanced Users**: You maintain reasonable usage. Stay consistent and watch for signs of increased conflicts or sleep reduction.")
        else:
            st.success("🌟 **Healthy Minimalists**: Excellent digital habits! You use social media purposefully without significant well-being trade-offs.")
