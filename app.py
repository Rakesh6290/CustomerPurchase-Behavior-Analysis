import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

st.set_page_config(page_title="Customer Purchase Analysis", layout="wide")
st.title("📊 Customer Purchase Behavior Analysis")

# =========================
# CSV Upload
# =========================
st.sidebar.header("Upload CSV (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload customer purchase data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/sample_data.csv")

# =========================
# Data Cleaning
# =========================
df = df.dropna()
df = df[df['product_name'].str.len() < 100]
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# =========================
# Feature Engineering
# =========================
customer_df = df.groupby('customer_id').agg(
    total_spent=('total_amount', 'sum'),
    purchase_frequency=('order_id', 'count'),
    recency=('purchase_date', lambda x: (pd.Timestamp.today() - x.max()).days)
).reset_index()

# =========================
# KPI Metrics
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"₹{customer_df['total_spent'].sum():,.0f}")
col2.metric("👥 Total Customers", customer_df.shape[0])
col3.metric("🛒 Total Orders", df['order_id'].nunique())

# =========================
# Customer Segmentation (KMeans)
# =========================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_df[['total_spent', 'purchase_frequency', 'recency']])

kmeans = KMeans(n_clusters=3, random_state=42)
customer_df['segment'] = kmeans.fit_predict(scaled_data)

segment_map = {0: "High Value", 1: "Medium Value", 2: "Low Value"}
customer_df['segment_label'] = customer_df['segment'].map(segment_map)

# =========================
# Churn Prediction
# =========================
customer_df['churn_label'] = (customer_df['recency'] > 90).astype(int)

if customer_df['churn_label'].nunique() > 1:
    X = customer_df[['total_spent', 'purchase_frequency', 'recency']]
    y = customer_df['churn_label']
    model = LogisticRegression()
    model.fit(X, y)
    customer_df['churn_probability'] = model.predict_proba(X)[:, 1]
else:
    st.warning("⚠️ Not enough churn class variety in data. Showing default churn probability.")
    customer_df['churn_probability'] = 0.5

# =========================
# Charts
# =========================
st.subheader("📈 Customer Segmentation")
fig1 = px.pie(customer_df, names='segment_label', title='Customer Segments')
st.plotly_chart(fig1, width='stretch')

st.subheader("📊 Revenue by Segment")
fig2 = px.bar(
    customer_df.groupby('segment_label')['total_spent'].sum().reset_index(),
    x='segment_label', y='total_spent',
    title='Revenue by Customer Segment',
    labels={'total_spent': 'Revenue'}
)
st.plotly_chart(fig2, width='stretch')

# =========================
# Sentiment Analysis
# =========================
if 'review_text' in df.columns:
    st.subheader("📝 Review Sentiment Analysis")
    df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    fig3 = px.histogram(df, x='sentiment', nbins=20, title='Review Sentiment Distribution')
    st.plotly_chart(fig3, width='stretch')

# =========================
# Business Insights
# =========================
st.subheader("💡 Business Insights")
high_value_pct = (customer_df[customer_df['segment_label']=="High Value"].shape[0] / customer_df.shape[0]) * 100
st.write(f"""
• **{high_value_pct:.1f}% customers generate majority of revenue**  
• High value customers purchase frequently with low recency  
• Customers with recency > 90 days show high churn probability  
""")

# =========================
# Download
# =========================
st.download_button(
    "⬇️ Download Customer Insights CSV",
    customer_df.to_csv(index=False),
    file_name="customer_insights.csv"
)

st.success("✅ Analysis Complete")
