import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dashboard Analitik CRM - E-commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    # You can replace this with file upload or direct CSV loading
    # For now, I'll create sample data based on your CSV structure
    df = pd.read_csv('customer.csv')   
    df.dropna(inplace=True)

    return df

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🛒 Dashboard Analitik CRM - E-commerce</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar
    page = st.sidebar.radio(
    "📊 Dashboard Navigation",
    ["Overview", "Customer Insights", "Customer Segmentation (RFM)","Customer Segmentation (K-Means)","Churn Analysis", "Strategic Recommendations"]
    )
    
    if page == "Overview":
        st.header("📈 Business Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pelanggan", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Total Pendapatan", f"${df['Total Spend'].sum():,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.metric("Rata-rata Nilai Pesanan", f"${df['Total Spend'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            satisfaction_rate = (df['Satisfaction Level'] == 'Satisfied').mean() * 100
            st.metric("Tingkat Kepuasan", f"{satisfaction_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("Indikator Kinerja Utama")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM Analyis segmentation
            rfm_data = perform_rfm_analysis(df)
            df_rfm = df.merge(rfm_data[['Customer ID', 'RFM_Segment']], on='Customer ID', how='left')
            segment_counts = rfm_data['RFM_Segment'].value_counts()
            fig1 = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title="Segmentasi Pelanggan RFM")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Churn Analysis Overview
            df['Churn_Risk'] = ((df['Satisfaction Level'] == 'Unsatisfied') | 
                            (df['Days Since Last Purchase'] > 45))
            high_risk = df[df['Churn_Risk'] == True].sort_values('Total Spend', ascending=False)
            if len(high_risk) > 0:
                st.write(f"**{len(high_risk)} pelanggan** berisiko tinggi untuk churn")
                st.write(f"**Potensi Kehilangan Pendapatan:** ${high_risk['Total Spend'].sum():.2f}")

            # Churn rate by segment
            churn_rate = df.groupby('Satisfaction Level').apply(
                lambda x: ((x['Days Since Last Purchase'] > 45) | 
                          (x['Satisfaction Level'] == 'Unsatisfied')).mean() * 100
            ).reset_index()
            churn_rate.columns = ['Satisfaction Level', 'Tingkat Churn (%)']
            
            fig1 = px.bar(churn_rate, x='Satisfaction Level', y='Tingkat Churn (%)',
                         title="Tingkat Churn berdasarkan Tingkat Kepuasan",
                         color='Tingkat Churn (%)',
                         color_continuous_scale='Reds')
            st.plotly_chart(fig1, use_container_width=True)
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by Membership Type
            membership_vs_spend = df.groupby('Membership Type')['Total Spend'].mean()
            fig1 = px.bar(x=membership_vs_spend.index, y=membership_vs_spend.values, labels={'x':'Tipe Keanggotaan', 'y':'Rata-rata Pengeluaran'}, title='Rata-rata Pengeluaran berdasarkan Tipe Keanggotaan')
            st.plotly_chart(fig1, use_container_width=True)

            # Customer Satisfaction by City
            satisfaction_by_city = df.groupby(['City', 'Satisfaction Level']).size().unstack(fill_value=0)
            fig2 = px.bar(satisfaction_by_city.reset_index(), x='City', 
                         y=['Satisfied', 'Neutral', 'Unsatisfied'],
                         title="Kepuasan Pelanggan berdasarkan Kota",
                         color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Average Spending with/without Discounts
            discount_vs_spend = df.groupby('Discount Applied')['Total Spend'].mean()
            fig1 = px.bar(x=discount_vs_spend.index, y=discount_vs_spend.values, labels={'x':'Diskon Diterapkan', 'y':'Rata-rata Pengeluaran'}, title='Rata-rata Pengeluaran dengan/tanpa Diskon')
            st.plotly_chart(fig1, use_container_width=True)

            # Average item purchased and Spending
            items_vs_spend = df.groupby('Items Purchased')['Total Spend'].mean()
            fig2 = px.bar(x=items_vs_spend.index, y=items_vs_spend.values, labels={'x':'Jumlah Item Dibeli', 'y':'Rata-rata Pengeluaran'}, title='Rata-rata Pengeluaran berdasarkan Jumlah Item Dibeli')
            st.plotly_chart(fig2, use_container_width=True)

    elif page == "Customer Insights":
        st.header("🔍 Customer Insights & Behavior Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Demografi", "Perilaku Pembelian", "Analisis Kepuasan"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                fig1 = px.histogram(df, x='Age', nbins=20, title="Distribusi Usia",
                                   color_discrete_sequence=['#2E86AB'])
                st.plotly_chart(fig1, use_container_width=True)
                
                # Gender distribution
                gender_counts = df['Gender'].value_counts()
                fig3 = px.pie(values=gender_counts.values, names=gender_counts.index,
                             title="Distribusi Gender")
                st.plotly_chart(fig3, use_container_width=True)

                # City distribution
                city_counts = df['City'].value_counts()
                fig3 = px.bar(x=city_counts.values, y=city_counts.index,
                             title="Pelanggan berdasarkan Kota", orientation='h',
                             color=city_counts.values,
                             color_continuous_scale='Blues')
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                #Age Group Distribution
                age_groups = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['0-18', '19-30', '31-45', '46-60', '60+'])
                df['Age Group'] = age_groups
                fig2 = px.histogram(df, x='Age Group', title='Distribusi Kelompok Usia')
                st.plotly_chart(fig2, use_container_width=True)

                # Age and Gender Distribution
                gender_with_age = df.groupby('Gender')['Age'].value_counts().reset_index()
                fig = px.bar(gender_with_age, x='Age', y='count', color='Gender', barmode='group', title='Distribusi Gender dan Usia')
                st.plotly_chart(fig, use_container_width=True)
                
                # Membership distribution
                membership_counts = df['Membership Type'].value_counts()
                fig4 = px.bar(x=membership_counts.index, y=membership_counts.values,
                             title="Distribusi Tipe Keanggotaan",
                             color=['Gold', 'Silver', 'Bronze'],
                             color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
                st.plotly_chart(fig4, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Total Spend Distribution
                fig = px.histogram(df, x='Total Spend', title='Distribusi Total Pengeluaran')
                st.plotly_chart(fig, use_container_width=True)

                # Days since last purchase distribution
                fig2 = px.histogram(df, x='Days Since Last Purchase', nbins=20,
                                   title="Distribusi Hari Sejak Pembelian Terakhir",
                                   color_discrete_sequence=['#F18F01'])
                st.plotly_chart(fig2, use_container_width=True)

                # Discount Applied Distribution
                fig = px.pie(df, names='Discount Applied', title='Distribusi Penerapan Diskon')
                st.plotly_chart(fig, use_container_width=True)

                # Spending vs Gender
                spending_by_gender = df.groupby('Gender')['Total Spend'].mean()
                fig = px.bar(x=spending_by_gender.index, y=spending_by_gender.values, labels={'x':'Gender', 'y':'Rata-rata Pengeluaran'}, title='Rata-rata Pengeluaran berdasarkan Gender')
                st.plotly_chart(fig, use_container_width=True)

                # Spending vs Age
                age_vs_spend = df.groupby('Age')['Total Spend'].mean()
                fig = px.bar(x=age_vs_spend.index, y=age_vs_spend.values, labels={'x':'Usia', 'y':'Rata-rata Pengeluaran'}, title='Rata-rata Pengeluaran berdasarkan Kelompok Usia')
                st.plotly_chart(fig, use_container_width=True)

                # Days Since Last Purchase vs Spending
                days_vs_spend = df.groupby('Days Since Last Purchase')['Total Spend'].mean()
                fig = px.bar(x=days_vs_spend.index, y=days_vs_spend.values, labels={'x':'Hari Sejak Pembelian Terakhir', 'y':'Rata-rata Pengeluaran'}, title='Rata-rata Pengeluaran berdasarkan Hari Sejak Pembelian Terakhir')
                st.plotly_chart(fig, use_container_width=True)

                # Spending vs Items correlation
                fig1 = px.scatter(df, x='Items Purchased', y='Total Spend',
                                 color='Membership Type', size='Average Rating',
                                 title="Pengeluaran vs Jumlah Item Dibeli",
                                 color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
                st.plotly_chart(fig1, use_container_width=True)
                
            
            with col2:
                # Item Purchased Distribution
                fig = px.histogram(df, x='Items Purchased', title='Distribusi Jumlah Item Dibeli')
                st.plotly_chart(fig, use_container_width=True)

                # Rating distribution
                fig4 = px.histogram(df, x='Average Rating', nbins=20,
                                   title="Distribusi Rating Pelanggan",
                                   color_discrete_sequence=['#A23B72'])
                st.plotly_chart(fig4, use_container_width=True)

                # Spending, Membership, and Gender
                fig = px.box(df, x='Membership Type', y='Total Spend', color='Gender', title='Pengeluaran berdasarkan Tipe Keanggotaan dan Gender')
                st.plotly_chart(fig, use_container_width=True) 

                # City, Item purchased, membership Type
                fig = px.box(df, x='City', y='Items Purchased', color='Membership Type', title='Pengeluaran berdasarkan Tipe Keanggotaan dan Kota')
                st.plotly_chart(fig, use_container_width=True)

                # Average Spending by Discount Applied
                discount_vs_spend = df.groupby('Discount Applied')['Total Spend'].mean()
                fig = px.bar(x=discount_vs_spend.index, y=discount_vs_spend.values, labels={'x':'Diskon Diterapkan', 'y':'Rata-rata Pengeluaran'}, title='Rata-rata Pengeluaran dengan/tanpa Diskon')
                st.plotly_chart(fig, use_container_width=True)  

                # Membership vs Discount vs Total Spend
                df_grouped = df.groupby(['Membership Type', 'Discount Applied'])['Total Spend'].sum().unstack()
                fig = px.bar(df_grouped, barmode='group', title='Total Pengeluaran berdasarkan Tipe Keanggotaan dan Diskon Diterapkan')              
                st.plotly_chart(fig, use_container_width=True) 

                # Average spending by membership
                avg_spending = df.groupby('Membership Type')['Total Spend'].mean().reset_index()
                fig3 = px.bar(avg_spending, x='Membership Type', y='Total Spend',
                             title="Rata-rata Pengeluaran berdasarkan Tipe Keanggotaan",
                             color='Membership Type',
                             color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
                st.plotly_chart(fig3, use_container_width=True)
                        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Satisfaction by membership
                satisfaction_membership = pd.crosstab(df['Membership Type'], df['Satisfaction Level'], normalize='index') * 100
                fig1 = px.bar(satisfaction_membership.reset_index(), 
                             x='Membership Type', y=['Satisfied', 'Neutral', 'Unsatisfied'],
                             title="Tingkat Kepuasan berdasarkan Tipe Keanggotaan (%)",
                             color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Rating vs Satisfaction correlation
                fig2 = px.box(df, x='Satisfaction Level', y='Average Rating',
                             title="Distribusi Rating berdasarkan Tingkat Kepuasan",
                             color='Satisfaction Level',
                             color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01'])
                st.plotly_chart(fig2, use_container_width=True)

    elif page == "Customer Segmentation (RFM)":
        st.header("🎯 Customer Segmentation (RFM Analysis)")
        
        # Perform RFM analysis
        rfm_data = perform_rfm_analysis(df)
        
        # Merge with original data for complete view
        df_rfm = df.merge(rfm_data[['Customer ID', 'RFM_Segment']], on='Customer ID', how='left')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment distribution
            segment_counts = rfm_data['RFM_Segment'].value_counts()
            fig1 = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title="Distribusi Segmen Pelanggan")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Revenue by segment
            revenue_by_segment = df_rfm.groupby('RFM_Segment')['Total Spend'].sum().sort_values(ascending=False)
            fig2 = px.bar(x=revenue_by_segment.index, y=revenue_by_segment.values,
                         title="Pendapatan berdasarkan Segmen Pelanggan",
                         color=revenue_by_segment.values,
                         color_continuous_scale='Blues')
            fig2.update_layout(xaxis_title="Segmen Pelanggan", yaxis_title="Total Pendapatan ($)")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed segment analysis
        st.subheader("📊 Analisis Segmen")
        
        segment_summary = df_rfm.groupby('RFM_Segment').agg({
            'Customer ID': 'count',
            'Total Spend': ['mean', 'sum'],
            'Items Purchased': 'mean',
            'Average Rating': 'mean',
            'Days Since Last Purchase': 'mean'
        }).round(2)
        
        segment_summary.columns = ['Jumlah Pelanggan', 'Rata-rata Pengeluaran', 'Total Pendapatan', 
                                 'Rata-rata Item', 'Rata-rata Rating', 'Rata-rata Hari Sejak Pembelian']
        
        st.dataframe(segment_summary, use_container_width=True)
        
        # Strategic insights for each segment
        st.subheader("💡 Wawasan Strategis per Segmen")
        
        insights = {
            'Top Customers': 'Pelanggan tingkat atas dengan loyalitas tinggi dan keterlibatan terbaru. Tawarkan akses VIP, penawaran eksklusif. Prioritaskan layanan pelanggan, retensi dan upselling yang dipersonalisasi.',
            'High Value Customers (Loyal)': 'Pelanggan dengan pengeluaran konsisten yang menunjukkan loyalitas. Jaga keterlibatan mereka dengan hadiah eksklusif dan penawaran yang disesuaikan.',
            'Medium Value Customers (Need Attention)': 'Pelanggan baru yang menunjukkan perilaku menjanjikan. Bina mereka untuk menjadi pelanggan utama melalui kampanye yang ditargetkan.',
            'Low Value Customers (At Risk)': 'Pelanggan dengan aktivitas menurun atau keterlibatan terbaru yang rendah. Kampanye re-engagement diperlukan untuk mengembalikan mereka.',
            'Hibernating': 'Pelanggan bernilai tinggi yang menunjukkan tanda-tanda risiko. Strategi retensi yang dipersonalisasi diperlukan segera.',
        }
        
        for segment, insight in insights.items():
            if segment in segment_counts.index:
                st.write(f"**{segment}:** {insight}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Customer Segmentation (K-Means)":
        st.header("🎯 Customer Segmentation (K-Means Segementation)")
        
        # Perform RFM analysis
        cluster_data = k_means_clustering(df)
        
        # Merge with original data for complete view
        df_cluster = df.merge(cluster_data[['Customer ID', 'Cluster1', 'EngagementCluster', 'SeasonalCluster']], on='Customer ID', how='left')
    
        tab1, tab2, tab3 = st.tabs(["Total Pengeluaran, Usia, dan Item Dibeli", "Segmentasi Berbasis Keterlibatan", "Segmentasi Musiman"])
        
        with tab1:
            st.subheader('Segmentasi Berdasarkan Total Pengeluaran, Usia, dan Item Dibeli')

            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df, x='Cluster1', y='Total Spend', title='Total Pengeluaran berdasarkan Cluster')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cluster_distribution = df['Cluster1'].value_counts().sort_index()
                fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, labels={'x':'Cluster', 'y':'Jumlah'}, title='Distribusi Cluster')
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader('Segmentasi Berdasarkan Keterlibatan')

            cluster_distribution = df['EngagementCluster'].value_counts().sort_index()
            fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, labels={'x':'Cluster', 'y':'Jumlah'}, title='Distribusi Cluster Keterlibatan')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.subheader('Segmentasi Berdasarkan Musiman')

            cluster_distribution = df['SeasonalCluster'].value_counts().sort_index()
            fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, labels={'x':'Cluster', 'y':'Jumlah'}, title='Distribusi Cluster Musiman')
            st.plotly_chart(fig, use_container_width=True)
            
    elif page == "Churn Analysis":
        st.header("⚠️ Customer Churn Analysis")
        
        # Build churn model
        model, feature_importance, y_test, y_pred = build_churn_model(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn rate by segment
            churn_rate = df.groupby('Satisfaction Level').apply(
                lambda x: ((x['Days Since Last Purchase'] > 45) | 
                          (x['Satisfaction Level'] == 'Unsatisfied')).mean() * 100
            ).reset_index()
            churn_rate.columns = ['Satisfaction Level', 'Tingkat Churn (%)']
            
            fig1 = px.bar(churn_rate, x='Satisfaction Level', y='Tingkat Churn (%)',
                         title="Tingkat Churn berdasarkan Tingkat Kepuasan",
                         color='Tingkat Churn (%)',
                         color_continuous_scale='Reds')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Feature importance for churn prediction
            fig2 = px.bar(feature_importance.head(8), x='Importance', y='Feature',
                         title="Faktor yang Mempengaruhi Churn Pelanggan",
                         orientation='h',
                         color='Importance',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Churn risk segments
        st.subheader("🎯 Identifikasi Pelanggan Berisiko Tinggi")
        
        # Identify high-risk customers
        df['Churn_Risk'] = ((df['Satisfaction Level'] == 'Unsatisfied') | 
                           (df['Days Since Last Purchase'] > 45))
        
        high_risk = df[df['Churn_Risk'] == True].sort_values('Total Spend', ascending=False)
        
        if len(high_risk) > 0:
            st.write(f"**{len(high_risk)} pelanggan** berisiko tinggi untuk churn")
            st.write(f"**Potensi Kehilangan Pendapatan:** ${high_risk['Total Spend'].sum():.2f}")
            
            # Show top at-risk customers
            st.subheader("10 Pelanggan Berisiko Tertinggi (berdasarkan pengeluaran)")
            risk_customers = high_risk[['Customer ID', 'Gender', 'Age', 'City', 'Membership Type', 
                                      'Total Spend', 'Days Since Last Purchase', 'Satisfaction Level']].head(10)
            st.dataframe(risk_customers, use_container_width=True)
        
        # Retention strategies
        st.subheader("🛡️ Strategi Retensi")
        st.write("""
        **Tindakan Segera:**
        - Kirim email re-engagement yang dipersonalisasi ke pelanggan yang tidak aktif selama 45+ hari
        - Tawarkan diskon eksklusif untuk pelanggan yang tidak puas
        - Implementasikan survei keluar untuk memahami alasan ketidakpuasan
        
        **Strategi Jangka Panjang:**
        - Tingkatkan layanan pelanggan untuk pengalaman dengan rating rendah
        - Kembangkan program loyalitas untuk pembeli yang sering
        - Buat konten yang ditargetkan untuk segmen pelanggan yang berbeda
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Strategic Recommendations":
        st.header("💡 Strategic CRM Recommendations")
        
        # Calculate key metrics for recommendations
        rfm_data = perform_rfm_analysis(df)
        df_rfm = df.merge(rfm_data[['Customer ID', 'RFM_Segment']], on='Customer ID', how='left')
        
        churn_rate = ((df['Satisfaction Level'] == 'Unsatisfied') | 
                     (df['Days Since Last Purchase'] > 45)).mean() * 100
        
        avg_clv = df['Total Spend'].mean()
        satisfaction_rate = (df['Satisfaction Level'] == 'Satisfied').mean() * 100
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("📊 Analisis Kondisi Saat Ini")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tingkat Churn", f"{churn_rate:.1f}%", delta="-2.3%" if churn_rate < 25 else "+1.5%")
        with col2:
            st.metric("Rata-rata Nilai Pelanggan", f"${avg_clv:.2f}", delta="+$45.20")
        with col3:
            st.metric("Tingkat Kepuasan", f"{satisfaction_rate:.1f}%", delta="+3.2%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Strategic recommendations
        st.subheader("🎯 Strategi CRM yang Diprioritaskan")
        
        strategies = [
            {
                "priority": "TINGGI",
                "title": "Program Retensi Pelanggan",
                "description": "Implementasikan kampanye retensi yang ditargetkan untuk pelanggan berisiko",
                "actions": [
                    "Kirim email win-back yang dipersonalisasi ke pelanggan yang tidak aktif selama 30+ hari",
                    "Tawarkan diskon eksklusif untuk pelanggan yang tidak puas",
                    "Buat saluran dukungan VIP untuk anggota Gold yang berisiko"
                ],
                "expected_impact": "Kurangi tingkat churn sebesar 15-20%",
                "timeline": "1-2 bulan"
            },
            {
                "priority": "TINGGI",
                "title": "Strategi Segmentasi Pelanggan",
                "description": "Kembangkan kampanye pemasaran yang ditargetkan untuk setiap segmen pelanggan",
                "actions": [
                    "Buat rekomendasi produk premium untuk Pelanggan Utama",
                    "Rancang hadiah loyalitas untuk pelanggan reguler",
                    "Implementasikan kampanye onboarding untuk pelanggan baru"
                ],
                "expected_impact": "Tingkatkan nilai seumur hidup pelanggan sebesar 25%",
                "timeline": "2-3 bulan"
            },
            {
                "priority": "SEDANG",
                "title": "Peningkatan Pengalaman Pelanggan",
                "description": "Tingkatkan kepuasan pelanggan dan kualitas layanan secara keseluruhan",
                "actions": [
                    "Implementasikan sistem umpan balik pelanggan real-time",
                    "Tingkatkan waktu respons layanan pelanggan",
                    "Personalisasi rekomendasi produk berdasarkan riwayat pembelian"
                ],
                "expected_impact": "Tingkatkan tingkat kepuasan hingga 85%+",
                "timeline": "3-4 bulan"
            },
            {
                "priority": "SEDANG",
                "title": "Implementasi Analitik Prediktif",
                "description": "Gunakan wawasan berbasis data untuk manajemen pelanggan proaktif",
                "actions": [
                    "Terapkan model prediksi churn untuk intervensi dini",
                    "Implementasikan prediksi nilai seumur hidup pelanggan",
                    "Buat peringatan otomatis untuk perubahan perilaku pelanggan"
                ],
                "expected_impact": "Tingkatkan retensi pelanggan sebesar 30%",
                "timeline": "4-6 bulan"
            }
        ]
        
        for i, strategy in enumerate(strategies, 1):
            with st.expander(f"{i}. {strategy['title']} - Prioritas: {strategy['priority']}"):
                st.write(f"**Deskripsi:** {strategy['description']}")
                st.write("**Tindakan Utama:**")
                for action in strategy['actions']:
                    st.write(f"• {action}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Dampak yang Diharapkan:** {strategy['expected_impact']}")
                with col2:
                    st.write(f"**Timeline Implementasi:** {strategy['timeline']}")
        
        # ROI Projections
        st.subheader("💰 Proyeksi ROI yang Diharapkan")
        
        roi_data = {
            'Strategy': ['Program Retensi', 'Segmentasi', 'Peningkatan Pengalaman', 'Analitik Prediktif'],
            'Investment ($)': [50000, 75000, 100000, 150000],
            'Expected Return ($)': [200000, 300000, 250000, 450000],
            'ROI (%)': [300, 300, 150, 200]
        }
        
        roi_df = pd.DataFrame(roi_data)
        
        fig = px.bar(roi_df, x='Strategy', y='ROI (%)',
                    title="ROI yang Diharapkan per Strategi CRM",
                    color='ROI (%)',
                    color_continuous_scale='Greens',
                    text='ROI (%)')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Implementation roadmap
        st.subheader("🗓️ Peta Jalan Implementasi")
        
        roadmap_data = {
            'Month': ['Bulan 1', 'Bulan 2', 'Bulan 3', 'Bulan 4', 'Bulan 5', 'Bulan 6'],
            'Program Retensi': [100, 100, 50, 0, 0, 0],
            'Strategi Segmentasi': [50, 100, 100, 50, 0, 0],
            'Peningkatan Pengalaman': [0, 25, 50, 100, 100, 50],
            'Analitik Prediktif': [0, 0, 25, 50, 100, 100]
        }
        
        roadmap_df = pd.DataFrame(roadmap_data)
        roadmap_melted = roadmap_df.melt(id_vars=['Month'], 
                                        var_name='Strategy', 
                                        value_name='Completion %')
        
        fig_roadmap = px.bar(roadmap_melted, x='Month', y='Completion %',
                           color='Strategy', title="Peta Jalan Implementasi 6 Bulan",
                           color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#6A994E'])
        st.plotly_chart(fig_roadmap, use_container_width=True)

# RFM Analysis
def perform_rfm_analysis(df):
    # Calculate RFM metrics
    current_date = df['Days Since Last Purchase'].min()

    rfm_table = df.groupby('Customer ID').agg({
            'Days Since Last Purchase': lambda x: current_date - x.iloc[0],  # Recency (lower is better)
            'Items Purchased': 'sum',  # Frequency
            'Total Spend': 'sum'  # Monetary
        }).reset_index()
    rfm_table.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    rfm_table['Recency'] = -rfm_table['Recency']  # Make recency positive (lower days = higher recency score)
    
    # Rank RFM
    rfm_table['R_Rank'] = rfm_table['Recency'].rank(ascending=False)
    rfm_table['F_Rank'] = rfm_table['Frequency'].rank(ascending=True)
    rfm_table['M_Rank'] = rfm_table['Monetary'].rank(ascending=True)
    rfm_table.head()

    # Normalize RFM
    rfm_table['R_rank_norm'] = (rfm_table['R_Rank'] / rfm_table['R_Rank'].max()) * 100
    rfm_table['F_rank_norm'] = (rfm_table['F_Rank'] / rfm_table['F_Rank'].max()) * 100
    rfm_table['M_rank_norm'] = (rfm_table['M_Rank'] / rfm_table['M_Rank'].max()) * 100

    # Drop individual rank
    rfm_table.drop(columns=['R_Rank', 'F_Rank', 'M_Rank'], inplace=True)
    rfm_table.head()

    # RFM Scoring menggunakan metode weighted dengan 55% Monetary, 30% Frequency, dan 15% Recency
    rfm_table['RFM_Score'] = 0.15 * rfm_table['R_rank_norm'] + 0.3 * rfm_table['F_rank_norm'] + 0.55 * rfm_table['M_rank_norm']
    rfm_table['RFM_Score'] *= 0.05
    rfm_table = rfm_table.round(2)
    
    # Segmentasi Customer berdasarkan RFM
    rfm_table['RFM_Segment'] = pd.cut(
        rfm_table['RFM_Score'],
        bins=[0, 1.6, 3, 4, 4.5, 5],
        labels=['Hibernating',
                'Low Value Customers (At Risk)',
                'Medium Value Customers (Need Attention)',
                'High Value Customers (Loyal)',
                'Top Customers'
                ])
        
    return rfm_table

def k_means_clustering(df):
    X = df[['Total Spend', 'Age', 'Items Purchased']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster1'] = kmeans.fit_predict(X_scaled)

    X_engagement = df[['Days Since Last Purchase', 'Average Rating', 'Discount Applied']]
    scaler_engagement = StandardScaler()
    X_engagement_scaled = scaler_engagement.fit_transform(X_engagement)

    k_engagement = 4

    kmeans_engagement = KMeans(n_clusters=k_engagement, random_state=42)
    df['EngagementCluster'] = kmeans_engagement.fit_predict(X_engagement_scaled)

    X_seasonal = df[['Days Since Last Purchase', 'Total Spend']]
    scaler_seasonal = StandardScaler()
    X_seasonal_scaled = scaler_seasonal.fit_transform(X_seasonal)

    k_seasonal = 5

    kmeans_seasonal = KMeans(n_clusters=k_seasonal, random_state=42)
    df['SeasonalCluster'] = kmeans_seasonal.fit_predict(X_seasonal_scaled)

    return df

# Churn Prediction Model
def build_churn_model(df):
    # Prepare features for churn prediction
    features = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
    
    # Create binary target (churn if unsatisfied or high days since last purchase)
    df['Churn'] = ((df['Satisfaction Level'] == 'Unsatisfied') | 
                   (df['Days Since Last Purchase'] > 45)).astype(int)
    
    # Prepare feature matrix
    X = df[features].copy()
    y = df['Churn']
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, feature_importance, y_test, y_pred

if __name__ == "__main__":
    main()