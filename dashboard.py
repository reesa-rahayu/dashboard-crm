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

# Define consistent color palette
COLOR_PALETTE = {
    'primary': '#2E86AB',      # Main blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#6A994E',      # Green
    'warning': '#E9C46A',      # Yellow
    'danger': '#E76F51',       # Red
    'neutral': '#264653',      # Dark blue
    'light': '#E8F4F8'         # Light blue
}

# Update color sequences for plots
COLOR_SEQUENCE = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
                 COLOR_PALETTE['accent'], COLOR_PALETTE['success'],
                 COLOR_PALETTE['warning'], COLOR_PALETTE['danger']]

# Update color scales
COLOR_SCALES = {
    'blues': ['#E8F4F8', '#2E86AB', '#264653'],
    'purples': ['#F3E5F5', '#A23B72', '#4A148C'],
    'oranges': ['#FFF3E0', '#F18F01', '#E65100'],
    'greens': ['#E8F5E9', '#6A994E', '#1B5E20']
}

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
    
    # Add dataset link at bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Dataset")
    st.sidebar.markdown("[Download Customer Dataset](https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-datasets)")
    
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
        
        # Customer Growth and Retention
        st.markdown("#### Customer Growth & Retention")
        col1, col2 = st.columns(2)
        with col1:
            active_customers = len(df[df['Days Since Last Purchase'] <= 30])
            st.metric("Costumer Aktif (30 days)", f"{active_customers:,}")

        with col2:
            returning_customers = len(df[df['Days Since Last Purchase'] <= 90])
            st.metric("Returning Customers (90 days)", f"{returning_customers:,}")

        # Diagrams
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer Activity Trends
            fig = px.line(df.groupby('Days Since Last Purchase').size().reset_index(name='count'),
                         x='Days Since Last Purchase', y='count',
                         title='Aktivitas Customer',
                         color_discrete_sequence=[COLOR_PALETTE['primary']])
            st.plotly_chart(fig, use_container_width=True)

            # Average item purchased and Spending
            items_vs_spend = df.groupby('Items Purchased')['Total Spend'].mean()
            fig2 = px.bar(x=items_vs_spend.index, y=items_vs_spend.values, 
                         labels={'x':'Jumlah Item Dibeli', 'y':'Rata-rata Pengeluaran'}, 
                         title='Rata-rata Pengeluaran berdasarkan Jumlah Item Dibeli',
                         color_discrete_sequence=[COLOR_PALETTE['secondary']])
            st.plotly_chart(fig2, use_container_width=True)

            # Average Spending with/without Discounts
            discount_vs_spend = df.groupby('Discount Applied')['Total Spend'].mean()
            fig1 = px.bar(x=discount_vs_spend.index, y=discount_vs_spend.values, 
                         labels={'x':'Diskon Diterapkan', 'y':'Rata-rata Pengeluaran'}, 
                         title='Rata-rata Pengeluaran dengan/tanpa Diskon',
                         color_discrete_sequence=[COLOR_PALETTE['accent']])
            st.plotly_chart(fig1, use_container_width=True)

        
        with col2:
            # RFM Analyis segmentation
            rfm_data = perform_rfm_analysis(df)
            df_rfm = df.merge(rfm_data[['Customer ID', 'RFM_Segment']], on='Customer ID', how='left')
            segment_counts = rfm_data['RFM_Segment'].value_counts()
            fig1 = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title="Segmentasi Pelanggan RFM",
                         color_discrete_sequence=COLOR_SEQUENCE)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Revenue by Membership Type
            membership_vs_spend = df.groupby('Membership Type')['Total Spend'].mean()
            fig1 = px.bar(x=membership_vs_spend.index, y=membership_vs_spend.values, 
                         labels={'x':'Tipe Keanggotaan', 'y':'Rata-rata Pengeluaran'}, 
                         title='Rata-rata Pengeluaran berdasarkan Tipe Keanggotaan',
                         color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']])
            st.plotly_chart(fig1, use_container_width=True)

            # Customer Satisfaction by City
            satisfaction_by_city = df.groupby(['City', 'Satisfaction Level']).size().unstack(fill_value=0)
            fig2 = px.bar(satisfaction_by_city.reset_index(), x='City', 
                         y=['Satisfied', 'Neutral', 'Unsatisfied'],
                         title="Kepuasan Pelanggan berdasarkan Kota",
                         color_discrete_sequence=[COLOR_PALETTE['success'], COLOR_PALETTE['warning'], COLOR_PALETTE['danger']])
            st.plotly_chart(fig2, use_container_width=True)
        
        # New: Key Insights
        st.markdown("### 💡 Key Insights")
        
        insights = [
            {
                "title": "Customer Base Health",
                "content": f"Pelanggan aktif menunjukkan {active_customers/len(df)*100:.1f}% keterlibatan dalam 30 hari terakhir, dengan {returning_customers/len(df)*100:.1f}% kembali dalam 90 hari."
            },
            {
                "title": "Revenue Distribution",
                "content": f"Tingkat keanggotaan Gold menyumbang {membership_vs_spend['Gold']/membership_vs_spend.sum()*100:.1f}% dari total pendapatan, menunjukkan basis pelanggan yang kuat."
            },
            {
                "title": "Customer Satisfaction",
                "content": f"Tingkat kepuasan keseluruhan sebesar {satisfaction_rate:.1f}% dengan rata-rata penilaian {df['Average Rating'].mean():.1f}/5,0, menunjukkan pengalaman pelanggan yang positif."
            },
            {
                "title": "Purchase Behavior",
                "content": f"Rata-rata nilai pesanan sebesar ${df['Total Spend'].mean():.2f} dengan {df['Items Purchased'].mean():.1f} item per pembelian, menunjukkan ukuran keranjang yang cukup baik."
            }
        ]
        
        for insight in insights:
            with st.expander(f"📌 {insight['title']}"):
                st.write(insight['content'])

    elif page == "Customer Insights":
        st.header("🔍 Customer Insights & Behavior Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Demografi", "Total Pembelian", "Analisis Kepuasan"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                fig1 = px.histogram(df, x='Age', nbins=20, title="Distribusi Usia",
                                   color_discrete_sequence=[COLOR_PALETTE['primary']])
                st.plotly_chart(fig1, use_container_width=True)
                
                # Gender distribution
                gender_counts = df['Gender'].value_counts()
                fig3 = px.pie(values=gender_counts.values, names=gender_counts.index,
                             title="Distribusi Gender",
                             color_discrete_sequence=[COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']])
                st.plotly_chart(fig3, use_container_width=True)

                # City distribution
                city_counts = df['City'].value_counts()
                fig3 = px.bar(x=city_counts.values, y=city_counts.index,
                             title="Pelanggan berdasarkan Kota", orientation='h',
                             color=city_counts.values,
                             color_continuous_scale=COLOR_SCALES['blues'])
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
                             color_discrete_sequence=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], COLOR_PALETTE['accent']])
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
                         title="Customer Segment Distribution")
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

        # New RFM Analysis Section
        st.subheader("📊 RFM Score Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM Score Distribution
            fig = px.histogram(rfm_data, x='RFM_Score', 
                             title="Distribusi score RFM",
                             nbins=30,
                             color_discrete_sequence=['#2E86AB'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Recency vs Frequency
            fig = px.scatter(rfm_data, x='Recency', y='Frequency',
                           color='RFM_Segment',
                           title="Recency dan Frequency berdasarkan Segment",
                           size='Monetary',
                           hover_data=['Customer ID'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monetary vs Frequency
            fig = px.scatter(rfm_data, x='Monetary', y='Frequency',
                           color='RFM_Segment',
                           title="Monetary dan Frequency berdasarkan Segment",
                           size='Recency',
                           hover_data=['Customer ID'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Recency vs Monetary
            fig = px.scatter(rfm_data, x='Recency', y='Monetary',
                           color='RFM_Segment',
                           title="Recency dan Monetary berdasarkan Segment",
                           size='Frequency',
                           hover_data=['Customer ID'])
            st.plotly_chart(fig, use_container_width=True)

        # New Segment Behavior Analysis
        st.subheader("🎯 Segment Behavior Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average Rating by Segment
            fig = px.box(df_rfm, x='RFM_Segment', y='Average Rating',
                        title="Distribusi Rating berdasarkan Segment",
                        color='RFM_Segment')
            st.plotly_chart(fig, use_container_width=True)
            
            # Items Purchased by Segment
            fig = px.box(df_rfm, x='RFM_Segment', y='Items Purchased',
                        title="Distribusi Jumlah Item berdasarkan Segment",
                        color='RFM_Segment')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Days Since Last Purchase by Segment
            fig = px.box(df_rfm, x='RFM_Segment', y='Days Since Last Purchase',
                        title="Hari Terakhir Pembelian berdasarkan Segment",
                        color='RFM_Segment')
            st.plotly_chart(fig, use_container_width=True)
            
            # Membership Type Distribution by Segment
            membership_by_segment = pd.crosstab(df_rfm['RFM_Segment'], df_rfm['Membership Type'], normalize='index') * 100
            fig = px.bar(membership_by_segment.reset_index(), 
                        x='RFM_Segment', y=['Gold', 'Silver', 'Bronze'],
                        title="Distribusi Membership berdasarkan Segment (%)",
                        color_discrete_map={'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'})
            st.plotly_chart(fig, use_container_width=True)

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
        st.subheader("💡 Strategic Insights by Segment")
        
        insights_recommendations = {
            'Top Customers': {
                'insight': 'Pelanggan dengan loyalitas dan keterlibatan tinggi.',
                'recommendation': [
                    "Memberikan **penawaran eksklusif** (bundling premium, diskon khusus).",
                    "Implementasi **layanan pelanggan prioritas**.",
                    "Dorong **upselling dan cross-selling** sesuai preferensi."
                ]
            },
            'High Value Customers (Loyal)': {
                'insight': 'Pelanggan dengan pengeluaran konsisten dan loyal.',
                'recommendation': [
                    "Memberikan **rewards loyalty berkala** (point-based atau hadiah kejutan).",
                    "Mengirimkan **email marketing personalisasi**.",
                    "Mempertahankan **frekuensi komunikasi** dengan chatbot/pesan otomatis terkait penawaran."
                ]
            },
            'Medium Value Customers (Need Attention)': {
                'insight': 'Pelanggan baru atau berkembang dengan potensi tinggi.',
                'recommendation': [
                    "Meningkatkan **onboarding campaign**.",
                    "Menampilkan **retargeting ads** untuk produk yang pernah dilihat/dicari.",
                    "Mengirimkan **penawaran time-sensitive/terbatas** untuk mendorong pembelian.",
                ]
            },
            'Low Value Customers (At Risk)': {
                'insight': 'Pelanggan yang mulai pasif atau tidak aktif.',
                'recommendation': [
                    "Mengirimkan kampanye **win-back email** dengan diskon besar atau pesan emosional.",
                    "Memberikan survei: **'Apa yang bisa kami tingkatkan?'**",
                    "Menawarkan rekomendasi berbasis histori.",
                    "Menawarkan **penawaran terbatas** untuk mengembalikan pelanggan."
                ]
            },
            'Hibernating': {
                'insight': 'Pernah bernilai tinggi, kini tidak aktif / berisiko hilang.',
                'recommendation': [
                    "Mengirimkan pesan **reaktivasi personalisasi** berdasarkan riwayat pembelian.",
                    "Memberikan **insentif besar untuk kembali** seperti: voucher besar, akses premium gratis.",
                    "Menawatkan program **loyalitas atau upgrade membership**.",
                ]
            }
        }

        # Menampilkan insight dna rekomendasi
        for segment, info in insights_recommendations.items():
            if segment in segment_counts.index:
                st.markdown(f"### 🧩 {segment}")
                st.markdown(f"**Insight:** {info['insight']}")
                st.markdown("**Rekomendasi:**")
                for rec in info['recommendation']:
                    st.markdown(f"- {rec}")
                st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Customer Segmentation (K-Means)":
        st.header("🎯 Customer Segmentation (K-Means Segmentation)")
        
        # Perform K-Means Clustering
        cluster_data = k_means_clustering(df)
        
        # Merge with original data for complete view
        df_cluster = df.merge(cluster_data[['Customer ID', 'AgeCluster', 'EngagementCluster', 'SeasonalCluster']], on='Customer ID', how='left')
    
        tab1, tab2, tab3 = st.tabs(["Age Clustering", "Engagement-Based Clustering", "Seasonal Clustering"])
        
        with tab1:
            st.subheader('Segmentasi Berdasarkan Usia dan Pembelian')

            col1, col2 = st.columns(2)
            
            with col1:
                cluster_distribution = df['AgeCluster'].value_counts().sort_index()
                fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, labels={'x':'Cluster', 'y':'Jumlah'}, title='Distribusi Cluster')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.scatter(df, x='Age', y='Items Purchased',
                               color='AgeCluster',
                               size='Total Spend',
                               title='Spending Patterns by Age Cluster')
                st.plotly_chart(fig, use_container_width=True)
                
            
            # New: Cluster Characteristics
            cluster_stats = df.groupby('AgeCluster').agg({
                'Total Spend': ['mean', 'min', 'max'],
                'Age': ['mean', 'min', 'max'],
                'Items Purchased': ['mean', 'min', 'max']
            }).round(2)

            # AgeCluster Characteristics Analysis
            st.markdown("### 📊 Analisis Karakteristik Cluster")
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Cluster Profiles
            st.markdown("#### Profil Cluster")
            cluster_profiles = {
                0: {
                    "name": "Pelanggan Bernilai Tinggi",
                    "description": "Pelanggan dengan pengeluaran tinggi, usia menengah, dan pembelian item yang konsisten. Memiliki rating tinggi dan keterlibatan aktif.",
                    "recommendation": "Fokus pada program loyalitas dan penawaran eksklusif."
                },
                1: {
                    "name": "Pelanggan Potensial",
                    "description": "Pelanggan dengan pengeluaran menengah, usia lebih bervariasi, dan pembelian item yang bervariasi. Memiliki potensi pertumbuhan.",
                    "recommendation": "Meningkatkan engagement dengan program onboarding dan rekomendasi produk."
                }
            }
            
            for cluster, profile in cluster_profiles.items():
                with st.expander(f"📌 Cluster {cluster}: {profile['name']}"):
                    st.write(f"**Deskripsi:** {profile['description']}")
                    st.write(f"**Rekomendasi:** {profile['recommendation']}")
                    

        with tab2:
            st.subheader('Segmentation Based on Engagement')

            col1, col2 = st.columns(2)
            
            with col1:
                cluster_distribution = df['EngagementCluster'].value_counts().sort_index()
                fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, 
                           labels={'x':'Cluster', 'y':'Count'}, title='Engagement Cluster Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # New: Engagement Patterns
                fig = px.scatter(df, x='Days Since Last Purchase', y='Average Rating',
                               color='EngagementCluster',
                               size='Total Spend',
                               title='Engagement Patterns by Cluster')
                st.plotly_chart(fig, use_container_width=True)
                
            # New: Engagement Metrics by Cluster
            engagement_stats = df.groupby('EngagementCluster').agg({
                'Days Since Last Purchase': ['mean', 'min', 'max'],
                'Average Rating': ['mean', 'min', 'max'],
                'Total Spend': ['mean', 'min', 'max']
            }).round(2)
            
            st.write("**Engagement Metrics by Cluster:**")
            st.dataframe(engagement_stats, use_container_width=True)
            
            st.markdown("#### Engagement Cluster Profile")

            cluster_profiles = {
                2: {
                    "name": "Pelanggan Bernilai Tinggi",
                    "description": "Pelanggan dengan ketertarikan tinggi,  pengeluaran tinggi, baru-baru ini melakukan transaksi, memiliki rating tinggi dan keterlibatan aktif. Merupakan pelanggan top",
                    "recommendation": "Fokus pada program loyalitas dan penawaran eksklusif."
                },
                3: {
                    "name": "Pelanggan Potensial",
                    "description": "Pelanggan dengan pengeluaran tinggi, memberikan rating bagus, dan sudah agak lama bertransaksi. Memiliki potensi pertumbuhan dan pembelian kembali.",
                    "recommendation": "Meningkatkan engagement dengan program onboarding dan rekomendasi produk."
                },
                1: {
                    "name": "Pelanggan Menegah",
                    "description": "Pelanggan dengan pengeluaran menengah, sudah jarang bertransaksi. Rating cenderung menurun dan tidak menunjukkan loyalitas yang kuat.",
                    "recommendation": "Perlu kampanye re-engagement yang agresif dan insentif untuk kembali bertransaksi."
                },
                0: {
                    "name": "Pelanggan Beresiko",
                    "description": "Pelanggan yang melakukan pembelian baru-baru ini, namun memiliki rating rendah dan total spend minimal.",
                    "recommendation": "Fokus pada strategi win-back dan memberikan survei untuk mengetahui keluhan."
                }
            }

            for cluster, profile in cluster_profiles.items():
                with st.expander(f"📌 Cluster {cluster}: {profile['name']}"):
                    st.write(f"**Deskripsi:** {profile['description']}")
                    st.write(f"**Rekomendasi:** {profile['recommendation']}")

        with tab3:
            st.subheader('Segmentation Based on Seasonal')

            col1, col2 = st.columns(2)
            
            with col1:
                cluster_distribution = df['SeasonalCluster'].value_counts().sort_index()
                fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values, 
                           labels={'x':'Cluster', 'y':'Count'}, title='Seasonal Cluster Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
            
            with col2:
                # New: Seasonal Patterns
                fig = px.scatter(df, x='Days Since Last Purchase', y='Total Spend',
                               color='SeasonalCluster',
                               size='Items Purchased',
                               title='Seasonal Purchase Patterns')
                st.plotly_chart(fig, use_container_width=True)

            # New: Seasonal Cluster Characteristics
            seasonal_stats = df.groupby('SeasonalCluster').agg({
                'Days Since Last Purchase': ['mean', 'min', 'max'],
                'Total Spend': ['mean', 'sum'],
                'Items Purchased': ['mean', 'sum']
            }).round(2)
            
            st.write("**Seasonal Cluster Metrics:**")
            st.dataframe(seasonal_stats, use_container_width=True)
            
            # New: Seasonal Cluster Profiles
            st.markdown("#### Seasonal Cluster Profiles")

            seasonal_profiles = {
                1: {
                    "name": "Pelanggan Musiman Bernilai Tinggi",
                    "description": "Melakukan pembelian besar namun jarang. Pola berulang secara musiman dengan kontribusi pendapatan tinggi.",
                    "recommendation": "Berikan penawaran eksklusif saat musim pembelian mereka tiba dan dorong pembelian tambahan melalui bundling musiman."
                },
                3: {
                    "name": "Pelanggan Musiman Reguler",
                    "description": "Melakukan pembelian secara konsisten di musim tertentu dengan pengeluaran sedang.",
                    "recommendation": "Kirimkan pengingat dan promosi pra-musim untuk meningkatkan keterlibatan dan frekuensi pembelian."
                },
                0: {
                    "name": "Pelanggan Musiman Bernilai Rendah",
                    "description": "Pola pembelian tidak konsisten dan nilai transaksi kecil, namun tetap mengikuti musim tertentu.",
                    "recommendation": "Dorong engagement melalui diskon musiman atau program loyalitas mikro (contoh: cashback kecil atau free shipping)."
                },
                4: {
                    "name": "Pelanggan Musiman Baru",
                    "description": "Melakukan pembelian baru-baru ini dengan nilai yang bervariasi, belum memiliki pola tetap.",
                    "recommendation": "Lakukan onboarding personalisasi dan analisis preferensi untuk mengarahkan mereka ke kategori musiman tertentu."
                },
                2: {
                    "name": "Pelanggan Musiman Tidak Aktif",
                    "description": "Sudah agak lama tidak bertransaksi, bahkan di musim-musim yang biasanya aktif.",
                    "recommendation": "Kampanye win-back dengan urgensi musiman (contoh: 'Promo Musim Terakhir') dan penawaran eksklusif."
                }
            }

            for cluster, profile in seasonal_profiles.items():
                with st.expander(f"📌 Cluster {cluster}: {profile['name']}"):
                    st.write(f"**Deskripsi:** {profile['description']}")
                    st.write(f"**Rekomendasi:** {profile['recommendation']}")

    elif page == "Churn Analysis":
        st.header("⚠️ Customer Churn Analysis")
        
        # Build churn model
        model, feature_importance, y_test, y_pred = build_churn_model(df)
        
        col1, col2 = st.columns(2)
        
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
        
        # Feature importance for churn prediction
        fig2 = px.bar(feature_importance.head(8), x='Importance', y='Feature',
                        title="Faktor yang Mempengaruhi Churn Pelanggan",
                        orientation='h',
                        color='Importance',
                        color_continuous_scale=COLOR_SCALES['purples'])
        st.plotly_chart(fig2, use_container_width=True)

        # Retention strategies
        st.subheader("🛡️ Strategi Retensi")
        st.write("""
        **Strategi Jangka Pendek:**
        - Mengirimkan email re-engagement yang dipersonalisasi ke pelanggan yang tidak aktif selama 45+ hari
        - Menawarkan diskon eksklusif untuk pelanggan
        - Mengimplementasikan survei untuk memahami alasan ketidakpuasan
        
        **Strategi Jangka Panjang:**
        - Meningkatkan layanan pelanggan untuk pengalaman dengan rating rendah
        - Mengembangkan program loyalitas untuk pembeli yang loyal
        - Membuat konten lebih terpersonalisasi dengan riwayat pengguna dan segmen pelanggan
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
        
        st.subheader("📊 Analisis Kondisi Saat Ini")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tingkat Churn", f"{churn_rate:.1f}%")
        with col2:
            st.metric("Rata-rata Nilai Pelanggan", f"${avg_clv:.2f}")
        with col3:
            st.metric("Tingkat Kepuasan", f"{satisfaction_rate:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Strategic recommendations based on analysis
        st.subheader("🎯 Rangkuman Strategi Berdasarkan Analisis")
        
        strategies = [
            {
                "priority": "TINGGI",
                "title": "Program Retensi Pelanggan",
                "description": "Implementasikan kampanye retensi yang ditargetkan untuk pelanggan berisiko berdasarkan analisis RFM dan K-Means",
                "actions": [
                    "Kirim email win-back yang dipersonalisasi ke pelanggan Hibernating dan Low Value",
                    "Tawarkan diskon eksklusif untuk pelanggan dengan rating rendah",
                    "Buat saluran dukungan VIP untuk anggota Gold yang berisiko churn",
                    "Implementasikan program loyalitas berbasis poin untuk pelanggan Medium Value"
                ],
                "expected_impact": "Kurangi tingkat churn sebesar 15-20%",
                "timeline": "1-2 bulan"
            },
            {
                "priority": "TINGGI",
                "title": "Strategi Segmentasi Pelanggan",
                "description": "Kembangkan kampanye pemasaran yang ditargetkan untuk setiap segmen pelanggan berdasarkan analisis RFM dan K-Means",
                "actions": [
                    "Buat rekomendasi produk premium untuk Top Customers dan High Value Customers",
                    "Rancang hadiah loyalitas untuk pelanggan dengan Engagement Cluster tinggi",
                    "Implementasikan kampanye onboarding untuk pelanggan baru",
                    "Kembangkan program khusus untuk pelanggan Seasonal dengan nilai tinggi"
                ],
                "expected_impact": "Tingkatkan nilai seumur hidup pelanggan sebesar 25%",
                "timeline": "2-3 bulan"
            },
            {
                "priority": "SEDANG",
                "title": "Peningkatan Pengalaman Pelanggan",
                "description": "Tingkatkan kepuasan pelanggan dan kualitas layanan berdasarkan analisis kepuasan dan rating",
                "actions": [
                    "Implementasikan sistem umpan balik pelanggan real-time",
                    "Tingkatkan waktu respons layanan pelanggan",
                    "Personalisasi rekomendasi produk berdasarkan riwayat pembelian",
                    "Kembangkan program pelatihan untuk tim layanan pelanggan"
                ],
                "expected_impact": "Tingkatkan tingkat kepuasan hingga 85%+",
                "timeline": "3-4 bulan"
            },
            {
                "priority": "SEDANG",
                "title": "Implementasi Analitik Prediktif",
                "description": "Gunakan wawasan berbasis data untuk manajemen pelanggan proaktif berdasarkan model prediksi",
                "actions": [
                    "Terapkan model prediksi churn untuk intervensi dini",
                    "Implementasikan prediksi nilai seumur hidup pelanggan",
                    "Buat peringatan otomatis untuk perubahan perilaku pelanggan",
                    "Kembangkan dashboard analitik real-time untuk monitoring"
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
        
        # Detailed Campaign Strategies
        st.subheader("📢 Detail Strategi Kampanye")
        
        campaign_strategies = {
            "Kampanye Retensi": {
                "Target": ["Hibernating", "Low Value Customers", "Pelanggan dengan Rating Rendah"],
                "Channel": ["Email", "SMS", "Push Notification"],
                "Taktik": [
                    "Email win-back dengan diskon 20%",
                    "Program 'Come Back' dengan hadiah loyalitas",
                    "Survei kepuasan dengan insentif",
                    "Penawaran eksklusif untuk pembelian pertama"
                ]
            },
            "Kampanye Loyalitas": {
                "Target": ["Top Customers", "High Value Customers", "Pelanggan dengan Engagement Tinggi"],
                "Channel": ["Email", "Aplikasi Mobile", "Website"],
                "Taktik": [
                    "Program poin loyalitas premium",
                    "Akses awal ke produk baru",
                    "Hadiah ulang tahun spesial",
                    "Event eksklusif untuk anggota"
                ]
            },
            "Kampanye Reaktivasi": {
                "Target": ["Medium Value Customers", "Pelanggan Musiman"],
                "Channel": ["Email", "Social Media", "Display Ads"],
                "Taktik": [
                    "Rekomendasi produk personalisasi",
                    "Flash sale dengan notifikasi",
                    "Program referral dengan bonus",
                    "Konten edukasi produk"
                ]
            }
        }
        
        for campaign, details in campaign_strategies.items():
            with st.expander(f"🎯 {campaign}"):
                st.write("**Target Audience:**")
                for target in details["Target"]:
                    st.write(f"• {target}")
                
                st.write("**Channel Komunikasi:**")
                for channel in details["Channel"]:
                    st.write(f"• {channel}")
                
                st.write("**Taktik Kampanye:**")
                for tactic in details["Taktik"]:
                    st.write(f"• {tactic}")
        
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
                    color_continuous_scale=COLOR_SCALES['greens'],
                    text='ROI (%)')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # Implementation Roadmap
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
                           color_discrete_sequence=[COLOR_PALETTE['primary'], 
                                                      COLOR_PALETTE['secondary'], 
                                                      COLOR_PALETTE['accent'], 
                                                      COLOR_PALETTE['success']])
        st.plotly_chart(fig_roadmap, use_container_width=True)
        
        # Success Metrics
        st.subheader("📈 Metrik Keberhasilan")
        
        success_metrics = {
            "Metrik Utama": [
                "Penurunan tingkat churn sebesar 15-20%",
                "Peningkatan nilai seumur hidup pelanggan sebesar 25%",
                "Peningkatan tingkat kepuasan hingga 85%+",
                "Peningkatan retensi pelanggan sebesar 30%"
            ],
            "Metrik Pendukung": [
                "Peningkatan engagement rate sebesar 40%",
                "Peningkatan konversi email sebesar 25%",
                "Peningkatan nilai rata-rata pesanan sebesar 15%",
                "Peningkatan skor NPS sebesar 20 poin"
            ]
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Metrik Utama:**")
            for metric in success_metrics["Metrik Utama"]:
                st.write(f"• {metric}")
        
        with col2:
            st.write("**Metrik Pendukung:**")
            for metric in success_metrics["Metrik Pendukung"]:
                st.write(f"• {metric}")

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
    df['AgeCluster'] = kmeans.fit_predict(X_scaled)

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