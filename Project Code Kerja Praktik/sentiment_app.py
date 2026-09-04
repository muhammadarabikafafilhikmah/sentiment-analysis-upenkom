import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import re
import string
from collections import Counter
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen UPENKOM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .positive-sentiment {
        color: #28a745;
        font-weight: bold;
    }
    .negative-sentiment {
        color: #dc3545;
        font-weight: bold;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi preprocessing (dari analisis asli)
def case_folding(text):
    if pd.isna(text) or text == '':
        return ''
    return str(text).lower()

def cleaning(text):
    if pd.isna(text) or text == '':
        return ''
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenizing(text):
    if pd.isna(text) or text == '':
        return []
    return text.split()

def create_simple_stopwords():
    return {
        'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 'oleh',
        'adalah', 'akan', 'telah', 'sudah', 'dapat', 'bisa', 'harus', 'atau', 'jika',
        'itu', 'ini', 'ada', 'tidak', 'juga', 'saya', 'kita', 'kami', 'mereka',
        'dia', 'ia', 'nya', 'mu', 'ku', 'se', 'ter', 'ber', 'per', 'an', 'al'
    }

def remove_stopwords(tokens):
    stop_words = create_simple_stopwords()
    return [token for token in tokens if token not in stop_words]

def preprocess_text(text):
    if pd.isna(text) or text == '':
        return ''
    text = case_folding(text)
    text = cleaning(text)
    tokens = tokenizing(text)
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)

def create_indonesian_lexicon():
    return {
        # Kata positif
        'baik': 1, 'bagus': 1, 'hebat': 1, 'luar': 1, 'biasa': 1, 'sempurna': 1,
        'mantap': 1, 'keren': 1, 'oke': 1, 'lancar': 1, 'mudah': 1, 'jelas': 1,
        'membantu': 1, 'profesional': 1, 'ramah': 1, 'cepat': 1, 'efisien': 1,
        'memuaskan': 1, 'terima': 1, 'kasih': 1, 'senang': 1, 'puas': 1,
        'lulus': 1, 'harap': 1, 'hasil': 1, 'nilai': 1, 'laksana': 1,
        
        # Kata negatif
        'buruk': -1, 'jelek': -1, 'payah': -1, 'lambat': -1, 'sulit': -1,
        'rumit': -1, 'ribet': -1, 'susah': -1, 'kurang': -1, 'tidak': -1,
        'bingung': -1, 'kecewa': -1, 'mahal': -1, 'lama': -1, 'error': -1,
        'gagal': -1, 'masalah': -1, 'trouble': -1, 'problem': -1, 'tambah': -1,
        'tulis': -1, 'talent': -1
    }

def lexicon_sentiment_analysis(text, lexicon):
    if pd.isna(text) or text == '':
        return 0, 'netral'
    
    tokens = text.split()
    sentiment_score = 0
    count = 0
    
    for token in tokens:
        if token in lexicon:
            sentiment_score += lexicon[token]
            count += 1
    
    if count > 0:
        sentiment_score = sentiment_score / count
    
    if sentiment_score > 0.05:
        label = 'positif'
    elif sentiment_score < -0.05:
        label = 'negatif'
    else:
        if any(word in text.lower() for word in ['baik', 'bagus', 'lancar', 'mudah']):
            label = 'positif'
        elif any(word in text.lower() for word in ['sulit', 'kurang', 'lambat', 'rumit']):
            label = 'negatif'
        else:
            label = 'positif'
        
    return sentiment_score, label

# Data hasil analisis dari dokumen
@st.cache_data
def load_analysis_results():
    # Data berdasarkan hasil analisis asli
    analysis_results = {
        'total_data': 198,
        'valid_timestamp': 131,
        'lexicon_results': {
            'positif': 171,
            'negatif': 27,
            'netral': 0
        },
        'ml_results': {
            'positif': 173,
            'negatif': 25
        },
        'model_performance': {
            'accuracy': 0.9275,
            'cross_val_mean': 0.8741,
            'cross_val_std': 0.0360,
            'agreement': 0.9697
        },
        'confusion_matrix': [[31, 4], [1, 33]],
        'classification_report': {
            'negatif': {'precision': 0.9688, 'recall': 0.8857, 'f1': 0.9254, 'support': 35},
            'positif': {'precision': 0.8919, 'recall': 0.9706, 'f1': 0.9296, 'support': 34}
        },
        'monthly_trends': {
            '2022-01': {'total': 40, 'positif': 36, 'negatif': 4},
            '2022-02': {'total': 21, 'positif': 18, 'negatif': 3},
            '2022-03': {'total': 70, 'positif': 60, 'negatif': 10}
        },
        'top_positive_keywords': [
            ('baik', 113), ('waktu', 80), ('yg', 71), ('lebih', 63), ('dapat', 57),
            ('lulus', 48), ('laksana', 47), ('harap', 45), ('hasil', 43), ('nilai', 42)
        ],
        'top_negative_keywords': [
            ('waktu', 18), ('kurang', 13), ('yg', 11), ('tambah', 10), ('masalah', 9),
            ('sulit', 8), ('tulis', 7), ('lama', 7), ('talent', 7), ('ada', 7)
        ]
    }
    return analysis_results

# Load data dummy untuk preview
@st.cache_data
def load_sample_data():
    # Sample data berdasarkan preview dari analisis asli
    sample_data = {
        'timestamp': [
            '1/24/2022 13:36:22', '1/24/2022 14:21:55', '1/24/2022 14:26:30',
            '1/24/2022 14:40:47', '1/24/2022 14:41:07'
        ],
        'instrumen_sulit': [
            'Waktu terlalu cepat', '-', 'Waktunya singkat dg masalah yg banyak',
            '-', 'Harus menulis banyak'
        ],
        'harapan_peserta': [
            'Harapan saya bisa lolos uji kompetensi',
            'Meningkat pemahaman terhadap nilai-nilai leadership',
            'Terpenuhi 9 kompetensi yg diasess',
            'Hasil dapat tersampaikan',
            'Dapat segera dilantik sebagai Pejabat Administrator'
        ],
        'saran_kritik': [
            'Sudah sangat baik.', 'Terus dipertahankan dan ditingkatkan',
            'Waktu lebih diperpanjang', '-', 'Lewat Model C. A. T'
        ]
    }
    return pd.DataFrame(sample_data)

# Header utama
st.markdown('<div class="main-header">🎯 Dashboard Analisis Sentimen UPENKOM</div>', unsafe_allow_html=True)

# Load hasil analisis
results = load_analysis_results()
df_sample = load_sample_data()

# Sidebar
st.sidebar.title("🔧 Pengaturan Dashboard")
st.sidebar.markdown("---")

# Informasi dataset di sidebar
st.sidebar.markdown("### 📊 Informasi Dataset")
st.sidebar.markdown(f"**Total Data:** {results['total_data']}")
st.sidebar.markdown(f"**Data Valid:** {results['valid_timestamp']}")
st.sidebar.markdown(f"**Periode:** Jan - Mar 2022")

# Menu navigasi (menghapus menu analisis teks baru)
menu = st.sidebar.selectbox(
    "Pilih Halaman:",
    ["📊 Dashboard Utama", "📈 Analisis Tren", "📋 Data & Statistik", "🎯 Model Performance"]
)

if menu == "📊 Dashboard Utama":
    st.header("📊 Ringkasan Analisis Sentimen")
    
    # Metrics row dengan data asli
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Respons",
            value=f"{results['total_data']}",
            delta="Data Feb-Mar 2022"
        )
    
    with col2:
        lexicon_pos_pct = (results['lexicon_results']['positif'] / results['total_data']) * 100
        st.metric(
            label="Sentimen Positif (Lexicon)",
            value=f"{lexicon_pos_pct:.1f}%",
            delta="86.4%"
        )
    
    with col3:
        ml_pos_pct = (results['ml_results']['positif'] / results['total_data']) * 100
        st.metric(
            label="Sentimen Positif (ML)", 
            value=f"{ml_pos_pct:.1f}%",
            delta="87.4%"
        )
    
    with col4:
        st.metric(
            label="Akurasi Model",
            value=f"{results['model_performance']['accuracy']:.1%}",
            delta=f"±{results['model_performance']['cross_val_std']:.1%}"
        )
    
    # Insight box
    st.markdown(f"""
    <div class="insight-box">
        <h4>🔍 Key Insights</h4>
        <ul>
            <li><strong>Sentimen Sangat Positif:</strong> {lexicon_pos_pct:.1f}% responden memberikan feedback positif</li>
            <li><strong>Agreement Tinggi:</strong> {results['model_performance']['agreement']:.1%} kesesuaian antara Lexicon dan ML</li>
            <li><strong>Model Akurat:</strong> Akurasi {results['model_performance']['accuracy']:.1%} dengan cross-validation {results['model_performance']['cross_val_mean']:.1%}</li>
            <li><strong>Bulan Terbaik:</strong> Januari 2022 dengan 90% sentimen positif</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualisasi utama yang diperbaiki
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Sentimen")
        
        # Pie chart untuk distribusi sentimen berdasarkan data asli
        labels = ['Positif', 'Negatif']
        lexicon_values = [results['lexicon_results']['positif'], results['lexicon_results']['negatif']]
        colors = ['#28a745', '#dc3545']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=lexicon_values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent+value',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Distribusi Sentimen (Lexicon-based)",
            annotations=[dict(text='Sentimen<br>Lexicon', x=0.5, y=0.5, font_size=16, showarrow=False)],
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Perbandingan Metode")
        
        # Bar chart perbandingan metode dengan data asli - diperbaiki
        methods = ['Lexicon', 'Machine Learning']
        positive = [
            (results['lexicon_results']['positif'] / results['total_data']) * 100,
            (results['ml_results']['positif'] / results['total_data']) * 100
        ]
        negative = [
            (results['lexicon_results']['negatif'] / results['total_data']) * 100,
            (results['ml_results']['negatif'] / results['total_data']) * 100
        ]
        
        fig = go.Figure()
        
        # Menambahkan trace secara terpisah dengan text yang benar
        fig.add_trace(go.Bar(
            name='Positif', 
            x=methods, 
            y=positive, 
            marker_color='#28a745',
            text=[f'{p:.1f}%' for p in positive],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Negatif', 
            x=methods, 
            y=negative, 
            marker_color='#dc3545',
            text=[f'{n:.1f}%' for n in negative],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Perbandingan Hasil Metode Analisis',
            barmode='group',
            yaxis_title='Persentase (%)',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data preview
    st.markdown("---")
    st.subheader("📋 Preview Data Respons")
    st.dataframe(df_sample, use_container_width=True)
    
    # Tambahan: Visualisasi sentiment distribution yang lebih menarik
    st.markdown("---")
    st.subheader("🎯 Agreement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart untuk agreement
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = results['model_performance']['agreement'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Agreement Rate (%)"},
            delta = {'reference': 95},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': "#1f77b4"},
                     'steps' : [
                         {'range': [0, 80], 'color': "#ffcccb"},
                         {'range': [80, 95], 'color': "#ffffcc"},
                         {'range': [95, 100], 'color': "#ccffcc"}],
                     'threshold' : {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 95}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary stats
        st.markdown("### 📈 Summary Statistics")
        st.markdown(f"""
        - **Total Agreement:** {results['model_performance']['agreement']:.1%}
        - **Model Accuracy:** {results['model_performance']['accuracy']:.1%}
        - **Cross-validation:** {results['model_performance']['cross_val_mean']:.1%} ± {results['model_performance']['cross_val_std']:.1%}
        - **Positive Sentiment Dominance:** {lexicon_pos_pct:.1f}%
        """)
        
        # Quick action buttons
        st.markdown("### 🚀 Quick Actions")
        if st.button("📊 Lihat Detail Tren", type="primary"):
            st.session_state.selected_menu = "📈 Analisis Tren"
            st.rerun()

elif menu == "📈 Analisis Tren":
    st.header("📈 Analisis Tren Sentimen")
    
    # Tren sentimen berdasarkan data asli
    st.subheader("📅 Tren Sentimen Bulanan (2022)")
    
    monthly_data = results['monthly_trends']
    months = list(monthly_data.keys())
    positive_counts = [monthly_data[month]['positif'] for month in months]
    negative_counts = [monthly_data[month]['negatif'] for month in months]
    positive_pcts = [monthly_data[month]['positif']/monthly_data[month]['total']*100 for month in months]
    negative_pcts = [monthly_data[month]['negatif']/monthly_data[month]['total']*100 for month in months]
    
    # Grafik tren persentase
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months, y=positive_pcts,
        mode='lines+markers',
        name='Sentimen Positif',
        line=dict(color='#28a745', width=3),
        marker=dict(size=10),
        text=[f'{p:.1f}% ({positive_counts[i]}/{monthly_data[months[i]]["total"]})' for i, p in enumerate(positive_pcts)],
        textposition="top center"
    ))
    
    fig.add_trace(go.Scatter(
        x=months, y=negative_pcts,
        mode='lines+markers',
        name='Sentimen Negatif',
        line=dict(color='#dc3545', width=3),
        marker=dict(size=10),
        text=[f'{n:.1f}% ({negative_counts[i]}/{monthly_data[months[i]]["total"]})' for i, n in enumerate(negative_pcts)],
        textposition="bottom center"
    ))
    
    fig.update_layout(
        title='Tren Sentimen UPENKOM (Januari - Maret 2022)',
        xaxis_title='Bulan',
        yaxis_title='Persentase (%)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistik tren
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bulan Terbaik", "Jan 2022", "90.0% positif")
    with col2:
        avg_positive = np.mean(positive_pcts)
        st.metric("Rata-rata Positif", f"{avg_positive:.1f}%")
    with col3:
        total_responses = sum([monthly_data[month]['total'] for month in months])
        st.metric("Total Respons", total_responses)
    
    # Analisis kata kunci
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Kata Kunci Positif Teratas")
        
        pos_words = [word for word, count in results['top_positive_keywords'][:5]]
        pos_counts = [count for word, count in results['top_positive_keywords'][:5]]
        
        fig = go.Figure([go.Bar(
            x=pos_counts, y=pos_words,
            orientation='h',
            marker_color='#28a745',
            text=pos_counts,
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Top 5 Kata Kunci Positif",
            xaxis_title="Frekuensi",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔴 Kata Kunci Negatif Teratas")
        
        neg_words = [word for word, count in results['top_negative_keywords'][:5]]
        neg_counts = [count for word, count in results['top_negative_keywords'][:5]]
        
        fig = go.Figure([go.Bar(
            x=neg_counts, y=neg_words,
            orientation='h',
            marker_color='#dc3545',
            text=neg_counts,
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Top 5 Kata Kunci Negatif",
            xaxis_title="Frekuensi",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif menu == "🎯 Model Performance":
    st.header("🎯 Performa Model Analisis Sentimen")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Akurasi", f"{results['model_performance']['accuracy']:.1%}")
    with col2:
        st.metric("Cross-Validation", f"{results['model_performance']['cross_val_mean']:.1%}")
    with col3:
        st.metric("Std Deviasi", f"{results['model_performance']['cross_val_std']:.1%}")
    with col4:
        st.metric("Agreement", f"{results['model_performance']['agreement']:.1%}")
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    
    cm_data = np.array(results['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif', 'Positif'],
                yticklabels=['Negatif', 'Positif'],
                ax=ax)
    plt.title('Confusion Matrix Model Machine Learning')
    plt.ylabel('Label Aktual')
    plt.xlabel('Prediksi Model')
    
    st.pyplot(fig)
    
    # Classification Report
    st.subheader("📊 Classification Report")
    
    report_data = []
    for label, metrics in results['classification_report'].items():
        report_data.append({
            'Kelas': label.capitalize(),
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1']:.3f}",
            'Support': int(metrics['support'])
        })
    
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True)
    
    # Agreement Analysis
    st.subheader("🤝 Agreement Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Agreement Rate", f"{results['model_performance']['agreement']:.1%}")
        st.write("Tingkat kesesuaian antara metode Lexicon dan Machine Learning sangat tinggi.")
    
    with col2:
        # Visualisasi agreement
        agreement_data = {
            'Metode': ['Lexicon', 'Machine Learning'],
            'Positif (%)': [
                (results['lexicon_results']['positif'] / results['total_data']) * 100,
                (results['ml_results']['positif'] / results['total_data']) * 100
            ]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agreement_data['Metode'],
            y=agreement_data['Positif (%)'],
            mode='lines+markers',
            name='Sentimen Positif',
            line=dict(color='#28a745', width=3),
            marker=dict(size=12)
        ))
        
        fig.update_layout(
            title='Perbandingan Hasil Kedua Metode',
            yaxis_title='Persentase Positif (%)',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif menu == "📋 Data & Statistik":
    st.header("📋 Data & Statistik Lengkap")
    
    # Dataset Info
    st.subheader("ℹ️ Informasi Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Dataset", "198 baris", "42 kolom")
    with col2:
        st.metric("Data Teranalisis", "198 baris", "4 kolom utama")
    with col3:
        st.metric("Timestamp Valid", f"{results['valid_timestamp']}", f"{(results['valid_timestamp']/results['total_data']*100):.1f}%")
    
    # Kolom yang dianalisis
    st.subheader("📊 Kolom yang Dianalisis")
    columns_info = {
        'Kolom': [
            'Timestamp',
            'Instrumen Sulit',
            'Harapan Peserta', 
            'Saran Kritik'
        ],
        'Deskripsi': [
            'Waktu pengisian kuesioner',
            'Alasan instrumen penilaian yang paling sulit',
            'Harapan setelah mengikuti penilaian kompetensi',
            'Saran untuk perbaikan penyelenggaraan'
        ],
        'Data Valid': [
            f"{results['valid_timestamp']}/{results['total_data']}",
            f"{results['total_data']}/{results['total_data']}",
            f"{results['total_data']}/{results['total_data']}",
            f"{results['total_data']}/{results['total_data']}"
        ]
    }
    
    columns_df = pd.DataFrame(columns_info)
    st.dataframe(columns_df, use_container_width=True)
    
    # Distribusi Detail Sentimen
    st.subheader("📈 Distribusi Detail Sentimen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Hasil Lexicon-based Analysis**")
        lexicon_data = {
            'Sentimen': ['Positif', 'Negatif', 'Netral'],
            'Jumlah': [
                results['lexicon_results']['positif'],
                results['lexicon_results']['negatif'],
                results['lexicon_results']['netral']
            ],
            'Persentase': [
                f"{(results['lexicon_results']['positif']/results['total_data']*100):.1f}%",
                f"{(results['lexicon_results']['negatif']/results['total_data']*100):.1f}%",
                f"{(results['lexicon_results']['netral']/results['total_data']*100):.1f}%"
            ]
        }
        lexicon_df = pd.DataFrame(lexicon_data)
        st.dataframe(lexicon_df, use_container_width=True)
    
    with col2:
        st.markdown("**🤖 Hasil Machine Learning**")
        ml_data = {
            'Sentimen': ['Positif', 'Negatif'],
            'Jumlah': [
                results['ml_results']['positif'],
                results['ml_results']['negatif']
            ],
            'Persentase': [
                f"{(results['ml_results']['positif']/results['total_data']*100):.1f}%",
                f"{(results['ml_results']['negatif']/results['total_data']*100):.1f}%"
            ]
        }
        ml_df = pd.DataFrame(ml_data)
        st.dataframe(ml_df, use_container_width=True)
    
    # Sample Data dengan Preprocessing
    st.subheader("🔧 Contoh Preprocessing Data")
    
    sample_texts = [
        "Pelayanan UPENKOM sudah sangat baik dan profesional",
        "Waktu terlalu cepat dan harus menulis banyak", 
        "Harapan saya bisa lolos uji kompetensi dengan hasil yang memuaskan"
    ]
    
    preprocessing_demo = []
    lexicon = create_indonesian_lexicon()
    
    for i, text in enumerate(sample_texts, 1):
        processed = preprocess_text(text)
        score, sentiment = lexicon_sentiment_analysis(processed, lexicon)
        
        preprocessing_demo.append({
            'No': i,
            'Teks Asli': text,
            'Setelah Preprocessing': processed,
            'Skor Sentimen': f"{score:.3f}",
            'Label': sentiment.capitalize(),
            'Warna': '🟢' if sentiment == 'positif' else '🔴'
        })
    
    demo_df = pd.DataFrame(preprocessing_demo)
    st.dataframe(demo_df, use_container_width=True)
    
    # Statistik Kata Kunci
    st.subheader("🔤 Analisis Kata Kunci")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 10 Kata Positif**")
        pos_keywords_df = pd.DataFrame(
            results['top_positive_keywords'][:10],
            columns=['Kata', 'Frekuensi']
        )
        st.dataframe(pos_keywords_df, use_container_width=True)
    
    with col2:
        st.markdown("**Top 10 Kata Negatif**")
        neg_keywords_df = pd.DataFrame(
            results['top_negative_keywords'][:10],
            columns=['Kata', 'Frekuensi']
        )
        st.dataframe(neg_keywords_df, use_container_width=True)
    
    # Export Data
    st.markdown("---")
    st.subheader("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Hasil Analisis", type="primary"):
            results_export = {
                'total_data': results['total_data'],
                'lexicon_positif': results['lexicon_results']['positif'],
                'lexicon_negatif': results['lexicon_results']['negatif'],
                'ml_positif': results['ml_results']['positif'],
                'ml_negatif': results['ml_results']['negatif'],
                'akurasi': results['model_performance']['accuracy'],
                'agreement': results['model_performance']['agreement']
            }
            st.json(results_export)
    
    with col2:
        if st.button("📈 Download Tren Data"):
            trend_export = results['monthly_trends']
            st.json(trend_export)
    
    with col3:
        if st.button("📝 Download Keywords"):
            keywords_export = {
                'positive_keywords': results['top_positive_keywords'][:20],
                'negative_keywords': results['top_negative_keywords'][:20]
            }
            st.json(keywords_export)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>📊 Dashboard Analisis Sentimen UPENKOM</h4>
    <p>Sistem Analisis Sentimen Feedback Peserta Uji Penilaian Kompetensi</p>
    <p><strong>Metode:</strong> Lexicon-based Analysis & Machine Learning (Naive Bayes)</p>
    <p><strong>Dataset:</strong> 198 respons dari Januari - Maret 2022</p>
    <p><strong>Akurasi Model:</strong> 92.75% | <strong>Agreement:</strong> 96.97%</p>
</div>
""", unsafe_allow_html=True)

# Sidebar tambahan - About
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Tentang Aplikasi")
st.sidebar.markdown("""
Dashboard Analisis Sentimen UPENKOM adalah aplikasi web untuk menganalisis 
sentimen feedback peserta Uji Penilaian Kompetensi menggunakan dua pendekatan:

1. **Lexicon-based Analysis**: Menggunakan kamus kata sentimen bahasa Indonesia
2. **Machine Learning**: Model Naive Bayes dengan akurasi 92.75%

**Fitur Utama:**
- 📊 Dashboard analisis real-time
- 📈 Visualisasi tren sentimen
- 📋 Statistik lengkap
- 🎯 Evaluasi performa model
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Kontak")
st.sidebar.markdown("""
**Tim Pengembang:**
- Data Science Team
- UPENKOM Analysis Division

**Versi:** 1.0.0  
**Update:** Maret 2024
""")