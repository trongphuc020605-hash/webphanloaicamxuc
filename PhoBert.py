import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import os
import re
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from underthesea import word_tokenize

# ==========================================
# CẤU HÌNH & BỘ NHỚ TRANG WEB
# ==========================================
st.set_page_config(page_title="Hệ thống Quản trị TQM & PCI", page_icon="📊", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>HỆ THỐNG PHÂN TÍCH CẢM XÚC BÌNH LUẬN KHÁCH HÀNG</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6B7280;'>Nghiên cứu đối sánh: PhoBERT - PhoBERT+PCA - CNN-LSTM - TF-IDF+SVM</h4>", unsafe_allow_html=True)
st.divider()

# Khởi tạo bộ nhớ Lịch sử Test
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ==========================================
# NẠP MÔ HÌNH & TỪ ĐIỂN
# ==========================================
@st.cache_resource
def load_all_models():
    # Đã sửa lại thư viện khởi tạo: Bỏ pca, phobert_svm đi, thay bằng phobert_pipeline
    models = {'phobert': None, 'phobert_extractor': None, 'cnn': None, 'cnn_tokenizer': None, 'svm': None, 'tfidf': None, 'phobert_pipeline': None, 'teencode': {}}
    
    # 1. PhoBERT Fine-tuned & Extractor (Lấy vector cho PCA)
    try:
        duong_dan_phobert = "PhucTG2k5P/phobert-cam-xuc-tmdt"
        if os.path.exists(duong_dan_phobert):
            models['phobert'] = pipeline("text-classification", model=duong_dan_phobert, tokenizer=duong_dan_phobert)
            models['phobert_extractor'] = pipeline("feature-extraction", model=duong_dan_phobert, tokenizer=duong_dan_phobert)
    except Exception as e:
        st.error(f"Lỗi nạp PhoBERT: {e}")

    # 2. CNN-LSTM
    try:
        models['cnn'] = load_model('best_cnn_lstm.keras')
        models['cnn_tokenizer'] = joblib.load('cnn_lstm_tokenizer.pickle')
    except Exception as e:
        st.error(f"Lỗi nạp CNN-LSTM: {e}")

    # 3. TF-IDF + SVM Truyền thống
    try:
        models['svm'] = joblib.load('svm_tfidf_model.pkl')
        models['tfidf'] = joblib.load('tfidf_vectorizer.pkl')
    except Exception as e:
        st.error(f"Lỗi nạp TF-IDF+SVM: {e}")

    # 4. PhoBERT + PCA + SVM (Dùng file Pipeline duy nhất)
    try:
        models['phobert_pipeline'] = joblib.load('pca_svm_pipeline.pkl')
    except Exception as e:
        st.warning("⚠️ Chưa tìm thấy file pca_svm_pipeline.joblib. Bỏ qua mô hình PCA+SVM.")
        
    # 5. Teencode
    try:
        with open('teencode_shopee.json', 'r', encoding='utf-8') as f:
            models['teencode'] = json.load(f)
    except Exception as e:
        st.warning("Không tìm thấy file teencode_shopee.json. Bỏ qua bước chuẩn hóa Teencode.")

    return models

with st.spinner("Đang khởi tạo hệ thống đa mô hình..."):
    ai_models = load_all_models()

tu_dien_nhan = {
    'LABEL_0': 'Tiêu cực', 'LABEL_1': 'Trung lập', 'LABEL_2': 'Tích cực',
    0: 'Tiêu cực', 1: 'Trung lập', 2: 'Tích cực'
}

vietnamese_stopwords = set(["và", "là", "có", "không", "của", "cho", "để", "với", "thì", "mà", "bị", "được", "những", "các", "một", "như", "này", "kia", "rất", "cũng", "đã", "đang", "sẽ", "nào", "từ", "tại", "vào", "ra", "lên", "xuống", "qua", "lại", "nữa", "rồi", "khi", "trong", "trên", "dưới", "ngoài", "đây", "đó", "vậy", "á", "nha", "nhé", "ạ", "đi"])

def clean_and_segment_text(text, teencode_dict):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    words = text.split()
    words = [teencode_dict.get(w, w) for w in words]
    text = " ".join(words)
    
    text = word_tokenize(text, format="text")
    return text

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in vietnamese_stopwords])

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
tab1, tab2 = st.tabs(["📊 Phân tích Dữ liệu Hàng loạt (Batch Processing)", "🔍 Kiểm thử Truy vấn Đơn lẻ (Single Inference)"])

# ------------------------------------------
# TÍNH NĂNG 1: BATCH PROCESSING & WORD CLOUD
# ------------------------------------------
with tab1:
    st.subheader("Trích xuất và Đánh giá Chỉ số Quản trị Chất lượng")
    uploaded_file = st.file_uploader("Tải lên tệp dữ liệu bình luận (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        else:
            df = pd.read_excel(uploaded_file)
            
        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox("Chọn trường dữ liệu chứa nội dung:", df.columns)
        with col2:
            date_col = st.selectbox("Chọn trường dữ liệu thời gian (PCI):", ["Không có"] + list(df.columns))

        # Tiền xử lý tạo cột phân đoạn thời gian ảo (Nếu Không có date)
        if date_col == "Không có":
            num_segments = 5
            df['Giai_Doan_Gia_Lap'] = pd.cut(df.index, bins=num_segments, labels=[f"Đợt {i+1}" for i in range(num_segments)], include_lowest=True)

        if st.button("Tiến hành phân tích", type="primary"):
            if ai_models['phobert'] is None:
                st.error("Lỗi: Không tìm thấy mô hình PhoBERT.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                st.divider()
                st.subheader("KẾT QUẢ ĐÁNH GIÁ (LIVE DASHBOARD)")
                metrics_placeholder = st.empty()
                charts_placeholder = st.empty()
                
                ket_qua_ai = []
                danh_sach_text_clean = []
                tong_so_dong = len(df)
                buoc_nhay = max(1, tong_so_dong // 20) 
                
                for i, row in df.iterrows():
                    cau_van_goc = str(row[text_col])[:256]
                    cau_van_sach = clean_and_segment_text(cau_van_goc, ai_models['teencode'])
                    danh_sach_text_clean.append(cau_van_sach)
                    
                    if len(cau_van_sach.strip()) == 0:
                        ket_qua_ai.append("Trung lập")
                    else:
                        nhan_goc = ai_models['phobert'](cau_van_sach)[0]['label']
                        ket_qua_ai.append(tu_dien_nhan.get(nhan_goc, nhan_goc))
                    
                    # LIVE UPDATE (Biểu đồ)
                    if i % buoc_nhay == 0 or i == tong_so_dong - 1:
                        progress_bar.progress((i + 1) / tong_so_dong)
                        status_text.text(f"Đang phân tích: {i+1} / {tong_so_dong} bình luận...")
                        
                        df_temp = df.iloc[:i+1].copy()
                        df_temp['KetQua_PhanLoai'] = ket_qua_ai
                        counts = df_temp['KetQua_PhanLoai'].value_counts()
                        
                        with metrics_placeholder.container():
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            metric_col1.metric("Đã phân tích", f"{i+1}")
                            metric_col2.metric("Tích cực", counts.get("Tích cực", 0))
                            metric_col3.metric("Trung lập", counts.get("Trung lập", 0))
                            metric_col4.metric("Tiêu cực", counts.get("Tiêu cực", 0))
                        
                        with charts_placeholder.container():
                            chart_col1, chart_col2 = st.columns(2)
                            with chart_col1:
                                tqm_data = counts.reset_index()
                                tqm_data.columns = ['Cảm xúc', 'Số lượng']
                                fig_pie = px.pie(tqm_data, values='Số lượng', names='Cảm xúc', color='Cảm xúc',
                                                 color_discrete_map={'Tích cực':'#10B981', 'Trung lập':'#F59E0B', 'Tiêu cực':'#EF4444'}, hole=0.4, title="Biểu đồ TQM (Tỷ lệ hài lòng)")
                                st.plotly_chart(fig_pie, use_container_width=True)
                                
                            with chart_col2:
                                # Tính toán xu hướng thời gian thực hoặc giả lập
                                truong_thoi_gian = date_col if date_col != "Không có" else 'Giai_Doan_Gia_Lap'
                                trend_data = df_temp.groupby([truong_thoi_gian, 'KetQua_PhanLoai'], observed=False).size().reset_index(name='Số lượng')
                                
                                title_chart = "Biểu đồ Xu hướng PCI" if date_col != "Không có" else "Biểu đồ PCI (Mô phỏng theo tiến trình Data)"
                                fig_bar = px.bar(trend_data, x=truong_thoi_gian, y='Số lượng', color='KetQua_PhanLoai', text='Số lượng', 
                                                 color_discrete_map={'Tích cực':'#10B981', 'Trung lập':'#F59E0B', 'Tiêu cực':'#EF4444'},
                                                 barmode='group', title=title_chart)
                                st.plotly_chart(fig_bar, use_container_width=True)

                status_text.text("🎉 Phân tích hoàn tất!")
                
                df['KetQua_PhanLoai'] = ket_qua_ai
                df['Text_Cleaned'] = danh_sach_text_clean
                
                # VẼ ĐÁM MÂY TỪ VỰNG AN TOÀN (CHỐNG CRASH)
                st.divider()
                st.subheader("☁️ ĐÁM MÂY TỪ VỰNG (NHẬN DIỆN TỪ KHÓA LÕI)")
                wc_col1, wc_col2 = st.columns(2)
                
                with wc_col1:
                    st.markdown("**Từ khóa KHÁCH KHEN nhiều nhất**")
                    pos_text = remove_stopwords(" ".join(df[df['KetQua_PhanLoai'] == 'Tích cực']['Text_Cleaned'].astype(str)))
                    if len(pos_text.split()) > 0:
                        try:
                            wc_pos = WordCloud(width=500, height=350, background_color='white', colormap='Greens').generate(pos_text)
                            fig_pos, ax_pos = plt.subplots()
                            ax_pos.imshow(wc_pos, interpolation='bilinear')
                            ax_pos.axis('off')
                            st.pyplot(fig_pos)
                        except ValueError:
                            st.info("Chưa đủ từ vựng hợp lệ để vẽ đám mây.")
                    else:
                        st.info("💡 Không có bình luận Tích cực trong tập dữ liệu.")
                        
                with wc_col2:
                    st.markdown("**Từ khóa KHÁCH CHÊ nhiều nhất**")
                    neg_text = remove_stopwords(" ".join(df[df['KetQua_PhanLoai'] == 'Tiêu cực']['Text_Cleaned'].astype(str)))
                    if len(neg_text.split()) > 0:
                        try:
                            wc_neg = WordCloud(width=500, height=350, background_color='white', colormap='Reds').generate(neg_text)
                            fig_neg, ax_neg = plt.subplots()
                            ax_neg.imshow(wc_neg, interpolation='bilinear')
                            ax_neg.axis('off')
                            st.pyplot(fig_neg)
                        except ValueError:
                            st.info("Chưa đủ từ vựng hợp lệ để vẽ đám mây.")
                    else:
                        st.info("💡 Tuyệt vời! Không có bình luận Tiêu cực nào trong tập dữ liệu.")

                csv = df.drop(columns=['Text_Cleaned', 'Giai_Doan_Gia_Lap'], errors='ignore').to_csv(index=False, encoding='utf-8-sig')
                st.download_button(label="Tải xuống báo cáo kết quả (CSV)", data=csv, file_name='BaoCao_PhanTich_CamXuc.csv', mime='text/csv')

# ------------------------------------------
# TÍNH NĂNG 2: SINGLE INFERENCE (4 MÔ HÌNH CHẠY ĐUA)
# ------------------------------------------
with tab2:
    st.subheader("Kiểm thử Tương tác: Đối chiếu Hiệu năng 4 Mô hình")
    user_input = st.text_area("Nhập nội dung bình luận cần phân tích:", placeholder="Ví dụ: Sản phẩm này tốt nhưng giá quá mắc...")
    
    if st.button("Phân loại đồng loạt", type="primary"):
        if user_input:
            text_clean = clean_and_segment_text(user_input, ai_models['teencode'])
            st.markdown(f"**Văn bản sau tiền xử lý:** `{text_clean}`")
            st.divider()
            
            if len(text_clean.split()) < 2:
                st.error("⚠️ CẢNH BÁO TỪ HỆ THỐNG: Bình luận quá ngắn, chứa dữ liệu rác hoặc không mang ý nghĩa tiếng Việt!")
            else:
                col1, col2, col3, col4 = st.columns(4)
                res_phobert, res_pca_svm, res_cnn, res_svm = "Lỗi", "Lỗi", "Lỗi", "Lỗi"
                
                with col1:
                    st.markdown("#### 🤖 PhoBERT (SOTA)")
                    if ai_models['phobert']:
                        with st.spinner("..."):
                            ket_qua = ai_models['phobert'](text_clean)[0] 
                            nhan_ai = tu_dien_nhan.get(ket_qua['label'], ket_qua['label'])
                            do_tin_cay = round(ket_qua['score'] * 100, 2)
                            st.success(f"**{nhan_ai}**\n\nTin cậy: {do_tin_cay}%")
                            res_phobert = f"{nhan_ai} ({do_tin_cay}%)"
                    else:
                        st.error("Chưa nạp.")
                        
                with col2:
                    st.markdown("#### 🧬 PhoBERT+PCA+SVM")
                    if ai_models.get('phobert_pipeline') and ai_models.get('phobert_extractor'):
                        with st.spinner("..."):
                            try:
                                # 1. Lấy vector 768 chiều từ PhoBERT
                                features = ai_models['phobert_extractor'](text_clean)
                                cls_embedding = np.array(features[0][0]).reshape(1, -1)
                                
                                # 2. Đưa thẳng vào Pipeline (Nó tự PCA rồi tự SVM luôn)
                                label_idx = int(ai_models['phobert_pipeline'].predict(cls_embedding)[0])
                                
                                st.success(f"**{tu_dien_nhan[label_idx]}**\n\n*(Cơ chế cực nhẹ)*")
                                res_pca_svm = f"{tu_dien_nhan[label_idx]}"
                            except Exception as e:
                                st.error("Lỗi dự đoán PCA+SVM")
                    else:
                        st.error("Thiếu File")
                        
                with col3:
                    st.markdown("#### 🧠 CNN-LSTM")
                    if ai_models['cnn'] and ai_models['cnn_tokenizer']:
                        with st.spinner("..."):
                            seq = ai_models['cnn_tokenizer'].texts_to_sequences([text_clean])
                            padded = pad_sequences(seq, maxlen=80) 
                            pred = ai_models['cnn'].predict(padded, verbose=0)
                            label_idx = int(np.argmax(pred, axis=1)[0])
                            do_tin_cay = round(float(np.max(pred)) * 100, 2)
                            st.success(f"**{tu_dien_nhan[label_idx]}**\n\nTin cậy: {do_tin_cay}%")
                            res_cnn = f"{tu_dien_nhan[label_idx]} ({do_tin_cay}%)"
                    else:
                        st.error("Chưa nạp.")
                        
                with col4:
                    st.markdown("#### ⚡ TF-IDF+SVM")
                    if ai_models['svm'] and ai_models['tfidf']:
                        with st.spinner("..."):
                            vec = ai_models['tfidf'].transform([text_clean])
                            label_idx = int(ai_models['svm'].predict(vec)[0])
                            try:
                                proba = ai_models['svm'].predict_proba(vec)[0]
                                do_tin_cay = round(max(proba) * 100, 2)
                                st.success(f"**{tu_dien_nhan[label_idx]}**\n\nTin cậy: {do_tin_cay}%")
                                res_svm = f"{tu_dien_nhan[label_idx]} ({do_tin_cay}%)"
                            except AttributeError:
                                st.success(f"**{tu_dien_nhan[label_idx]}**\n\n*Không xuất XS*")
                                res_svm = f"{tu_dien_nhan[label_idx]}"
                    else:
                        st.error("Chưa nạp.")
                
                st.session_state['history'].append({
                    "📝 Câu bình luận": user_input,
                    "🤖 PhoBERT": res_phobert,
                    "🧬 PCA+SVM": res_pca_svm,
                    "🧠 CNN-LSTM": res_cnn,
                    "⚡ TF-IDF": res_svm
                })
        else:
            st.warning("Vui lòng nhập văn bản.")

    st.divider()
    st.subheader("🕒 Bảng Ghi nhận Lịch sử Kiểm thử")
    if st.session_state['history']:
        df_history = pd.DataFrame(st.session_state['history']).iloc[::-1].reset_index(drop=True)
        st.dataframe(df_history, use_container_width=True)
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.info("Chưa có lịch sử kiểm thử nào được ghi nhận.")
