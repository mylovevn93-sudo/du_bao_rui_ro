import streamlit as tf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import io

# ==========================================
# CẤU HÌNH TRANG WEB STREAMLIT BẮT BUỘC ĐẦU TIÊN
# ==========================================
tf.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# ==========================================
# CÁC HÀM CACHE DÙNG CHUNG
# ==========================================
@tf.cache_data
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes để đảm bảo tính hashable cho st.cache_data
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            tf.error("Định dạng tệp không được hỗ trợ. Vui lòng tải lên file CSV hoặc Excel.")
            return None
        return df
    except Exception as e:
        tf.error(f"Lỗi khi đọc file: {e}")
        return None

# ==========================================
# KHỞI TẠO SESSION STATE
# ==========================================
if 'model' not in tf.session_state:
    tf.session_state['model'] = None
if 'trained_features' not in tf.session_state:
    tf.session_state['trained_features'] = None
if 'train_metrics' not in tf.session_state:
    tf.session_state['train_metrics'] = None
if 'confusion_matrix' not in tf.session_state:
    tf.session_state['confusion_matrix'] = None
if 'roc_data' not in tf.session_state:
    tf.session_state['roc_data'] = None

# ==========================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with tf.sidebar:
    tf.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu huấn luyện
    uploaded_file = tf.file_uploader(
        "Tải lên dữ liệu huấn luyện (CSV/Excel)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu mẫu chứa các đặc trưng từ X_1 đến X_14 và cột mục tiêu 'default'"
    )
    
    tf.divider()
    
    # 2. Cấu hình Tham số Mô hình (Trích xuất từ RandomForest trong notebook)
    tf.subheader("Tham số mô hình AI")
    tf.caption("Thuật toán: RandomForestClassifier")
    
    n_estimators = tf.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định trong rừng độc lập."
    )
    
    max_depth = tf.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=1, 
        max_value=30, 
        value=10, 
        step=1,
        help="Độ sâu tối đa của mỗi cây quyết định. Giúp kiểm soát quá khớp."
    )
    
    min_samples_split = tf.slider(
        "Số mẫu tối thiểu để tách nút (min_samples_split)", 
        min_value=2, 
        max_value=20, 
        value=2, 
        step=1,
        help="Số lượng mẫu tối thiểu cần thiết để phân tách một nút nội bộ."
    )
    
    random_state = tf.number_input(
        "Trạng thái ngẫu nhiên (random_state)", 
        min_value=0, 
        max_value=9999, 
        value=42, 
        step=1,
        help="Hạt giống ngẫu nhiên để đảm bảo kết quả có thể tái hiện lại."
    )
    
    tf.divider()
    
    # 3. Nút kích hoạt Huấn luyện
    trigger_train = tf.button(
        "🚀 Huấn luyện Mô hình", 
        type="primary", 
        use_container_width=True,
        help="Nhấp để bắt đầu quá trình trích xuất đặc trưng và huấn luyện thuật toán RandomForest."
    )

# ==========================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
tf.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận & Rủi ro Mặc định")
tf.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch tài chính, tự động phát hiện các hành vi gian lận hoặc rủi ro dựa trên học máy.")

if uploaded_file is None:
    tf.info("👋 Chào mừng! Vui lòng tải tệp dữ liệu giao dịch mẫu ở thanh Sidebar bên trái để bắt đầu.")
    tf.stop()

# Nạp dữ liệu qua hàm cache chung
file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    tf.stop()

tf.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | Quy mô: {df_raw.shape[0]} hàng, {df_raw.shape[1]} cột")
tf.divider()

# Xác định danh sách biến tự động dựa trên cấu trúc file đã phân tích
expected_features = [f"X_{i}" for i in range(1, 15)]
target_col = "default"

# Kiểm tra tính hợp lệ của cấu trúc file dữ liệu
missing_cols = [col for col in expected_features + [target_col] if col not in df_raw.columns]
if missing_cols:
    tf.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc sau: {missing_cols}. Vui lòng kiểm tra lại cấu trúc file mẫu.")
    tf.stop()

# ==========================================
# KHỐI HUẤN LUYỆN (Chỉ chạy khi bấm nút ở sidebar)
# ==========================================
if trigger_train:
    with tf.spinner("🔄 Đang xử lý dữ liệu và huấn luyện mô hình..."):
        X = df_raw[expected_features]
        y = df_raw[target_col]
        
        # Chia dữ liệu Train/Test theo tỷ lệ chuẩn 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
        
        # Khởi tạo và huấn luyện mô hình RandomForest theo tham số người dùng nhập
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        # Dự báo trên tập kiểm định
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        
        # Lưu kết quả vào Session State
        tf.session_state['model'] = clf
        tf.session_state['trained_features'] = expected_features
        
        # Tính toán chỉ số đánh giá
        tf.session_state['train_metrics'] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        }
        
        tf.session_state['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            tf.session_state['roc_data'] = (fpr, tpr)
            
    tf.success("🎉 Huấn luyện mô hình thành công! Chuyển sang các Tab bên dưới để xem chi tiết.")

# ==========================================
# THIẾT KẾ GIAO DIỆN TABS CHÍNH
# ==========================================
tab1, tab2, tab3, tab4 = tf.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🔬 Kết quả kiểm định", 
    "🔮 Sử dụng mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------
with tab1:
    tf.subheader("Phân tích Thống kê Mô tả")
    
    col_m1, col_m2, col_m3 = tf.columns(3)
    col_m1.metric("Tổng số bản ghi", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số đặc trưng mô hình (X)", len(expected_features))
    col_m3.metric("Kích thước tệp tin", f"{len(file_bytes) / (1024*1024):.2f} MB")
    
    tf.write("##### 📋 Xem trước 5 dòng dữ liệu đầu tiên")
    tf.dataframe(df_raw.head(5), use_container_width=True)
    
    tf.write("##### 📐 Bảng tóm tắt thống kê các biến mô hình")
    # Chỉ hiển thị thống kê cho các biến đưa vào mô hình theo quy tắc
    tf.dataframe(df_raw[expected_features + [target_col]].describe().T, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------
with tab2:
    tf.subheader("Trực quan phân phối biến đặc trưng")
    
    # Cho phép người dùng tùy chọn hiển thị các biến quan tâm nếu danh sách quá dài
    selected_vis_features = tf.multiselect(
        "Chọn các biến đặc trưng muốn trực quan hóa (Mặc định chọn 4 biến đầu):",
        options=expected_features,
        default=expected_features[:4]
    )
    
    # Lưới hiển thị biểu đồ phân bổ
    tf.write("##### 📊 Phân phối của biến mục tiêu và các đặc trưng lựa chọn")
    
    # Hàng 1: Biến mục tiêu và Biến đặc trưng thứ nhất
    c1, c2 = tf.columns(2)
    with c1:
        # Biểu đồ phân phối biến mục tiêu (y)
        target_counts = df_raw[target_col].value_counts().reset_index()
        target_counts.columns = [target_col, 'Số lượng']
        target_counts[target_col] = target_counts[target_col].map({0: 'Bình thường (0)', 1: 'Rủi ro/Gian lận (1)'})
        fig_target = px.bar(target_counts, x=target_col, y='Số lượng', color=target_col,
                            title="Tỷ lệ phân phối biến mục tiêu (default)", height=350)
        tf.plotly_chart(fig_target, use_container_width=True)
        
    with c2:
        if len(selected_vis_features) > 0:
            feat = selected_vis_features[0]
            fig_f1 = px.histogram(df_raw, x=feat, color=target_col, marginal="box",
                                  title=f"Phân phối đặc trưng {feat} theo biến mục tiêu", height=350)
            tf.plotly_chart(fig_f1, use_container_width=True)

    # Hàng 2: Biến đặc trưng tiếp theo nếu có
    c3, c4 = tf.columns(2)
    with c3:
        if len(selected_vis_features) > 1:
            feat = selected_vis_features[1]
            fig_f2 = px.histogram(df_raw, x=feat, color=target_col, marginal="box",
                                  title=f"Phân phối đặc trưng {feat} theo biến mục tiêu", height=350)
            tf.plotly_chart(fig_f2, use_container_width=True)
    with c4:
        if len(selected_vis_features) > 2:
            feat = selected_vis_features[2]
            fig_f3 = px.histogram(df_raw, x=feat, color=target_col, marginal="box",
                                  title=f"Phân phối đặc trưng {feat} theo biến mục tiêu", height=350)
            tf.plotly_chart(fig_f3, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ------------------------------------------
with tab3:
    tf.subheader("Đánh giá độ chính xác thuật toán")
    
    if tf.session_state['model'] is None:
        tf.info("💡 Vui lòng bấm nút **🚀 Huấn luyện Mô hình** ở thanh Sidebar để xem kết quả đánh giá.")
    else:
        metrics = tf.session_state['train_metrics']
        
        # Hiển thị các chỉ số cốt lõi qua st.metric
        met_c1, met_c2, met_c3, met_c4, met_c5 = tf.columns(5)
        met_c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        met_c2.metric("Precision", f"{metrics['Precision']:.4f}")
        met_c3.metric("Recall", f"{metrics['Recall']:.4f}")
        met_c4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        met_c5.metric("AUC Score", f"{metrics['AUC']:.4f}")
        
        tf.divider()
        
        graph_c1, graph_c2 = tf.columns(2)
        
        with graph_c1:
            tf.write("##### 🟦 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = tf.session_state['confusion_matrix']
            # Trực quan hóa ma trận nhầm lẫn bằng heatmap của plotly
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                x=['Bình thường (0)', 'Rủi ro (1)'],
                y=['Bình thường (0)', 'Rủi ro (1)'],
                color_continuous_scale="Blues",
                height=400
            )
            tf.plotly_chart(fig_cm, use_container_width=True)
            
        with graph_c2:
            tf.write("##### 📈 Đường cong ROC (Receiver Operating Characteristic)")
            if tf.session_state['roc_data'] is not None:
                fpr, tpr = tf.session_state['roc_data']
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"RandomForest (AUC={metrics['AUC']:.3f})", line=dict(color='darkorange', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Đường ngẫu nhiên', line=dict(color='navy', dash='dash')))
                fig_roc.update_layout(xaxis_title='Tỷ lệ dương tính giả (FPR)', yaxis_title='Tỷ lệ dương tính thật (TPR)', height=400, margin=dict(l=20, r=20, t=30, b=20))
                tf.plotly_chart(fig_roc, use_container_width=True)
            else:
                tf.warning("Không có dữ liệu xác suất để vẽ đường cong ROC.")

# ------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ------------------------------------------
with tab4:
    tf.subheader("Dự báo dữ liệu giao dịch mới")
    
    if tf.session_state['model'] is None:
        tf.info("💡 Vui lòng bấm nút **🚀 Huấn luyện Mô hình** ở thanh Sidebar trước khi thực hiện dự báo.")
    else:
        model = tf.session_state['model']
        
        predict_mode = tf.radio(
            "Chọn phương thức nhập dữ liệu:",
            options=["Nhập thủ công từng giao dịch", "Tải tệp danh sách cần chấm điểm (X_new)"],
            horizontal=True
        )
        
        # Tính toán giá trị mặc định dựa trên dữ liệu huấn luyện ban đầu
        X_src = df_raw[expected_features]
        
        if predict_mode == "Nhập thủ công từng giao dịch":
            tf.write("##### 📝 Điền thông số các biến đặc trưng:")
            
            # Sử dụng st.form để gom cụm sự kiện tránh rerun liên tục
            with tf.form("single_prediction_form"):
                form_cols = tf.columns(2)
                input_data = {}
                
                for idx, feat in enumerate(expected_features):
                    # Phân bổ đều các cột nhập liệu vào lưới 2 cột
                    target_col_form = form_cols[0] if idx % 2 == 0 else form_cols[1]
                    
                    with target_col_form:
                        min_val = float(X_src[feat].min())
                        max_val = float(X_src[feat].max())
                        median_val = float(X_src[feat].median())
                        
                        input_data[feat] = tf.number_input(
                            f"Nhập giá trị cho {feat}",
                            min_value=min_val - abs(min_val)*0.5,
                            max_value=max_val + abs(max_val)*0.5,
                            value=median_val,
                            format="%.6f",
                            help=f"Khoảng giá trị trong tập mẫu: [{min_val:.4f} -> {max_val:.4f}]"
                        )
                
                submit_predict = tf.form_submit_button("🔍 Tiến hành phân tích rủi ro", type="primary")
            
            if submit_predict:
                # Chuyển đổi dữ liệu nhập vào thành DataFrame đúng định dạng cột
                input_df = pd.DataFrame([input_data])
                
                # Thực hiện dự đoán
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                tf.divider()
                tf.write("##### 🛑 Kết quả phân tích:")
                res_col1, res_col2 = tf.columns(2)
                
                if prediction == 1:
                    res_col1.error("🚨 CẢNH BÁO: Giao dịch có dấu hiệu rủi ro / gian lận cao!")
                    res_col2.metric("Xác suất rủi ro", f"{probabilities[1]*100:.2f}%")
                else:
                    res_col1.success("🟢 AN TOÀN: Giao dịch được đánh giá bình thường.")
                    res_col2.metric("Xác suất an toàn", f"{probabilities[0]*100:.2f}%")
                    
        else:
            tf.write("##### 📁 Tải file danh sách tổng hợp")
            tf.caption("Yêu cầu file định dạng Excel hoặc CSV chứa đầy đủ cấu trúc 14 cột đặc trưng từ X_1 đến X_14.")
            
            scoring_file = tf.file_uploader(
                "Chọn tệp dữ liệu cần chấm điểm hàng loạt", 
                type=["csv", "xlsx"],
                key="scoring_uploader"
            )
            
            if scoring_file is not None:
                scoring_bytes = scoring_file.getvalue()
                df_scoring_raw = load_data(scoring_bytes, scoring_file.name)
                
                if df_scoring_raw is not None:
                    # Kiểm tra sự tương thích của hệ thống cột dữ liệu mới
                    missing_score_cols = [col for col in expected_features if col not in df_scoring_raw.columns]
                    
                    if missing_score_cols:
                        tf.error(f"❌ Không thể chấm điểm. File tải lên thiếu các trường thông tin sau: {missing_score_cols}")
                    else:
                        X_scoring = df_scoring_raw[expected_features]
                        
                        # Dự báo hàng loạt
                        predictions = model.predict(X_scoring)
                        probabilities = model.predict_proba(X_scoring)[:, 1]
                        
                        # Đóng gói kết quả đầu ra
                        df_results = df_scoring_raw.copy()
                        df_results['Du_Bao_Default'] = predictions
                        df_results['Xac_Suat_Rui_Ro'] = probabilities
                        
                        tf.success(f"⚡ Đã chấm điểm xong cho tất cả {len(df_results)} bản ghi trong danh sách!")
                        
                        # Phân tích thống kê kết quả dự báo mới
                        total_fraud = int(predictions.sum())
                        tf.metric("Số lượng giao dịch rủi ro phát hiện", total_fraud, f"Tỷ lệ {total_fraud/len(predictions)*100:.2f}%")
                        
                        # Cho phép tải kết quả về máy
                        tf.write("##### 📥 Bảng kết quả dự báo chi tiết (Tải về qua nút bên dưới)")
                        tf.dataframe(df_results, use_container_width=True)
                        
                        # Tạo buffer để export csv dạng utf-8-sig chống lỗi font Excel
                        csv_buffer = df_results.to_csv(index=False).encode('utf-8-sig')
                        tf.download_button(
                            label="📥 Tải xuống tệp kết quả (.CSV)",
                            data=csv_buffer,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv"
                        )
