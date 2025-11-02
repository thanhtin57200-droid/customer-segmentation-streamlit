import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # noqa

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Phân cụm khách hàng bằng K-Means")

# --- sidebar ---
st.sidebar.header("⚙️ Thiết lập")
uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type=["csv"])
k_input = st.sidebar.slider("Số cụm (K)", min_value=2, max_value=10, value=4, step=1)

if uploaded_file is None:
    st.info("⬅️ Hãy upload file `Mall_Customers.csv` (bạn đã có sẵn).")
    df = pd.read_csv("Mall_Customers.csv")
else:
    df = pd.read_csv(uploaded_file)

st.subheader("1. Dữ liệu gốc")
st.dataframe(df.head())

# chọn cột
cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
data = df[cols].dropna()

# chuẩn hóa
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# train kmeans
kmeans = KMeans(n_clusters=k_input, init='k-means++', random_state=42)
kmeans.fit(data_scaled)

df['Cluster'] = kmeans.labels_

st.subheader("2. Kết quả phân cụm")
st.write("Số khách hàng trong từng cụm:")
st.write(df['Cluster'].value_counts())

# bảng trung bình
cluster_profile = df.groupby('Cluster')[cols].mean().round(2)
st.write("Đặc trưng trung bình từng cụm:")
st.dataframe(cluster_profile)

# --- vẽ 2D ---
st.subheader("3. Biểu đồ 2D: Thu nhập vs. Điểm chi tiêu")
fig, ax = plt.subplots()
scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
)
ax.set_xlabel("Thu nhập (k$)")
ax.set_ylabel("Điểm chi tiêu")
ax.set_title("Phân cụm khách hàng (2D)")
st.pyplot(fig)

# --- vẽ 3D ---
st.subheader("4. Biểu đồ 3D: Tuổi – Thu nhập – Chi tiêu")
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
p = ax3d.scatter(
    df['Age'],
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)
ax3d.set_xlabel("Tuổi")
ax3d.set_ylabel("Thu nhập (k$)")
ax3d.set_zlabel("Chi tiêu")
st.pyplot(fig3d)

st.success("✅ Phân cụm xong rồi. Bạn có thể thay file / đổi K ở sidebar để xem khác nhau.")
