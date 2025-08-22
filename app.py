import streamlit as st
import numpy as np
import pandas as pd
import io

# -----------------------------
# 1. TinyDL Models Definitions
# -----------------------------

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# 1) TinyNN (1 hidden layer)
class TinyNN:
    def __init__(self, input_size, hidden_size=5, output_size=1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        h = relu(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out

# 2) MicroMLP (2 hidden layers)
class MicroMLP:
    def __init__(self, input_size, h1=5, h2=3, output_size=1):
        self.W1 = np.random.randn(input_size, h1)*0.1
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2)*0.1
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, output_size)*0.1
        self.b3 = np.zeros(output_size)

    def forward(self, x):
        h1 = relu(np.dot(x, self.W1) + self.b1)
        h2 = relu(np.dot(h1, self.W2) + self.b2)
        out = np.dot(h2, self.W3) + self.b3
        return out

# 3) MiniCNN (1D convolution)
class MiniCNN:
    def __init__(self, input_size, filter_size=3, num_filters=2, output_size=1):
        self.filters = np.random.randn(num_filters, filter_size)*0.1
        self.W_out = np.random.randn(num_filters*(input_size-filter_size+1), output_size)*0.1
        self.b_out = np.zeros(output_size)

    def conv1d(self, x, f):
        return np.array([np.sum(x[i:i+len(f)]*f) for i in range(len(x)-len(f)+1)])

    def forward(self, x):
        conv_outputs = [relu(self.conv1d(x, f)) for f in self.filters]
        conv_concat = np.concatenate(conv_outputs)
        out = np.dot(conv_concat, self.W_out) + self.b_out
        return out

# 4) Perceptron (single layer)
class Perceptron:
    def __init__(self, input_size, output_size=1):
        self.W = np.random.randn(input_size, output_size)*0.1
        self.b = np.zeros(output_size)

    def forward(self, x):
        out = sigmoid(np.dot(x, self.W) + self.b)
        return out

# 5) MiniRNN (vanilla, small)
class MiniRNN:
    def __init__(self, input_size, hidden_size=5, output_size=1):
        self.Wx = np.random.randn(input_size, hidden_size)*0.1
        self.Wh = np.random.randn(hidden_size, hidden_size)*0.1
        self.bh = np.zeros(hidden_size)
        self.Wy = np.random.randn(hidden_size, output_size)*0.1
        self.by = np.zeros(output_size)

    def forward(self, x_seq):
        h = np.zeros(self.Wh.shape[0])
        for x in x_seq:
            h = np.tanh(np.dot(x, self.Wx) + np.dot(h, self.Wh) + self.bh)
        out = np.dot(h, self.Wy) + self.by
        return out

# 6) MiniAutoencoder
class MiniAutoencoder:
    def __init__(self, input_size, encoding_size=3):
        self.W_enc = np.random.randn(input_size, encoding_size)*0.1
        self.b_enc = np.zeros(encoding_size)
        self.W_dec = np.random.randn(encoding_size, input_size)*0.1
        self.b_dec = np.zeros(input_size)

    def forward(self, x):
        encoded = relu(np.dot(x, self.W_enc) + self.b_enc)
        decoded = np.dot(encoded, self.W_dec) + self.b_dec
        return decoded

# -----------------------------
# 2. Streamlit Interface
# -----------------------------
st.title("Tiny Deep Learning Models Comparison")

# Use a selectbox to choose between demo CSVs and uploading a new one
csv_options = ["energy_hourly.csv", "sales_monthly.csv", "stock_daily.csv", "Upload your own CSV"]
selected_option = st.selectbox("Select a demo CSV or upload your own:", csv_options)

df = None
if selected_option == "Upload your own CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    # Read the demo CSV files
    # Note: In a real-world scenario, you'd have to provide these files in the
    # same directory as the app.py file
    if selected_option == "energy_hourly.csv":
        data = """
datetime,consumption
2021-01-01 00:00:00,102
2021-01-01 01:00:00,98
2021-01-01 02:00:00,95
"""
    elif selected_option == "sales_monthly.csv":
        data = """
month,sales
1,150
2,165
3,180
"""
    elif selected_option == "stock_daily.csv":
        data = """
date,price,volume
2023-01-01,150,1000
2023-01-02,152,1100
2023-01-03,151,1050
"""
    df = pd.read_csv(io.StringIO(data))


if df is not None:
    st.write(f"Using {selected_option} data:")
    st.dataframe(df.head())

    # Use numeric columns only
    numeric_data = df.select_dtypes(include=np.number).values
    if numeric_data.shape[0] == 0:
        st.error("No numeric data found in CSV!")
    else:
        st.write(f"Using numeric data with shape: {numeric_data.shape}")

        # -----------------------------
        # 3. Initialize Models
        # -----------------------------
        input_size = numeric_data.shape[1]
        tinynn = TinyNN(input_size)
        micromlp = MicroMLP(input_size)
        minicnn = MiniCNN(input_size)
        perceptron = Perceptron(input_size)
        minirnn = MiniRNN(input_size)
        minaec = MiniAutoencoder(input_size)

        # -----------------------------
        # 4. Run Models
        # -----------------------------
        results = {}
        for i, x in enumerate(numeric_data):
            results.setdefault("TinyNN", []).append(tinynn.forward(x))
            results.setdefault("MicroMLP", []).append(micromlp.forward(x))
            results.setdefault("MiniCNN", []).append(minicnn.forward(x))
            results.setdefault("Perceptron", []).append(perceptron.forward(x))
            results.setdefault("MiniRNN", []).append(minirnn.forward(x))
            results.setdefault("MiniAutoencoder", []).append(minaec.forward(x))

        # Convert to arrays
        for k in results:
            results[k] = np.array(results[k])

        st.subheader("Predictions of Each Model (first 5 rows)")
        for k in results:
            st.write(f"**{k}**")
            st.write(results[k][:5])

        # -----------------------------
        # 5. Comparison Table
        # -----------------------------
        st.subheader("Comparison Table (Mean of Outputs)")
        comparison = {k: np.mean(results[k]) for k in results}
        comparison_df = pd.DataFrame.from_dict(comparison, orient="index", columns=["Mean_Output"])
        st.dataframe(comparison_df)
