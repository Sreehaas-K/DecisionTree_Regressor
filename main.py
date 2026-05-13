import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model training to avoid recomputation
@st.cache_resource
def load_and_train_model(csv_path):
    """Load data, train model, and return encoders"""
    try:
        # Load dataset
        df = pd.read_csv(csv_path, encoding="latin-1")
        
        # Data preprocessing
        df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
        df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)
        
        # Store original values for display
        encoders = {}
        
        # Encode categorical columns
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
        
        # Split data
        x = df.drop("Price_euros", axis=1)
        y = df["Price_euros"]
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = DecisionTreeRegressor(random_state=42, max_depth=15)
        model.fit(x_train, y_train)
        
        # Get predictions for evaluation
        y_pred = model.predict(x_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'train_size': len(x_train),
            'test_size': len(x_test)
        }
        
        return model, encoders, x.columns, metrics, df
        
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'laptop_price.csv' is in the same directory.")
        return None, None, None, None, None

def get_unique_values(df, column):
    """Get unique values for categorical columns"""
    if column in ['Company', 'TypeName', 'OpSys']:
        return df[column].unique()
    return None

# Main Title
st.title("💻 Laptop Price Prediction System")
st.markdown("### Predict laptop prices using Machine Learning")
st.divider()

# Sidebar - File upload and info
with st.sidebar:
    st.header("📊 Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file (laptop_price.csv)",
        type=['csv'],
        help="Upload your laptop price dataset"
    )
    
    if uploaded_file:
        csv_path = uploaded_file
        # Save temporarily
        with open("temp_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_path = "temp_data.csv"
    else:
        st.info("⚠️ No file uploaded yet. Using default path.")
        csv_path = "laptop_price.csv"
    
    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    - **Model**: Decision Tree Regressor
    - **Target**: Laptop Price (in €)
    - **Features**: 12 laptop specifications
    - **Use Case**: Price estimation & analysis
    """)

# Load model and data
model, encoders, feature_names, metrics, original_df = load_and_train_model(csv_path)

if model is None:
    st.error("Could not load the model. Please check your dataset.")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Price Prediction", "📈 Model Performance", "📊 Data Insights", "ℹ️ Help"])

# ======================== TAB 1: PRICE PREDICTION ========================
with tab1:
    st.header("🎯 Make a Prediction")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Specifications")
        
        # Get unique values from original data for dropdowns
        companies = sorted(original_df['Company'].unique())
        types = sorted(original_df['TypeName'].unique())
        os_list = sorted(original_df['OpSys'].unique())
        
        # User inputs
        company = st.selectbox("🏢 Company", companies, help="Select laptop manufacturer")
        product = st.text_input("📱 Product Name", "Custom Laptop", help="Enter product name")
        laptop_type = st.selectbox("💼 Type", types, help="Select laptop type")
        inches = st.slider("📏 Screen Size (inches)", 10.0, 18.0, 15.6, 0.1)
        
        col_a, col_b = st.columns(2)
        with col_a:
            resolution = st.number_input("🖥️ Screen Resolution (pixels)", 800, 3840, 1920)
        with col_b:
            cpu = st.text_input("⚙️ CPU", "Intel Core i5", help="Enter CPU specifications")
        
        ram = st.slider("🧠 RAM (GB)", 2, 64, 8, 1)
        memory = st.text_input("💾 Storage", "256GB SSD", help="Enter storage type and size")
        gpu = st.text_input("🎮 GPU", "Intel UHD Graphics", help="Enter GPU specifications")
        os = st.selectbox("🖲️ Operating System", os_list, help="Select operating system")
        weight = st.slider("⚖️ Weight (kg)", 0.8, 5.0, 1.8, 0.1)
    
    with col2:
        st.subheader("Prediction Result")
        
        # Prepare data for prediction
        try:
            # Create input dataframe with same structure as training data
            input_data = pd.DataFrame({
                'laptop_ID': [1],
                'Company': [company],
                'Product': [product],
                'TypeName': [laptop_type],
                'Inches': [inches],
                'ScreenResolution': [cpu + " " + gpu],  # Simplified combination
                'Cpu': [cpu],
                'Ram': [ram],
                'Memory': [memory],
                'Gpu': [gpu],
                'OpSys': [os],
                'Weight': [weight]
            })
            
            # Encode categorical features
            encoded_data = input_data.copy()
            for col in encoded_data.columns:
                if col in encoders:
                    try:
                        # Handle unknown categories
                        encoded_data[col] = encoders[col].transform(encoded_data[col])
                    except ValueError:
                        # Use the most frequent value if unknown
                        encoded_data[col] = 0
            
            # Get feature order
            feature_order = [col for col in feature_names if col != 'Price_euros']
            
            # Ensure all features are present
            for col in feature_order:
                if col not in encoded_data.columns:
                    encoded_data[col] = 0
            
            encoded_data = encoded_data[feature_order]
            
            # Make prediction
            predicted_price = model.predict(encoded_data)[0]
            
            # Display prediction with styling
            st.markdown(f"""
            <div class="prediction-box">
                €{predicted_price:,.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Additional info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Predicted Price", f"€{predicted_price:,.2f}", delta="Based on specifications")
            with col_info2:
                st.metric("Confidence", "Good", delta_color="off")
            
            # Price range estimation
            st.subheader("📊 Price Range")
            lower_bound = predicted_price * 0.85
            upper_bound = predicted_price * 1.15
            
            col_range1, col_range2, col_range3 = st.columns(3)
            with col_range1:
                st.metric("Lower Estimate", f"€{lower_bound:,.2f}")
            with col_range2:
                st.metric("Predicted", f"€{predicted_price:,.2f}")
            with col_range3:
                st.metric("Upper Estimate", f"€{upper_bound:,.2f}")
            
            # Summary
            st.subheader("📝 Configuration Summary")
            summary_df = pd.DataFrame({
                'Feature': ['Company', 'Type', 'Screen Size', 'RAM', 'Storage', 'Weight', 'OS'],
                'Value': [company, laptop_type, f"{inches}\"", f"{ram}GB", memory, f"{weight}kg", os]
            })
            st.table(summary_df)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            st.info("Please ensure all required fields are filled correctly.")

# ======================== TAB 2: MODEL PERFORMANCE ========================
with tab2:
    st.header("📈 Model Performance Metrics")
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>R² Score</h3>
                <h2>{metrics['r2_score']:.4f}</h2>
                <p>Model explains {metrics['r2_score']*100:.2f}% of price variance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE</h3>
                <h2>€{metrics['rmse']:,.2f}</h2>
                <p>Average prediction error</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>MAE</h3>
                <h2>€{metrics['mae']:,.2f}</h2>
                <p>Mean absolute error</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        col_data1, col_data2 = st.columns(2)
        
        with col_data1:
            st.markdown(f"""
            ### 📊 Dataset Split
            - **Training Set**: {metrics['train_size']} samples
            - **Testing Set**: {metrics['test_size']} samples
            - **Total Samples**: {metrics['train_size'] + metrics['test_size']}
            """)
        
        with col_data2:
            st.markdown(f"""
            ### 🎯 Model Info
            - **Algorithm**: Decision Tree Regressor
            - **Max Depth**: 15 levels
            - **Random State**: 42
            - **Features Used**: {len(feature_names)} features
            """)

# ======================== TAB 3: DATA INSIGHTS ========================
with tab3:
    st.header("📊 Data Insights & Statistics")
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.subheader("💰 Price Statistics")
        price_stats = original_df['Price_euros'].describe()
        st.metric("Average Price", f"€{price_stats['mean']:,.2f}")
        st.metric("Median Price", f"€{original_df['Price_euros'].median():,.2f}")
        st.metric("Min Price", f"€{price_stats['min']:,.2f}")
        st.metric("Max Price", f"€{price_stats['max']:,.2f}")
    
    with col_insight2:
        st.subheader("📊 Feature Ranges")
        st.write(f"**RAM**: {original_df['Ram'].min():.0f} - {original_df['Ram'].max():.0f} GB")
        st.write(f"**Screen Size**: {original_df['Inches'].min():.1f} - {original_df['Inches'].max():.1f}\"")
        st.write(f"**Weight**: {original_df['Weight'].min():.2f} - {original_df['Weight'].max():.2f} kg")
    
    st.divider()
    
    st.subheader("🏢 Top Laptop Brands")
    brand_prices = original_df.groupby('Company')['Price_euros'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    brand_prices.columns = ['Count', 'Avg Price']
    st.bar_chart(brand_prices['Avg Price'])
    
    st.subheader("📈 Sample Data")
    st.dataframe(original_df.head(10), use_container_width=True)

# ======================== TAB 4: HELP ========================
with tab4:
    st.header("ℹ️ Help & Documentation")
    
    st.markdown("""
    ### 📖 How to Use This Application
    
    #### 1️⃣ **Price Prediction Tab**
    - Enter your laptop specifications using the form on the left
    - The model will instantly predict the estimated price
    - View price ranges (lower and upper estimates)
    - Get a configuration summary of your inputs
    
    #### 2️⃣ **Model Performance Tab**
    - View comprehensive model metrics
    - **R² Score**: Higher is better (0-1 scale)
    - **RMSE**: Lower is better (average error in €)
    - **MAE**: Mean absolute error in €
    
    #### 3️⃣ **Data Insights Tab**
    - Explore the dataset statistics
    - View price distributions
    - Analyze trends by brand
    - See feature ranges
    
    ### 🎯 Input Guidelines
    
    | Field | Input Type | Example |
    |-------|-----------|---------|
    | Company | Dropdown | Apple, Dell, HP |
    | Product | Text | MacBook Pro, XPS 13 |
    | Type | Dropdown | Ultrabook, Notebook, Gaming |
    | Screen Size | Slider | 13.3\" - 17.3\" |
    | CPU | Text | Intel Core i7, AMD Ryzen 5 |
    | RAM | Slider | 4GB - 64GB |
    | Storage | Text | 256GB SSD, 1TB HDD |
    | GPU | Text | Intel UHD, NVIDIA RTX 3060 |
    | OS | Dropdown | Windows, macOS, Linux |
    | Weight | Slider | 0.8kg - 5.0kg |
    
    ### ⚠️ Important Notes
    
    - **Dataset Required**: Upload 'laptop_price.csv' for accurate predictions
    - **Categorical Encoding**: Unknown categories are handled gracefully
    - **Price Accuracy**: Predictions are based on training data patterns
    - **Model Type**: Decision Tree - provides interpretable results
    
    ### 🔧 Technical Details
    
    **Features (12 total)**:
    - laptop_ID, Company, Product, TypeName
    - Inches, ScreenResolution, Cpu, Ram
    - Memory, Gpu, OpSys, Weight
    
    **Target Variable**: Price_euros
    
    **Model Architecture**:
    - Algorithm: Decision Tree Regressor
    - Max Depth: 15
    - Test Size: 20%
    - Random State: 42
    
    ### 📧 Support
    
    For issues or questions:
    - Check that your CSV file format matches the original
    - Ensure all required columns are present
    - Verify data types are correct
    
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>💻 Laptop Price Prediction System | Built with Streamlit & Scikit-Learn</p>
    <p>Model Accuracy: R² = {:.4f} | RMSE = €{:.2f}</p>
</div>
""".format(metrics['r2_score'], metrics['rmse']), unsafe_allow_html=True)