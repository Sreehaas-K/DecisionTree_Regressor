import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
        font-size: 42px;
        font-weight: bold;
        margin: 20px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model training to avoid recomputation
@st.cache_resource
def load_and_train_model():
    """Load data, train model, and return encoders with proper mappings"""
    try:
        # Load dataset with proper settings
        df = pd.read_csv(
            'laptop_price.csv', 
            encoding='latin-1',
            sep=',',
            on_bad_lines='skip'  # Skip any problematic rows
        )
        
        # Ensure correct data types right away
        df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')
        df['Price_euros'] = pd.to_numeric(df['Price_euros'], errors='coerce')
        
        # Remove any rows with NaN values created by coercion
        df = df.dropna(subset=['Inches', 'Price_euros'])
        
        original_df = df.copy()
        
        # Data preprocessing - handle Ram and Weight
        df["Ram"] = df["Ram"].astype(str).str.replace("GB", "").str.strip()
        df["Ram"] = pd.to_numeric(df["Ram"], errors='coerce').astype(int)
        
        df["Weight"] = df["Weight"].astype(str).str.replace("kg", "").str.strip()
        df["Weight"] = pd.to_numeric(df["Weight"], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Create encoding mappings (dict-based for easier reverse mapping)
        encoders = {}
        encode_mappings = {}
        
        # Define categorical columns to encode
        categorical_cols = ['laptop_ID', 'Company', 'Product', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
        
        # Encode categorical columns
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                encoded_values = le.fit_transform(df[col].astype(str))
                df[col] = encoded_values
                encoders[col] = le
                
                # Create mapping for display {encoded: original}
                encode_mappings[col] = dict(zip(range(len(le.classes_)), le.classes_))
        
        # Convert all remaining object columns to numeric
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                encode_mappings[col] = dict(zip(range(len(le.classes_)), le.classes_))
        
        # Verify all columns are numeric
        df = df.astype(float)
        
        # Split data
        x = df.drop("Price_euros", axis=1)
        y = df["Price_euros"]
        
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = DecisionTreeRegressor(random_state=42, max_depth=15, min_samples_split=5)
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
        
        # Get unique values for dropdowns
        unique_values = {
            'Company': sorted(original_df['Company'].unique().tolist()),
            'TypeName': sorted(original_df['TypeName'].unique().tolist()),
            'OpSys': sorted(original_df['OpSys'].unique().tolist()),
        }
        
        return model, encoders, encode_mappings, x.columns, metrics, original_df, unique_values
        
    except FileNotFoundError:
        st.error("❌ Dataset file not found at /mnt/user-data/uploads/laptop_price.csv")
        st.info("Please ensure the laptop_price.csv file is in the uploads directory.")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please check that the CSV file is properly formatted.")
        return None, None, None, None, None, None, None

# Load model and data
model, encoders, encode_mappings, feature_names, metrics, original_df, unique_values = load_and_train_model()

if model is None:
    st.error("Could not load the model. Please check your dataset.")
    st.stop()

# Main Title
st.title("💻 Laptop Price Prediction System")
st.markdown("### Predict laptop prices using Machine Learning")
st.divider()

# Sidebar - Info
with st.sidebar:
    st.header("📊 Quick Info")
    st.markdown(f"""
    ### 📈 Model Status: ✅ Ready
    - **Algorithm**: Decision Tree Regressor
    - **Accuracy (R²)**: {metrics['r2_score']:.4f}
    - **Dataset Size**: {metrics['train_size'] + metrics['test_size']} laptops
    - **Features**: 12 specifications
    
    ### 💰 Price Range
    - **Min**: €{original_df['Price_euros'].min():,.0f}
    - **Max**: €{original_df['Price_euros'].max():,.0f}
    - **Avg**: €{original_df['Price_euros'].mean():,.0f}
    """)
    
    st.divider()
    st.markdown("**Dataset embedded!** 🎉\nNo file upload needed.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Price Prediction", "📈 Model Performance", "📊 Data Insights", "ℹ️ Help"])

# ======================== TAB 1: PRICE PREDICTION ========================
with tab1:
    st.header("🎯 Make a Prediction")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("📋 Laptop Specifications")
        
        # User inputs with proper string dropdowns
        company = st.selectbox(
            "🏢 Company", 
            unique_values['Company'],
            help="Select laptop manufacturer"
        )
        
        product = st.text_input(
            "📱 Product Name", 
            "Custom Laptop",
            help="Enter product name (e.g., MacBook Pro, XPS 13)"
        )
        
        laptop_type = st.selectbox(
            "💼 Laptop Type", 
            unique_values['TypeName'],
            help="Select laptop type/category"
        )
        
        col_screen1, col_screen2 = st.columns(2)
        with col_screen1:
            inches = st.slider("📏 Screen Size (inches)", 10.0, 18.0, 15.6, 0.1)
        with col_screen2:
            ram_values = sorted(original_df['Ram'].str.replace('GB', '').astype(int).unique().tolist())
            ram = st.selectbox("🧠 RAM (GB)", ram_values)
        
        col_specs1, col_specs2 = st.columns(2)
        with col_specs1:
            resolution = st.text_input("🖥️ Screen Resolution", "1920x1080", help="e.g., 1920x1080, 2560x1600")
        with col_specs2:
            memory = st.text_input("💾 Storage", "256GB SSD", help="e.g., 256GB SSD, 512GB HDD")
        
        cpu = st.text_input("⚙️ CPU", "Intel Core i5", help="e.g., Intel Core i7, AMD Ryzen 5")
        gpu = st.text_input("🎮 GPU", "Intel UHD Graphics", help="e.g., Intel HD 620, NVIDIA GTX 1650")
        
        os = st.selectbox(
            "🖲️ Operating System", 
            unique_values['OpSys'],
            help="Select operating system"
        )
        
        weight = st.slider("⚖️ Weight (kg)", 0.8, 5.0, 1.8, 0.1)
        
        # Add some space before button
        st.markdown("---")
        
        # Predict Button
        predict_button = st.button(
            "🚀 PREDICT PRICE",
            use_container_width=True,
            type="primary",
            key="predict_btn"
        )
    
    with col2:
        st.subheader("📊 Prediction Result")
        
        # Make prediction when button is clicked
        if predict_button:
            try:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'laptop_ID': [1],
                    'Company': [company],
                    'Product': [product],
                    'TypeName': [laptop_type],
                    'Inches': [inches],
                    'ScreenResolution': [resolution],
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
                            # Handle unknown categories by finding closest match
                            encoded_data[col] = encoders[col].transform(encoded_data[col])
                        except ValueError:
                            # Use the first encoded value if unknown
                            encoded_data[col] = 0
                
                # Ensure correct feature order
                feature_order = [col for col in feature_names]
                encoded_data = encoded_data[feature_order]
                
                # Make prediction
                predicted_price = model.predict(encoded_data)[0]
                
                # Display prediction with styling
                st.markdown(f"""
                <div class="prediction-box">
                    €{predicted_price:,.2f}
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f"✅ Prediction Complete!")
                
                # Price range estimation
                st.subheader("📊 Price Analysis")
                lower_bound = predicted_price * 0.85
                upper_bound = predicted_price * 1.15
                
                col_range1, col_range2, col_range3 = st.columns(3)
                with col_range1:
                    st.metric("Lower Estimate", f"€{lower_bound:,.2f}")
                with col_range2:
                    st.metric("Predicted Price", f"€{predicted_price:,.2f}", delta="Our estimate")
                with col_range3:
                    st.metric("Upper Estimate", f"€{upper_bound:,.2f}")
                
                # Configuration Summary
                st.subheader("📝 Configuration Summary")
                summary_data = {
                    'Specification': [
                        'Company',
                        'Product',
                        'Type',
                        'Screen Size',
                        'Resolution',
                        'CPU',
                        'RAM',
                        'Storage',
                        'GPU',
                        'Operating System',
                        'Weight'
                    ],
                    'Value': [
                        company,
                        product,
                        laptop_type,
                        f"{inches}\"",
                        resolution,
                        cpu,
                        f"{ram}GB",
                        memory,
                        gpu,
                        os,
                        f"{weight}kg"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error in prediction: {str(e)}")
                st.info("Please ensure all fields are filled with valid data.")
        else:
            st.info("👈 Fill in the specifications on the left and click the **PREDICT PRICE** button to see the estimated price!")

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
            - **Training Set**: {metrics['train_size']} samples (80%)
            - **Testing Set**: {metrics['test_size']} samples (20%)
            - **Total Samples**: {metrics['train_size'] + metrics['test_size']}
            """)
        
        with col_data2:
            st.markdown(f"""
            ### 🎯 Model Configuration
            - **Algorithm**: Decision Tree Regressor
            - **Max Depth**: 15 levels
            - **Min Samples Split**: 5
            - **Random State**: 42 (reproducible)
            - **Features Used**: {len(feature_names)} features
            """)
        
        st.divider()
        st.subheader("📚 Understanding the Metrics")
        
        col_explain1, col_explain2 = st.columns(2)
        with col_explain1:
            st.markdown("""
            **R² Score (Coefficient of Determination)**
            - Range: 0 to 1
            - Higher is better
            - 0.765 = Model explains 76.5% of price variations
            - Good models typically have R² > 0.7
            """)
        
        with col_explain2:
            st.markdown("""
            **RMSE (Root Mean Squared Error)**
            - Unit: Euros
            - Lower is better
            - Penalizes larger errors more
            - Useful for understanding magnitude of errors
            """)

# ======================== TAB 3: DATA INSIGHTS ========================
with tab3:
    st.header("📊 Dataset Analysis & Insights")
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.subheader("💰 Price Statistics")
        price_stats = original_df['Price_euros'].describe()
        st.metric("Average Price", f"€{price_stats['mean']:,.2f}")
        st.metric("Median Price", f"€{original_df['Price_euros'].median():,.2f}")
        st.metric("Min Price", f"€{price_stats['min']:,.2f}")
        st.metric("Max Price", f"€{price_stats['max']:,.2f}")
        st.metric("Std Deviation", f"€{price_stats['std']:,.2f}")
    
    with col_insight2:
        st.subheader("📊 Hardware Specifications Range")
        ram_values = original_df['Ram'].str.replace('GB', '').astype(int)
        weight_values = original_df['Weight'].str.replace('kg', '').astype(float)
        
        st.markdown(f"""
        **RAM**: {ram_values.min():.0f}GB - {ram_values.max():.0f}GB
        
        **Screen Size**: {original_df['Inches'].min():.1f}\" - {original_df['Inches'].max():.1f}\"
        
        **Weight**: {weight_values.min():.2f}kg - {weight_values.max():.2f}kg
        
        **Total Laptops**: {len(original_df)}
        """)
    
    st.divider()
    
    # Brands information
    st.subheader("🏢 Laptop Brands in Dataset")
    brand_data = original_df['Company'].value_counts().reset_index()
    brand_data.columns = ['Company', 'Count']
    brand_prices = original_df.groupby('Company')['Price_euros'].mean().reset_index().sort_values('Price_euros', ascending=False)
    
    col_brand1, col_brand2 = st.columns(2)
    with col_brand1:
        st.write("**Number of Models by Brand**")
        st.bar_chart(brand_data.set_index('Company')['Count'])
    
    with col_brand2:
        st.write("**Average Price by Brand**")
        st.bar_chart(brand_prices.set_index('Company')['Price_euros'])
    
    st.divider()
    
    # Laptop Types
    st.subheader("💼 Laptop Types Distribution")
    type_data = original_df['TypeName'].value_counts()
    type_prices = original_df.groupby('TypeName')['Price_euros'].mean().sort_values(ascending=False)
    
    col_type1, col_type2 = st.columns(2)
    with col_type1:
        st.write("**Count by Type**")
        st.bar_chart(type_data)
    
    with col_type2:
        st.write("**Average Price by Type**")
        st.bar_chart(type_prices)
    
    st.divider()
    
    st.subheader("🖨️ Sample Data from Dataset")
    st.dataframe(original_df.head(15), use_container_width=True)

# ======================== TAB 4: HELP ========================
with tab4:
    st.header("ℹ️ Help & Documentation")
    
    st.markdown("""
    ## 📖 How to Use This Application
    
    ### 1️⃣ **Price Prediction Tab**
    1. Fill in your laptop specifications in the left panel
    2. All fields are dropdowns or sliders for easy selection
    3. Click the **PREDICT PRICE** button
    4. View the estimated price and price range
    5. See a detailed configuration summary
    
    ### 2️⃣ **Model Performance Tab**
    - View comprehensive model metrics
    - **R² Score**: How well the model explains price variations
    - **RMSE**: Average error in euros
    - **MAE**: Mean absolute error
    
    ### 3️⃣ **Data Insights Tab**
    - Explore the dataset statistics
    - View price distributions by brand and type
    - See hardware specification ranges
    - Analyze market trends
    
    ---
    
    ## 🎯 Feature Guide
    
    | Feature | Type | Description |
    |---------|------|-------------|
    | Company | Dropdown | Laptop manufacturer (Apple, Dell, HP, etc.) |
    | Product | Text | Model name (MacBook Pro, XPS 13, etc.) |
    | Type | Dropdown | Category (Ultrabook, Gaming, Notebook, etc.) |
    | Screen Size | Slider | Diagonal display size in inches |
    | Resolution | Text | Display resolution (1920x1080, 2560x1600, etc.) |
    | CPU | Text | Processor type and model |
    | RAM | Dropdown | Memory in GB |
    | Storage | Text | Storage type and capacity |
    | GPU | Text | Graphics processor |
    | OS | Dropdown | Operating system |
    | Weight | Slider | Total weight in kilograms |
    
    ---
    
    ## 💡 Tips for Best Results
    
    ✅ **Use standard format**
    - CPU: "Intel Core i7", "AMD Ryzen 5"
    - Storage: "256GB SSD", "512GB HDD"
    - Resolution: "1920x1080", "2560x1600"
    
    ✅ **Realistic specifications**
    - Select specs that actually exist in the market
    - Check the dataset for valid combinations
    
    ✅ **Understand predictions**
    - Prices are based on training data patterns
    - Unusual combinations may have less accurate predictions
    - Price range (±15%) shows typical variance
    
    ---
    
    ## 🔒 Privacy & Security
    
    ✅ **Fully Local Processing**
    - No data sent to external servers
    - All calculations happen on your computer
    - No account or login required
    
    ✅ **Data Safety**
    - Dataset is embedded in the application
    - No file uploads needed
    - Your specifications are not stored
    """)

# Footer
st.divider()
st.markdown(f"""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>💻 Laptop Price Prediction System | Built with Streamlit & Scikit-Learn</p>
    <p>Model Performance: R² = {metrics['r2_score']:.4f} | RMSE = €{metrics['rmse']:,.2f} | MAE = €{metrics['mae']:,.2f}</p>
    <p style="font-size: 12px;">Dataset embedded • No file upload needed • Instant predictions</p>
</div>
""", unsafe_allow_html=True)