import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Vehicle Emission Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for styling
st.markdown(
    """
<style>
    .main {max-width: 1200px;}
    h1, h2, h3 {color: #2C3E50; font-family: 'Inter', sans-serif;}
    .fact-box {background-color: #3498DB; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;}
    .tip-box {background-color: #26A65B; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;}
    .emission-stat {height: 250px; text-align: center; padding: 20px; border-radius: 10px; margin: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .low-emission {background-color: #2ecc71; color: white;}
    .moderate-emission {background-color: #f1c40f; color: white;}
    .high-emission {background-color: #e74c3c; color: white;}
    .insight-card {background-color: #1F2633; height: 200px; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #3498DB;}
</style>
""",
    unsafe_allow_html=True,
)


# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        with open("model/vehicle_emission_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully!")
        return model_data
    except FileNotFoundError:
        st.sidebar.error("❌ Model file not found. Please run training script first.")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None


# Create model information visualizations
def create_model_info_visualizations(model_data):
    if model_data is None:
        st.warning("Model data not available. Visualizations will be limited.")
        return

    # Feature importance visualization
    if model_data.get("feature_importance"):
        feat_imp = pd.DataFrame(
            model_data["feature_importance"][:10],
            columns=["Feature", "Importance"],
        )
        fig_importance = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Feature Importance",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("---")

    # Model performance comparison
    if model_data.get("all_models_performance"):
        model_perf = pd.DataFrame(
            [
                (name, cv, test)
                for name, (cv, test) in model_data["all_models_performance"].items()
            ],
            columns=["Model", "Cross-Validation Score", "Test Accuracy"],
        )
        fig_models = go.Figure()
        fig_models.add_trace(
            go.Bar(
                x=model_perf["Model"],
                y=model_perf["Cross-Validation Score"],
                name="Cross-Validation Score",
                marker_color="#3498DB",
            )
        )
        fig_models.add_trace(
            go.Bar(
                x=model_perf["Model"],
                y=model_perf["Test Accuracy"],
                name="Test Accuracy",
                marker_color="#2ECC71",
            )
        )
        fig_models.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_models, use_container_width=True)

    # Emission category distribution
    if model_data.get("category_distribution"):
        cat_df = pd.DataFrame(
            list(model_data["category_distribution"].items()),
            columns=["Category", "Count"],
        )
        fig_cat = px.pie(
            cat_df,
            names="Category",
            values="Count",
            title="Distribution of Emission Categories in Training Data",
            color="Category",
            color_discrete_map={
                "Low Emission": "#2ecc71",
                "Moderate Emission": "#f1c40f",
                "High Emission": "#e74c3c",
            },
            hole=0.4,
        )
        fig_cat.update_traces(textinfo="percent+label", textfont=dict(color="white"))
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")

    # Feature correlation heatmap
    if model_data.get("feature_correlations") and model_data.get("selected_features"):
        selected_corrs = [
            (feat, corr)
            for feat, corr in model_data["feature_correlations"]
            if feat in model_data["selected_features"]
        ]
        corr_df = pd.DataFrame(
            selected_corrs, columns=["Feature", "Correlation with Emissions"]
        ).sort_values("Correlation with Emissions", ascending=False)
        fig_corr = px.bar(
            corr_df,
            x="Feature",
            y="Correlation with Emissions",
            title="Feature Correlation with Emissions",
            color="Correlation with Emissions",
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # Visual explanation of how the model works
    st.subheader("How the Model Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="insight-card">
                <h4>1️⃣ Data Collection</h4>
                <p>Vehicle specifications and their measured CO2 emissions are collected and processed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="insight-card">
                <h4>2️⃣ Feature Engineering</h4>
                <p>Raw data is transformed into meaningful features that help predict emissions accurately.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="insight-card">
                <h4>3️⃣ Model Training</h4>
                <p>Machine learning algorithms learn patterns from the data to predict emission categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Create prediction visualizations
def create_prediction_visualizations(prediction_label, proba=None):
    # Prediction confidence
    if proba is not None and len(proba) > 0:
        st.markdown("---")
        st.subheader("Prediction Confidence")
        categories = list(proba.keys())
        values = list(proba.values())
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        fig_confidence = go.Figure(
            [go.Bar(x=categories, y=values, marker_color=colors[: len(categories)])]
        )
        fig_confidence.update_layout(
            title="Prediction Confidence by Category",
            yaxis=dict(title="Probability", range=[0, 1]),
            xaxis_title="Emission Category",
        )
        st.plotly_chart(fig_confidence, use_container_width=True)

    st.markdown("---")

    # Environmental impact visualization
    st.subheader("Environmental Impact")
    impact_mapping = {
        "Low Emission": {
            "Trees": 2,
            "CO2": "Low impact: ~100-150 g/km",
            "Icon": "🌱",
            "Text": "This vehicle has a relatively small carbon footprint.",
        },
        "Moderate Emission": {
            "Trees": 4,
            "CO2": "Moderate impact: ~150-200 g/km",
            "Icon": "🌿",
            "Text": "This vehicle has an average carbon footprint.",
        },
        "High Emission": {
            "Trees": 8,
            "CO2": "High impact: >200 g/km",
            "Icon": "🌲",
            "Text": "This vehicle has a significant carbon footprint.",
        },
    }

    if prediction_label in impact_mapping:
        impact = impact_mapping[prediction_label]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="insight-card" style="height: 250px; width: 375px">
                    <h3>{impact['Icon']} Environmental Impact</h3>
                    <p><b>CO2 Output:</b> {impact['CO2']}</p>
                    <p><b>Trees needed to offset annual emissions:</b> {impact['Trees']} trees per 10,000 km driven</p>
                    <p>{impact['Text']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            tree_fig = go.Figure()
            tree_x, tree_y, tree_texts = [], [], []
            num_trees = impact["Trees"]
            cols = 3 if num_trees > 4 else 2
            rows = max(1, (num_trees + cols - 1) // cols)
            for i in range(num_trees):
                row = i // cols
                col = i % cols
                x_pos = col + 0.1 * (np.random.rand() - 0.5)
                y_pos = rows - row - 1 + 0.1 * (np.random.rand() - 0.5)
                tree_x.append(x_pos)
                tree_y.append(y_pos)
                tree_texts.append("🌲")
            tree_fig.add_trace(
                go.Scatter(
                    x=tree_x,
                    y=tree_y,
                    mode="text",
                    text=tree_texts,
                    textposition="middle center",
                    textfont=dict(size=40),
                    name="Trees Needed",
                )
            )
            tree_fig.update_layout(
                title="Trees Needed to Offset Annual Emissions",
                showlegend=False,
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-0.5, cols - 0.5],
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    range=[-0.5, rows - 0.5],
                ),
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(tree_fig, use_container_width=True)


# New function for dataset information visualizations
# Dataset visualizations
def create_dataset_visualizations(model_data):
    if model_data is None:
        st.error(
            "❌ No model data loaded. Ensure model/vehicle_emission_model.pkl exists and run train_and_save.py if needed."
        )
        return
    if "df" not in model_data:
        st.error(
            "❌ Dataset DataFrame not found in model data. Re-run train_and_save.py to include the dataset."
        )
        return
    if not isinstance(model_data["df"], pd.DataFrame):
        st.error(
            "❌ Invalid dataset format. Expected a pandas DataFrame. Re-run train_and_save.py."
        )
        return

    df = model_data["df"]

    # Dataset Statistics
    st.subheader("Dataset Statistics")
    num_rows, num_cols = df.shape
    unique_makes = len(model_data["makes"])
    unique_vehicle_classes = len(model_data["vehicle_classes"])
    avg_co2 = df["CO2 Emissions (g/km)"].mean()
    min_engine = df["Engine Size (L)"].min()
    max_engine = df["Engine Size (L)"].max()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>Dataset Size</h4>
                <p><b>{num_rows}</b> vehicles<br><b>{num_cols}</b> features</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>Vehicle Diversity</h4>
                <p><b>{unique_makes}</b> unique makes<br><b>{unique_vehicle_classes}</b> vehicle classes</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>Key Metrics</h4>
                <p>Avg. CO2: <b>{avg_co2:.1f} g/km</b><br>Engine Size: <b>{min_engine:.1f}L - {max_engine:.1f}L</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Key Features
    st.subheader("Key Features Used in the Model")
    features = model_data.get("selected_features", [])
    feature_desc = {
        "Engine Size (L)": "Engine displacement in liters",
        "Cylinders": "Number of engine cylinders",
        "Fuel Consumption (L/100Km)": "City fuel consumption per 100 km",
        "Hwy (L/100 km)": "Highway fuel consumption per 100 km",
        "Comb (L/100 km)": "Combined fuel consumption",
        "Fuel Type_Encoded": "Type of fuel (e.g., Gasoline, Diesel)",
        "Vehicle Class_Encoded": "Vehicle category (e.g., Compact, SUV)",
        "Transmission_Type_Encoded": "Transmission type (e.g., Automatic, Manual)",
        "Make_Encoded": "Vehicle manufacturer",
        "Model_Encoded": "Specific vehicle model",
        "Gear_Count": "Number of transmission gears",
        "Power_to_Weight": "Engine size relative to vehicle class",
        "Fuel_Efficiency": "Estimated CO2 per fuel consumption",
        "City_Hwy_Ratio": "Ratio of city to highway fuel consumption",
        "Engine_Size_Per_Cylinder": "Engine size per cylinder",
        "Vehicle_Size": "Categorized vehicle size (Small, Medium, Large)",
        "Is_4WD": "Presence of four/all-wheel drive",
    }
    for feature in features:
        desc = feature_desc.get(feature, "Derived feature")
        st.markdown(f"- **{feature}**: {desc}")

    st.markdown("---")

    # Visualizations
    st.subheader("Dataset Visualizations")

    # CO2 Emissions Distribution
    fig_co2 = px.histogram(
        df,
        x="CO2 Emissions (g/km)",
        nbins=30,
        title="Distribution of CO2 Emissions",
        color_discrete_sequence=["#3498DB"],
        labels={"CO2 Emissions (g/km)": "CO2 Emissions (g/km)"},
    )
    fig_co2.update_layout(
        xaxis_title="CO2 Emissions (g/km)", yaxis_title="Number of Vehicles", bargap=0.1
    )
    st.plotly_chart(fig_co2, use_container_width=True)

    # Vehicle Class Distribution
    vehicle_class_counts = df["Vehicle Class"].value_counts().reset_index()
    vehicle_class_counts.columns = ["Vehicle Class", "Count"]
    fig_vehicle_class = px.pie(
        vehicle_class_counts,
        names="Vehicle Class",
        values="Count",
        title="Distribution of Vehicle Classes",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        hole=0.4,
    )
    fig_vehicle_class.update_traces(
        textinfo="percent+label", textfont=dict(color="white")
    )
    st.plotly_chart(fig_vehicle_class, use_container_width=True)

    # Engine Size vs. CO2 Emissions
    st.subheader("Engine Size vs. CO2 Emissions")
    fig_scatter = px.scatter(
        df,
        x="Engine Size (L)",
        y="CO2 Emissions (g/km)",
        color="Fuel Type",
        title="Engine Size vs. CO2 Emissions by Fuel Type",
        color_discrete_map={
            "X": "#3498DB",
            "Z": "#E74C3C",
            "D": "#2ECC71",
            "E": "#F1C40F",
        },
        labels={
            "Engine Size (L)": "Engine Size (L)",
            "CO2 Emissions (g/km)": "CO2 Emissions (g/km)",
        },
    )
    fig_scatter.update_layout(
        xaxis_title="Engine Size (L)",
        yaxis_title="CO2 Emissions (g/km)",
        legend_title="Fuel Type",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # CO2 Emissions by Fuel Type
    st.subheader("CO2 Emissions by Fuel Type")
    fig_box = px.box(
        df,
        x="Fuel Type",
        y="CO2 Emissions (g/km)",
        title="CO2 Emissions Distribution by Fuel Type",
        color="Fuel Type",
        color_discrete_map={
            "X": "#3498DB",
            "Z": "#E74C3C",
            "D": "#2ECC71",
            "E": "#F1C40F",
        },
        labels={
            "Fuel Type": "Fuel Type",
            "CO2 Emissions (g/km)": "CO2 Emissions (g/km)",
        },
    )
    fig_box.update_layout(
        xaxis_title="Fuel Type", yaxis_title="CO2 Emissions (g/km)", showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)


# Main function
def main():
    # Load the model
    model_data = load_model()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Home", "Dataset Information", "Model Information", "Make Predictions"]
    page = st.sidebar.radio("Go to", pages)

    # Emission facts in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Emission Facts")
    facts = [
        "Transportation accounts for ~29% of greenhouse gas emissions in the US.",
        "Electric vehicles produce zero direct emissions.",
        "Keeping tires properly inflated can reduce emissions by up to 3%.",
        "Idling for more than 10 seconds uses more fuel than restarting your car.",
        "Regular maintenance can reduce your vehicle's emissions by up to 10%.",
    ]
    st.sidebar.info(np.random.choice(facts))

    if page == "Home":
        st.title("🚗 Welcome to Vehicle Emission Classifier")
        st.write("Predict vehicle emission levels based on your specifications!")
        st.markdown("---")

        # Key statistics dashboard
        st.subheader("📊 Vehicle Emissions Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div class="emission-stat low-emission">
                    <h2>Low Emission</h2>
                    <h3 style="margin-top: -25px">< 150 g/km</h3>
                    <p>Electric, Hybrid & Efficient Compact Cars</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="emission-stat moderate-emission">
                    <h2>Moderate Emission</h2>
                    <h3>150-200 g/km</h3>
                    <p>Most Sedans & Small SUVs</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div class="emission-stat high-emission">
                    <h2>High Emission</h2>
                    <h3>> 200 g/km</h3>
                    <p>Large SUVs, Trucks & Sports Cars</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # About Vehicle Emissions
        st.subheader("About Vehicle Emissions")
        st.write(
            """
            Vehicle emissions are a major contributor to air pollution and climate change. 
            Understanding how different vehicle characteristics affect emissions can help you make more environmentally friendly choices.
            
            This tool uses machine learning to predict whether a vehicle will have low, moderate, or high emissions based on specifications 
            like engine size, fuel type, and transmission.
            """
        )

        st.markdown("---")

        # Emission factors infographic
        st.subheader("Top Factors Affecting Vehicle Emissions")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                """
                <div class="insight-card">
                    <h4>Engine Size</h4>
                    <p>Larger engines typically consume more fuel and produce more emissions.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="insight-card">
                    <h4>Fuel Type</h4>
                    <p>Diesel, gasoline, and alternative fuels each have different emission profiles.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div class="insight-card">
                    <h4>Vehicle Weight</h4>
                    <p>Heavier vehicles require more energy to move, increasing fuel consumption.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                """
                <div class="insight-card">
                    <h4>Transmission Type</h4>
                    <p>Automatic vs. manual transmissions affect fuel efficiency and emissions.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "Dataset Information":
        st.title("📊 Dataset Information")
        st.write(
            """
            The Vehicle Emission Classifier is powered by the **Fuel Consumption Ratings 2023** dataset, 
            sourced from Natural Resources Canada (or similar authority). 
            This dataset contains detailed information about vehicles sold in 2023, including their 
            fuel consumption, CO2 emissions, and technical specifications.
            """
        )
        st.markdown("---")

        create_dataset_visualizations(model_data)

    # [Preserving existing pages without modification]
    elif page == "Model Information":
        st.title("Model Information")
        # ... (Original Model Information page code unchanged, as in the provided app.py)
        st.subheader("Performance Summary")
        if model_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Model", model_data.get("best_model_name", "N/A"))
            with col2:
                st.metric("Accuracy", f"{model_data.get('best_accuracy', 0):.2%}")
            with col3:
                st.metric(
                    "Cross-Validation Score", f"{model_data.get('cv_accuracy', 0):.2%}"
                )
        st.markdown("---")
        create_model_info_visualizations(model_data)
        st.markdown("---")
        st.subheader("Understanding Emission Categories")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div class="emission-stat low-emission">
                    <h3>Low Emission</h3>
                    <p>Vehicles that produce minimal CO2 and other harmful gases. These include electric vehicles, hybrids, and highly efficient compact cars.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="emission-stat moderate-emission">
                    <h3>Moderate Emission</h3>
                    <p style="margin-top: -15px">Average-performing vehicles in terms of emissions. Most conventional sedans and smaller SUVs fall into this category.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div class="emission-stat high-emission">
                    <h3>High Emission</h3>
                    <p>Vehicles with significant emissions impact. Typically includes larger SUVs, trucks, sports cars, and older vehicles with less efficient technology.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "Make Predictions":
        st.title("Predict Vehicle Emissions")
        # ... (Original Make Predictions page code unchanged, as in the provided app.py)
        st.write("Enter vehicle details to see its emission category.")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                make = st.selectbox("Make", options=[""] + model_data["makes"])
                model = st.text_input("Model")
                vehicle_class = st.selectbox(
                    "Vehicle Class", options=[""] + model_data["vehicle_classes"]
                )
                engine_size = st.number_input(
                    "Engine Size (L)",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                )
                cylinders = st.number_input(
                    "Cylinders", min_value=1, max_value=16, value=4, step=1
                )
            with col2:
                transmission = st.text_input("Transmission (e.g., A6, M5)")
                fuel_type = st.selectbox(
                    "Fuel Type", options=[""] + model_data["fuel_types"]
                )
                city_consumption = st.number_input(
                    "City Fuel Consumption (L/100Km)",
                    min_value=1.0,
                    max_value=30.0,
                    value=10.0,
                    step=0.1,
                )
                hwy_consumption = st.number_input(
                    "Highway Fuel Consumption (L/100Km)",
                    min_value=1.0,
                    max_value=30.0,
                    value=8.0,
                    step=0.1,
                )
                comb_consumption = st.number_input(
                    "Combined Fuel Consumption (L/100Km)",
                    min_value=1.0,
                    max_value=30.0,
                    value=9.0,
                    step=0.1,
                )
            co2_rating = st.slider("CO2 Rating (optional)", 1, 10, 5)
            smog_rating = st.slider("Smog Rating (optional)", 1, 10, 5)
            submitted = st.form_submit_button("Predict")
        st.write("\n")
        st.write("\n")
        if submitted:
            try:
                input_data = pd.DataFrame(
                    {
                        "Make": [make],
                        "Model": [model],
                        "Vehicle Class": [vehicle_class],
                        "Engine Size (L)": [engine_size],
                        "Cylinders": [cylinders],
                        "Transmission": [transmission],
                        "Fuel Type": [fuel_type],
                        "Fuel Consumption (L/100Km)": [city_consumption],
                        "Hwy (L/100 km)": [hwy_consumption],
                        "Comb (L/100 km)": [comb_consumption],
                        "CO2 Rating": [co2_rating],
                        "Smog Rating": [smog_rating],
                    }
                )
                input_data["Transmission_Type"] = input_data[
                    "Transmission"
                ].str.extract(r"([A-Z]+)")
                input_data["Gear_Count"] = (
                    input_data["Transmission"].str.extract(r"(\d+)").astype(float)
                )
                input_data["Gear_Count"] = input_data["Gear_Count"].fillna(5)
                input_data["Power_to_Weight"] = (
                    input_data["Engine Size (L)"]
                    * 1000
                    / input_data["Vehicle Class"].map(lambda x: len(x))
                )
                input_data["Fuel_Efficiency"] = 200 / input_data["Comb (L/100 km)"]
                input_data["City_Hwy_Ratio"] = (
                    input_data["Fuel Consumption (L/100Km)"]
                    / input_data["Hwy (L/100 km)"]
                )
                input_data["Engine_Size_Per_Cylinder"] = input_data[
                    "Engine Size (L)"
                ] / input_data["Cylinders"].replace(0, np.nan)
                input_data["Engine_Size_Per_Cylinder"] = input_data[
                    "Engine_Size_Per_Cylinder"
                ].fillna(input_data["Engine Size (L)"])
                categorical_cols = [
                    "Model",
                    "Vehicle Class",
                    "Transmission_Type",
                    "Fuel Type",
                    "Make",
                ]
                for col in categorical_cols:
                    if col in model_data["encoders"]:
                        encoder = model_data["encoders"][col]
                        try:
                            input_data[f"{col}_Encoded"] = encoder.transform(
                                input_data[col]
                            )
                        except:
                            input_data[f"{col}_Encoded"] = 0

                def get_size_category(vehicle_class):
                    if any(
                        term in vehicle_class.lower()
                        for term in ["small", "compact", "mini"]
                    ):
                        return 0
                    elif any(
                        term in vehicle_class.lower() for term in ["mid", "medium"]
                    ):
                        return 1
                    else:
                        return 2

                input_data["Vehicle_Size"] = input_data["Vehicle Class"].apply(
                    get_size_category
                )
                input_data["Is_4WD"] = input_data["Model"].apply(
                    lambda x: (
                        1 if any(term in x for term in ["4WD", "4X4", "AWD"]) else 0
                    )
                )
                selected_features = model_data["selected_features"]
                X_input = input_data[selected_features].values
                X_input_scaled = model_data["scaler"].transform(X_input)
                prediction = model_data["best_model"].predict(X_input_scaled)[0]
                prediction_label = model_data["emission_mapping"][prediction]
                proba_dict = {}
                if hasattr(model_data["best_model"], "predict_proba"):
                    proba = model_data["best_model"].predict_proba(X_input_scaled)[0]
                    proba_dict = {
                        model_data["emission_mapping"][i]: prob
                        for i, prob in enumerate(proba)
                    }
                st.markdown("### Prediction Result")
                st.markdown(
                    f"<h3 style='color: {'#2ecc71' if prediction_label == 'Low Emission' else '#e74c3c' if prediction_label == 'High Emission' else '#f1c40f'}'>Emission Category: {prediction_label}</h3>",
                    unsafe_allow_html=True,
                )
                create_prediction_visualizations(prediction_label, proba_dict)
            except Exception as e:
                st.error(
                    "ERROR: Please make sure all required fields are filled correctly."
                )
                st.write("")


if __name__ == "__main__":
    main()
