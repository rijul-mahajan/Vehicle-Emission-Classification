# Vehicle Emission Classifier

This project is a machine learning-based web application that predicts vehicle emission levels (Low, Moderate, High) based on vehicle specifications such as engine size, fuel type, and transmission. The application uses a trained classification model and provides an interactive interface built with Streamlit for users to input vehicle details and visualize predictions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Dependencies](#dependencies)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features

- Predicts vehicle emission categories (Low, Moderate, High) based on user inputs.
- Visualizes model performance, feature importance, and prediction confidence using interactive Plotly charts.
- Displays environmental impact metrics, such as CO2 output and trees needed to offset emissions.
- Provides insights into factors affecting vehicle emissions.
- Handles data preprocessing, feature engineering, and model training with robust error handling.
- Supports clustering analysis to identify patterns in vehicle emission data.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rijul-mahajan/vehicle-emission-classifier.git
   cd vehicle-emission-classifier
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Ensure you have Python 3.8+ installed.

4. Download the dataset (if not included) and place it in the `dataset/` folder:
   - The project uses `Fuel_Consumption_Ratings_2023.csv`. You can obtain it from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/fuel-concumption-ratings-2023/data) or a similar source.

## Usage

1. **Train the Model**:
   Run the training script to process the dataset, engineer features, train models, and save the best model:

   ```bash
   python train_and_save.py
   ```

   This generates `vehicle_emission_model.pkl` in the `model/` directory.

2. **Launch the Web Application**:
   Start the Streamlit app to interact with the model:

   ```bash
   streamlit run app.py
   ```

   Open your browser and navigate to `http://localhost:8501`.

3. **Explore the Application**:
   - **Home**: Learn about vehicle emissions and key factors.
   - **Model Information**: View model performance metrics and visualizations.
   - **Make Predictions**: Input vehicle details to predict emission categories and see environmental impact.

## Project Structure

```
vehicle-emission-classifier/
├── dataset/
│   └── Fuel_Consumption_Ratings_2023.csv  # Input dataset
├── model/
│   └── vehicle_emission_model.pkl         # Trained model and components
├── app.py                                # Streamlit web application
├── train_and_save.py                     # Script for training and saving the model
├── vehicle_emission_classification.py    # Core data processing and model training logic
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

## Dataset

The project uses the **Fuel Consumption Ratings 2023** dataset, which includes:

- Vehicle specifications (Make, Model, Engine Size, Fuel Type, etc.).
- Fuel consumption metrics (City, Highway, Combined).
- CO2 emissions (g/km) and ratings.

The dataset is preprocessed to handle missing values, encode categorical variables, and engineer features like `Power_to_Weight` and `Fuel_Efficiency`.

## Model Training

The `train_and_save.py` script:

1. Loads and cleans the dataset.
2. Engineers features (e.g., `Transmission_Type`, `Vehicle_Size`).
3. Creates emission categories based on CO2 emissions percentiles.
4. Selects relevant features using correlation analysis.
5. Trains multiple models (Random Forest, Gradient Boosting, SVM, etc.) using GridSearchCV.
6. Performs clustering (KMeans, DBSCAN) for additional analysis.
7. Saves the best model and components (scaler, encoders, etc.) as a pickle file.

The best model is chosen based on cross-validation F1 scores, balancing precision and recall.

## Web Application

The `app.py` script powers the Streamlit interface, offering:

- A user-friendly form to input vehicle details.
- Visualizations of prediction confidence, feature importance, and emission distributions.
- Environmental impact insights, including CO2 estimates and tree offset calculations.
- Model performance metrics and explanations of emission categories.

## Dependencies

Key dependencies include:

- `pandas`, `numpy`: Data manipulation.
- `scikit-learn`: Machine learning models and preprocessing.
- `streamlit`: Web application framework.
- `plotly`: Interactive visualizations.

See `requirements.txt` for the full list.

## Future Improvements

- Add real-time data integration for newer vehicle models.
- Enhance model interpretability with SHAP or LIME.
- Include more advanced feature engineering based on domain knowledge.
- Deploy the application to a cloud platform (e.g., Heroku, AWS).
- Support multi-language interfaces for broader accessibility.

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit) for details.
