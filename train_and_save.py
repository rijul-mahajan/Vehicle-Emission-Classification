import pandas as pd
import pickle
import os
from tqdm import tqdm
from vehicle_emission_classification import process_data

# Clear the terminal before starting
os.system("cls" if os.name == "nt" else "clear")

# === Step 1: Load the dataset ===
file_path = "dataset/Fuel_Consumption_Ratings_2023.csv"
try:
    print("\n")
    df = pd.read_csv(file_path)
    print(
        f"✅ Dataset loaded successfully: {file_path} ({len(df)} rows, {len(df.columns)} columns)"
    )
    print("\n")
except Exception as e:
    print("\n")
    raise Exception(f"❌ Failed to load dataset: {str(e)}")

# === Step 2: Process data, train, and evaluate ===
try:
    # Use tqdm to show progress for the entire process
    with tqdm(total=100, desc="Processing and Training", unit="%") as pbar:
        results = process_data(file_path, pbar=pbar)  # Pass pbar to update progress
        pbar.update(100 - pbar.n)  # Ensure completion
    print("✅ Data processing and model training completed")
    print("\n")
except Exception as e:
    print("\n")
    raise Exception(f"❌ Error during processing/training: {str(e)}")

# === Step 3: Save the trained model and related components ===
output_file = "model/vehicle_emission_model.pkl"
try:
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Show progress for saving
    with tqdm(total=1, desc="Saving Model", unit="file") as pbar:
        with open(output_file, "wb") as f:
            pickle.dump(
                {
                    "best_model": results["best_model"],
                    "best_model_name": results["best_model_name"],
                    "selected_features": results["selected_features"],
                    "feature_importance": results["feature_importance"],
                    "emission_mapping": results["emission_mapping"],
                    "scaler": results["scaler"],
                    "encoders": results["encoders"],
                    "feature_correlations": results["feature_correlations"],
                    "category_distribution": results["category_distribution"],
                    "best_accuracy": results["best_accuracy"],
                    "cv_accuracy": results["cv_accuracy"],
                    "makes": (
                        results["df"]["Make"].unique().tolist()
                        if "Make" in results["df"].columns
                        else []
                    ),
                    "vehicle_classes": (
                        results["df"]["Vehicle Class"].unique().tolist()
                        if "Vehicle Class" in results["df"].columns
                        else []
                    ),
                    "fuel_types": (
                        results["df"]["Fuel Type"].unique().tolist()
                        if "Fuel Type" in results["df"].columns
                        else []
                    ),
                    "df": results["df"],
                },
                f,
            )
        pbar.update(1)
    print(f"✅ Model and components saved to '{output_file}'")
except Exception as e:
    print("\n")
    raise Exception(f"❌ Error saving model: {str(e)}")

# === Step 4: Print summary ===
print("\n📝 Training Summary:")
print(f"\n\t🔧 Selected Features:")
for feature in results["selected_features"]:
    print(f"\t\t{feature}")
print(
    f"\n\t🏆 Best Model:\n"
    f"\t\tName: {results['best_model_name']}\n"
    f"\t\tTest Accuracy: {results['best_accuracy']*100:.4f}%\n"
    f"\t\tCross-Validation Accuracy: {results['cv_accuracy']*100:.4f}%"
)
print(f"\n\t📊 Emission Category Distribution:")
for category, count in results["category_distribution"].items():
    print(f"\t\t{category}: {count}")
