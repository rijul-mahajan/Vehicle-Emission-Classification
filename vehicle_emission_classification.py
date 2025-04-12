import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.decomposition import PCA
import warnings
from tqdm import tqdm  # Added for progress bars

warnings.filterwarnings("ignore")


def load_and_prepare_data(file_path):
    """Load and prepare the dataset with basic cleaning"""
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df


def engineer_features(df):
    """Create engineered features to improve model performance"""
    data = df.copy()
    data["Transmission_Type"] = data["Transmission"].str.extract(r"([A-Z]+)")
    data["Gear_Count"] = data["Transmission"].str.extract(r"(\d+)").astype(float)
    data["Gear_Count"] = data["Gear_Count"].fillna(
        data["Gear_Count"].median(skipna=True)
    )
    data["Power_to_Weight"] = (
        data["Engine Size (L)"] * 1000 / data["Vehicle Class"].map(lambda x: len(x))
    )
    data["Fuel_Efficiency"] = data["CO2 Emissions (g/km)"] / data["Comb (L/100 km)"]
    data["City_Hwy_Ratio"] = data["Fuel Consumption (L/100Km)"] / data["Hwy (L/100 km)"]
    data["Engine_Size_Per_Cylinder"] = data["Engine Size (L)"] / data[
        "Cylinders"
    ].replace(0, np.nan)
    data["Engine_Size_Per_Cylinder"] = data["Engine_Size_Per_Cylinder"].fillna(
        data["Engine Size (L)"]
    )

    encoders = {}
    categorical_cols = [
        "Model",
        "Vehicle Class",
        "Transmission_Type",
        "Fuel Type",
        "Make",
    ]
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[f"{col}_Encoded"] = le.fit_transform(data[col])
            encoders[col] = le

    def get_size_category(vehicle_class):
        if any(term in vehicle_class.lower() for term in ["small", "compact", "mini"]):
            return 0
        elif any(term in vehicle_class.lower() for term in ["mid", "medium"]):
            return 1
        else:
            return 2

    data["Vehicle_Size"] = data["Vehicle Class"].apply(get_size_category)
    data["Is_4WD"] = data["Model"].apply(
        lambda x: 1 if any(term in x for term in ["4WD", "4X4", "AWD"]) else 0
    )
    return data, encoders


def create_emission_categories(df, num_categories=3):
    """Create emission categories based on percentiles"""
    thresholds = [
        df["CO2 Emissions (g/km)"].quantile(i / num_categories)
        for i in range(1, num_categories)
    ]

    def categorize_emission(x, thresholds):
        for i, threshold in enumerate(thresholds):
            if x <= threshold:
                return i
        return len(thresholds)

    df["Emission_Label"] = df["CO2 Emissions (g/km)"].apply(
        lambda x: categorize_emission(x, thresholds)
    )
    categories = ["Low Emission", "Moderate Emission", "High Emission"]
    emission_mapping = {i: categories[i] for i in range(len(categories))}
    df["Emission_Category"] = df["Emission_Label"].map(emission_mapping)
    return df, emission_mapping


def select_features(df, target="Emission_Label", corr_threshold=0.1, max_features=10):
    """Select the most relevant features based on correlation with target"""
    potential_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [target, "CO2 Emissions (g/km)"]
    potential_features = [col for col in potential_features if col not in exclude_cols]

    correlations = []
    for col in potential_features:
        if col in df.columns:
            corr = np.abs(np.corrcoef(df[col], df[target])[0, 1])
            correlations.append((col, corr))

    correlations.sort(key=lambda x: x[1], reverse=True)
    selected_features = [feat for feat, corr in correlations if corr > corr_threshold]

    if len(selected_features) > max_features:
        selected_features = [feat for feat, _ in correlations[:max_features]]
    if len(selected_features) < 3:
        selected_features = [feat for feat, _ in correlations[:5]]

    return selected_features, correlations


def train_classification_models(X_train, X_test, y_train, y_test):
    """Build and optimize multiple classification models"""
    models = {
        "Random Forest": RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
    }

    param_grid = {
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 15, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting": {
            "max_iter": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 6, 9],
            "l2_regularization": [0, 1.0, 10.0],
        },
        "Decision Tree": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
            "min_samples_leaf": [1, 2, 4],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "p": [1, 2],
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "poly"],
            "gamma": ["scale", "auto", 0.1],
        },
    }

    results = {}
    for name, model in models.items():
        try:
            # Use GridSearchCV with stratified k-fold for better validation
            from sklearn.model_selection import StratifiedKFold

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            grid = GridSearchCV(
                model,
                param_grid[name],
                cv=skf,
                scoring=["accuracy", "f1_weighted", "roc_auc_ovr"],
                refit="f1_weighted",  # Optimize for F1 score which balances precision and recall
                n_jobs=-1,
                verbose=0,
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)

            # More comprehensive evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Cross-validation with the best model
            cv_scores = cross_val_score(
                best_model,
                np.vstack((X_train, X_test)),
                np.concatenate((y_train, y_test)),
                cv=skf,
                scoring="f1_weighted",
            )

            # Store results
            results[name] = {
                "model": best_model,
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix,
                "classification_report": class_report,
                "cv_scores": cv_scores,
                "best_params": grid.best_params_,
                "f1_score": class_report["weighted avg"]["f1-score"],
            }

        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue

    return results


def perform_clustering(X):
    """Perform clustering analysis"""
    clustering_results = {}

    # Find optimal number of clusters for KMeans
    max_k = min(10, X.shape[0] // 10)
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
            silhouette_scores.append((k, score))
        except:
            continue

    if silhouette_scores:
        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        clustering_results["KMeans"] = {
            "model": kmeans,
            "labels": kmeans_labels,
            "silhouette_score": silhouette_score(X, kmeans_labels),
            "optimal_k": optimal_k,
        }

    # DBSCAN clustering
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2)
    nbrs = nn.fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, 1])

    # Simple heuristic for eps value
    eps_value = np.percentile(distances, 90)

    best_dbscan, best_score, best_labels = None, -1, None
    for min_samples in [3, 5, 10, 15]:
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters >= 2 and n_noise < len(X) * 0.5:
            valid_indices = labels != -1
            if sum(valid_indices) > n_clusters:
                try:
                    score = silhouette_score(X[valid_indices], labels[valid_indices])
                    if score > best_score:
                        best_score = score
                        best_dbscan = dbscan
                        best_labels = labels
                except:
                    continue

    if best_dbscan is not None:
        n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
        n_noise = list(best_labels).count(-1)
        clustering_results["DBSCAN"] = {
            "model": best_dbscan,
            "labels": best_labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette_score": best_score,
        }

    return clustering_results


def analyze_results(classification_results, X, feature_names):
    """Analyze classification results and return best model info"""
    models = list(classification_results.keys())
    cv_means = [classification_results[model]["cv_scores"].mean() for model in models]
    test_accuracies = [classification_results[model]["accuracy"] for model in models]

    best_model_idx = np.argmax(cv_means)
    best_model_name = models[best_model_idx]
    best_model = classification_results[best_model_name]["model"]

    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_importance = [(feature_names[i], importances[i]) for i in indices]

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_accuracy": test_accuracies[best_model_idx],
        "cv_accuracy": cv_means[best_model_idx],
        "feature_importance": feature_importance,
        "all_models_performance": dict(zip(models, zip(cv_means, test_accuracies))),
    }


def process_data(file_path, pbar=None):
    """
    Main function to process data and build models.
    Args:
        file_path: Path to the dataset CSV.
        pbar: Optional tqdm progress bar to update progress.
    Returns:
        Dictionary with model results and metadata.
    """
    # Initialize progress steps
    progress_steps = {
        "load_prepare": 10,
        "feature_engineering": 20,
        "emission_categories": 10,
        "feature_selection": 10,
        "data_preparation": 10,
        "model_training": 30,
        "clustering": 10,
    }
    total_progress = sum(progress_steps.values())

    def update_progress(step_name):
        if pbar is not None:
            increment = progress_steps.get(step_name, 0)
            pbar.update(increment * 100 / total_progress)
            pbar.set_postfix({"Step": step_name})

    # Load and prepare data
    df = load_and_prepare_data(file_path)
    update_progress("load_prepare")

    # Feature engineering
    processed_df, encoders = engineer_features(df)
    update_progress("feature_engineering")

    # Create emission categories
    processed_df, emission_mapping = create_emission_categories(processed_df)
    category_distribution = dict(
        processed_df["Emission_Category"].value_counts().items()
    )
    update_progress("emission_categories")

    # Check for class imbalance
    class_counts = processed_df["Emission_Label"].value_counts()
    imbalance_ratio = (
        class_counts.max() / class_counts.min()
        if class_counts.min() > 0
        else float("inf")
    )

    # Feature selection
    selected_features, feature_correlations = select_features(
        processed_df, corr_threshold=0.15
    )
    update_progress("feature_selection")

    # Prepare data for modeling
    X = processed_df[selected_features].values
    y = processed_df["Emission_Label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    update_progress("data_preparation")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train classification models
    classification_results = train_classification_models(
        X_train, X_test, y_train, y_test
    )
    update_progress("model_training")

    # Perform clustering
    clustering_results = perform_clustering(X_scaled)
    update_progress("clustering")

    # Analyze results
    analysis = analyze_results(classification_results, X_scaled, selected_features)

    return {
        "df": df,
        "processed_df": processed_df,
        "selected_features": selected_features,
        "feature_correlations": feature_correlations,
        "best_model": analysis["best_model"],
        "best_model_name": analysis["best_model_name"],
        "best_accuracy": analysis["best_accuracy"],
        "cv_accuracy": analysis["cv_accuracy"],
        "feature_importance": analysis["feature_importance"],
        "all_models_performance": analysis["all_models_performance"],
        "category_distribution": category_distribution,
        "emission_mapping": emission_mapping,
        "scaler": scaler,
        "encoders": encoders,
        "classification_results": classification_results,
        "clustering_results": clustering_results,
        "pca": PCA(n_components=2).fit(X_scaled),
        "imbalance_ratio": imbalance_ratio,
    }
