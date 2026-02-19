"""
Feature Selection Comparison Script

Compares three feature selection approaches:
1. tsfresh FeatureSelection (statistical significance only)
2. RandomForest feature_importances_ (permutation importance only)
3. Combined two-stage (tsfresh → RandomForest) [RECOMMENDED]

This script demonstrates the enhanced feature selection capabilities
for volcanic eruption forecasting using seismic tremor data.
"""

import os
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

from eruption_forecast import FeaturesBuilder, ForecastModel
from eruption_forecast.decorators import timer
from eruption_forecast.features import FeatureSelector
from eruption_forecast.logger import logger


@timer("Feature Selection Comparison")
def main(
    use_existing_features: bool = True,
    output_base_dir: str = r"D:\Projects\eruption-forecast\output\VG.OJN.00.EHZ\features",
):
    """
    Compare three feature selection approaches for eruption forecasting.

    Args:
        use_existing_features: If True, loads pre-extracted features.
                              If False, runs full pipeline from raw data.
        output_base_dir: Directory for comparison results
    """
    # ========== PARAMETERS ==========
    sds_dir = r"D:\Data\OJN"

    params = {
        "station": "OJN",
        "channel": "EHZ",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "window_size": 2,
        "volcano_id": "Lewotobi Laki-laki",
        "verbose": True,
        "debug": False,
    }

    eruptions = [
        "2025-03-20",
        "2025-04-22",
        "2025-05-18",
        "2025-06-17",
        "2025-07-07",
        "2025-08-01",
        "2025-08-17",
    ]

    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)

    features_csv = None
    label_csv = None

    # ========== STEP 1: Extract Features (or Load Existing) ==========
    if use_existing_features:
        logger.info("=" * 80)
        logger.info("Loading pre-extracted features...")
        logger.info("=" * 80)

        # Look for existing features in the default output directory
        default_output = (
            f"output/AM.{params['station']}.00.{params['channel']}/features"
        )

        # Find the most recent all_features file
        if os.path.exists(default_output):
            feature_files = [
                f
                for f in os.listdir(default_output)
                if f.startswith("all_features_") and f.endswith(".csv")
            ]
            label_files = [
                f
                for f in os.listdir(default_output)
                if f.startswith("label_features_") and f.endswith(".csv")
            ]

            if feature_files and label_files:
                features_csv = os.path.join(default_output, sorted(feature_files)[-1])
                label_csv = os.path.join(default_output, sorted(label_files)[-1])
                logger.info(f"Found features: {features_csv}")
                logger.info(f"Found labels: {label_csv}")
            else:
                raise FileNotFoundError(
                    f"No feature files found in {default_output}. "
                    "Set use_existing_features=False to extract features."
                )
        else:
            raise FileNotFoundError(
                f"Directory not found: {default_output}. "
                "Set use_existing_features=False to extract features."
            )

    else:
        logger.info("=" * 80)
        logger.info("Extracting features from raw data...")
        logger.info("=" * 80)

        fm = ForecastModel(
            overwrite=False,
            n_jobs=4,
            **params,
        )

        fm.calculate(
            source="sds",
            sds_dir=sds_dir,
            plot_tmp=True,
            save_plot=True,
            remove_outlier_method="maximum",
        ).build_label(
            start_date="2025-01-01",
            end_date="2025-07-24",
            day_to_forecast=2,
            window_step=6,
            window_step_unit="hours",
            eruption_dates=eruptions,
            verbose=True,
        ).extract_features(
            select_tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4"],
            save_tremor_matrix_per_method=True,
            save_tremor_matrix_per_id=False,
            exclude_features=[
                "agg_linear_trend",
                "linear_trend_timewise",
                "length",
                "has_duplicate_max",
                "has_duplicate_min",
                "has_duplicate",
            ],
            use_relevant_features=False,  # Extract ALL features first
            overwrite=False,
        )

        if isinstance(fm.FeaturesBuilder, FeaturesBuilder):
            features_csv = fm.FeaturesBuilder.all_features_csv
            label_csv = fm.FeaturesBuilder.label_features_csv

    # ========== STEP 2: Load Features and Labels ==========
    logger.info("=" * 80)
    logger.info("Loading features and labels...")
    logger.info("=" * 80)

    if features_csv is None or label_csv is None:
        raise FileNotFoundError("CSV files not found")

    X = pd.read_csv(features_csv, index_col=0)
    y_df = pd.read_csv(label_csv, index_col=0)
    y = y_df["is_erupted"]

    # Ensure X and y have matching indices (required by tsfresh)
    y.index = X.index

    logger.info(f"Total features: {X.shape[1]}")
    logger.info(f"Total samples: {X.shape[0]}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    # ========== STEP 3: Train/Test Split ==========
    logger.info("=" * 80)
    logger.info("Splitting data (80% train, 20% test)...")
    logger.info("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # type: ignore
    )

    logger.info(f"Train samples: {X_train.shape[0]}")
    logger.info(f"Test samples: {X_test.shape[0]}")

    # ========== STEP 4: Compare Feature Selection Methods ==========
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------ Method 1: tsfresh Only ------------------
    logger.info("\n" + "=" * 80)
    logger.info("METHOD 1: tsfresh Statistical Selection")
    logger.info("=" * 80)

    selector_tsfresh = FeatureSelector(
        method="tsfresh",
        n_jobs=4,
        random_state=42,
        verbose=True,
    )

    X_train_tsfresh = selector_tsfresh.fit_transform(X_train, y_train, fdr_level=0.05)
    X_test_tsfresh = selector_tsfresh.transform(X_test)

    # Train RandomForest on tsfresh-selected features
    rf_tsfresh = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=4,
    )
    rf_tsfresh.fit(X_train_tsfresh, y_train)
    y_pred_tsfresh = rf_tsfresh.predict(X_test_tsfresh)  # type: ignore

    # Evaluate
    results["tsfresh"] = {
        "n_features": X_train_tsfresh.shape[1],
        "accuracy": accuracy_score(y_test, y_pred_tsfresh),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_tsfresh),
        "f1_score": f1_score(y_test, y_pred_tsfresh),
        "features": X_train_tsfresh.columns.tolist(),
    }

    logger.info(f"Selected features: {results['tsfresh']['n_features']}")
    logger.info(f"Test Accuracy: {results['tsfresh']['accuracy']:.4f}")
    logger.info(
        f"Test Balanced Accuracy: {results['tsfresh']['balanced_accuracy']:.4f}"
    )
    logger.info(f"Test F1-Score: {results['tsfresh']['f1_score']:.4f}")

    # Save feature list
    tsfresh_features_file = os.path.join(
        output_base_dir, f"tsfresh_features_{timestamp}.csv"
    )
    pd.DataFrame({"feature": results["tsfresh"]["features"]}).to_csv(
        tsfresh_features_file, index=False
    )
    logger.info(f"Saved feature list: {tsfresh_features_file}")

    # ------------------ Method 2: RandomForest Only ------------------
    logger.info("\n" + "=" * 80)
    logger.info("METHOD 2: RandomForest Permutation Importance")
    logger.info("=" * 80)

    selector_rf = FeatureSelector(
        method="random_forest",
        n_jobs=4,
        random_state=42,
        verbose=True,
    )

    X_train_rf = selector_rf.fit_transform(
        X_train,
        y_train,
        top_n=30,
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        n_repeats=10,
    )
    X_test_rf = selector_rf.transform(X_test)

    # Train RandomForest on RF-selected features
    rf_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=4,
    )
    rf_rf.fit(X_train_rf, y_train)  # type: ignore
    y_pred_rf = rf_rf.predict(X_test_rf)  # type: ignore

    # Evaluate
    results["random_forest"] = {
        "n_features": X_train_rf.shape[1],
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_rf),
        "f1_score": f1_score(y_test, y_pred_rf),
        "features": X_train_rf.columns.tolist(),
    }

    logger.info(f"Selected features: {results['random_forest']['n_features']}")
    logger.info(f"Test Accuracy: {results['random_forest']['accuracy']:.4f}")
    logger.info(
        f"Test Balanced Accuracy: {results['random_forest']['balanced_accuracy']:.4f}"
    )
    logger.info(f"Test F1-Score: {results['random_forest']['f1_score']:.4f}")

    # Save feature list
    rf_features_file = os.path.join(
        output_base_dir, f"random_forest_features_{timestamp}.csv"
    )
    pd.DataFrame({"feature": results["random_forest"]["features"]}).to_csv(
        rf_features_file, index=False
    )
    logger.info(f"Saved feature list: {rf_features_file}")

    # ------------------ Method 3: Combined (Two-Stage) ------------------
    logger.info("\n" + "=" * 80)
    logger.info("METHOD 3: Combined Two-Stage (tsfresh → RandomForest) [RECOMMENDED]")
    logger.info("=" * 80)

    selector_combined = FeatureSelector(
        method="combined",
        n_jobs=4,
        random_state=42,
        verbose=True,
    )

    X_train_combined = selector_combined.fit_transform(
        X_train,
        y_train,
        fdr_level=0.05,
        top_n=30,
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        n_repeats=10,
    )
    X_test_combined = selector_combined.transform(X_test)

    # Train RandomForest on combined-selected features
    rf_combined = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=4,
    )
    rf_combined.fit(X_train_combined, y_train)  # type: ignore
    y_pred_combined = rf_combined.predict(X_test_combined)  # type: ignore

    # Evaluate
    results["combined"] = {
        "n_features": X_train_combined.shape[1],
        "accuracy": accuracy_score(y_test, y_pred_combined),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_combined),
        "f1_score": f1_score(y_test, y_pred_combined),
        "features": X_train_combined.columns.tolist(),
    }

    logger.info(f"Selected features: {results['combined']['n_features']}")
    logger.info(f"Test Accuracy: {results['combined']['accuracy']:.4f}")
    logger.info(
        f"Test Balanced Accuracy: {results['combined']['balanced_accuracy']:.4f}"
    )
    logger.info(f"Test F1-Score: {results['combined']['f1_score']:.4f}")

    # Save feature list and scores
    combined_features_file = os.path.join(
        output_base_dir, f"combined_features_{timestamp}.csv"
    )
    feature_scores = selector_combined.get_feature_scores()
    feature_scores.to_csv(combined_features_file)
    logger.info(f"Saved feature list with scores: {combined_features_file}")

    # ========== STEP 5: Compare Results ==========
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)

    comparison_df = pd.DataFrame(
        {
            "Method": ["tsfresh", "random_forest", "combined"],
            "N_Features": [
                results["tsfresh"]["n_features"],
                results["random_forest"]["n_features"],
                results["combined"]["n_features"],
            ],
            "Accuracy": [
                results["tsfresh"]["accuracy"],
                results["random_forest"]["accuracy"],
                results["combined"]["accuracy"],
            ],
            "Balanced_Accuracy": [
                results["tsfresh"]["balanced_accuracy"],
                results["random_forest"]["balanced_accuracy"],
                results["combined"]["balanced_accuracy"],
            ],
            "F1_Score": [
                results["tsfresh"]["f1_score"],
                results["random_forest"]["f1_score"],
                results["combined"]["f1_score"],
            ],
        }
    )

    print("\n" + comparison_df.to_string(index=False))

    # Save comparison
    comparison_file = os.path.join(output_base_dir, f"comparison_{timestamp}.csv")
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\nSaved comparison: {comparison_file}")

    # ========== STEP 6: Feature Overlap Analysis ==========
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE OVERLAP ANALYSIS")
    logger.info("=" * 80)

    tsfresh_set = set(results["tsfresh"]["features"])
    rf_set = set(results["random_forest"]["features"])
    combined_set = set(results["combined"]["features"])

    logger.info(f"Features in tsfresh only: {len(tsfresh_set - rf_set - combined_set)}")
    logger.info(f"Features in RF only: {len(rf_set - tsfresh_set - combined_set)}")
    logger.info(
        f"Features in combined only: {len(combined_set - tsfresh_set - rf_set)}"
    )
    logger.info(f"Features in all three: {len(tsfresh_set & rf_set & combined_set)}")
    logger.info(f"Features in tsfresh ∩ combined: {len(tsfresh_set & combined_set)}")

    overlap_analysis = {
        "tsfresh_only": list(tsfresh_set - rf_set - combined_set),
        "rf_only": list(rf_set - tsfresh_set - combined_set),
        "combined_only": list(combined_set - tsfresh_set - rf_set),
        "all_three": list(tsfresh_set & rf_set & combined_set),
        "tsfresh_and_combined": list(tsfresh_set & combined_set),
    }

    overlap_file = os.path.join(output_base_dir, f"feature_overlap_{timestamp}.csv")
    max_len = max(len(v) for v in overlap_analysis.values())
    overlap_df = pd.DataFrame(
        {k: v + [""] * (max_len - len(v)) for k, v in overlap_analysis.items()}
    )
    overlap_df.to_csv(overlap_file, index=False)
    logger.info(f"Saved feature overlap analysis: {overlap_file}")

    # ========== STEP 7: Detailed Classification Reports ==========
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED CLASSIFICATION REPORTS")
    logger.info("=" * 80)

    logger.info("\n--- tsfresh Method ---")
    print(
        classification_report(
            y_test, y_pred_tsfresh, target_names=["No Eruption", "Eruption"]
        )
    )

    logger.info("\n--- RandomForest Method ---")
    print(
        classification_report(
            y_test, y_pred_rf, target_names=["No Eruption", "Eruption"]
        )
    )

    logger.info("\n--- Combined Method ---")
    print(
        classification_report(
            y_test, y_pred_combined, target_names=["No Eruption", "Eruption"]
        )
    )

    # ========== STEP 8: Recommendations ==========
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)

    best_method = comparison_df.loc[
        comparison_df["Balanced_Accuracy"].idxmax(), "Method"
    ]
    best_score = comparison_df["Balanced_Accuracy"].max()

    logger.info(
        f"✓ Best performing method: {best_method.upper()} "  # type: ignore
        f"(Balanced Accuracy: {best_score:.4f})"
    )

    if best_method == "combined":
        logger.info("\n✓ The COMBINED two-stage method is recommended because it:")
        logger.info(
            "  - Combines statistical rigor (tsfresh) with model optimization (RF)"
        )
        logger.info("  - Reduces overfitting through statistical pre-filtering")
        logger.info("  - Captures feature interactions with RandomForest")
        logger.info(
            "  - Provides both p-values and importance scores for interpretability"
        )
    elif best_method == "tsfresh":
        logger.info("\n✓ The TSFRESH method performed best. Consider:")
        logger.info("  - Your features may have strong univariate relationships")
        logger.info("  - Feature interactions may not be critical for this dataset")
        logger.info("  - This is the fastest and most interpretable approach")
    else:
        logger.info("\n✓ The RANDOM FOREST method performed best. Consider:")
        logger.info("  - Feature interactions are likely important")
        logger.info(
            "  - You may want to try the combined method for better generalization"
        )

    logger.info(f"\n✓ All results saved to: {output_base_dir}")

    return results, comparison_df


if __name__ == "__main__":
    # Set to True if you already have extracted features
    # Set to False to run the full pipeline from raw data
    USE_EXISTING_FEATURES = False

    results, comparison = main(use_existing_features=USE_EXISTING_FEATURES)
