from data_loader import load_data
from data_preprocessing import preprocess_data
from data_exploration import (
    plot_histograms, plot_boxplots, plot_categorical_counts, 
    plot_correlation_heatmap, plot_risk_distribution, 
    plot_risk_vs_features, plot_pairplot
)
from model import train_model_classification


if __name__ == "__main__":
    file_path = "dataset/german_credit_dataset.csv"  # Veri seti yolu

    print("🔹 Loading dataset...")
    df = load_data(file_path)

    if df is not None:
        # 🛠 Data Preprocessing
        print("🔹 Preprocessing data...")
        processed_df, encoders = preprocess_data(df)
        processed_df.to_csv("dataset/preprocessed_dataset.csv", index=False)
        print("✅ Data preprocessing complete. Saved as 'preprocessed_dataset.csv'.")

        # 🔍 Data Exploration & Visualization
        print("🔹 Performing data exploration and visualization...")
        df_cleaned = df.drop(df.columns[0], axis=1, errors="ignore")  # Gereksiz index sütunu kaldırıldı

        try:
            plot_histograms(df_cleaned)
            plot_boxplots(df_cleaned)
            plot_categorical_counts(df_cleaned)
            plot_correlation_heatmap(df_cleaned)
            plot_risk_distribution(df_cleaned)
            plot_risk_vs_features(df_cleaned)
            plot_pairplot(df_cleaned)
            print("✅ Data exploration complete.")
        except Exception as e:
            print(f"⚠️ Error during data visualization: {e}")

    # 🎯 Model Training
    print("🔹 Training model...")
    model = train_model_classification("dataset/preprocessed_dataset.csv")
    
    if model:
        print("✅ Model training complete. Saved as 'models/risk_classification_model.pkl'.")
    else:
        print("❌ Model training failed!")

    
