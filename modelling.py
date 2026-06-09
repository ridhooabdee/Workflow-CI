import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Menyimpan hasil log lokal untuk dibaca oleh Docker saat CI berjalan
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_Pipeline_Beasiswa")

def main():
    mlflow.sklearn.autolog()
    
    # Dataset diakses dari dalam folder MLProject
    data_path = 'dataset-kelayakan-beasiswa_preprocessing/dataset_kelayakan_beasiswa_clean.csv'
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Status_Kelayakan'])
    y = df['Status_Kelayakan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), num_cols),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ])
    
    # MENGGUNAKAN PARAMETER TERBAIK DARI HASIL TUNING KRITERIA 2
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=200,          # <-- Parameter Terbaik
            max_depth=6,               # <-- Parameter Terbaik
            min_samples_split=10,      # <-- Parameter Terbaik
            class_weight='balanced', 
            random_state=42
        ))
    ])
    
    with mlflow.start_run(run_name="CI_Run") as run:
        rf_pipeline.fit(X_train, y_train)
        
        # Simpan Run ID ke file teks. Ini TRIK PENTING agar GitHub Actions tahu ID modelnya!
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        
        print(f"Selesai melatih model juara! Run ID untuk Docker: {run.info.run_id}")

if __name__ == "__main__":
    main()