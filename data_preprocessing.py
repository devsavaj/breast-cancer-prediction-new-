import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def preprocess_data(file_path="data/data.csv", n_components=2):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['Unnamed: 32'], axis=1)
    
    # Encode target
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
    # Split features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA for visualization
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train_pca, X_test_pca, pca
