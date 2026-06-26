"""
Advanced ML Model Optimization System
Week 4 - Advanced Machine Learning Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.datasets import make_classification
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(page_title="Advanced ML Optimization", page_icon="🚀", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-font { font-size:30px !important; font-weight:bold; color:#1E88E5; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               padding: 20px; border-radius: 10px; color: white; text-align: center; }
.best-model { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
              padding: 15px; border-radius: 10px; color: white; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'data' not in st.session_state:
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, weights=[0.8, 0.2], random_state=42)
    feature_names = [f'feature_{i}' for i in range(20)]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')
    st.session_state.data = pd.concat([X, y], axis=1)
    st.session_state.models = {}
    st.session_state.results = None
    st.session_state.best_model = None
    st.session_state.best_name = None
    st.session_state.tuned_model = None
    st.session_state.scaler = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None

# Sidebar
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "📊 Dataset", "🧠 Model Training", "🔧 Optimization", "🎯 Predictions"])

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown("<p class='big-font'>🚀 Advanced ML Model Optimization System</p>", unsafe_allow_html=True)
    st.markdown("### Week 4 - Advanced Machine Learning Models, Feature Engineering & Model Optimization")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>📊 Dataset</h3>
            <p style='font-size:24px'>{st.session_state.data.shape[0]:,}</p>
            <p>Samples</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🤖 Models</h3>
            <p style='font-size:24px'>5</p>
            <p>Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        target_dist = st.session_state.data['target'].value_counts()
        pos_pct = (target_dist[1]/(target_dist[0]+target_dist[1])*100)
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🎯 Target</h3>
            <p style='font-size:24px'>{pos_pct:.1f}%</p>
            <p>Positive Class</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📋 Task Description
    **Objective:** Build advanced ML models, perform feature engineering, hyperparameter tuning, and create a production-ready application.
    
    **Models:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM
    
    **Technologies:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit
    
    **Deliverables:**
    - ✅ ML Notebook
    - ✅ Feature Engineered Dataset
    - ✅ Streamlit Application
    - ✅ GitHub Repository
    - ✅ Model Comparison Report
    - ✅ Screenshots
    """)

# ==================== DATASET PAGE ====================
elif page == "📊 Dataset":
    st.markdown("<p class='big-font'>📊 Dataset Overview</p>", unsafe_allow_html=True)
    
    df = st.session_state.data
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Data Preview")
        st.dataframe(df.head(20))
    with col2:
        st.markdown("### Statistical Summary")
        st.dataframe(df.describe())
    
    st.markdown("### Target Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    df['target'].value_counts().plot(kind='bar', ax=ax, color=['#4CAF50','#FF6B6B'])
    ax.set_title('Target Variable Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    for i, v in enumerate(df['target'].value_counts().values):
        ax.text(i, v+50, str(v), ha='center', va='bottom')
    st.pyplot(fig)
    
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,10))
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

# ==================== MODEL TRAINING PAGE ====================
elif page == "🧠 Model Training":
    st.markdown("<p class='big-font'>🧠 Model Training & Comparison</p>", unsafe_allow_html=True)
    
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training models..."):
            df = st.session_state.data
            X = df.drop('target', axis=1)
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
            
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
                'KNN': KNeighborsClassifier(n_jobs=-1),
                'SVM': SVC(random_state=42, probability=True)
            }
            
            results = {}
            models_dict = {}
            
            progress = st.progress(0)
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred)
                }
                models_dict[name] = model
                progress.progress((i+1)/len(models))
            
            st.session_state.models = models_dict
            st.session_state.results = pd.DataFrame(results).T
            st.session_state.best_name = st.session_state.results['F1-Score'].idxmax()
            st.session_state.best_model = models_dict[st.session_state.best_name]
            
            os.makedirs('models', exist_ok=True)
            joblib.dump(st.session_state.best_model, 'models/best_model.pkl')
            joblib.dump(scaler, 'models/scaler.pkl')
            
            st.success(f"✅ Training complete! Best model: {st.session_state.best_name}")
    
    if st.session_state.results is not None:
        st.markdown("### 📊 Model Performance")
        st.dataframe(st.session_state.results.style.background_gradient(cmap='YlOrRd'))
        
        best_f1 = st.session_state.results.loc[st.session_state.best_name, 'F1-Score']
        st.markdown(f"""
        <div class='best-model'>
            <h3>⭐ Best Model: {st.session_state.best_name}</h3>
            <p style='font-size:20px'>F1-Score: {best_f1:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10,6))
            st.session_state.results.plot(kind='bar', ax=ax)
            ax.set_title('Model Comparison')
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.legend(loc='lower right')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(st.session_state.results, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax)
            ax.set_title('Performance Heatmap')
            st.pyplot(fig)
        
        # Confusion Matrix
        st.markdown("### 📈 Confusion Matrix - Best Model")
        X_test_scaled = st.session_state.scaler.transform(st.session_state.X_test)
        y_pred = st.session_state.best_model.predict(X_test_scaled)
        
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

# ==================== OPTIMIZATION PAGE ====================
elif page == "🔧 Optimization":
    st.markdown("<p class='big-font'>🔧 Hyperparameter Tuning</p>", unsafe_allow_html=True)
    
    if st.session_state.best_model is None:
        st.warning("⚠️ Please train models first in 'Model Training' page")
    else:
        st.markdown(f"### Optimizing: {st.session_state.best_name}")
        
        param_grid = {
            'Random Forest': {'n_estimators': [50,100,200], 'max_depth': [10,15,None], 'min_samples_split': [2,5,10]},
            'Decision Tree': {'max_depth': [10,15,None], 'min_samples_split': [2,5,10], 'min_samples_leaf': [1,2,4]},
            'Logistic Regression': {'C': [0.1,1,10], 'solver': ['liblinear','saga']},
            'KNN': {'n_neighbors': [3,5,7,9], 'weights': ['uniform','distance']},
            'SVM': {'C': [0.1,1,10], 'kernel': ['rbf','linear']}
        }
        
        grid = None
        for key in param_grid:
            if key in st.session_state.best_name:
                grid = param_grid[key]
                break
        
        if grid:
            st.markdown("#### Parameter Grid")
            st.json(grid)
            
            if st.button("🔧 Start Optimization", type="primary", use_container_width=True):
                with st.spinner("Performing GridSearchCV..."):
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    if 'Random Forest' in st.session_state.best_name:
                        base = RandomForestClassifier(random_state=42, n_jobs=-1)
                    elif 'Decision Tree' in st.session_state.best_name:
                        base = DecisionTreeClassifier(random_state=42)
                    elif 'Logistic Regression' in st.session_state.best_name:
                        base = LogisticRegression(random_state=42, max_iter=1000)
                    elif 'KNN' in st.session_state.best_name:
                        base = KNeighborsClassifier(n_jobs=-1)
                    else:
                        base = SVC(random_state=42, probability=True)
                    
                    search = GridSearchCV(base, grid, cv=5, scoring='f1', n_jobs=-1)
                    search.fit(X_train_scaled, y_train)
                    
                    st.session_state.tuned_model = search.best_estimator_
                    
                    st.success("✅ Optimization complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Best Parameters")
                        st.json(search.best_params_)
                    with col2:
                        st.metric("Best CV F1-Score", f"{search.best_score_:.4f}")
                    
                    X_test_scaled = scaler.transform(st.session_state.X_test)
                    y_pred = search.predict(X_test_scaled)
                    
                    tuned_metrics = {
                        'Accuracy': accuracy_score(st.session_state.y_test, y_pred),
                        'Precision': precision_score(st.session_state.y_test, y_pred),
                        'Recall': recall_score(st.session_state.y_test, y_pred),
                        'F1-Score': f1_score(st.session_state.y_test, y_pred)
                    }
                    
                    st.markdown("#### Tuned Model Performance")
                    st.dataframe(pd.DataFrame([tuned_metrics]).T.rename(columns={0:'Value'}))
                    
                    joblib.dump(search.best_estimator_, 'models/tuned_model.pkl')

# ==================== PREDICTIONS PAGE ====================
elif page == "🎯 Predictions":
    st.markdown("<p class='big-font'>🎯 Make Predictions</p>", unsafe_allow_html=True)
    
    if st.session_state.best_model is None:
        st.warning("⚠️ Please train a model first")
    else:
        model_choice = st.radio("Select Model", ["Best Model", "Tuned Model (if available)"])
        
        if model_choice == "Tuned Model (if available)" and st.session_state.tuned_model is not None:
            model = st.session_state.tuned_model
            st.success("✅ Using tuned model")
        else:
            model = st.session_state.best_model
            st.info("ℹ️ Using best model")
        
        st.markdown("### Enter Feature Values")
        feature_cols = st.session_state.data.drop('target', axis=1).columns
        
        cols = st.columns(4)
        input_data = {}
        for i, col in enumerate(feature_cols):
            with cols[i % 4]:
                input_data[col] = st.number_input(f"{col}", value=float(st.session_state.data[col].mean()), step=0.1)
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            input_scaled = st.session_state.scaler.transform(input_df)
            
            prediction = model.predict(input_scaled)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
            else:
                proba = [0.5, 0.5]
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div style='background:#FF6B6B; padding:20px; border-radius:10px; text-align:center;'>
                        <h2 style='color:white;'>⚠️ Positive</h2>
                        <p style='color:white;'>Class 1</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background:#4CAF50; padding:20px; border-radius:10px; text-align:center;'>
                        <h2 style='color:white;'>✅ Negative</h2>
                        <p style='color:white;'>Class 0</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Prediction", "Positive" if prediction == 1 else "Negative")
            
            with col3:
                st.metric("Confidence", f"{proba.max():.2%}")
            
            st.markdown("#### Prediction Probabilities")
            proba_df = pd.DataFrame({'Class': ['Negative (0)', 'Positive (1)'], 'Probability': proba})
            st.dataframe(proba_df.style.background_gradient(cmap='Blues'))