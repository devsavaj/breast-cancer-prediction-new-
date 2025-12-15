from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

def train_ml_models(X_train, X_test, y_train, y_test):
    predictions = {}
    models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    predictions['Logistic Regression'] = lr.predict(X_test)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    params_rf = {'n_estimators':[100,150], 'max_depth':[None,10,20]}
    grid_rf = GridSearchCV(rf, params_rf, cv=cv, scoring='accuracy')
    grid_rf.fit(X_train, y_train)
    models['Random Forest'] = grid_rf.best_estimator_
    predictions['Random Forest'] = grid_rf.predict(X_test)

    # SVM
    svc = SVC(probability=True)
    params_svc = {'C':[1,10], 'kernel':['linear','rbf']}
    grid_svc = GridSearchCV(svc, params_svc, cv=cv, scoring='accuracy')
    grid_svc.fit(X_train, y_train)
    models['SVM'] = grid_svc.best_estimator_
    predictions['SVM'] = grid_svc.predict(X_test)

    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_model.predict(X_test)

    # Voting Classifier Ensemble
    ensemble = VotingClassifier(estimators=[
        ('lr', lr), ('rf', grid_rf.best_estimator_), ('svc', grid_svc.best_estimator_), ('xgb', xgb_model)
    ], voting='soft')
    ensemble.fit(X_train, y_train)
    models['Voting Ensemble'] = ensemble
    predictions['Voting Ensemble'] = ensemble.predict(X_test)

    return predictions, models
