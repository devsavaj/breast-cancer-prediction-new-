import shap
import matplotlib.pyplot as plt
import os

def shap_explain_ml(models, X_train, X_test, feature_names):
    if not os.path.exists("outputs/figures"):
        os.makedirs("outputs/figures")
    
    for name, model in models.items():
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, show=False)
            plt.savefig(f'outputs/figures/{name}_shap_summary.png')
            plt.close()
        except:
            print(f"SHAP explanation not supported for {name}")
