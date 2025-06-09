from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import seaborn as sns


####### metrics ########
def performance(y_test, y_pred):
    # Calcul des métriques de performance
    r2 = r2_score(y_test, y_pred)  # R² (proportion de variance expliquée)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Erreur quadratique moyenne
    mae = mean_absolute_error(y_test, y_pred)  # Erreur absolue moyenne

    print(f'\nR² Score: {r2:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    return r2,rmse,mae

def performance_both(y_train, y_train_pred, y_test, y_pred):
    print('\ntraining performance')
    r2_train,rmse_train,mae_train = performance(y_train, y_train_pred)
    print('\ntest performance')
    r2_test,rmse_test,mae_test = performance(y_test, y_pred)
    return (r2_train,rmse_train,mae_train),(r2_test,rmse_test,mae_test)

def plot_pred_vs_actual(y_pred, y_actual):
    plt.scatter(y_pred, y_actual)
    xrange = np.linspace(min(y_actual), max(y_actual), 1000)
    plt.plot(xrange, xrange, color='red', linestyle='--', label='45° Line (y = x)')
    plt.xlabel('pred')
    plt.ylabel('actual')

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42)


######## models #########
# Linear Regression
from sklearn.linear_model import LinearRegression
def lr(X_train_1, X_test_1, y_train_1, y_test_1, X_test_real_1, fit_intercept=True, weights=None):
    if weights is not None:
        linear_all = LinearRegression(fit_intercept=fit_intercept).fit(X_train_1, y_train_1,  sample_weight=weights)
    else:
        linear_all = LinearRegression(fit_intercept=fit_intercept).fit(X_train_1, y_train_1)

    y_pred_train_linear = linear_all.predict(X_train_1)
    y_pred_linear = linear_all.predict(X_test_1)
    y_pred_test_real = linear_all.predict(X_test_real_1)
    (r2_train,rmse_train,mae_train),(r2_test,rmse_test,mae_test) = performance_both(y_train_1, y_pred_train_linear, y_test_1, y_pred_linear)
    plt.figure(figsize=(5,3))
    plt.scatter(y_pred_linear,y_test_1- y_pred_linear)
    plt.xlabel('y')
    plt.ylabel('resid')
    y_fit_train = pd.DataFrame(y_pred_train_linear, index=X_train_1.index, columns=y_train_1.columns)
    y_pred_test = pd.DataFrame(y_pred_linear, index=X_test_1.index, columns=y_test_1.columns)
    y_pred_test_real =  pd.DataFrame(y_pred_test_real, index=X_test_real_1.index, columns=y_test_1.columns)
    return y_fit_train, y_pred_test, y_pred_test_real

# Feature forward Selection using aic
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def aic_lr(n,k,mse):
    """
    Compute the Akaike Information Criterion (AIC) for a linear regression model.
    """
    log_likelihood = -n/2 * np.log(2 * np.pi * mse) - n/2
    return -2 * log_likelihood + 2 * k

def forward_select_features_aic(X_train, y_train, X_test, y_test):
    selected_features_aic = []
    remaining_features = list(X_train.columns)
    best_score = float('inf')
    n = len(X_train)
    while remaining_features:
        print('len(selected_features_aic)', len(selected_features_aic), best_score)
        best_new_score, best_candidate = float('inf'), None
        for candidate in remaining_features:
            features_to_try = selected_features_aic + [candidate]
            model = LinearRegression().fit(X_train[features_to_try], y_train)
            y_pred = model.predict(X_test[features_to_try])
            mse = mean_squared_error(y_test, y_pred)
            aic=aic_lr(n, len(features_to_try), mse)
            if aic < best_new_score:
                best_new_score = aic
                best_candidate = candidate
        if best_candidate is not None and best_new_score < best_score:
            selected_features_aic.append(best_candidate)
            remaining_features.remove(best_candidate)
            best_score = best_new_score
        else:
            break
    return selected_features_aic

# WLS
def wls(errors,X_train, y_train, X_test, y_test):

    # Calculate absolute errors from previous predictions
    # errors = np.abs(y_train.values.flatten() - model_lr_aic.predict(X_train[selected_features_aic]).flatten())
    weights = 1 / (errors + 1e-6)  # Add small constant to avoid division by zero

    # Fit WLS using sample_weight (no feature selection but use the )
    wls_sklearn = LinearRegression()
    wls_sklearn.fit(X_train, y_train, sample_weight=weights)

    # Predict on test set
    y_pred_wls_sklearn = wls_sklearn.predict(X_test)

    # Evaluate performance
    (r2_train,rmse_train,mae_train), (r2_test,rmse_test,mae_test) = performance_both(y_train, wls_sklearn.predict(X_train), y_test, y_pred_wls_sklearn)
    return wls_sklearn

# Lasso
from sklearn.linear_model import LassoCV
def lasso_plot(lasso):
    alpha_range = lasso.alphas_
    mse_values = lasso.mse_path_.mean(axis = 1)
    sd_values = lasso.mse_path_.std(axis = 1)

    plt.plot(np.log(alpha_range), mse_values, "k-o")
    plt.plot(np.log(alpha_range), mse_values + sd_values, "r--")
    plt.plot(np.log(alpha_range), mse_values - sd_values, "r--")
    plt.fill_between(np.log(alpha_range),
                    mse_values + sd_values,
                    mse_values - sd_values, alpha = .2)
    print("alpha (regularization):", lasso.alpha_) #regularization
    plt.plot([np.log(lasso.alpha_)] * 2, [0, (mse_values + sd_values).max()])  # 
    plt.xlabel(r"$log(\alpha)$")
    plt.ylabel("f{n_folds}-Fold CV MSE")
    plt.show()


def lasso_model(X_train, y_train, X_test, y_test, n_folds = 5):
    n_folds = 5
    lasso = LassoCV(n_alphas = 100, cv = n_folds)
    lasso.fit(X_train, y_train)
    lasso_plot(lasso)

    coef = pd.DataFrame(lasso.coef_, index=X_train.columns, columns=['betas'])
    selected_cols_lasso = coef[coef["betas"] != 0].index.tolist()
    print(len(selected_cols_lasso), selected_cols_lasso)

    y_pred_lasso = lasso.predict(X_test)
    r2,rmse,mae = performance(y_test, y_pred_lasso)
    return y_pred_lasso,lasso

# Dtree
from sklearn.tree import DecisionTreeRegressor
def dtree(X_train_1, X_test_1, y_train_1, y_test_1,X_test_real_1):
    dtree_all = DecisionTreeRegressor(max_depth=5).fit(X_train_1, y_train_1)
    y_pred_all = dtree_all.predict(X_test_1)
    y_pred_all_train = dtree_all.predict(X_train_1)
    (r2_train,rmse_train,mae_train), (r2_test,rmse_test,mae_test) = performance_both(y_train_1, y_pred_all_train, y_test_1, y_pred_all)

    y_fit_train = pd.DataFrame(y_pred_all_train, index=X_train_1.index)
    y_pred_test = pd.DataFrame(y_pred_all, index=X_test_1.index)
    plot_pred_vs_actual(y_pred_all_train, y_train_1.iloc[:,0])

    dtree_importance = pd.DataFrame(dtree_all.feature_importances_, 
                                    index=X_train_1.columns, 
                                    columns=['importance']).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.title("Decision Tree Feature Importances")
    plt.barh(dtree_importance.index[:30], dtree_importance['importance'][:30])
    plt.gca().invert_yaxis()
    plt.show()
    y_pred_test_real = dtree_all.predict(X_test_real_1)
    y_pred_test_real =  pd.DataFrame(y_pred_test_real, index=X_test_real_1.index)
    return y_fit_train, y_pred_test, y_pred_test_real


from sklearn.ensemble import RandomForestRegressor
def rf(X_train_1, X_test_1, y_train_1, y_test_1, X_test_real_1,max_depth=7):
    rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
    rf.fit(X_train_1, y_train_1)
    y_pred_rf = pd.DataFrame(rf.predict(X_test_1),index=X_test_1.index)
    y_fit_rf = pd.DataFrame(rf.predict(X_train_1), index=X_train_1.index)
    (r2_train,rmse_train,mae_train), (r2_test,rmse_test,mae_test) = performance_both(y_train_1, y_fit_rf, y_test_1, y_pred_rf)

    rf_importances = pd.DataFrame(rf.feature_importances_, 
                                  index=X_train_1.columns, 
                                  columns=['importance']).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.title("Random Forest Feature Importances")
    plt.barh(rf_importances.index[:30], rf_importances['importance'][:30])
    plt.gca().invert_yaxis()
    plt.show()
    y_pred_test_real = rf.predict(X_test_real_1)
    y_pred_test_real =  pd.DataFrame(y_pred_test_real, index=X_test_real_1.index)
    return y_fit_rf, y_pred_rf, y_pred_test_real


from xgboost.sklearn import XGBRegressor
def xgb_predictor(X_train_1, X_test_1, y_train_1, y_test_1,X_test_real_1):
    xgb_sklearn = XGBRegressor(max_depth= 6, # How deep should each tree be?
            eta= 0.1, # The learning rate: how big are our steps?
            verbosity= 0, # Whether to print out messages as you fit. 1 makes it quiet
            objective='reg:squarederror', # What is our loss function? For now, logistic - outputs probability. Would change for multiclass
            gamma= 1, # Minimum loss reduction
            subsample= 1, # How much to subsample for each tree. Smaller number (like 0.5) would subsample
            colsample_bytree=1, # like the random forest restriction on variables, sample of columns to get
            lam=0, # How much L2 penalty on the node weights
            alpha=0, # How much L1 penalty on the node weights
            eval_metric='mae', # this is the default eval_metric for logistic 
            min_child_weight = 1,# Minimum sum of instance weights (hessian) in a child. 
            seed= 1).fit(X_train_1, y_train_1)

    y_pred_all = xgb_sklearn.predict(X_test_1)
    y_pred_all_train = xgb_sklearn.predict(X_train_1)
    (r2_train,rmse_train,mae_train), (r2_test,rmse_test,mae_test) = performance_both(y_train_1, y_pred_all_train, y_test_1, y_pred_all)

    y_fit_train = pd.DataFrame(y_pred_all_train, index=X_train_1.index)
    y_pred_test = pd.DataFrame(y_pred_all, index=X_test_1.index)
    plot_pred_vs_actual(y_pred_all_train, y_train_1['sales'])

    dtree_importance = pd.DataFrame(xgb_sklearn.feature_importances_, 
                                    index=X_train_1.columns, 
                                    columns=['importance']).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.title("XGBoost Feature Importances")
    plt.barh(dtree_importance.index[:30], dtree_importance['importance'][:30])
    plt.gca().invert_yaxis()
    plt.show()
    y_pred_test_real = xgb_sklearn.predict(X_test_real_1)
    y_pred_test_real =  pd.DataFrame(y_pred_test_real, index=X_test_real_1.index)
    return y_fit_train, y_pred_test, y_pred_test_real

from sklearn.model_selection import GridSearchCV 
def gridSearch(xgb_sklearn, X_train, X_test, y_train, y_test, X_test_real):
    param_grid1 = {
        'max_depth': range(2, 10, 2),
        'min_child_weight': range(1, 6, 2),
        'gamma': np.arange(0, 1, 0.5),
        'subsample': np.arange(0,1,0.3)

    }

    grid_search = GridSearchCV(estimator = xgb_sklearn, param_grid = param_grid1, 
                            scoring = 'neg_mean_squared_error', cv = 5).fit(X_train, y_train)
    grid_search.best_params_
    best_xgb_sklearn = grid_search.best_estimator_
    y_pred_xgb_sklearn = best_xgb_sklearn.predict(X_test)
    y_pred_xgb_sklearn_train = best_xgb_sklearn.predict(X_train)
    y_pred_test_real = xgb_sklearn.predict(X_test_real)
    (r2_train,rmse_train,mae_train), (r2_test,rmse_test,mae_test) = performance_both(y_train, y_pred_xgb_sklearn_train, y_test, y_pred_xgb_sklearn)
    return y_pred_xgb_sklearn_train, y_pred_xgb_sklearn,y_pred_test_real
