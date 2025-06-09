from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pd.set_option('display.max_columns', None)  # Show all columns
# import warnings
# warnings.filterwarnings("ignore")
# df.explode('col'):reverse of agg

# seperate categorical and numerical features
def seperate_cat_num(df):
    # use a rough seperation
    cat_columns = df.select_dtypes(include = ['object']).columns
    num_columns = df.select_dtypes(include = ['float64', 'int64']).columns
    return cat_columns, num_columns

# plot numerical col
def plot_numeric_columns(df,num_columns):
    df[num_columns].hist(figsize=(10, 5), bins=50, xlabelsize=8, ylabelsize=8)
    plt.show()

def plot_categorical_columns(df,cat_columns):
    # Plot all categorical columns
    if len(cat_columns)>=4:
        m,n=4, int(np.ceil(len(cat_columns)/4))
        fig, axes = plt.subplots(m,n, figsize=(28, 24))
        axes = axes.flatten()
    else:
        m,n= 2, len(cat_columns) # a dummy row
        fig, axes = plt.subplots(m,n, figsize=(28, 24))
    for i, col in enumerate(cat_columns):
        df[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

# zscore method to detect outliers for col {num_columns}, and display with col {ycol}
def outliers_zscore(df, num_columns, ycols= ['SalePrice'], zscore_cutoff=10):
    # display ycols
    for col in ycols:
        df[col].hist(figsize=(10,2), bins=10)
    plt.show()

    # display outliers
    for col in num_columns:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = df[(z_scores.abs() > zscore_cutoff)]
        if not outliers.empty:
            print(f"Column '{col}' has {len(outliers)} outliers.")
            display(outliers[[col]+ ycols])
            df[col].hist(figsize=(5,2), bins=10)
            plt.show()
    return plt



def OHE(X_train_1, X_test_1, X_test_real_1, cat_columns = ['family']):
    OH = OneHotEncoder(sparse_output=False, handle_unknown='ignore',drop = 'first')
    X_train_1_oh = pd.DataFrame(OH.fit_transform(X_train_1[cat_columns]), 
                            columns=OH.get_feature_names_out(cat_columns), index=X_train_1.index)
    X_train_1 =pd.concat([X_train_1.drop(columns=cat_columns),X_train_1_oh], axis=1)

    X_test_1_oh = pd.DataFrame(OH.transform(X_test_1[cat_columns]), 
                            columns=OH.get_feature_names_out(cat_columns), index=X_test_1.index)
    X_test_1 = pd.concat([X_test_1.drop(columns=cat_columns), X_test_1_oh], axis=1)

    X_test_real_1_oh = pd.DataFrame(OH.transform(X_test_real_1[cat_columns]), 
                            columns=OH.get_feature_names_out(cat_columns), index=X_test_real_1.index)
    X_test_real_1 = pd.concat([X_test_real_1.drop(columns=cat_columns), X_test_real_1_oh], axis=1)
    return X_train_1, X_test_1, X_test_real_1

from sklearn.preprocessing import OrdinalEncoder
# Ordinal_map = {
#     'ExterQual' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
#     'ExterCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],}
def encode_ordinal_features(train, test, ordinal_map, ord_columns):
    # order the ordinal columns
    ordered_categories=list(map(lambda x: ordinal_map[x], ord_columns))
    # Encode ordinal 
    ord_encoder = OrdinalEncoder(categories = ordered_categories,
                                  handle_unknown='use_encoded_value', 
                                  unknown_value=np.nan).fit(train[ord_columns])
    train[ord_columns] = ord_encoder.transform(train[ord_columns])
    test[ord_columns] = ord_encoder.transform(test[ord_columns])
    return train, test,ord_encoder

# qqplot
import scipy.stats as stats

def qqplot(series):
    # Create a Q-Q plot
    plt.figure(figsize=(5, 3))
    stats.probplot(series, dist='norm', plot=plt)
    plt.title('Normal Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Ordered Values')
    plt.show()