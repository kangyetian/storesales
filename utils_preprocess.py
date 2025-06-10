from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# pd.set_option('display.max_columns', None)  # Show all columns
# import warnings
# warnings.filterwarnings("ignore")
# df.explode('col'):reverse of agg

########### plot target ###########

# hist
from scipy.stats import norm
def hist_plot(series):
    # histogram with kde
    sns.histplot(series, bins=100, kde=True, stat="density", alpha=0.4)
    # normal 
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, loc=np.mean(series), scale=np.std(series))

    plt.plot(x, p, 'g', linewidth=2, label='Normal distribution fit')
    plt.legend()
    plt.title("Histogram with KDE and Normal Fit")
    plt.show()

# box
def box_plot_px(df, col):
    p = px.box(data_frame=df,  y=col)
    p.show()

def box_plot_sns(df, col):
    #box
    plt.figure(figsize=(5, 4))
    sns.boxplot(df[col])
    plt.show()
    

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


########### plot features ###########
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

########### outliers detection ###########
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

# detect extremely imbalanced features (categorical)
def detect_imbalanced_features(df, threshold=0.9):
    cols=[]
    for col in df.columns:
        most_common_value = df[col].value_counts(dropna=False).idxmax()
        count = df[col].value_counts(dropna=False).max()
        
        if count > len(df) * threshold:
            print(f"Column '{col}': value '{most_common_value}' appears {count} times out of {len(df)}, ({count/len(df):.1%})")
            cols.append(col)
    return cols

########### missing vals ###########
# impute
from sklearn.impute import KNNImputer
def impute_knn(train_features, test, features_to_impute, num_columns=5, num_neighbors=5):
    for feat in features_to_impute:
        # use the 5 most correlated features to impute
        corrs = train_features.corr()[feat].sort_values(key=abs, ascending=False)
        imputecols = corrs[:num_columns+1].index.tolist()
        knnImputer = KNNImputer(n_neighbors=num_neighbors)
        knnImputer.fit(train_features[imputecols])
        train_features[feat] = knnImputer.transform(train_features[imputecols])[:, 0]
        test[feat] = knnImputer.transform(test[imputecols])[:, 0]
    return train_features, test

# linear interpolate
# oil = oil.set_index('date').reindex(all_dates).interpolate(method='linear').reset_index()


########### encoding for cat/nominal feat ###########
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

# not one hot-> simply map to integer
def encode_categorical_features(train, test, cat_columns):
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    cat_encoder.fit(train[cat_columns])
    train[cat_columns] = cat_encoder.transform(train[cat_columns])
    test[cat_columns] = cat_encoder.transform(test[cat_columns])
    return train, test, cat_encoder

def print_ordinalEncoder_relationship(encoder, df):
    # Print the mapping
    for i, categories in enumerate(encoder.categories_):
        print(f"Feature {df.columns[i]}:")
        for idx, category in enumerate(categories):
            print(f"  {category} --> {idx}")

def decode_ordinalEncoder(encoder, df_encoded):
    decoded = encoder.inverse_transform(df_encoded)
    return decoded


# standardize 
from sklearn.preprocessing import StandardScaler
def standardize_num_columns(train, test, num_columns):
    scaler = StandardScaler()
    train[num_columns] = scaler.fit_transform(train[num_columns])
    test[num_columns] = scaler.transform(test[num_columns])
    return train, test

#### train_test_split
from sklearn.model_selection import train_test_split
def split_traintest(X, y, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, shuffle=shuffle)
def split_traintest(X, y, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)