import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns


# trend
def get_trend(df, order=2, datecol='date'):
    res=df.copy()
    base_date = df[datecol].min()-pd.Timedelta(days=1)
    trend_1 = (df['date'] - base_date).dt.days
    for i in range(1, order+1):
        res[f'trend_{i}']= trend_1**i
    return res


# seasonality
def get_seasonality_columns(df, date_col='date'):
    df[date_col] = df[date_col].astype('datetime64[ns]')
    df['day'] = df[date_col].dt.day 
    df['month'] =  df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['dayofweek'] = df[date_col].dt.dayofweek #not iso
    year_week_day_iso  = df[date_col].dt.isocalendar().rename(columns={'day':'dayofweek_iso',
                                                                        'year':'year_iso', 
                                                                        'week':'weekofyear_iso'})
    df = pd.concat([df, year_week_day_iso], axis=1)
    return df


# Time Related Features
def create_date_features(df, date_col='date'):
    df['month'] = df[date_col].dt.month.astype("int8")
    df['dayofmonth'] = df[date_col].dt.day.astype("int8")
    df['dayofyear'] = df[date_col].dt.dayofyear.astype("int16")
    df['weekofmonth'] = (df[date_col].apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df['dayofweek'] = (df[date_col].dt.dayofweek + 1).astype("int8")
    df['year'] = df[date_col].dt.year.astype("int32")
    df["is_wknd"] = (df[date_col].dt.weekday // 4).astype("int8")
    df["quarter"] = df[date_col].dt.quarter.astype("int8")
    df['is_month_start'] = df[date_col].dt.is_month_start.astype("int8")
    df['is_month_end'] = df[date_col].dt.is_month_end.astype("int8")
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype("int8")
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype("int8")
    df['is_year_start'] = df[date_col].dt.is_year_start.astype("int8")
    df['is_year_end'] = df[date_col].dt.is_year_end.astype("int8")
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df['month'].isin([12,1,2]), 0, 1)
    df["season"] = np.where(df['month'].isin([6,7,8]), 2, df["season"])
    df["season"] = np.where(df['month'].isin([9, 10, 11]), 3, df["season"])
    df["season"] = df["season"].astype('int8')
    year_week_day_iso  = df[date_col].dt.isocalendar().rename(columns={'day':'dayofweek_iso',
                                                                        'year':'year_iso', 
                                                                        'week':'weekofyear_iso'})
    df = pd.concat([df, year_week_day_iso], axis=1)

    return df


# each datapoint is 'dayofweek'(datapoint), each line is one 'week'(same_color)
def plot_seasonality(df, target, datapoint='dayofweek', same_color='week', ax=None, title=''):
    if not ax:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=df[same_color].nunique())
    ax = sns.lineplot(
        x=datapoint,
        y=target,
        hue=same_color,
        data=df,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"{title} Seasonal Plot ({same_color}/{datapoint})")
    # labels
    for line, name in zip(ax.lines, df[same_color].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax

# each datapoint is 'month'(datapoint), each line is one 'year' same_color)
def plot_seasonality_px(df, target, datapoint='month', same_color='year',title='', markers=True):
    a = df.groupby([same_color,datapoint])[[target]].mean().reset_index()
    plot=px.line(a, x=datapoint, y = target, color=same_color, title=title, markers=markers)
    return plot

# each box will contain data every {same_color}, and seperate into categories {category}
def plot_seasonality_box(df, target, category='year', same_color='month', title="", points=False):
    if not points:
        plot = px.box(df, x=category, y=target, color = same_color, title =title)
    else:
        plot = px.box(df, x=category, y=target, color = same_color, title = title, points='all')
    plot.show()
    return plot

from scipy.signal import periodogram
# ts index is date
def plot_periodogram(ts, fs = 365, detrend='linear', ax=None):
    freqencies, spectrum = periodogram(
        ts,
        fs=fs, # how many datapoints in a cycle 
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple") # step plot
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


## lag
# only used in ts models (not available for the whole dataset)
# the target should be unique with columns groupby + timecol

def create_lags_nogroup(df, target, timecol='date', numlag=2):
    if numlag<1:
        return df
    df = df.set_index(timecol).sort_index()
    reindexed_df= df[target]
    lags = [reindexed_df.shift(0)] + [reindexed_df.shift(i).rename(f'lag_{i}') for i in range(1, numlag+1)]
    return pd.concat(lags, axis=1)

def create_lags(df, target, groupby, timecol='date', numlag=2): 
    if numlag<1:
        return df
    df = df.set_index(groupby+[timecol]).sort_index()
    reindexed_df= df.groupby(groupby)[target]
    lags = [reindexed_df.shift(0)] + [reindexed_df.shift(i).rename(f'lag_{i}') for i in range(1, numlag+1)]
    return pd.concat(lags, axis=1)
        
import math
def lagplot(x, y=None, lag=1, standardize=False, ax=None):
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', alpha=0.75, linewidth= 1)
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax)

    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax

# x,y: pd.Series
def plot_lags(x, y=None, lags=6, nrows=1):
    ncols =math.ceil(lags / nrows)
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, nrows=nrows, ncols=ncols, figsize=(ncols*2, nrows*2) )
    for ax, k in zip(fig.get_axes(), range(nrows* ncols)):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax)
            ax.set_title(f"Lag {k + 1}")
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

import statsmodels.api as sm
def plot_acf_pacf(series,lags=365, title='' ):
    try:
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        sm.graphics.tsa.plot_acf(series, lags=lags, ax=ax[0], title = "acf "+title)
        sm.graphics.tsa.plot_pacf(series, lags=lags, ax=ax[1], title =  "pacf "+title)
        plt.show()
    except:
        pass

## arima model
from pmdarima import auto_arima
## lungbox: high p means fail to reject H0 => residuals uncorrelated => model is good
def arima(series, seasonality=7, pred_periods = 7):
    model = auto_arima(
        series,
        # with_intercept=True,
        # trend='linear', # need  with_intercept=True
        seasonal=True,
        m=seasonality,  # weekly seasonality: seasonality=7
        stepwise=True,
        trace=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion = 'aic'
    )
    display(model.summary())
    display(model.order)
    pred = pd.DataFrame(model.predict(n_periods=pred_periods))
    return pred, model