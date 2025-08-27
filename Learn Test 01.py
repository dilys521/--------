### GRUBBS TEST
import numpy as np
from scipy import cluster
import scipy.stats as stats  
x=np.array([12,13,14,19,21,23])
y=np.array([12,13,14,19,21,23,45])

def grubbs_test(x):
    n=len(x)
    mean_x=np.mean(x)
    sd_x=np.std(x)
# 这行代码求的是数组 x 中每个元素与数组 x 的均值 mean_x 差值的绝对值的最大值，通常用于 Grubbs 检验中计算分子部分。
    numerator = np.max(np.abs(x - mean_x))
    g_calcluated=numerator/sd_x
    print("Grubbs Calculated Value:",g_calcluated)

# 这行代码的作用是使用 SciPy 库中的 stats.t.ppf 函数来计算学生 t 分布的百分位点。
# stats.t.ppf 是百分点函数（Percent Point Function），它是累积分布函数（CDF）的反函数。
# 第一个参数 1 - 0.05 / (2 * n) 表示显著性水平，这里 0.05 是常见的显著性水平，除以 2 * n 是为了进行双侧检验的调整。
# 第二个参数 n - 2 是学生 t 分布的自由度，在 Grubbs 检验中通常取样本数量 n 减去 2。
# 计算得到的 t_value 会用于后续 Grubbs 检验临界值的计算。
    t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_critical=((n-1)*np.sqrt(np.square(t_value)))/(np.sqrt(n)*np.sqrt(n-2+np.square(t_value)))
    print("Grubbs Critical Value:",g_critical)
    if g_calcluated>g_critical:
        print("Outlier Detected")
    else:
        print("No Outlier Detected")
grubbs_test(x)
grubbs_test(y)

# 这个是两个样本，每个样本自己去计算异常值，其中45距离其他样本的距离比较远，所以算是理解上的异常值的点

###Z-Score
import pandas as pd
train=pd.read_csv("./house-prices-advanced-regression-techniques/train.csv")
out=[]
def Zscore_outlier(df):
    m=np.mean(df)
    sd=np.std(df)
    for i in df:
        z=(i-m)/sd
        if np.abs(z)>3:
            out.append(i)
    print("Outliers:",out)
Zscore_outlier(train["LotArea"])


###Robust Z-score
train=pd.read_csv("./house-prices-advanced-regression-techniques/train.csv")
out=[]
def ZRscore_outlier(df):
    med=np.median(df)
# 这行代码的意思是使用 SciPy 库中的 median_absolute_deviation 函数来计算输入数据 df 的中位数绝对偏差（Median Absolute Deviation, MAD）。
# 中位数绝对偏差是一种衡量数据离散程度的稳健统计量，计算公式为 MAD = median(|Xi - median(X)|)，其中 Xi 是数据集中的每个观测值，median(X) 是数据集的中位数。
# 计算得到的中位数绝对偏差会存储在变量 ma 中，后续可能会用于基于稳健 Z 分数的异常值检测。
    ma=stats.median_abs_deviation(df)
    for i in df:
        z=(0.675*(i-med))/(np.median(ma))
        if np.abs(z)>3:
            out.append(i)
    print("Outliers:",out)
ZRscore_outlier(train["LotArea"])

###IQR METHOD
out=[]
def iqr_outliers(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    for i in df:
        if i<lower or i>upper:
            out.append(i)
    print("Outliers:",out)
iqr_outliers(train["LotArea"])


### 截尾处理
train=pd.read_csv("./titanic/train.csv")
out=[]
def Winsorization_outliers(df):
    q1=np.percentile(df,1)
    q3=np.percentile(df,99)
    for i in df:
        if i>q3 or i<q1:
            out.append(i)
    print("Outliers:",out)
Winsorization_outliers(train["Fare"])


###DBSCAN
from sklearn.cluster import DBSCAN
train=pd.read_csv("./titanic/train.csv")
def DB_outliers(df):
    outlier_detection=DBSCAN(eps=2,metric='euclidean',min_samples=5)
    clusters=outlier_detection.fit_predict(df.values.reshape(-1,1))
    data=pd.DataFrame()
    data['cluster']=clusters
    print(data['cluster'].value_counts().sort_values(ascending=False))
DB_outliers(train["Fare"])
       
### 孤立森林
from sklearn.ensemble import IsolationForest
train=pd.read_csv("./titanic/train.csv")
train["Fare"].fillna(train[train.Pclass==3]["Fare"].median(),inplace=True)
def Iso_outliers(df):
    iso=IsolationForest(random_state=42,contamination='auto')
    preds=iso.fit_predict(df.values.reshape(-1,1))
    data=pd.DataFrame()
    data['cluster']=preds
    print(data['cluster'].value_counts().sort_values(ascending=False))
Iso_outliers(train["Fare"])

### 可视化
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
train = pd.read_csv('./titanic/train.csv')
def Box_plots(df):
    plt.figure(figsize=(10, 4))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()
Box_plots(train['Age'])

def hist_plots(df):
    plt.figure(figsize=(10, 4))
    plt.hist(df)
    plt.title("Histogram Plot")
    plt.show()
hist_plots(train['Age'])

def scatter_plots(df1,df2):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(df1,df2)
    ax.set_xlabel('Age')
    ax.set_ylabel('Fare')
    plt.title("Scatter Plot")
    plt.show()
scatter_plots(train['Age'],train['Fare'])

def dist_plots(df):
    plt.figure(figsize=(10, 4))
    sns.distplot(df)
    plt.title("Distribution plot")
    sns.despine()
    plt.show()
dist_plots(train['Fare'])

def qq_plots(df):
    plt.figure(figsize=(10, 4))
    qqplot(df,line='s')
    plt.title("Normal QQPlot")
    plt.show()
qq_plots(train['Fare'])

 




