# -*- coding: utf-8 -*-
"""
Carseats 多元线性回归与 VIF 分析

依赖安装：
    pip install ISLP pandas statsmodels

运行：
    python carseats_regression.py
"""

from ISLP import load_data
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

def main():
    # 1. 加载数据集
    Carseats = load_data('Carseats')

    # 2. 建立多元线性回归模型
    model = smf.ols('Sales ~ Price + Income + Advertising + ShelveLoc', data=Carseats).fit()

    # 3. 提取模型拟合报告
    print("================ 模型拟合报告 ================")
    print(model.summary())

    # 4. 计算 VIF 评估多重共线性
    # 获取模型的设计矩阵（不包含截距项）
    X = model.model.exog[:, 1:]
    feature_names = model.model.exog_names[1:]

    # 创建 DataFrame 来存储 VIF 结果
    vif_data = pd.DataFrame()
    vif_data["Variable"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

    print("\n================ VIF 计算结果 ================")
    print(vif_data)

if __name__ == "__main__":
    main()


# 由数据得到：
# 1.基准组是Bad，因为在系数表中只显示了ShelveLoc[T.Good] 和 ShelveLoc[T.Medium]，说明Bad被作为参照标准
# 2. ShelveLoc [Good] 系数的实际商业含义：在控制价格（Price）、收入（Income）和广告投入（Advertising）保持不变的情况下，
# 与货架位置差（Bad）的商店相比，货架位置好（Good）的商店平均能多卖出 4.8359 千（即 4835.9 个）汽车座椅。
# 3.VIF的均值远小于5，因此不存在多重共线性风险。