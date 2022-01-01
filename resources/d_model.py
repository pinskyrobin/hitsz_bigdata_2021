# -*- coding: UTF-8 -*-
# 为方便测试，请统一使用 numpy、pandas、sklearn 三种包，如果实在有特殊需求，请单独跟助教沟通
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import argparse

# 设定随机数种子，保证代码结果可复现
np.random.seed(1024)


class Model:
    """
    要求：
        1. 需要有__init__、train、predict三个方法，且方法的参数应与此样例相同
        2. 需要有self.X_train、self.y_train、self.X_test三个实例变量，请注意大小写
        3. 如果划分出验证集，请将实例变量命名为self.X_valid、self.y_valid
    """

    # 模型初始化，数据预处理，仅为示例
    def __init__(self, train_path, test_path):
        df_train = pd.read_csv(train_path, encoding='gbk', index_col='id')
        df_test = pd.read_csv(test_path, encoding='gbk', index_col='id')
        self.y_train = df_train['血糖'].values

        self.__data_preprocess(df_train)
        self.__data_preprocess(df_test)

        self.__nan_analysis(df_train)
        self.__corr_analysis(df_train)

        df_train = self.__delete_nan(df_train)

        self.__drop_col(df_train, "血糖")
        self.__drop_col(df_test, "血糖")

        data_preprocessing = SimpleImputer(strategy='mean')
        self.X_train = data_preprocessing.fit_transform(df_train.values)
        self.X_test = data_preprocessing.transform(df_test.values)

        self.regression_model = ensemble.RandomForestRegressor(n_estimators=20)
        self.df_predict = pd.DataFrame(index=df_test.index)

    # 模型训练，输出训练集MSE
    def train(self):
        self.regression_model.fit(self.X_train, self.y_train)
        y_train_pred = self.regression_model.predict(self.X_train)
        return mean_squared_error(self.y_train, y_train_pred)

    # 模型测试，输出测试集预测结果，要求此结果为DataFrame格式，可以通过to_csv方法保存为Kaggle的提交文件
    def predict(self):
        y_test_pred = self.regression_model.predict(self.X_test)
        self.df_predict['Predicted'] = y_test_pred
        self.df_predict.to_csv("rgs_predicted.csv")
        return self.df_predict

    def __data_preprocess(self, df):
        gender_mapper = {'男': 1, '女': 0}
        df['性别'] = df['性别'].map(gender_mapper)

        # 特征降维
        col_list = ["血小板体积分布宽度", "单核细胞%", "乙肝表面抗原", "白球比例",
                    "乙肝e抗体", "乙肝e抗原", "乙肝表面抗体", "乙肝核心抗体", "体检日期",
                    "白蛋白", "嗜酸细胞%", "肌酐", "血小板比积", "红细胞平均血红蛋白浓度",
                    "*总蛋白", "嗜碱细胞%", "血小板计数", "红细胞平均体积", "红细胞平均血红蛋白量",
                    "红细胞体积分布宽度", "*球蛋白", "*碱性磷酸酶", "*r-谷氨酰基转换酶",
                    "*丙氨酸氨基转换酶", "甘油三酯", "*天门冬氨酸氨基转换酶"]

        for col_name in col_list:
            self.__drop_col(df, col_name)

    # 缺省数据分析
    def __nan_analysis(self, df):
        df.isna().sum().to_csv("rgs_nan_col.csv")
        df.isna().sum(axis=1).to_csv("rgs_nan_row.csv")

    # 删除缺省数据
    def __delete_nan(self, df):
        has_val_list = df.shape[1] - df.apply(lambda x: x.count(), axis=1).values

        # 针对行数据的删除
        delete_row_list = []
        for i in range(len(has_val_list)):
            if has_val_list[i] >= 15:
                delete_row_list.append(i)

        # 针对列数据的删除
        df.drop(delete_row_list, axis=0, inplace=True)
        self.y_train = np.delete(self.y_train, delete_row_list, axis=0)

        # 删除过大的血糖数据,threshold=20
        self.y_train = np.delete(self.y_train, df[df['血糖'] > 20].index.values, axis=0)
        df = df.drop(df[df['血糖'] > 20].index)

        return df

    # 适用于DataFrame列的删除函数,封装了重复而必要的参数
    def __drop_col(self, df, col_name):
        df.drop(labels=col_name, axis=1, inplace=True)

    # 对DataFrame数据进行相关性分析,采用kendall相关系数
    def __corr_analysis(self, df):
        df.corr(method='kendall').to_csv("rgs_corr.csv")


# 以下部分请勿改动！
if __name__ == '__main__':
    # 解析输入参数。在终端执行以下语句即可运行此代码： python d_model.py --train_path "d_train.csv" --test_path "d_test.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="d_train.csv", help="path to train dataset")
    parser.add_argument("--test_path", type=str, default="d_test.csv", help="path to test dataset")
    opt = parser.parse_args()

    model = Model(opt.train_path, opt.test_path)
    print('训练集维度:{}\n测试集维度:{}'.format(model.X_train.shape, model.X_test.shape))
    MSE_train = model.train()
    print('MSE_train={:.6f}'.format(MSE_train))

    d_predict = model.predict()
    d_predict.to_csv('d_predict.csv')
