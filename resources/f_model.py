# -*- coding: UTF-8 -*-
# 为方便测试，请统一使用 numpy、pandas、sklearn 三种包，如果实在有特殊需求，请单独跟助教沟通
import numpy as np
import pandas as pd

from sklearn.svm import SVC

from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
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
        self.y_train = df_train['label'].values

        self.__nan_analysis(df_train)
        self.__corr_analysis(df_train)

        # 特征降维
        col_list = ["身高", "孕前体重", "孕前BMI", "RBP4", "分娩时",
                    "SNP54", "SNP55", "ACEID", "SNP21", "SNP22",
                    "SNP23", "SNP49", "SNP30", "SNP9", "SNP7",
                    "SNP44", "SNP37", "SNP5", "LDLC", "SNP8",
                    "SNP19", "糖筛孕周", "ALT", "CHO", "ApoB"]

        for col_name in col_list:
            self.__drop_col(df_train, col_name)
            self.__drop_col(df_test, col_name)

        df_train = self.__delete_nan(df_train)

        # 按label分类,利用均值填补空值
        # cnt = 0
        # for index, col in df_train.iteritems():
        #     cnt += 1
        #     if cnt >= df_train.shape[1]:
        #         break
        #     df_train[[index]] = df_train[[index, "label"]].groupby("label").transform(lambda x: x.fillna(x.mean()))
        #     df_test[[index]] = df_test[[index]].transform(lambda x: x.fillna(x.mean()))

        self.__drop_col(df_train, "label")
        self.__drop_col(df_test, "label")

        self.df_predict = pd.DataFrame(index=df_test.index)
        # df_train = (df_train - df_train.min()) / (df_train.max() - df_train.min())
        #
        # df_train.var().to_csv("clf_var.csv")

        # self.X_train = df_train.values
        # self.X_test = df_test.values

        data_preprocessing = SimpleImputer(strategy='most_frequent')
        self.X_train = data_preprocessing.fit_transform(df_train.values)
        self.X_test = data_preprocessing.transform(df_test.values)

        self.classification_model = SVC(C=1, kernel='linear', shrinking=True, decision_function_shape='ovo')

    # 模型训练，输出训练集f1_score
    def train(self):
        self.classification_model.fit(self.X_train, self.y_train)
        y_train_pred = self.classification_model.predict(self.X_train)
        return f1_score(self.y_train, y_train_pred)

    # 模型测试，输出测试集预测结果，要求此结果为DataFrame格式，可以通过to_csv方法保存为Kaggle的提交文件
    def predict(self):
        y_test_pred = self.classification_model.predict(self.X_test)
        self.df_predict['Predicted'] = y_test_pred
        self.df_predict.to_csv("clf_predicted.csv")
        return self.df_predict

    # 缺省数据分析
    def __nan_analysis(self, df):
        df.isna().sum().to_csv("clf_nan_col.csv")
        df.isna().sum(axis=1).to_csv("clf_nan_row.csv")

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

        return df

    # 适用于DataFrame列的删除函数,封装了重复而必要的参数
    def __drop_col(self, df, col_name):
        df.drop(labels=col_name, axis=1, inplace=True)

    # 对DataFrame数据进行相关性分析,采用kendall相关系数
    def __corr_analysis(self, df):
        df.corr(method='kendall').to_csv("clf_corr.csv")


# 以下部分请勿改动！
if __name__ == '__main__':
    # 解析输入参数。在终端执行以下语句即可运行此代码： python f_model.py --train_path "f_train.csv" --test_path "f_test.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="f_train.csv", help="path to train dataset")
    parser.add_argument("--test_path", type=str, default="f_test.csv", help="path to test dataset")
    opt = parser.parse_args()

    model = Model(opt.train_path, opt.test_path)
    print('训练集维度:{}\n测试集维度:{}'.format(model.X_train.shape, model.X_test.shape))
    f1_score_train = model.train()
    print('f1_score_train={:.6f}'.format(f1_score_train))

    f_predict = model.predict()
    f_predict.to_csv('f_predict.csv')
