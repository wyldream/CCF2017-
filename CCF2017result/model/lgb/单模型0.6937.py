# -*- coding: utf-8 -*-
__title__ = 'byes'
__author__ = 'JieYuan'
__mtime__ = '2017/12/12'

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('all_data.csv')  # ÊØÍûÕß413ÌØÕ÷£¨train+test£©
df.drop('ENDDATE', 1, inplace=True)
df.ALTDIFF = df.ALTDIFF.replace(-np.inf, -9999)
df.fillna(-9999, inplace=True)
data = df.copy() \
    .assign(EID=lambda x: x.EID.astype(str)) \
    .assign(_num_EID=lambda x: x.EID.apply(lambda x: int(''.join(filter(str.isdigit, x))))) \
    .assign(_encode_EID=lambda x: LabelEncoder().fit_transform(x.EID)) \
    .assign(_dup_EID=lambda x: x._num_EID.duplicated(False).astype(int)) \
    .assign(_rank_EID=lambda x: x._num_EID.rank())

train = data[data.TARGET != -9999]
test = data[data.TARGET == -9999]
X = train.drop(['EID', 'TARGET'], 1).values
y = train.TARGET.values.ravel()

clf = LGBMClassifier(boosting_type='gbdt',
                     objective='binary',
                     max_depth=-1,
                     learning_rate=0.01,
                     n_estimators=2000,
                     subsample=0.6,
                     colsample_bytree=0.6,
                     reg_alpha=5.39,
                     reg_lambda=10,
                     num_leaves=2 ** 6,
                     min_child_weight=10,
                     min_split_gain=0.05,
                     scale_pos_weight=1,
                     random_state=999,
                     n_jobs=-1)

clf.fit(X, y)


def get_res(clf, path='cv.07470.csv'):
    res = clf.predict_proba(test.drop(['EID', 'TARGET'], 1))[:, 1]
    test[['EID']].assign(FORTARGET=0, PROB=res).to_csv(path, index=False)


if __name__ == '__main__':
    get_res(clf)
