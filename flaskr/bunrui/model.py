import sys
#print(sys.path)
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import joblib

df = pd.read_csv('flaskr/bunrui/spam.csv', encoding='latin-1')
# 未使用列を削除
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# リネーム
df.rename(columns={"v1":"class", "v2":"text"}, inplace=True)

X = pd.DataFrame(df['text'])
Y = pd.DataFrame(df['class'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=10)

# 単語の出現回数取得
vec_count = CountVectorizer(min_df=3)
vec_count.fit(X_train['text'])

# トレーニング・評価データをベクトル化
X_train_vec = vec_count.transform(X_train['text'])
X_text_vec = vec_count.transform(X_test['text'])

# マルチノビア　ベルヌーイモデル
model = BernoulliNB()
model.fit(X_train_vec, Y_train['class'])

joblib.dump(model,"model.pkl",compress=True)
joblib.dump(vec_count,"vec_count.pkl",compress=True)

print('Train accuracy = %.3f' % model.score(X_train_vec, Y_train))
print(' Test accuracy = %.3f' % model.score(X_text_vec, Y_test))