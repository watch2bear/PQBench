import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# datasets: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data
# code_notebook: https://www.kaggle.com/code/elenasm/fake-news-detection-using-tf-idf
data = pd.read_csv('/fakenews/WELFake_Dataset.csv')
data=data.dropna()
X=data[['title','text']]
y=data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

column_transformer = ColumnTransformer([(x, TfidfVectorizer(), x) for x in X_train.columns])
model1 = make_pipeline(column_transformer, MultinomialNB())
model2 = make_pipeline(column_transformer, LinearSVC())
#model = make_pipeline(column_transformer, MLPClassifier(hidden_layer_sizes=(20, 10, 10), max_iter=10, random_state=42))

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
predicted1 = model1.predict(X_test)
print(classification_report(y_test, predicted1))
print("Accuracy1:", round(accuracy_score(y_test, predicted1)*100),'%')
predicted2 = model2.predict(X_test)
print(classification_report(y_test, predicted2))
print("Accuracy2:", round(accuracy_score(y_test, predicted2)*100),'%')

from skl2onnx.common.data_types import StringTensorType
from skl2onnx import to_onnx
onx1 = to_onnx(model1, initial_types=[("title", StringTensorType([None, 1])), ("text", StringTensorType([None, 1]))])
onx2 = to_onnx(model2, initial_types=[("title", StringTensorType([None, 1])), ("text", StringTensorType([None, 1]))])
with open('/fakenews/fakenews_nb.onnx', "wb") as f:
    f.write(onx1.SerializeToString())
with open('/fakenews/fakenews_svm.onnx', "wb") as f:
    f.write(onx2.SerializeToString())
