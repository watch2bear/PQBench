import numpy as np
import pandas as pd
from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType, DoubleTensorType, StringTensorType
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# 表路径
path1 = "/home/uw1/snippets/py_onnx/expedia/data/S_listings.csv"
path2 = "/home/uw1/snippets/py_onnx/expedia/data/R1_hotels.csv"
path3 = "/home/uw1/snippets/py_onnx/expedia/data/R2_searches.csv"
# 读取csv表
S_listings = pd.read_csv(path1)
R1_hotels = pd.read_csv(path2)
R2_searches = pd.read_csv(path3)
# 连接3张表
data = pd.merge(pd.merge(S_listings, R1_hotels, how='inner'), R2_searches, how='inner')
# print(data.isnull().any())    #检测缺失值
data.dropna(inplace=True)  # 删除NaN

# 获取分类label
y = np.array(data.loc[:, 'promotion_flag'])
# 8 numerical, 20 categorical
numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd',
                     'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks',
                       'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id',
                       'visitor_location_country_id',
                       'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
                       'srch_adults_count',
                       'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
                       'random_bool']

X = data.loc[:, numerical_columns+categorical_columns]

input_columns = numerical_columns + categorical_columns

type_map = {
    "int64": Int64TensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1]),
}


init_types = [(elem, type_map[data[elem].dtype.name]) for elem in input_columns]

numerical_preprocessor = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ],
    verbose=True
)

normal_preprocessor = Pipeline(
    steps=[
        ("onehot", OneHotEncoder()),
    ],
    verbose=True
)

preprocessor = ColumnTransformer(
    [
        ("numerical", numerical_preprocessor, numerical_columns),
        ("categorical", normal_preprocessor, categorical_columns),
    ],
    verbose=True,
)

model = make_pipeline(preprocessor, LogisticRegression(solver='liblinear'))


model.fit(X, y)

print("Training done.")


model_onnx = convert_sklearn(model, initial_types=init_types)
with open("expedia.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

