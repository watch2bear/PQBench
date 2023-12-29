import numpy as np
import pandas as pd
import onnxruntime as ort

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

X = data.loc[:, numerical_columns + categorical_columns]

input_columns = numerical_columns + categorical_columns

batch = X.iloc[: 4096, :]

type_map = {
    "int64": np.int64,
    "float64": np.float32,
    "object": str,
}

infer_batch = {
    elem: batch[elem].to_numpy().astype(type_map[batch[elem].dtype.name]).reshape((-1, 1))
    for elem in input_columns
}

ort_sess = ort.InferenceSession('expedia.onnx')

label = ort_sess.get_outputs()[0]

outputs = ort_sess.run([label.name], infer_batch)[0]

print(outputs.shape)
print(outputs)
