import time
import numpy as np
import pandas as pd
import onnxruntime as ort

path1 = "/home/RealWorldDatasets/Expedia/S_listings.csv"
path2 = "/home/RealWorldDatasets/Expedia/R1_hotels.csv"
path3 = "/home/RealWorldDatasets/Expedia/R2_searches.csv"
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

#batch = X.iloc[: 1024, :]

size = len(data)

type_map = {
    "int64": np.int64,
    "float64": np.float32,
    "object": str,
}

#infer_batch = {
#    elem: batch[elem].to_numpy().astype(type_map[batch[elem].dtype.name]).reshape((-1, 1))
#    for elem in input_columns
#}
ort_opts = ort.SessionOptions()
#ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
ort_opts.intra_op_num_threads = 1
#ort_opts.inter_op_num_threads = 1
print(ort_opts.graph_optimization_level)

ort_sess = ort.InferenceSession('expedia.onnx', sess_options=ort_opts)

label1 = ort_sess.get_outputs()[0]
# label2 = ort_sess.get_outputs()[2]

outputs = []
elapsed = 0
cut = 500
start = time.time()
for i in range(cut):
    batch = X.iloc[int(i * (size / cut)): int((i + 1) * (size / cut)), :]
    infer_batch = {
        elem: batch[elem].to_numpy().astype(type_map[batch[elem].dtype.name]).reshape((-1, 1))
        for elem in input_columns
    }
    #start = time.process_time()
    outputs = ort_sess.run([label1.name], infer_batch)
    #finish = time.process_time()
    #print(1000 * (finish - start))
    #elapsed += (finish - start)
    #print(label1.name, outputs[0])
# print(label2.name, outputs[1])

#print(elapsed*1000, "ms")

end = time.time()
print('运算时间： {} sec'.format(end - start))
