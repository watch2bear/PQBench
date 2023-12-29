import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#表路径
path1 = "/home/test/csv/RealWorldDatasets/Expedia/S_listings.csv"
path2 = "/home/test/csv/RealWorldDatasets/Expedia/R1_hotels.csv"
path3 = "/home/test/csv/RealWorldDatasets/Expedia/R2_searches.csv"
#读取csv表
S_listings = pd.read_csv(path1)
R1_hotels = pd.read_csv(path2)
R2_searches = pd.read_csv(path3)
#连接3张表
data = pd.merge(pd.merge(S_listings, R1_hotels, how = 'inner'), R2_searches, how = 'inner')
#print(data.isnull().any())    #检测缺失值
data.dropna(inplace=True)      #删除NaN

#获取分类label
y = np.array(data.loc[:, 'promotion_flag'])
#8 numerical, 20 categorical
numerical = np.array(data.loc[:, ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd', 
                                  'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']])
categorical = np.array(data.loc[:, ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks', 
                                    'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id', 
                                    'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 
                                    'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']])

#standard scaling & one-hot encoding
scaler = StandardScaler()
standard_scale_model = scaler.fit(numerical)
enc = OneHotEncoder()
one_hot_model = enc.fit(categorical)

#获取训练数据
X = np.hstack((scaler.transform(numerical) , enc.transform(categorical).toarray()))
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))

#训练过程
lr = LogisticRegression(solver='liblinear')				            #逻辑回归模型
lr.fit(X_train, y_train)						                    #训练

y_prob = lr.predict_proba(X_test)[:, 1]					            #预测结果为1的概率
y_pred = lr.predict(X_test)						                    #预测结果
fpr_lr, tpr_lr, threshold_lr = metrics.roc_curve(y_test, y_prob)	#真阳率、伪阳率、阈值
auc_lr = metrics.auc(fpr_lr,tpr_lr)					                #AUC
score_lr = metrics.accuracy_score(y_test, y_pred)			        #模型准确率
print('模型准确率:{}\nAUC得分:{}'.format(score_lr, auc_lr))

#保存模型
scaler_path = '/home/test/model/expedia_standard_scale_model.pkl'
enc_path = '/home/test/model/expedia_one_hot_encoder.pkl'
file_path = '/home/test/model/expedia_lr_model.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
with open(enc_path, 'wb') as f:
    pickle.dump(one_hot_model, f)
with open(file_path, 'wb') as f:
    pickle.dump(lr, f)
