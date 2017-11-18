# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
path='./'
df=pd.read_csv(path+u'train-ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+u'train-ccf_first_round_shop_info.csv')
test=pd.read_csv(path+u'AB-evaluation_public.csv')
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
df['time_stamp']=pd.to_datetime(df['time_stamp'])


############feature engineering#####################
people_shop_location_longitude = df.groupby(['user_id','shop_id'],as_index=False).longitude.agg(
            {'p_shop_location_longitude':np.median})
people_shop_location_latitude = df.groupby(['user_id','shop_id'],as_index=False).latitude.agg(
            {'p_shop_location_latitude':np.median})
df = pd.merge(df,people_shop_location_longitude,on=['user_id','shop_id'],how='left')
df = pd.merge(df,people_shop_location_latitude,on=['user_id','shop_id'],how='left')
del people_shop_location_longitude;gc.collect()
del people_shop_location_latitude;gc.collect()

df=df.drop(['longitude','latitude'],axis=1)
#df.rename(columns={'p_shop_location_longitude':'longitude','p_shop_location_latitude':'latitude'})

test['p_shop_location_longitude']=test['longitude']
test['p_shop_location_latitude']=test['latitude']
test=test.drop(['longitude','latitude'],axis=1)





train=pd.concat([df,test])
del df,test
gc.collect()
# 时间串信息组合
train['time_stamp'] = pd.to_datetime(train['time_stamp'])
train['history_hour'] =  pd.DatetimeIndex(train.time_stamp).hour
train['history_day'] =  pd.DatetimeIndex(train.time_stamp).day





mall_list=list(set(list(shop.mall_id)))
result=pd.DataFrame()
for mall in mall_list:
    train1=train[train.mall_id==mall].reset_index(drop=True)
    print(train1.head(5))
    l=[]
    wifi_dict = {}
    for index,row in train1.iterrows():
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            r[i[0]]=int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]]=1
            else:
                wifi_dict[i[0]]+=1
        l.append(r)    
    delate_wifi=[]
    for i in wifi_dict:
        if wifi_dict[i]<20:
            delate_wifi.append(i)
    m=[]
    for row in l:
        new={}
        for n in row.keys():
            if n not in delate_wifi:
                new[n]=row[n]
        m.append(new)
    train1 = pd.concat([train1,pd.DataFrame(m)], axis=1)
    df_train=train1[train1.shop_id.notnull()]
    df_test=train1[train1.shop_id.isnull()]
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    num_class=df_train['label'].max()+1    
    params = {
            'objective': 'multi:softmax',
            'eta': 0.1,
            'max_depth': 9,
            'eval_metric': 'merror',
            'seed': 0,
            'missing': -999,
            'num_class':num_class,
            'silent' : 1
            }
    # print(train1.columns)
    feature=[x for x in train1.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]
    # print(feature)
    #print(df_test[feature])
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
    num_rounds=200
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
    xgb.plot_importance(model)
    df_test['label']=model.predict(xgbtest)
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])
    result['row_id']=result['row_id'].astype('int')
    result.to_csv(path+'sub20.csv',index=False)