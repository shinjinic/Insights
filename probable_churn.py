import findspark
findspark.init()
import pyspark
import numpy as np
import os
import gc
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
#import seaborn as sns
from functools import reduce
import os
import pandas as pd
#import pyecharts
import time
# try:
#     import xgboost as xgb
# except:
#     os.system('python -m pip install xgboost')
#     import xgboost as xgb 
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.naive_bayes import BernoulliNB,MultinomialNB
# from sklearn.svm import SVC
# from sklearn.externals import joblib
# from sklearn.preprocessing import StandardScaler
from collections import Counter
from getset import GetSet
from  mysqlplugin import MySqlConnector
import glob
from pathlib import Path
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer,OneHotEncoderEstimator, OneHotEncoderModel, VectorAssembler, StandardScaler
#import pickle
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit,CrossValidatorModel
from pyspark.ml.classification import LogisticRegression,NaiveBayes,LinearSVC 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
    

class probchurn:
    
    def removenullcols(self,df):
        df1 = df.agg(*[F.count(c).alias(c) for c in df.columns])
        nonNull_cols = [c for c in df1.columns if df1[[c]].first()[c] > 0]
        df = df.select(*nonNull_cols)
        return df

    def data_process(self,spark,columns):
            churn_data_semifinal1=spark.read.format('csv').options(header='true', inferschema='true').load('./'+mysqldetails[1]+'/churn_data_semifinal1.csv')            
            churn_data_semifinal1=churn_data_semifinal1.filter(churn_data_semifinal1['total_visits']>1)
            #churn_data_semifinal1=churn_data_semifinal1.drop(['state','tier','gender','city'],axis=1)   
            churn_data_semifinal1 = churn_data_semifinal1.withColumn('gender',when(col('gender').isNull(),'Male').otherwise(col('gender')))    
            '''
            churn_data_semifinal1 = churn_data_semifinal1.select('mobile', 'gender', 'tier', 'state', 'city', 'total_txn',
            'total_visits', 'total_amount', 'pointscollected', 'lapsedpoints',
            'availablepoints', 'max_latency', 'recency', 'tenure', 'latency',
            'labels')
            '''
            churn_data_semifinal1.printSchema()
            
            churn_data_semifinal1 = churn_data_semifinal1.withColumn('latency',when(col('latency').isNull(),999).otherwise(col('latency')))    
            churn_data_semifinal1 = churn_data_semifinal1.withColumn('max_latency',col('max_latency').cast('int')) 
            churn_data_semifinal1 = churn_data_semifinal1.withColumn('latency',col('latency').cast('int')) 
            
            churn_data_semifinal1 = probchurn.removenullcols(self, churn_data_semifinal1)          
            
            categorical_columns=['gender', 'tier', 'state', 'city']
            categorical_columns = list(set(categorical_columns).intersection(set(columns)))
            indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categorical_columns] 
            
            encoder = OneHotEncoderEstimator(
            inputCols=[indexer.getOutputCol() for indexer in indexers],
            outputCols=[
                "{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]
            )
            numericCols = ['total_txn','total_visits','total_amount','pointscollected','lapsedpoints','availablepoints','max_latency','recency','tenure','latency']
            numericCols = list(set(numericCols).intersection(set(columns)))
            assemblerinputs = numericCols + [c+"_indexed_encoded" for c in categorical_columns]
            print(assemblerinputs)
            assembler = VectorAssembler(
                inputCols=assemblerinputs,
                outputCol="features"
            )
            pipeline = Pipeline(stages=indexers +[encoder]+[assembler])
            data = pipeline.fit(churn_data_semifinal1).transform(churn_data_semifinal1)
            try:
                data = data.select('mobile','features','labels')
            except:
                data = data.select('mobile','features')    
           
            return data
    
    def classify(self,mysqldetails,method,data):
            # try:
            #     from sparkxgb import XGBoostEstimator
            # except:
            #     os.system('python -m pip install sparkxgb')
            #     from sparkxgb import XGBoostEstimator            
            #spark.sparkContext.addPyFile("/home/vivek/Downloads/sparkxgb.zip")                              
            
            #.config("spark.driver.extraClassPath", "/home/vivek/Downloads/xgboost4j-0.72.jar:/home/vivek/Downloads/xgboost4j-spark-0.72.jar") \
            #.config("spark.executor.extraClassPath", "/home/vivek/Downloads/xgboost4j-0.72.jar:/home/vivek/Downloads/xgboost4j-spark-0.72.jar") \
            #######################################################
            ######## Data Preparation #############################
            #######################################################
            
            #churn_data_semifinal1=spark.read.format('csv').options(header='true', inferschema='true').load('/home/vivek/Desktop/vivek_churn/final_data.csv')                        
            #churn_data_semifinal1=churn_data_semifinal1.drop(['state','tier','gender','city'],axis=1)               
            #.rdd.map(lambda x: x.labels).collect()
            data_train, data_test = data.select('features','labels').randomSplit([0.7, 0.3], seed = 2019)      
            
            my_eval = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',
                                                   labelCol='labels',metricName="areaUnderROC")
            
            #####################################################
            ######### Fit the model #############################
            #####################################################            
            
            if method.lower() == 'svm':
                
                scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                                    withStd=True, withMean=False)
            
                pipeline = Pipeline(stages=scaler)
                data_train = pipeline.fit(data_train).transform(data_train)
                data_test = pipeline.fit(data_test).transform(data_test)
                model = LinearSVC(labelCol="labels", featuresCol="scaledFeatures")
                paramGrid = ParamGridBuilder() \
                .addGrid(model.maxIter, [10, 20, 50]) \
                .addGrid(model.regParam, [0.001, 0.01, 0.1, 1]) \
                .build()
                mname = './'+mysqldetails[1]+'/model_svm'
                         
            #             elif method.lower() == 'xgboost':
            #                 # GridSearch
            #                 model = XGBoostEstimator(
            #                 featuresCol="features", 
            #                 labelCol="labels", 
            #                 predictionCol="prediction"
            #                 )
            #                 #===========================================================================
            #                 # learning_rate = [0.001, 0.01, 0.1, 0.3]
            #                 # n_estimators = [10, 50, 100]
            #                 # max_depth = [10, 20]
            #                 #===========================================================================
            #                 
            #                 paramGrid = ParamGridBuilder() \
            #                 .addGrid(model.elasticNetParam, [0,1]) \
            #                 .addGrid(model.regParam, np.logspace(-3,3,7)) \
            #                 .addGrid(model.maxIter, [10,50,100])\
            #                 .build()
            #             
            #                 mname = './model_xgboost'
              
            elif method.lower() == 'naive_bayes':    
                model = NaiveBayes()
                model = model.fit(data_train)
                predict_test = model.transform(data_test)
                mname = './'+mysqldetails[1]+'/model_naivebayes.pkl'
                with open(mname, 'wb') as f:
                    pickle.dump(model, f)
                print("The area under ROC for test set after CV  is {}".format(my_eval.evaluate(predict_test)))
                score = {'score_'+method.lower():my_eval.evaluate(predict_test)}
                with open('./'+mysqldetails[1]+'/score_'+method.lower()+'.pkl', 'wb') as f:
                        pickle.dump(score, f)
                return "model saved"     
            
            elif method.lower() == 'logistic_regression':
                model = LogisticRegression(labelCol="labels", featuresCol="features", maxIter=10)
                paramGrid = ParamGridBuilder() \
                .addGrid(model.elasticNetParam,[0.0, 0.5, 1.0])\
                .addGrid(model.fitIntercept,[False, True])\
                .addGrid(model.maxIter,[10, 100])\
                .addGrid(model.regParam,[0.01, 0.5, 2.0])\
                .build()
                mname = './'+mysqldetails[1]+'/model_logistic'
                
            elif method.lower() == 'lightgbm':    
                model = LightGBMClassifier(featuresCol="features",labelCol="labels")
                paramGrid = ParamGridBuilder() \
                .addGrid(model.numIterations, [10, 100, 1000]) \
                .addGrid(model.earlyStoppingRound, [10]) \
                .addGrid(model.learningRate, [0.01,0.1,0.2,0.4]) \
                .addGrid(model.numLeaves, [30, 50]) \
                .build()
                mname = './'+mysqldetails[1]+'/model_lightgbm'
                    
            # GridSearch
            crossval = CrossValidator(estimator=model,
                                      estimatorParamMaps=paramGrid,
                                      evaluator=my_eval,
                                      numFolds=5)
            data_train.show(2)
            data_test.show(2)
            params_res = crossval.fit(data_train)
            predict_test=params_res.transform(data_test)
            predict_test.show(5,False)
            print("The area under ROC for test set after CV  is {}".format(my_eval.evaluate(predict_test)))
            params_res.save(mname) 
            score = {'score_'+method.lower():my_eval.evaluate(predict_test)}
            with open('./'+mysqldetails[1]+'/score_'+method.lower()+'.pkl', 'wb') as f:
                    pickle.dump(score, f)
            print("model saved") 
               
    def predict(self,mysqldetails,method,churn_data_semifinal1):
        params_res=CrossValidatorModel.load(mname)
        predict_test=params_res.transform(data_test)
        X = churn_data_semifinal1
        if method.lower() == 'logistic_regression':
            params_res=CrossValidatorModel.load('./'+mysqldetails+'/model_logistic')
        elif method.lower() == 'naive_bayes':
            with open('./'+mysqldetails+'/model_naivebayes.pkl','rb') as f:
                params_res=pickle.load(f)
        #elif method.lower() == 'xgboost':
            #params_res=CrossValidatorModel.load('./model_xgboost') 
        elif method.lower() == 'svm':           
            params_res=CrossValidatorModel.load('./'+mysqldetails+'/model_svm')
        elif method.lower() ==  'lightgbm': 
            params_res=CrossValidatorModel.load('./'+mysqldetails+'/model_lightgbm')     
        predict_test=params_res.transform(X)
        return predict_test.select('mobile','prediction')
    
    def checkcols(self,spark,mysqldetails):
        sku_report_loyalty_cols = ['txnmappedmobile','modifiedbillno','uniqueitemcode','itemqty'] 
        member_report_cols = ['Mobile', 'tier', 'gender','EnrolledStoreCode']
        store_master_cols = ['StoreCode','state','city']
        txn_report_accrual_redemption_cols = ['Mobile','ModifiedBillNo','TxnDate','amount','StoreCode','PointsCollected','LapsedPoints','AvailablePoints']
        modelcols = [sku_report_loyalty_cols,member_report_cols,store_master_cols,txn_report_accrual_redemption_cols]
        tables = ['sku_report_loyalty','member_report','store_master','txn_report_accrual_redemption']
        allcols=[]
        for table,clm in zip(tables,modelcols):
            q = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = N'"+table+"'"
            cols = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], q, mysqldetails[2], mysqldetails[3])
            cols = [str(c.COLUMN_NAME) for c in cols.collect()]
            mycols = ','.join(e for e in list(set(cols).intersection(set(clm))))
            allcols.append(mycols)
        return allcols    
    
    def churners(self,spark,mysqldetails,InsightName,memberDF,tagsDict):
        getset = GetSet()
        pw = getset.get_predictorwindow()
        mydate = getset.get_churnmaxdate()

        #churn_data_semifinal1=pd.read_csv('final_data.csv',low_memory=False)
                   
        query = "(SELECT txnmappedmobile as Mobile,modifiedbillno,uniqueitemcode,itemqty FROM sku_report_loyalty where ModifiedTxnDate>DATE_SUB('"+mydate+"',INTERVAL "+str(pw)+" DAY)) as df"
        jdbcDFsku = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        jdbcDFsku = jdbcDFsku.groupBy('Mobile','modifiedbillno').agg(countDistinct('uniqueitemcode').alias('uniqueitemcode'),sum('itemqty').alias('itemqty'))
        jdbcDFsku = jdbcDFsku.groupBy('Mobile').agg(sum('uniqueitemcode').alias('uniqueitems'),sum('itemqty').alias('qty'))
         
        query = "(SELECT Mobile, tier, gender,EnrolledStoreCode FROM member_report) as df" 
        jdbcDFmem =MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
         
        query = "(select StoreCode,state,city from store_master) as df"
        jdbcDFstr = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        
        jdbcDFmem = jdbcDFmem.alias('a').join(jdbcDFstr.alias('b'),col('a.EnrolledStoreCode')==col('b.StoreCode'),'left')
         
        query = "(SELECT TxnMappedMobile as Mobile,modifiedbillno,departmentcode FROM (SELECT a.uniqueitemcode,a.departmentcode,b.modifiedbillno,b.TxnMappedMobile FROM item_master a inner join sku_report_loyalty b on a.uniqueitemcode=b.uniqueitemcode where b.ModifiedTxnDate>DATE_SUB('"+mydate+"',INTERVAL "+str(pw)+" DAY)) t ) as df"
        jdbcDFdep = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        jdbcDFdep = jdbcDFdep.groupBy('Mobile','modifiedbillno').agg(countDistinct('departmentcode').alias('departmentcode'))
        jdbcDFdep = jdbcDFdep.groupBy('Mobile').agg(sum('departmentcode').alias('ndept'))    
             
        #query = "(select Mobile,ModifiedBillNo,TxnDate,amount,StoreCode,PointsCollected,LapsedPoints,AvailablePoints from txn_report_accrual_redemption where TxnDate<DATE_SUB('2019-04-01',INTERVAL "+cw+" DAY) and TxnDate>DATE_SUB('2019-04-01',INTERVAL "+lw+" DAY))) as df"
        query = "(select Mobile,ModifiedBillNo,TxnDate,amount,StoreCode,PointsCollected,LapsedPoints,AvailablePoints from txn_report_accrual_redemption where TxnDate>DATE_SUB('"+mydate+"',INTERVAL "+str(pw)+" DAY)) as df"
        
        jdbcDF3 = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        jdbcDF3.show(5)
        #jdbcDF3c = jdbcDF3.filter(col('TxnDate')>date_sub(lit(datetime(2019,4,1).date()),lit(cw)))
        #mlist = jdbcDF3c.rdd.map(lambda r: r.Mobile).collect()
        #jdbcDF3 = jdbcDF3.withColumn('labels',when(jdbcDF3.Mobile.isin(mlist),0).otherwise(1))
        
        window = Window.partitionBy('Mobile').orderBy("TxnDate")
        jdbcDF3 = jdbcDF3.withColumn("diff", datediff(lag('TxnDate',1).over(window),'TxnDate'))
        
        #jdbcDF3=jdbcDF3.groupby('Mobile').agg(sum('amount').alias('total_amount'),countDistinct('ModifiedBillNo').alias('total_txn'),countDistinct('TxnDate').alias('total_visits'),\
        #max('TxnDate').alias('MaxDate'),min('TxnDate').alias('MinDate'),countDistinct('StoreCode').alias('nstores'),max('diff').alias('max_latency'),max('labels').alias('labels'))
        jdbcDF3=jdbcDF3.groupby('Mobile').agg(sum('amount').alias('total_amount'),countDistinct('ModifiedBillNo').alias('total_txn'),countDistinct('TxnDate').alias('total_visits'),sum('LapsedPoints').alias('lapsedpoints'),sum('AvailablePoints').alias('availablepoints'),\
        max('TxnDate').alias('MaxDate'),min('TxnDate').alias('MinDate'),countDistinct('StoreCode').alias('nstores'),max('diff').alias('max_latency'),sum('PointsCollected').alias('pointscollected'))
        jdbcDF3=jdbcDF3.withColumn('current_date', datetime.strptime(mydate,"%Y-%m-%d"))
        jdbcDF3=jdbcDF3.withColumn("recency",datediff(col("MaxDate"),col("current_date")))
        jdbcDF3=jdbcDF3.withColumn("Tenure",datediff(col("MinDate"),col("current_date")))
        jdbcDF3=jdbcDF3.withColumn("latency",datediff(col("MinDate"),col("MaxDate"))/(col('total_visits')-1))
        
        jdbcDF3=jdbcDF3.join(jdbcDFsku,'Mobile','left')
        jdbcDF3=jdbcDF3.join(jdbcDFdep,'Mobile','left')
        jdbcDF3 = jdbcDF3.join(jdbcDFmem,'Mobile','left')
        
        jdbcDF3 = jdbcDF3.toDF(*[c.lower() for c in jdbcDF3.columns])
        jdbcDF3 = jdbcDF3.select('mobile', 'gender', 'tier', 'state', 'city', 'total_txn',
               'total_visits', 'total_amount', 'pointscollected', 'lapsedpoints',
               'availablepoints', 'max_latency', 'recency', 'tenure', 'latency')
        jdbcDF3.write.option("header", "true").mode("overwrite").csv('./'+mysqldetails[1]+'/churn_data_semifinal1.csv')
        '''
        myvars = jdbcDF3.columns
        modelvars = ['mobile', 'gender', 'tier', 'state', 'city', 'total_txn',
               'total_visits', 'total_amount', 'pointscollected', 'lapsedpoints',
               'availablepoints', 'max_latency', 'recency', 'tenure', 'latency',
               'labels'] 
        columns = list(set(myvars).intersection(set(modelvars)))
        '''
        columns = jdbcDF3.columns
        data_predict=probchurn.data_process(self,spark,columns)
        #probchurn.classify(self,mysqldetails,'logistic_regression',churn_data_semifinal1)
        mlist = memberDF.rdd.map(lambda r: r.mobile).collect()
        churn_data_semifinal1=churn_data_semifinal1.Mobile.isin(mlist)
        method='lightgbm'
        memberDF = probchurn.predict(self,mysqldetails,method,data_predict)
        return memberDF
    
    def modelfit(self,spark,mysqldetails): 
        getset = GetSet()
        pw = getset.get_predictorwindow()
        cw = getset.get_churnwindow()
        lw = int(pw)+int(cw)
        mydate = getset.get_churnmaxdate()
        query = "(SELECT TxnMappedMobile as Mobile,modifiedbillno,uniqueitemcode,itemqty FROM sku_report_loyalty where TxnDate<DATE_SUB('"+mydate+"',INTERVAL "+str(cw)+" DAY) and TxnDate>DATE_SUB('"+mydate+"',INTERVAL "+str(lw)+" DAY)) as df"
        jdbcDFsku = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        jdbcDFsku = jdbcDFsku.groupBy('Mobile','modifiedbillno').agg(countDistinct('uniqueitemcode').alias('uniqueitemcode'),sum('itemqty').alias('itemqty'))
        jdbcDFsku = jdbcDFsku.groupBy('Mobile').agg(sum('uniqueitemcode').alias('uniqueitems'),sum('itemqty').alias('qty'))
         
        query = "(SELECT Mobile, tier, gender,EnrolledStoreCode FROM member_report) as df" 
        jdbcDFmem =MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
         
        query = "(select StoreCode,state,city from store_master) as df"
        jdbcDFstr = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        
        jdbcDFmem = jdbcDFmem.alias('a').join(jdbcDFstr.alias('b'),col('a.EnrolledStoreCode')==col('b.StoreCode'),'left')
         
        query = "(SELECT TxnMappedMobile as Mobile,modifiedbillno,departmentcode FROM (SELECT a.uniqueitemcode,a.departmentcode,b.modifiedbillno,b.TxnMappedMobile FROM item_master a inner join sku_report_loyalty b on a.uniqueitemcode=b.uniqueitemcode where b.TxnDate<DATE_SUB('"+mydate+"',INTERVAL "+str(cw)+" DAY) and b.TxnDate>DATE_SUB('"+mydate+"',INTERVAL "+str(lw)+" DAY)) t ) as df"
        jdbcDFdep = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        jdbcDFdep = jdbcDFdep.groupBy('Mobile','modifiedbillno').agg(countDistinct('departmentcode').alias('departmentcode'))
        jdbcDFdep = jdbcDFdep.groupBy('Mobile').agg(sum('departmentcode').alias('ndept'))    
             
        #query = "(select Mobile,ModifiedBillNo,TxnDate,amount,StoreCode,PointsCollected,LapsedPoints,AvailablePoints from txn_report_accrual_redemption where TxnDate<DATE_SUB('2019-04-01',INTERVAL "+cw+" DAY) and TxnDate>DATE_SUB('2019-04-01',INTERVAL "+lw+" DAY))) as df"
        query = "(select Mobile,ModifiedBillNo,TxnDate,amount,StoreCode,PointsCollected,LapsedPoints,AvailablePoints from txn_report_accrual_redemption where TxnDate>DATE_SUB('"+mydate+"',INTERVAL "+str(lw)+" DAY)) as df"
        
        jdbcDF3 = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        #jdbcDF3.show(5)
        jdbcDF3c = jdbcDF3.filter(col('TxnDate')>datetime.strptime(mydate,"%Y-%m-%d")-timedelta(days=int(cw)))
        mlist = jdbcDF3c.rdd.map(lambda r: r.Mobile).collect()
        print("now",datetime.strptime(mydate,"%Y-%m-%d"))
        print("churn window start",datetime.strptime(mydate,"%Y-%m-%d")-timedelta(days=int(cw)))
        jdbcDF3 = jdbcDF3.filter(col('TxnDate')<datetime.strptime(mydate,"%Y-%m-%d")-timedelta(days=int(cw)))
        jdbcDF3 = jdbcDF3.withColumn('labels',when(jdbcDF3.Mobile.isin(mlist),0).otherwise(1))        
        window = Window.partitionBy('Mobile').orderBy("TxnDate")
        jdbcDF3 = jdbcDF3.withColumn("diff", datediff(lag('TxnDate',1).over(window),'TxnDate'))
        jdbcDF3.show(5)
        jdbcDF3=jdbcDF3.groupby('Mobile').agg(sum('amount').alias('total_amount'),countDistinct('ModifiedBillNo').alias('total_txn'),countDistinct('TxnDate').alias('total_visits'),sum('LapsedPoints').alias('lapsedpoints'),sum('AvailablePoints').alias('availablepoints'),\
        max('TxnDate').alias('MaxDate'),min('TxnDate').alias('MinDate'),countDistinct('StoreCode').alias('nstores'),max('diff').alias('max_latency'),sum('PointsCollected').alias('pointscollected'),max('labels').alias('labels'))
        #jdbcDF3=jdbcDF3.groupby('Mobile').agg(sum('amount').alias('total_amount'),countDistinct('ModifiedBillNo').alias('total_txn'),countDistinct('TxnDate').alias('total_visits'),\
        #max('TxnDate').alias('MaxDate'),min('TxnDate').alias('MinDate'),countDistinct('StoreCode').alias('nstores'),max('diff').alias('max_latency'))
        jdbcDF3=jdbcDF3.withColumn('current_date', date_sub('MaxDate', int(cw)))
        jdbcDF3=jdbcDF3.withColumn("recency",datediff(col("MaxDate"),col("current_date")))
        jdbcDF3=jdbcDF3.withColumn("Tenure",datediff(col("MinDate"),col("current_date")))
        jdbcDF3=jdbcDF3.withColumn("latency",datediff(col("MinDate"),col("MaxDate"))/(col('total_visits')-1))
        
        jdbcDF3=jdbcDF3.join(jdbcDFsku,'Mobile','left')
        jdbcDF3=jdbcDF3.join(jdbcDFdep,'Mobile','left')
        jdbcDF3 = jdbcDF3.join(jdbcDFmem,'Mobile','left')
        
        jdbcDF3 = jdbcDF3.toDF(*[c.lower() for c in jdbcDF3.columns])
        #row = jdbcDF3.limit(1).collect()[0].asDict()
        #myvars = [i for i,j in row.items()]        
        '''
        myvars = jdbcDF3.columns
        modelvars = ['mobile', 'gender', 'tier', 'state', 'city', 'total_txn',
               'total_visits', 'total_amount', 'pointscollected', 'lapsedpoints',
               'availablepoints', 'max_latency', 'recency', 'tenure', 'latency',
               'labels'] 
        columns = list(set(myvars).intersection(set(modelvars)))
        '''
        columns = jdbcDF3.columns
        jdbcDF3 = jdbcDF3.select(*columns)
        os.system('mkdir -p '+mysqldetails[1])
        jdbcDF3.write.option("header", "true").mode("overwrite").csv('./'+mysqldetails[1]+'/churn_data_semifinal1.csv')
        #jdbcDF3.write.format('com.databricks.spark.csv').save('churn_data_semifinal1.csv',header= 'true')
        data=probchurn.data_process(self,spark,columns)
        probchurn.classify(self,mysqldetails,'lightgbm',data)
        print("churn modelling done!")
        

if __name__=='__main__':
    Probchurn = probchurn()
    getset = GetSet()
    mysqldetails = ["10.105.3.10","vmm_raw","rp_app","easy@123"]
    spark = SparkSession \
            .builder \
            .appName("InsightsFetch" + str(datetime.now())) \
            .master("spark://"+getset.get_sparkMasterIp()+":"+getset.get_port()) \
            .config("spark.executor.memory", getset.get_sparkexecutormemory()) \
            .config("spark.cores.max", getset.get_sparkcoresmax())  \
            .config("spark.driver.memory", getset.get_sparkdrivermemory())  \
            .config("spark.executor.memoryOverhead", getset.get_sparkexecutormemoryOverhead())  \
            .config("spark.driver.extraClassPath", getset.get_mysqljar()+":"+getset.get_elasticjar()) \
            .config("spark.executor.extraClassPath", getset.get_mysqljar()+":"+getset.get_elasticjar()) \
            .config("spark.jars.packages", "Azure:mmlspark:0.17")\
            .getOrCreate()
    
    from mmlspark import LightGBMClassifier
    Probchurn.modelfit(spark,mysqldetails)
         