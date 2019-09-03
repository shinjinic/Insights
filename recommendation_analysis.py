import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import os
#from blaze.compute.tests.test_dask_dataframe import df
#from openpyxl.utils.cell import col
#from lxml.html.builder import COL
#import ml_metrics
#import recmetrics
try:
    import sklearn
except:
    os.system('python -m pip install sklearn')
    import sklearn    
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import pickle
from datetime import datetime
#import seaborn as sns
from functools import reduce
import pandas as pd
from mysqlplugin import MySqlConnector
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
#from tqdm import tqdm
from collections import defaultdict
import random
random.seed(100)
import os
from functools import reduce
from pyspark.sql.functions import *
from getset import GetSet
from pyspark.sql import Window
from pyspark.sql.types import *

class RecommendationEngine:
    
    def data_sparse(self,spark,mysqldetails,memberDF,InsightName):
        getset = GetSet()
        '''
        if isinstance(memberDF, pd.DataFrame):
            winter_data=RecommendationEngine.data_atvpluspoints(self,spark,mysqldetails,memberDF)        
        else:          
            mobile_list = [str(r.Mobile) for r in memberDF.collect()]       
            length = int(len(mobile_list)/4)
            mobileList1 = mobile_list[:length]
            mobileList2 = mobile_list[length:2*length]
            mobileList3 = mobile_list[2*length:3*length]
            mobileList4 = mobile_list[3*length:4*length]
                    
            mobileString1 = ','.join(str(m) for m in mobileList1)
            mobileString2 = ','.join(str(m) for m in mobileList2)
            mobileString3 = ','.join(str(m) for m in mobileList3)
            mobileString4 = ','.join(str(m) for m in mobileList4)
            
            #table="(select lastrundate from insights_id_run_information where insightid in (select insightid from program_insight_master where insightname="+InsightName+") a) as df"
            #lastrun=MySqlConnector.mysqlRead(self,spark,mysqldetails[0],mysqldetails[1],table,mysqldetails[2],mysqldetails[3])
            #lastrundate = lastrun.rdd.map(lambda row:row.lastrundate).collect()[0]
            
            mobs = [mobileString1,mobileString2,mobileString3,mobileString4]
            dfs = []
            for mobileString in mobs:
                query = "(SELECT TxnMappedMobile,UniqueItemCode,ModifiedTxnDate as TxnDate,ItemQty from sku_report_loyalty where ModifiedTxnDate>date_sub('"+getset.get_recommaxdate()+"',INTERVAL "+getset.get_recomdays()+" DAY) and TxnMappedMobile in ("+mobileString+")) as df"
                query = "(SELECT TxnMappedMobile,UniqueItemCode,ModifiedTxnDate as TxnDate,ItemQty from sku_report_loyalty where ModifiedTxnDate>'2019-03-01' and TxnMappedMobile in ("+mobileString+")) as df"            
                #print(query)
                df=MySqlConnector.mysqlRead(self,spark,mysqldetails[0],mysqldetails[1],query,mysqldetails[2],mysqldetails[3])
                df = df.toPandas()
                dfs.append(df)
            winter_data=reduce(lambda x,y: x.append(y,ignore_index=True),dfs)
            print(winter_data.head(5))
        
        #Creating a look up table keeping a track of item name with item code
        
        #item_lookup = winter_data[['UniqueItemCode', 'UniqueItemName']].drop_duplicates() # Only get unique item/description pairs
        #item_lookup['UniqueItemCode'] = item_lookup.UniqueItemCode.astype(str) # Encode as strings for future lookup ease
        
        #item_lookup['UniqueItemCode']=item_lookup['UniqueItemCode'].astype('int')
        
        winter_data['TxnDate'] = pd.to_datetime(winter_data.TxnDate)
        winter_data[["UniqueItemCode","ItemQty"]]=winter_data[['UniqueItemCode','ItemQty']].astype(int)
        
        winter_data=winter_data.groupby(['TxnMappedMobile','UniqueItemCode','TxnDate']).agg({'ItemQty':'sum'}).reset_index()
        #winter_data.to_csv('winter_data.csv',header=True)
        '''
        winter_data=pd.read_csv('winter_data.csv')
                
        w = winter_data.groupby('TxnMappedMobile').agg({'TxnDate':'nunique'}).reset_index()
        w['TxnDate'] = pd.to_numeric(w['TxnDate'],errors='coerce')
        w['TxnMappedMobile'] = w['TxnMappedMobile'].astype('str') 
        winter_data['TxnMappedMobile'] = winter_data['TxnMappedMobile'].astype('str')
        print("w",w.head(5))
        wd = winter_data[winter_data['TxnMappedMobile'].isin(list(w['TxnMappedMobile'][w['TxnDate']>1]))]
        print("wd",wd.head(5))
        onetime_rs1 = winter_data['TxnMappedMobile'][~winter_data['TxnMappedMobile'].isin(list(w['TxnMappedMobile'][w['TxnDate']>1]))]
        onetime_rs = onetime_rs1.to_frame()
        onetime_rs = spark.createDataFrame(onetime_rs)
        onetime_rs = onetime_rs.selectExpr("TxnMappedMobile as Mobile")
        try:
            leftouts=winter_data['TxnMappedMobile'][~winter_data['TxnMappedMobile'].isin(list(wd['TxnMappedMobile']).extend(list(onetime_rs1)))]
        except:
            leftouts = winter_data['TxnMappedMobile']
        leftouts = leftouts.to_frame()
        leftouts = spark.createDataFrame(leftouts)
        leftouts = leftouts.selectExpr("TxnMappedMobile as Mobile")
            
        with open('./'+mysqldetails[1]+'/cusprod.pkl','rb') as f:
            cusprod = pickle.load(f)
            
        b=[list(t) for t in list(zip(*cusprod))]
        customers = [str(c) for c in b[0]]
        products = [str(p) for p in b[1]]
        
        print("pickled customers example: ",customers[:10])
        
        wd = wd.sort_values('TxnDate')
        
        wd=wd[['TxnMappedMobile','UniqueItemCode','ItemQty']]
        wd['TxnMappedMobile'] = wd['TxnMappedMobile'].astype('str')
        wd['UniqueItemCode'] = wd['UniqueItemCode'].astype('str')
        #print("common mobiles",len(wd[wd['TxnMappedMobile'].isin(customers)]))
        #print("common products",len(wd[(wd['UniqueItemCode'].isin(products))]))
        wd = wd[(wd['TxnMappedMobile'].isin(customers)) & (wd['UniqueItemCode'].isin(products))]
        print("customer data shape",wd.shape)
        print(wd.head(10))
        if len(wd)<1:
            return pd.DataFrame({'Mobile':[],InsightName:[]}),1,1,1,1,leftouts
        #customers = list(wd.TxnMappedMobile.unique()) # Get our unique customers
        #products = list(wd.UniqueItemCode.unique()) # Get our unique products that were purchased
        quantity = list(wd.ItemQty) # All of our purchases
        customers_arr=np.array(customers)
        products_arr=np.array(products)
        rows = wd.TxnMappedMobile.astype('category', categories = customers).cat.codes 
        # Get the associated row indices
        cols = wd.UniqueItemCode.astype('category', categories = products).cat.codes 
        # Get the associated column indices
        purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))
        
        matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] # Number of possible interactions in the matrix
        num_purchases = len(purchases_sparse.nonzero()[0]) # Number of items interacted with
        sparsity = 100*(1 - (num_purchases/matrix_size))
        return purchases_sparse,customers_arr,products_arr,wd,onetime_rs,leftouts
    
    def data_atvpluspoints(self,spark,mysqldetails,memberDF):
        getset = GetSet()
        memberDF['Mobile'] = memberDF['Mobile'].astype('str')
        mobile_list = list(memberDF['Mobile'])
        memberDF=spark.createDataFrame(memberDF)
        print("ATV memberDF")
        memberDF.show()
        length = int(len(mobile_list)/4)
        mobileList1 = mobile_list[:length]
        mobileList2 = mobile_list[length:2*length]
        mobileList3 = mobile_list[2*length:3*length]
        mobileList4 = mobile_list[3*length:4*length]
                
        mobileString1 = ','.join(str(m) for m in mobileList1)
        mobileString2 = ','.join(str(m) for m in mobileList2)
        mobileString3 = ','.join(str(m) for m in mobileList3)
        mobileString4 = ','.join(str(m) for m in mobileList4)
        
        mobs = [mobileString1,mobileString2,mobileString3,mobileString4]
        i=0
        for mobileString in mobs:
            query = "(SELECT TxnMappedMobile,UniqueItemCode,ModifiedTxnDate as TxnDate,ItemQty,ItemNetAmount from sku_report_loyalty where ModifiedTxnDate>date_sub('"+getset.get_recommaxdate()+"',INTERVAL "+getset.get_recomdays()+" DAY) and TxnMappedMobile in ("+mobileString+")) as df"
            query = "(SELECT TxnMappedMobile,UniqueItemCode,ModifiedTxnDate as TxnDate,ItemQty,ItemNetAmount from sku_report_loyalty where ModifiedTxnDate>'2019-03-01' and TxnMappedMobile in ("+mobileString+")) as df"            
            #print(query)
            df=MySqlConnector.mysqlRead(self,spark,mysqldetails[0],mysqldetails[1],query,mysqldetails[2],mysqldetails[3])
            if i==0:
                df1 = df
            else:
                df1 = df1.union(df)                
            i=i+1
        df1 = df1.alias('a').join(memberDF.alias('b'),col('a.TxnMappedMobile')==col('b.Mobile'),'inner').select('a.TxnMappedMobile','a.UniqueItemCode','a.TxnDate','a.ItemQty','a.ItemNetAmount','b.Value')    
        w = Window.partitionBy('TxnMappedMobile')
        df = df1.withColumn('maxamount', max('ItemNetAmount').over(w))\
                .where(col('ItemNetAmount').between(0.7*col('Value')<=1.3*col('Value')))\
                .drop('maxamount')\
                .drop('ItemNetAmount')\
                .drop('Value')
        winter_data = df.toPandas()        
        print(winter_data.head(5))
        return winter_data
    
    
    def make_train(self,ratings, pct_test = 0.2):
       test_set = ratings.copy() # Make a copy of the original set to be the test set. 
       test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
       training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
       nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
       nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))# Zip these pairs together of user,item index into list
       random.seed(0) # Set the random seed to zero for reproducibility
       num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
       samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
       user_inds = [index[0] for index in samples] # Get the user row indices
       #print("user_inds",user_inds[:10])
       item_inds = [index[1] for index in samples] # Get the item column indices
       training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
       training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
       return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  
    
    
    
    #def get_items_purchased(customer_id,mf_train,customers_list,products_list,item_lookup):
        
        #This just tells me which items have been already purchased by a specific user in the training set. 
        
        #parameters: 
        
        #customer_id - Input the customer's id number that you want to see prior purchases of at least once
        
        #mf_train - The initial ratings training set used (without weights applied)
        
        #customers_list - The array of customers used in the ratings matrix
        
        #products_list - The array of products used in the ratings matrix
        
        #item_lookup - A simple pandas dataframe of the unique product ID/product descriptions available
        
        #returns:
        
        #A list of item IDs and item descriptions for a particular customer that were already purchased in the training set
    '''
        cust_ind = np.where(customers_list == customer_id)[0][0] # Returns the index row of our customer id
        purchased_ind = mf_train[cust_ind,:].nonzero()[1] # Get column indices of purchased items
        prod_codes = products_list[purchased_ind] # Get the stock codes for our purchased items
        print(prod_codes)
        return item_lookup.loc[item_lookup.UniqueItemCode.isin(prod_codes)]
    '''
    
    def rec_items(self,customer_id, mf_train, user_vecs, item_vecs, customer_list, item_list, num_items):
        cust_ind = np.where(customer_list == customer_id)[0][0] # Returns the index row of our customer id
        pref_vec = mf_train[cust_ind,:].toarray() # Get the ratings from the training set ratings matrix
        pref_vec = pref_vec.reshape(-1) + 1 # Add 1 to everything, so that items not purchased yet become equal to 1
        pref_vec[pref_vec > 1] = 0 # Make everything already purchased zero
        #print("cust_ind",cust_ind)
        #print("user_vecs",user_vecs[cust_ind,:])
        rec_vector = user_vecs[cust_ind,:].dot(item_vecs.T) # Get dot product of user vector and all item vectors
        # Scale this recommendation vector between 0 and 1
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] 
        recommend_vector = pref_vec*rec_vector_scaled 
        # Items already purchased have their recommendation multiplied by zero
        product_idx = np.argsort(recommend_vector)[::-1][:num_items] # Sort the indices of the items into order 
        # of best recommendations
        rec_list = [] # start empty list to store items
        for index in product_idx:
            code = item_list[index]
            rec_list.append(code) 
            # Append our descriptions to the list
        codes = rec_list
        final_frame = pd.DataFrame({'UniqueItemCode': codes}) # Create a dataframe 
        return final_frame[['UniqueItemCode']] # Switch order of columns around
    
    #Creating a final list of recommendations for all users
    def recommend(self,wd,mysqldetails,product_train,customers_arr, products_arr,num_items):
        print("recommend...")
        i=0
        
        with open('./'+mysqldetails[1]+'/user_vecs.pkl','rb') as f:
            user_vecs=pickle.load(f)
    
        with open('./'+mysqldetails[1]+'/item_vecs.pkl','rb') as f:
            item_vecs=pickle.load(f)
            
        print("----Model read----")    
         
        #print(wd['TxnMappedMobile'].unique())           
        for customer_id in wd['TxnMappedMobile'].unique():    
            rec1 = RecommendationEngine.rec_items(self,str(customer_id), product_train, user_vecs, item_vecs, customers_arr, products_arr, num_items)['UniqueItemCode'].tolist()
            #print("rec1",rec1[:2])
            items=','.join(str(e) for e in rec1)
            rec2=pd.DataFrame({'Mobile':[str(customer_id)],'recom_items':[items]})
            if i==0:
                #print("rec created")
                rec=rec2
            else:
                #print("rec appending") 
                rec=rec.append(rec2,ignore_index=True)
            i=i+1
        #print("rec referenced before assignment...")    
        print(rec.head())  
        return rec      
                    
    #get_items_purchased('9212345677.0', product_train, customers_arr, products_arr, item_lookup)
    
    #rec_items('9212345677.0', product_train, user_vecs, item_vecs, customers_arr, products_arr,num_items = 10)
    
    def onetimer_recom(self,skudf,Start):
        print("skudf---------------->",skudf.head(20))        
        skudf = skudf.replace('nan',np.nan)
        skudf = skudf.replace('None',np.nan)
        skudf = skudf.replace('none',np.nan)
        skudf['Age'] = pd.to_numeric(skudf['Age'],errors='coerce')
        skudf['AgeBracket'] = pd.cut(skudf.Age, [0,15,25,35,45,100])
        
        m,n =skudf.shape
        
        skudf['Condition']=1
        print("skudf again---------------->",skudf.head(20))
        skudf=skudf.copy()
        
        def f(row):
            if(pd.isnull(row['AgeBracket']) and pd.isnull(row['Gender'])):
                return 0
            elif(pd.isnull(row['AgeBracket'])==False and pd.isnull(row['Gender'])):
                return 1
            elif(pd.isnull(row['AgeBracket']) and pd.isnull(row['Gender'])==False):
                return 2
            elif(pd.isnull(row['AgeBracket'])==False and pd.isnull(row['Gender'])==False):
                return 3
        
        skudf['Condition'] = skudf.apply(lambda row: f(row),axis=1)   
        
        df3 = skudf.groupby(['unique_item','Gender','AgeBracket'])['Txns'].sum().reset_index()
        df3 =df3[df3['unique_item'].str.contains("Carry Bag")==False]
        rp= df3.pivot_table(values='Txns',index=['AgeBracket', 'Gender'], columns='unique_item')
        map_df = rp.index.values
        Q = rp.values
        
        item_ids1 = (np.array(np.argsort(Q,axis=1))).T[::-1][:Start]
        item_ids2=item_ids1.T
        
        m1, n1 = item_ids1.T.shape
        item_code = np.array(rp.columns)
        
        d = [] 
        for user in range(m1):
            a = list(set(item_code[item_ids2[user]]))
            a = ','.join(str(i) for i in a)
            d.append({'set': map_df[user], 'Items': a})
                
        type(d)
        df_age_gen=pd.DataFrame(d)
        final = df_age_gen
        
        '''Condition 2- Gender is present but Age is null --> recommend the most popular products based on Gender'''
        df_Gender = skudf[(skudf['Condition']==3)|(skudf['Condition']==2)]
        df_Gender =df_Gender[df_Gender['unique_item'].str.contains("Carry Bag")==False]
        dfg3 = df_Gender.groupby(['unique_item','Gender'])['Txns'].sum().reset_index()
        rp= dfg3.pivot_table(values='Txns',index=['Gender'], columns='unique_item')
        map_df = rp.index.values
        Q = rp.values
        
        item_ids1 = (np.array(np.argsort(Q,axis=1))).T[::-1][:Start]
        item_ids2=item_ids1.T
        m1, n1 = item_ids1.T.shape
        item_code = np.array(rp.columns)
        d = [] 
        for user in range(m1):
            a = list(set(item_code[item_ids2[user]]))
            a = ','.join(str(i) for i in a)
            d.append({'set': map_df[user], 'Items': a})            
        
        type(d)
        df_gen=pd.DataFrame(d)
        final = final.append(df_gen,ignore_index=True)
        
        
        '''Condition 0- Gender and Age is null --> recommend the most popular products'''
        df = skudf[(skudf['Condition']==0)]
        df =df[df['unique_item'].str.contains("Carry Bag")==False]
        #print('Condition = 0 df len: ',len(df))
        df3 = df.groupby(['unique_item'])['Txns'].sum().reset_index()
        rp= df3.pivot_table(values='Txns', columns='unique_item')
        Q = rp.values
        
        item_ids1 = (np.array(np.argsort(Q,axis=1))).T[::-1][:Start]
        item_ids2=item_ids1.T
        m1, n1 = item_ids1.T.shape
        item_code = np.array(rp.columns)
        d = []
        #print("Overall item codes: ",item_code) 
        a = list(set(item_code[item_ids2[0]]))
        a = ','.join(str(i) for i in a)
        df=d.append({'set':'overall_popularlity','Items':a})
        df=pd.DataFrame(d)
        final=final.append(df,ignore_index=True)
        #print(final)
        final['AgeGender']=['(0-15],Female',
                            '(0-15],Male',
                            '(15-25],Female',
                            '(15-25],Male',
                            '(25-35],Female',
                            '(25-35],Male',
                            '(35-45],Female',
                            '(35-45],Male',
                            '(45-100],Female',
                            '(45-100],Male',
                            ',Female',
                            ',Male',
                            ',']
        #print("Final--------------->",final)
        nsu=skudf[['Mobile','AgeBracket','Gender','unique_item']]
        nsu['AgeBracket'] = nsu['AgeBracket'].astype('str')
        nsu=nsu.fillna('')
        nsu['AgeBracket']=nsu['AgeBracket'].str.replace('nan','')
        nsu['AgeGender']=nsu['AgeBracket']+','+nsu['Gender']
        #print(set(nsu.AgeGender))
        counts = nsu.groupby(['Mobile']).size().reset_index(name='count')
        #print(counts)
        mobs=counts[counts['count']==1]['Mobile']
        mm = nsu[nsu['Mobile'].isin(mobs)]
        mobs2=mm[mm['unique_item'].str.contains('Carry Bag')]['Mobile']
        nsu[nsu['Mobile'].isin(mobs2)]['AgeGender']=','
        recoms = nsu.merge(final,on='AgeGender',how='inner')[['Mobile','Items']]
        recoms=recoms.drop_duplicates(subset=['Mobile'])
        del [[nsu,mm,counts,final,df,df3,df_Gender,skudf]]
        gc.collect()
        #print(recoms)
        return recoms

    def cassandraRead(self,spark, keys_space_name, table_name):
            df = spark.read\
                .format("org.apache.spark.sql.cassandra")\
                .options(table=table_name, keyspace=keys_space_name)\
                .option("spark.cassandra.read.timeout_ms", "70000000")\
                .load()
            return df
        
    def run_recom(self,spark,mysqldetails,memberDF,InsightName,Start):
        
        Start = int(Start)
        skuDF = RecommendationEngine.cassandraRead(self,spark, mysqldetails[1], "fav_item600000000")
        skuDF=skuDF.dropna()
        skuDF = skuDF.select((col('mobile').cast('string')).alias('Mobile'),(col('uniqueitemcode').cast('string')).alias('unique_item'),(col('fav_item').cast('int')).alias('Txns'))
        skuDF = skuDF.filter(length(col('unique_item'))<2)
        #skuDF.show()
        purchases_sparse,customers_arr,products_arr,wd,onetime_rs,leftouts = RecommendationEngine.data_sparse(self,spark,mysqldetails,memberDF,InsightName)           
        
        if customers_arr!=1:            
            product_train,product_test,product_users_altered = RecommendationEngine.make_train(self,purchases_sparse, pct_test = 0.2)
            recomDF = RecommendationEngine.recommend(self,wd,mysqldetails,product_train,customers_arr, products_arr,Start)
            recomDF.columns=['Mobile',InsightName]
            #onetime_rs = ",".join(str(e) for e in list(onetime_rs))
            onetimers = skuDF.join(onetime_rs,'Mobile','inner')
            #print("original onetimers count",onetimers.count())
            print("recommending for one timers..")
            if isinstance(memberDF,pd.DataFrame):
                memberDF=memberDF.astype('str')
                memberDF = spark.createDataFrame(memberDF)
            onetimers=memberDF.join(onetimers,"Mobile","inner")
            #print("onetimers count after memberDF join",onetimers.count())
            try:
                onetimers = onetimers.select("Mobile","Age","Gender","unique_item").na.drop(subset=['unique_item']).toPandas()
                print(onetimers.head(5))
                favorites1 = RecommendationEngine.onetimer_recom(self,onetimers,Start)
                favorites1.columns = ['Mobile',InsightName]
            except:
                print("No onetimers info available")
                favorites1 = pd.DataFrame({'Mobile':[],InsightName:[]})
            
            recomDF = recomDF.append(favorites1,ignore_index=True)
            del favorites1
        else:
            print("No matching customers.. recommending based on popularity...")
            recomDF = purchases_sparse            
        
        '''
        query = "(SELECT TxnMappedMobile as Mobile,UniqueItemCode as unique_item,itemqty as Txns from sku_report_loyalty where TxnMappedMobile in ("+onetime_rs+")) as df"
        onetimers=MySqlConnector.mysqlRead(self,spark,mysqldetails[0],mysqldetails[1],query,mysqldetails[2],mysqldetails[3])
        '''
        if isinstance(memberDF,pd.DataFrame):
            memberDF=memberDF.astype('str')
            memberDF = spark.createDataFrame(memberDF)               
            
        newones=skuDF.join(leftouts,'Mobile','inner')        
        print("Recommending for the rest..")
        newones=memberDF.join(newones,"Mobile","inner") 
        newones = newones.select("Mobile","Age","Gender","unique_item").na.drop(subset=['unique_item']).toPandas()
        recomDF.columns = ['Mobile',InsightName]
        try:
            favoritesnew = RecommendationEngine.onetimer_recom(self,newones,Start)
            favoritesnew.columns = ['Mobile',InsightName]
        except:
            print("No info available for the rest..")    
            favoritesnew = pd.DataFrame({'Mobile':[],InsightName:[]})    
        
        recomDF1 = recomDF.append(favoritesnew,ignore_index=True)
        del [[recomDF,favoritesnew]]
        gc.collect()

        #recomDF1=recomDF1.append(recomDF1,ignore_index=True)
        recomDF1=recomDF1.astype('str')

        return recomDF1


