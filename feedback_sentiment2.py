import pandas
import os
try:
    import nltk
except:      
    nltk.download_shell()
try:
    from textblob import TextBlob
except:
    os.system('python -m pip install textblob')
    os.system('python -m textblob.download_corpora')
    from textblob import TextBlob
import numpy as np
import itertools
from collections import Counter
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.context import SparkContext
from itertools import dropwhile
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import pickle
import gc
from getset import GetSet

class FeedbackEngine:
    
    def mysqlRead(self,spark,mysqlServerIp,database,query,username,password):
        mysqlUrl="jdbc:mysql://"+mysqlServerIp+":3306/"+database+"?zeroDateTimeBehavior=convertToNull&autoReconnect=true&characterEncoding=UTF-8&characterSetResults=UTF-8"
        jdbcDF = spark.read \
        .format("jdbc") \
        .option("url", mysqlUrl) \
        .option("dbtable", query) \
        .option("user", username) \
        .option("password", password) \
        .option("numPartitions", "1000") \
        .option("fetchsize", "10000") \
.load()
        jdbcDF=jdbcDF.toPandas()


        return jdbcDF
        
            
        
        
    
    def NaiveBayesClassifier(self,train,df):
        train['user_response'] = train['user_response'].apply(nltk.word_tokenize)
        stemmer = PorterStemmer()
        train['user_response'] = train['user_response'].apply(lambda x: [stemmer.stem(y) for y in x])
        train['user_response'] = train['user_response'].apply(lambda x: ' '.join(x))
        df['user_response'] = df['user_response'].apply(nltk.word_tokenize)
        df['user_response'] = df['user_response'].apply(lambda x: [stemmer.stem(y) for y in x])
        df['user_response'] = df['user_response'].apply(lambda x: ' '.join(x))
        
        
        newdf = train[['user_response']].append(df[['user_response']])
        '''
        print(newdf.head(10))
        
        count_vect = CountVectorizer()  
        
        counts = count_vect.fit_transform(newdf['user_response'])
        
        print("counts done")
               
        transformer = TfidfTransformer().fit(counts)
        
        print("fitted to transformer")
        
        
        counts = transformer.transform(counts)
        #print(counts)
        
        print("transformed to tf-idf")
        '''
        '''
        file = open(sentiment_MNB_path + 'sentiment_MNB_model.pickle', 'wb')
        pickle.dump(bow_transformer, file)
        pickle.dump(nb, file)
        '''
        '''
        #counts1 = count_vect1.fit_transform(df['user_response'])
        #transformer1 = TfidfTransformer().fit(counts1)
        counts1 = transformer.transform(df['user_response'])
        
        #X_train, X_test, y_train, y_test = train_test_split(counts, df['classifier'], test_size=0.1, random_state=609)
        X_train= counts
        y_train=train['classifier']
        X_test = counts1
        #y_test = df['classifier']  
        '''
        
        
        def text_process(text):
            nopunc = [char for char in text.decode('utf8') if char not in string.punctuation]
            nopunc = ''.join(nopunc)
            noDig = ''.join(filter(lambda x: not x.isdigit(), nopunc)) 
        
            ## stemming
            stemmi = u''.join(sno.stem(unicode(x)) for x in noDig)
        
            stop = [word for word in stemmi.split() if word.lower() not in stopwords]
            stop = ' '.join(stop)
        
            return [word for word in stemmi.split() if word.lower() not in stopwords]
        
        
        transformer = CountVectorizer().fit(newdf['user_response'])
        counts = transformer.transform(train['user_response'])        
        counts1 = transformer.transform(df['user_response'])
        X_train= counts
        y_train=train['classifier']
        X_test = counts1
        
        
        model = MultinomialNB().fit(X_train, y_train)
        predicted = model.predict(X_test)
        print("predicted")
        return predicted
        
    
    def DataCleaning(self,dftable):
        
        dftable['user_response'] = dftable['user_response'].replace('[^A-Za-z\s]+', '',regex=True)
        dftable['user_response'].replace({'':np.nan},inplace=True)
        dftable['user_response']=dftable['user_response'].astype(str)
        dftable['user_response'] = dftable['user_response'].apply(lambda x: x.lower())

        dftable=dftable.dropna(subset=['user_response'])
        return dftable
        
    
    def mostfreqwords(self,df,n):
        lt_n = df['user_response'].apply(lambda x : [x for x,t in (TextBlob(x).tags) if(t=='NN' or t=='NNP' or  t=='NNS' or t=='NNPS')]).tolist()
        tot = Counter(i for i in itertools.chain.from_iterable(lt_n))

        

        for key, count in dropwhile(lambda key_count: key_count[1] >= 15, tot.most_common()):
            del tot[key]
            
        return tot.most_common(n)
        
        
    
    def getNegSentimentDF(self,dftable):
        
        dftable['polarity']= dftable['user_response'].apply(lambda x: TextBlob(x).sentiment.polarity)
        dftable['subjectivity'] = dftable['user_response'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        dftable.loc[dftable['polarity']<0,'sentiment']='Negative'
        df_NounTag = dftable[['mobile','user_response','sentiment']]
        df_negative = df_NounTag[df_NounTag['sentiment']=='Negative']
        return df_negative
        #print(skudf1.head(10))
    
        return skudf
    
    
    def getuntrainedDF(self,traindf,df_negative):
        new_df = pandas.merge(traindf, df_negative,  how='right', left_on=['mobile','user_response'], right_on = ['mobile','user_response'])
        new_df=new_df[new_df.classifier.isnull()]
        return new_df
        
        

    def CreateTrainingDF(self,df_negative):

        train_neg =df_negative['user_response']
        train_neg= df_negative[(df_negative['user_response'].str.contains("staff")==True) | (df_negative['user_response'].str.contains("Staff")==True)]
        train_neg['classifier'] = 'improve staff friendliness'
        train_neg=train_neg[['mobile','user_response','classifier']]
        
        train_neg2 =df_negative['user_response']
        train_neg2= df_negative[(df_negative['user_response'].str.contains("variety")==True) & (df_negative['user_response'].str.contains("staff")==False)]
        train_neg2['classifier'] = 'more variety'
        train_neg2=train_neg2[['mobile','user_response','classifier']]
        train = train_neg.append(pandas.DataFrame(data = train_neg2), ignore_index=True)
        
        train_neg3 =df_negative['user_response']
        train_neg3= df_negative[(df_negative['user_response'].str.contains("size")==True) & (df_negative['user_response'].str.contains("staff")==False) & (df_negative['user_response'].str.contains("variety")==False)]
        train_neg3['classifier'] = 'size not available'
        train_neg3=train_neg3[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg3), ignore_index=True)
        
        
        train_neg4 =df_negative['user_response']
        train_neg4= df_negative[(df_negative['user_response'].str.contains("experience")==True) & (df_negative['user_response'].str.contains("staff")==False) & (df_negative['user_response'].str.contains("variety")==False) & (df_negative['user_response'].str.contains("size")==False)]
        train_neg4['classifier'] = 'improve experience'
        train_neg4=train_neg4[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg4), ignore_index=True)
        
        train_neg5 =df_negative['user_response']
        train_neg5= df_negative[(df_negative['user_response'].str.contains("range")==True) & (df_negative['user_response'].str.contains("products")==True) & (df_negative['user_response'].str.contains("staff")==False) & (df_negative['user_response'].str.contains("variety")==False) & (df_negative['user_response'].str.contains("size")==False) & (df_negative['user_response'].str.contains("experience")==False)]
        train_neg5['classifier'] = 'improve range of products'
        train_neg5=train_neg5[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg5), ignore_index=True)
        
        train_neg6 =df_negative['user_response']
        train_neg6= df_negative[(df_negative['user_response'].str.contains("range")==True) & (df_negative['user_response'].str.contains("price")==True) & (df_negative['user_response'].str.contains("staff")==False) & (df_negative['user_response'].str.contains("variety")==False) & (df_negative['user_response'].str.contains("size")==False) & (df_negative['user_response'].str.contains("experience")==False)]
        train_neg6['classifier'] = 'range is high'
        train_neg6=train_neg6[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg6), ignore_index=True)
        
        train_neg7 =df_negative['user_response']
        train_neg7= df_negative[(df_negative['user_response'].str.contains("sale")==True) & (df_negative['user_response'].str.contains("person")==True) & (df_negative['user_response'].str.contains("variety")==False) & (df_negative['user_response'].str.contains("size")==False) & (df_negative['user_response'].str.contains("experience")==False)]
        train_neg7['classifier'] = 'issue with sales person'
        train_neg7=train_neg7[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg7), ignore_index=True)
        
        
        train_neg8 =df_negative['user_response']
        train_neg8= df_negative[(df_negative['user_response'].str.contains("billing")==True) & (df_negative['user_response'].str.contains("counter")==True) & (df_negative['user_response'].str.contains("variety")==False) & (df_negative['user_response'].str.contains("size")==False) & (df_negative['user_response'].str.contains("experience")==False)]
        train_neg8['classifier'] = 'issue at the billing counter'
        train_neg8=train_neg8[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg8), ignore_index=True)
        
        
        train_neg9 =df_negative['user_response']
        train_neg9= df_negative[(df_negative['user_response'].str.contains("discount")==True)  & (df_negative['user_response'].str.contains("counter")==False) & (df_negative['user_response'].str.contains("variety")==False) & (df_negative['user_response'].str.contains("size")==False) & (df_negative['user_response'].str.contains("experience")==False) & (df_negative['user_response'].str.contains("billing")==False)]
        train_neg9['classifier'] = 'discount related'
        train_neg9=train_neg9[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg9), ignore_index=True)
        
        
        train_neg10 =df_negative['user_response']
        train_neg10= df_negative[(df_negative['user_response'].str.contains("points")==True)  & (df_negative['user_response'].str.contains("variety")==False)| (df_negative['user_response'].str.contains("issued")==True) ]
        train_neg10['classifier'] = 'points related'
        train_neg10=train_neg10[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg10), ignore_index=True)
        
        train_neg11 =df_negative['user_response']
        train_neg11= df_negative[(df_negative['user_response'].str.contains("product")==True) & (df_negative['user_response'].str.contains("billing")==False) & (df_negative['user_response'].str.contains("counter")==False) &  (df_negative['user_response'].str.contains("discount")==False)  &  (df_negative['user_response'].str.contains("products")==True) & (df_negative['user_response'].str.contains("salesman")==False) & (df_negative['user_response'].str.contains("range")==False)  & (df_negative['user_response'].str.contains("staff")==False)]
        train_neg11['classifier'] = 'product related issues'
        train_neg11=train_neg11[['mobile','user_response','classifier']]
        train = train.append(pandas.DataFrame(data = train_neg11), ignore_index=True)

        return train
     

    def run_feedback(self,spark,mysqldetails,InsightName): 
        getset=GetSet()                     
        ''' Get the feedback data from the SQL Server'''
        query = "(SELECT feed_date,user_response, mobile FROM feed_master where char_length(user_response)>4 and user_response is not null and user_response !='') as df"
        query=getset.get_FeedbackQuery()
        feedbackdf= FeedbackEngine.mysqlRead(self,spark,mysqldetails[0],mysqldetails[1],query,mysqldetails[2],mysqldetails[3])
        #feedbackdf.head(10)
        '''Feedback Data is cleaned'''
        df=FeedbackEngine.DataCleaning(self,feedbackdf)
        #df.head(5)
        
        '''Pull out the data showing negative sentiemnt'''
        df_neg = FeedbackEngine.getNegSentimentDF(self,df)
        
        '''Check most Freq words in the user response having negative sentiment'''
        freqwords = FeedbackEngine.mostfreqwords(self,df_neg,50)
    
        '''Based on most freq words create a training data which classifies the negative user response into certain labels'''
        trainData= FeedbackEngine.CreateTrainingDF(self,df_neg)
        
        '''Get the data frame from df_neg which has not been classified into labels based on str.contain() function'''
        df_rem = FeedbackEngine.getuntrainedDF(self,trainData, df_neg)
        
        '''Apply Naive - Bayes - Algorithm on the untrained Data'''
        Y_pred = FeedbackEngine.NaiveBayesClassifier(self,trainData, df_rem)
        df_rem['classifier'] = Y_pred
        #print(df_rem[['user_response','classifier']].head(10))
        #print(df_rem.head(10))
        
        query = "(Select * from feedback_msg) as df"
        msg_df= FeedbackEngine.mysqlRead(self,spark,mysqldetails[0],mysqldetails[1],query,mysqldetails[2],mysqldetails[3])
        
        lst=[]
        for i in list(df_rem['classifier']):
            try:
                lst.append(msg_df['Message'][msg_df['Type']==i].values[0])
            except:
                lst.append('none')    
            #print(lst)
        #print(lst)    
        df_rem['Question']=pandas.Series(lst).values        
        df=df_rem[['mobile','Question']]
        df = df.rename(columns={'mobile':'Mobile','Question':InsightName})
        print("Feedback Questions")
        print(df.head(5))
        df[['Mobile',InsightName]] = df[['Mobile',InsightName]].astype('str')
        sdf = spark.createDataFrame(df)
        del [[df,msg_df,df_rem,feedbackdf]]
        gc.collect()
        return sdf

    
    
    
    
    
    
    
    
    
    
    
    
        
    
