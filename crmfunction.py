from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql import Row
from datetime import datetime, timedelta
from pyspark.sql.types import *
#from astropy.units import merg
#from sqlalchemy.sql.sqltypes import INTEGERTYPE
#from skimage.feature.tests.test_util import plt
import functools 
from  mysqlplugin import MySqlConnector
from programInsight import  ProgramInsightFetchEngine
from tags import Tagging
from message import Message
import re
from collections import OrderedDict
import mysql.connector as conn
from ast import literal_eval
from loyaltyfunction import Loyalty
from pyspark.sql.functions import row_number
from pyspark.sql.window import Window
from cltv import CLTV
from atv_points_recom import value_recommend
from feedback_sentiment import FeedbackEngine
from configparser import ConfigParser
from getset import GetSet
from insight_output import TagOutputs
from logger import logg
Logg=logg()
import logging
from segment_connect import SegmentConnect
from probable_churn import probchurn

# Logg.createFolder(logger_path,"basket")
# logger_path='/datadrive/Document/Projects/shinjini/logger/insight/basket/'
logger_1 = logging.getLogger("infolog")
logger_2 = logging.getLogger("errorlog")
EmailTrigger=1
mail_body = 'Insight process for @InsightName has failed or has ended with errors. Error_description : "@error" This is an automated mail alert, do not reply to this mail.'
 

class CRM:
    
    def NPS(self,spark,mysqldetails,InsightName,insights_repeater_total,npsFeed):
        getset=GetSet()
        npsFeed=npsFeed.withColumn("Type",when(npsFeed['user_response']<7,"'Detractor'").when(npsFeed['user_response']<9,"'Passive'").otherwise("'Promoter'"))
        querylang = "(select distinct(LanguageCode) from nps_revert_questions) as df"
        querylang = getset.get_NPSLangQuery()
        langs=MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], querylang, mysqldetails[2], mysqldetails[3])
        l=langs.rdd.map(lambda row: row.LanguageCode).collect()
        for i in ["'Detractor'","'Passive'","'Promoter'"]:
            m=npsFeed.where(col('Type')==i).select("Mobile")
            j=0
            for lang in l:
                question="(select Question from nps_revert_questions where CustomerType="+i+") as df"
                npsq=MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], question, mysqldetails[2], mysqldetails[3])
                q=npsq.rdd.map(lambda row: row.Question).collect()[0]
                memberDF = m.withColumn(InsightName,lit(q)).withColumn('Language',lit(lang))
                if(j==0):
                    memberDF=memberDF
                    j=j+1
                else:    
                    memberDF=memberDF.union(memberDF)       
        #memberDF.show(5,False)
        
        return memberDF
            
    def clv(self, spark,mysqldetails,Start,End,tagsDict,memberDF,InsightName):
        memberDF=CLTV.cltv_output(self,spark,mysqldetails,InsightName,memberDF)
        memberDF=memberDF.select(col("Mobile"),col(InsightName).cast('int'))
        memberDF=memberDF.filter((col(InsightName)>=Start) & (col(InsightName)<=End))
        memberDF=Loyalty.tags_calculation(self,spark,mysqldetails,tagsDict,memberDF,InsightName)       
        #memberDF.show(5,False)
        return memberDF
    
    def atvPlusPoints(self, spark,mysqldetails,InsightName,Start,End,Start1,End1,insights_repeater_total,memberDF,tagsDict):
            #memberDF=memberDF.limit(2000)
            MobileList=memberDF.rdd.map(lambda row: "%s" %(row.Mobile))
            MobileString=Loyalty.listToString(self,MobileList.collect())
            query1= "(Select Mobile,Amount,ModifiedBillNo,TxnDate,AvailablePoints from txn_report_accrual_redemption where Mobile in ("+MobileString+"))as df"
            txnDF = MySqlConnector.mysqlRead(self,spark,mysqldetails[0],mysqldetails[1],query1,mysqldetails[2],mysqldetails[3])          
            df = txnDF.groupBy("Mobile").agg((sum(col("Amount"))/countDistinct(col("ModifiedBillNo"))).alias("ATV"))
            w = Window.partitionBy('Mobile')
            ndf = txnDF.withColumn('maxTxnDate', max('TxnDate').over(w))\
                .where(col('TxnDate') == col('maxTxnDate'))\
                .select('Mobile','AvailablePoints')
            new=df.join(ndf,'Mobile','inner')
            #new.show(5)
            df.unpersist()    
            DF=new.filter(((col('AvailablePoints')*Start1*100/col('ATV'))>=Start) & ((col('AvailablePoints')*Start1*100/col('ATV'))<=End))   
            new.unpersist()
            sendDF=DF.withColumn('Value',col('ATV')+(col('AvailablePoints')*Start1)).select('Mobile','Value') 
            #sendDF.show(5)
            DF.unpersist()
            memberDF=value_recommend.value_recom(self,spark,mysqldetails,sendDF,InsightName,End1)
            field = [StructField('Mobile',StringType(), True),StructField(InsightName, StringType(), True)]
            schema = StructType(field)
            memberDF = spark.createDataFrame(memberDF,schema)
            memberDF = Loyalty.tags_calculation(self, spark,mysqldetails,tagsDict,memberDF,InsightName)
            #memberDF.show(5,False)
            return memberDF
    
    def NPV(self,spark,mysqldetails,InsightName,Start,End,insights_repeater_total,memberDF,npsFeed,tagsDict):
         #memberDF=memberDF.limit(2000)
         npsFeed=npsFeed.select(col("Mobile"),col("user_response").cast('int'))
         npsFeed=npsFeed.groupBy('Mobile').agg(avg('user_response').alias('user_response'))
         clt = CLTV.custvalue(self,spark,mysqldetails,InsightName,memberDF)
         df=npsFeed.join(clt,"Mobile","right_outer")
         df.show(5)
         #df = df.na.fill(0)
         memberDF=df.withColumn(InsightName,when(col('user_response').isNull(),col(InsightName)).otherwise(col('user_response')*0.5+(0.5*col(InsightName)))).select("Mobile",InsightName)
         memberDF = memberDF.filter((col(InsightName)<=End) & (col(InsightName)>=Start))
         memberDF = Loyalty.tags_calculation(self, spark,mysqldetails,tagsDict,memberDF,InsightName)
         #memberDF.show(5,False)
         return memberDF   
  
    def feedback_message(self,spark,mysqldetails,InsightName,insights_repeater_total,npsFeed):
         logger_1.info("creating feedback message for insight "+InsightName+"..")                    
         memberDF=FeedbackEngine.run_feedback(self,spark,mysqldetails,InsightName)
         return memberDF  
    
    def prob_churn(self,spark,mysqldetails,InsightName,memberDF,tagsDict):
        memberDF = probchurn.churners(self,spark,mysqldetails,InsightName,memberDF,tagsDict)
        return churnDF
    
    def processing(self, spark,mysqldetails,InsightName,Start,End,Start1,End1,Start2,End2,insights_repeater_total,Type,y,memberDF,npsFeed,tagsDict):
             
                if(Type == "Varchar" or Type == "Range"):
                            
                            if(y=="13"):
                                 memberDF=CRM.NPS(self, spark,mysqldetails,InsightName,insights_repeater_total,npsFeed)
                                 
                            elif(y=="14"):
                                 memberDF=CRM.clv(self, spark,mysqldetails,Start,End,tagsDict,memberDF,InsightName)  
                            
                            elif(y=="12"):
                                 memberDF=CRM.NPV(self,spark,mysqldetails,InsightName,Start,End,insights_repeater_total,memberDF,npsFeed,tagsDict)     
                            
                            elif(y=="19"):
                                 memberDF=CRM.feedback_message(self, spark, mysqldetails, InsightName, insights_repeater_total, memberDF, npsFeed, tagsDict, l)     
                
                elif(Type == "Double"):
                    
                            if(y=="16"):     
                                 memberDF=CRM.atvPlusPoints(self, spark,mysqldetails,InsightName,Start,End,Start1,End1,insights_repeater_total,memberDF,tagsDict)
                            
                            elif(y=="b"):
                                 memberDF=CRM.dummy(self, spark,mysqldetails,InsightName,Start,End,Start1,End1,insights_repeater_total,memberDF,tagsDict)
                              
                elif(Type == "Triple"):
                    
                            if(y=="c"):
                                 memberDF=CRM.dummy(self, spark,mysqldetails,InsightName,Start,End,Start1,End1,Start2,End2,insights_repeater_total,memberDF,tagsDict)
                               
                            elif(y=="d"):
                                 memberDF=CRM.dummy(self, spark,mysqldetails,InsightName,Start,End,Start1,End1,Start2,End2,insights_repeater_total,memberDF,tagsDict)
                
                return  memberDF
             
    
    def member_data(self,spark,mysqldetails):
    
        programDF1 = MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], "(select mobile from member_report) as m",mysqldetails[2], mysqldetails[3])
        #programDF1 = MySqlConnector.mysqlRead(self,spark,"182.18.183.247", "rituwears", "member_report", "neeraj2_247", "easy@247")
        memberDF = programDF1.selectExpr("Mobile")      
        return memberDF
    
    def tier(self,spark,mysqldetails):
        #programDF = programDF.selectExpr("Mobile","BillNo","ModifiedBillNo","TxnDate","TxnTime","Amount","PointsCollected","PointsSpent","LapsedPoints","AvailablePoints","Store","StoreCode")
        tier_DF= MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], "tier_detail_report", mysqldetails[2], mysqldetails[3])  
        tier_DF.printSchema()          
        return tier_DF
    
    def nps_data(self,spark,mysqldetails):
        getset=GetSet()
        query="(select * from feed_master where query_type='NPS' and user_response is not null) as df";        
        query = getset.get_NPSDataQuery()
        try:
            nps_feed=MySqlConnector.mysqlRead(self,spark,mysqldetails[0], mysqldetails[1], query, mysqldetails[2], mysqldetails[3])
        except Exception as e:
            logger_2.error("Error reading table feed_master ",exc_info=e)
            nps_feed=1
        return nps_feed    
        
         
    def crm_engine(self,spark,query_list,listdict,mysqldetails,tag_query_dictionary,tagdict,segmobDF,mem_segDF,insights_tagsDF):
        mem_segDF=spark.createDataFrame(mem_segDF)
        if len(insights_tagsDF)>0:
            insights_tagsDF=spark.createDataFrame(insights_tagsDF)
        else:
            insights_tagsDF=spark.createDataFrame(spark.sparkContext.emptyRDD(), StructType([]))
        logger_1.info("######### Creating CRM type insights - process begins #########")
        getset=GetSet()
        segcon = SegmentConnect()
        #print(len(query_list))
        query_list_id = {}
        #memberDF,memberDFfull = Loyalty.member_data(self, spark,mysqldetails)
        npsFeed = CRM.nps_data(self,spark,mysqldetails)
        memberDF = CRM.member_data(self,spark,mysqldetails)
        ##tierDF = CRM.tier(self,spark,mysqldetails)
        
        i=0
        idlist=[]
              
        print(query_list)
        for query in query_list:
            query_split = query.split(",")
            Insightid= query_split[0]
            query_list_id.setdefault(Insightid,[]).append(query)
        insights_repeater_total=""  
            
        j=0
        listdict = list(query_list_id.keys())
        #print(listdict)
        logger_1.info("--Insights creation started--")
        
        for y in list(set(listdict)):
            
            y=str(y)
            i=0
            query = query_list_id[y]
            for x in range(len( query_list_id[y])):
                i=i+1
                query_split = query[i-1].split(",")
                Type= query_split[3]
                InsightName = query_split[1]
                Start = query_split[4]
                End = query_split[5]
                Unit = query_split[6]
                Start1= query_split[7]
                End1= query_split[8]
                Unit1 = query_split[9]
                Start2= query_split[10]
                End2= query_split[11]
                Group = query_split[12]
                id = query_split[13]
                mobiles=segcon.unionmobs(segmobDF,id)
                field = [StructField('Mobile',StringType(), True)]
                schema = StructType(field)
                mobiles = spark.createDataFrame([[x] for x in mobiles], schema)
                #mobiles = mobiles.repartition(8)
                    
                #memberDF=memberDF.repartition(8)
                #memberDFfull=memberDFfull.filter(memberDFfull.Mobile.isin(mobiles.value))
                memberDF = memberDF.join(mobiles, "Mobile","inner")
                
                #npsFeed=npsFeed.repartition(8)
                try:
                    npsFeed = npsFeed.join(mobiles, "Mobile","inner")
                except:
                    npsFeed="hai hi nahi"    
                
                #tierDF = tierDF.repartition(8)
                ##tierDF = tierDF.join(mobiles, "Mobile","inner")
                
                idlist.append(id)
                
                if(InsightName=='NPS'):
                   try:
                       logger_1.info("Creating insight "+InsightName) 
                       #memberDF=memberDF.filter(memberDF.Mobile.isin(mobiles))
                       memberDF=CRM.NPS(self,spark,mysqldetails,InsightName,insights_repeater_total,npsFeed) 
                   except Exception as e:
                       logger_2.error("Error creating insight "+InsightName)
                       body= mail_body.replace("@InsightName", str(InsightName).upper())
                       body = Logg.firstLineError(e,InsightName,body)
                       Logg.mailAlert(literal_eval(getset.get_mailRecipientsList()), body, EmailTrigger)  
                       continue
                    
                   memberDF = memberDF.withColumn('InsightID',lit(y)).withColumn('InsightGroup',lit(Group))
                   memberDF=TagOutputs.outputtags(memberDF)
                   memberDF=memberDF.join(mem_segDF,memberDF.Mobile==mem_segDF.Mobile)
                   #memberDF = memberDF.withColumn('ExpiryDate',date_add(lit(datetime.now().date()),30))
                   #memberDF = memberDF.withColumn('InsertionDate',lit(datetime.now().date()))
                   #memberDF = memberDF.withColumnRenamed(InsightName,'Message') 
                   #memberDF = memberDF.withColumn('lastRunDate',lit(datetime.datetime.now()))
                   #memberDF.show(5)
                   logger_1.info("Insight "+InsightName+" created. Updating message table...") 
                   Loyalty.create_table(self, spark, memberDF, mysqldetails, insights_tagsDF)    
                   memberDF.unpersist()
                
                elif(InsightName=='feedback_message'):
                   try: 
                       logger_1.info("Creating insight "+InsightName) 
                       memberDF=CRM.feedback_message(self,spark,mysqldetails,InsightName,insights_repeater_total,npsFeed)
                   except Exception as e:
                       logger_2.error("Error creating insight "+InsightName)
                       body= mail_body.replace("@InsightName", str(InsightName).upper())
                       body = Logg.firstLineError(e,InsightName,body)
                       Logg.mailAlert(literal_eval(getset.get_mailRecipientsList()), body, EmailTrigger)  
                       continue
                        
                   memberDF = memberDF.withColumn('InsightID',lit(y)).withColumn('InsightGroup',lit(Group))
                   memberDF=TagOutputs.outputtags(memberDF)
                   memberDF=memberDF.join(mem_segDF,memberDF.Mobile==mem_segDF.Mobile)
                   #memberDF = memberDF.withColumn('lastRunDate',lit(datetime.datetime.now()))
                   #memberDF.show(5)
                   logger_1.info("Insight "+InsightName+" created. Updating message table...")
                   Loyalty.create_table(self, spark, memberDF, mysqldetails, insights_tagsDF)    
                   memberDF.unpersist()
                
                else:
                   
                    tagsDict=dict([ (k, tag_query_dictionary.get(k)) for k in tagdict[id] ])

                    if (j==0):
                        insights_repeater_total= InsightName
                        j=j+1
                        try:
                            logger_1.info("Creating Insight: "+InsightName)
                            #memberDF=memberDF.filter(memberDF.Mobile.isin(mobiles))
                            memberDF = CRM.processing(self, spark,mysqldetails,InsightName,Start,End,Start1,End1,Start2,End2,insights_repeater_total,Type,y,memberDF,npsFeed,tagsDict)
                        except Exception as e:
                            logger_2.error("Error creating insight "+InsightName,exc_info=e) 
                            body= mail_body.replace("@InsightName", str(InsightName).upper())
                            body = Logg.firstLineError(e,InsightName,body)
                            Logg.mailAlert(literal_eval(getset.get_mailRecipientsList()), body, EmailTrigger)  
                            continue

                        memberDF = memberDF.withColumn('InsightID',lit(y)).withColumn('InsightGroup',lit(Group))
                        #memberDF = memberDF.withColumn('ExpiryDate',date_add(lit(datetime.now().date()),30))
                        #memberDF=memberDF.withColumn('InsertionDate',lit(datetime.now().date())) 
                        #memberDF = memberDF.withColumn('lastRunDate',lit(datetime.datetime.now()))
                        #memberDF.show(5)
                    else :
                        insights_repeater_total=insights_repeater_total+","+InsightName
                        #print(insights_repeater_total)
                        try: 
                            logger_1.info("Creating Insight: "+InsightName)
                            #memberDF=memberDF.filter(memberDF.Mobile.isin(mobiles))
                            #memberDF=memberDF.filter(memberDF.Mobile.isin(mobiles))
                            memberDF = CRM.processing(self, spark,mysqldetails,InsightName,Start,End,Start1,End1,Start2,End2,insights_repeater_total,Type,y,memberDF,npsFeed,tagsDict)
                            if(memberDF == 1):
                                continue
                            memberDF.persist()
                        except Exception as e:
                            logger_2.error("Error creating insight "+InsightName,exc_info=e)  
                            body= mail_body.replace("@InsightName", str(InsightName).upper())
                            body = Logg.firstLineError(e,InsightName,body)
                            Logg.mailAlert(literal_eval(getset.get_mailRecipientsList()), body, EmailTrigger)                        
                            continue
                        memberDF = memberDF.withColumn('InsightID',lit(y)).withColumn('InsightGroup',lit(Group))
                        #memberDF = memberDF.withColumn('ExpiryDate',date_add(lit(datetime.now().date()),30))
                        #memberDF=memberDF.withColumn('InsertionDate',lit(datetime.now().date()))                           
                        #memberDF = memberDF.withColumn('lastRunDate',lit(datetime.datetime.now()))  
                                                        
                    memberDF=TagOutputs.outputtags(spark,memberDF)
                    #mem_segDF.show(10,False)
                    memberDF=memberDF.join(mem_segDF,"Mobile","inner")
                    #memberDF.show(10,False)            
                    print("Updating message table....") 
                    logger_1.info("Insight "+InsightName+" created. Updating message table...")
                    try:
                        Loyalty.create_table(self,memberDF,mysqldetails,insights_tagsDF)
                    except Exception as e:
                        logger_2.error("Error updating tags table",exc_info=e)    
        #memberDF=memberDF.drop_duplicates(subset=['Mobile'])
        indexTypeName=mysqldetails[1]+"insights"+"/data"
        #memberDF = memberDF.withColumn("ExpiryDate",lit(None).cast("date")).withColumn("flag",lit(None).cast('int'))
        print("-------------------------------------------------------------------")
        logger_1.info("All CRM insights created")
        
        """
        memberDF.write.format("org.elasticsearch.spark.sql").option("es.nodes", "10.105.2.7")\
            .option("es.nodes.wan.only", "true")\
            .option("es.batch.write.retry.count", "25")\
            .option("es.batch.write.retry.wait", "2500")\
            .option("es.batch.size.entries", "500")\
            .option("es.mapping.id", "Mobile")\
            .option("es.write.operation", "upsert")\
            .mode("append").save(indexTypeName)
        """    
            
        ab1 = ','.join(str(e) for e in idlist)
        ab1="("+ab1+")"
        print(ab1)
        Loyalty.update_list(mysqldetails,ab1)        
