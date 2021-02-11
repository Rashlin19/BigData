// Databricks notebook source
/** Title: Wisconsin Diagnostic Breast Cancer (WDBC)
Databricks notebook 2020-02-17 rr 

Wisconsin data, stored at UCI
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found at [Web Link] 
Given 10 features, predict malignant growth or benign growth, i.e. a binary classification
I downloaded this file and and then uploaded to Databricks' dbfs
1. Note the construction of a case class to describe the schema
2. I then just focused on one feature,"cThick",  to predict cancerous growth denoted by "clas" label 
NOTEs from the Wisconsin site:
Results:
	- predicting field 2, diagnosis: B = benign, M = malignant
	- sets are linearly separable using all 30 input features
	- best predictive accuracy obtained using one separating plane
		in the 3-D space of Worst Area, Worst Smoothness and
		Mean Texture.  Estimated accuracy 97.5% using repeated
		10-fold crossvalidations.  Classifier has correctly
		diagnosed 176 consecutive new patients as of November
		1995. 
1) ID number 
2) Diagnosis (M = malignant, B = benign) 
3-32) 
5. Number of instances: 569 
6. Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)
7. Attribute information

Ten real-valued features are computed for each cell nucleus: 

a) radius (mean of distances from center to points on the perimeter) 
b) texture (standard deviation of gray-scale values) 
c) perimeter 
d) area 
e) smoothness (local variation in radius lengths) 
f) compactness (perimeter^2 / area - 1.0) 
g) concavity (severity of concave portions of the contour) 
h) concave points (number of concave portions of the contour) 
i) symmetry 
j) fractal dimension ("coastline approximation" - 1)
*/
import org.apache.spark.sql.SQLImplicits
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.classification.{LogisticRegression }
//1017023,4,1,1,3,2,1,3,1,1,2
//1017122,8,10,10,8,7,10,9,7,1,4
//1018099,1,1,1,1,2,10,3,1,1,2
case class CancerClass(sample : Long,
                       cThick : Double,
                       uCSize: Int,
                       UCShape: Int,
                       mAdhes : Int,
                       sECSize : Int,
                          bNuc : Int,
                         bChrom: Int,
                           nNuc: Int,
                         mitosis: Int,
                            clas: Double)
//I will later cut down the original data set to just two features, cThick and clas ( the label)
case class Data(cThick: Double, clas : Double)
//here is original data set
val breastData = "/FileStore/tables/breastCancer.txt"
val recordSchema = new StructType()
                    .add("sample","long")
                    .add("cThick","double")
                    .add("uCSize","integer")
                    .add("uCShape","integer")
                    .add("mAdhes","integer")
                    .add("sECSize","integer")
                    .add("bNuc","integer")
                    .add("bChrom","integer")
                    .add("nNuc","integer")
                    .add("mitosis","integer")
                    .add("clas","double")  
val dfBreast = spark.read.format("csv").option("header", false).schema(recordSchema).load(breastData)
  dfBreast.show(2) 
println(f" Count of observations :${dfBreast.count()}%4.0f ")
val dsBreast= dfBreast.as[CancerClass] 
//dsBreast.show(3)
// val dfTruncated = dsBreast.selectExpr("cThick", "clas")
// I am going to just use ONE feature, cThick, to predict malignant cell growth growth or not
val dsTruncated = dsBreast.selectExpr("cThick", "clas").as[Data]
println(s"dsTruncated $dsTruncated showing 3 rows")
dsTruncated.show(3)
val correlation = dsTruncated.stat.corr("cThick", "clas")
println(f" Correlation, cThick, clas   = $correlation%2.2f ")
val stats = dsTruncated.describe().show()
display(dsTruncated)
// NOw 
//I converted column values, the "clas" feature,  to 0.0(no cancer) to 1.0( cancer)
//I directly then made a feature vector from the cThick feature, i.e. Vectors.dense(row.cThick)
 val dsData =  dsTruncated.map{  row =>{
             val index = if (row.clas == 2.0) 0.0 else 1.0
            (Vectors.dense(row.cThick),index ) }
            }.toDF("features", "label")
dsData.show(3)

val Array(trainingData, testData)= dsData.randomSplit(Array(0.8, 0.2))
println(f" Training data count = ${trainingData.count()}%4.0f , Test data count = ${testData.count()}")

val lr = new LogisticRegression()
val model = lr.fit(trainingData)
println(f" model.coefficients = ${model.coefficients} , model.intercept = ${model.intercept}%2.2f")



// COMMAND ----------


