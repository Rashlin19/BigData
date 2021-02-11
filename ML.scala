// Databricks notebook source
import org.apache.spark.ml.regression.{LinearRegression}
import org.apache.spark.ml.linalg.{ Vectors, Vector}
import org.apache.spark.ml.feature.VectorAssembler
val tdf = spark.read.format("csv")
.option("header", "false")
.option("inferSchema", "true")
.option("delimiter", " ")
.load("/FileStore/tables/wickensBivariate.txt")
.toDF("F", "label")
val vectorAssembler = new VectorAssembler().
  setInputCols(Array("F")).
  setOutputCol("features")

val trainingData = vectorAssembler.transform(tdf).select("label","features")

trainingData.show()                               
val lr = new LinearRegression()                   
val lrModel = lr.fit(trainingData)                                       

val trainingSummary = lrModel.summary 
println(f"Coeff : ${lrModel.coefficients(0)}%2.2f")
println(f"Intercept: ${lrModel.intercept}%2.2f")
println(f"MSE: ${trainingSummary.meanSquaredError}%2.2f")
println(f"RMSE: ${trainingSummary.rootMeanSquaredError}%2.2f")
println(f"r2 : ${trainingSummary.r2}%2.2f")


// COMMAND ----------


