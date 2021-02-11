// Databricks notebook source
import Math._
def ln2(a:Double) = log(a)/log(2)
def entropy(chk : Double , total : Double) : Double = {
  val p = chk/total
  val q = 1 - p
  if (p*q > 0 ) - (p*ln2(p) + q*ln2(q))
  else 0.0
} 
val parent = entropy(7,12)
val square = entropy(5,9)
val round = entropy(2,3)
val avg1 = 9/12*square + 3/12*round
val igBody = parent - avg1  
  
val rectangle = entropy(5,6)
val oval = entropy(1,6)
val avg2 = 6.0/12*rectangle + 6.0/12*oval
val IG_Head = parent - avg2  
   
val SQ_REC = entropy(5,9)
val ROUND_REC = entropy(2,3)
val avg3 = 5.0/6*square + 1/12*round
val IG_Part1 = parent - avg1  
  

val SQ_OVAL= entropy(4,6)
val ROUND_OVAL = entropy(2,6)
val avg4 = 4.0/6*square + 2.0/12*round
val IG_Part2 = parent - avg4

val avg34 = (avg3 + avg4)/2
val IG= parent - avg34

// COMMAND ----------


