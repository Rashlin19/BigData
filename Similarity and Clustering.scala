// Databricks notebook source
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SparkSession, Encoder, SQLImplicits, DataFrame, Dataset        }
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types._
import scala.math._
import org.apache.spark.sql.Row            
val spark = SparkSession.builder .master("local[*]").appName("IFT598Similarity") .getOrCreate()
                                                  
import spark.implicits._

//creating one  case class to represent a person
case class P( name: String, age: Double, income : Double, cards: Double,
        responseToCCOffer : Double, distance : Double,
  invWeightingFactor : Double , finalWeightContribution : Double)

//math to calculate distance between two points
def pairDistances ( p : P,  q : P  ):Double = {
   val  (a1, i1, c1)  = (p.age, p.income, p.cards)
   val (a2, i2, c2)  = (q.age, q.income,  q.cards)
  sqrt(pow((a1 - a2), 2) + pow((i1 - i2) ,2) + pow(( c1 - c2), 2) )
        }  

//math to calculate distance between main point and other points
def  AddDistances( target: P, persons: List[P]) :List[P] =
         persons. map{ p => { val d =pairDistances( target, p)
                                           p.copy( distance = d ) } }

//reciprocal of distance
def AddReciprocalDistances( personsWithDistance : List[P] ) : List[P] = 
                                                 personsWithDistance.map{ p => {
                                                        val d = p.distance
                                                        val recipDist2 =  if (d != 0.0)  1/( d * d) else 0.0
                                                         p.copy( invWeightingFactor = recipDist2 )
                                                          } }  
                                                  
//sum of all reciprocal distance
def SumReciprocalDistances( personsWithReciprocalDistances : List[ P ]) =
             personsWithReciprocalDistances.map{_.invWeightingFactor}.sum  
   
//contributions of each person
def AddContributions( personsWithReciprocals : List[P], sumRecips : Double ) : List[P] =
      personsWithReciprocals.map{ p =>
  p.copy(finalWeightContribution= p.invWeightingFactor/sumRecips)
               }    

val fn = "/FileStore/tables/sim_clus.csv"
//loading the raw data file and defining its schema                                             
val mySchema = StructType( Array(StructField("name", StringType, false),
                                 StructField("age", DoubleType, false),
                                      StructField("income", DoubleType, false),
                                          StructField("cards", DoubleType, false),
                                               StructField("responseToCCOffer", DoubleType, false),
                                                   StructField("distance",DoubleType, false),
                                                     StructField("invWeightingFactor",DoubleType, false),
                                                        StructField("finalWeightContribution",DoubleType, false)
                                                      ))
                                                  
val rawData = spark.read.format("csv")
.option("header", "false")
.schema(mySchema).load(fn)                                          
rawData.show()                                  
 
//conver data into a list
val ds = rawData.as[P]                           
val persons = ds.collect().toList 

//taking David as the main variable
val targetDavid = persons(0)                    

//calculate distance between each person and david
val personsWithDistances = AddDistances(targetDavid, persons)
val personWithReciprocalDistance = personsWithDistances.sortBy(_.distance)
                      .foreach{p =>  println(f"${p.name}%10s ${p.age}%3.2f ${p.income}%3.2f ${p.distance}%4.1f")}         
println() 

//similarity weight for all persons
val similarityWeight = AddReciprocalDistances(personsWithDistances)
similarityWeight.sortBy(_.distance)
                      .foreach{p =>  
println(f"${p.name}%10s ${p.age}%3.2f ${p.income}%3.2f ${p.distance}%4.1f ${p.invWeightingFactor}%1.6f")}
println()

//each persons contribution
val sumRecips = similarityWeight.foldLeft(0.0)((accum,p)=>accum + p.invWeightingFactor)
val contributions = AddContributions(similarityWeight,sumRecips)
val sortedPersons = contributions.sortBy(p=>(p.finalWeightContribution, p.responseToCCOffer))
sortedPersons.foreach{p =>  println(f"${p.name}%10s ${p.age}%3.2f ${p.income}%3.2f ${p.distance}%4.1f ${p.invWeightingFactor}%1.6f ${p.finalWeightContribution}%1.3f")
}
println()

//probability of David accepting the credit card
val prob = sortedPersons.filter(_.responseToCCOffer==1)
val probability = prob.foldLeft(0.0)((accum,p) => accum + p.finalWeightContribution)
println(f"Probability of David accepting the Credit Card is : ${probability}%2.2f")
println()

// COMMAND ----------


