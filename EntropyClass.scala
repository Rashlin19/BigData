// Databricks notebook source
import scala.math._
type D = Double
def ln2(x:D):D = log10(x)/log10(2.0)
def entropy(yes:D,no:D ,total:D):D = {
  val p= yes/total
  val q= no/total
  val r=1-p-q
  if(p>0) - (p *ln2(p)+ q *ln2(q) +r*ln2(r))
else 0
}
/*entropy(6,12)
//entropy(2,3)
entropy(1,3)
val parentEntropy = entropy(7,12)
val bodyEntropy = 7/12.0 * entropy(5,7) +5/12.0 *entropy(1,5)
val IG = parentEntropy - bodyEntropy*/

entropy(5,6,15)

val parentEntropy = entropy(5,6,15)
val bodyEntropy = 5/15.0 * entropy(5,7,15) +6/15.0 *entropy(5,6,15)+ 4/15.0*entropy(3,4,15)
val IG = parentEntropy - bodyEntropy

// COMMAND ----------


