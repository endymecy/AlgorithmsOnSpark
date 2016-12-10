package org.enme.sampling

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.StructType

/**
  * Created by endy on 16-12-8.
  */

trait OverSamplingParams extends Params{
  final val threshold = new DoubleParam(this, "threshold", "The threshold whether to  " +
    "undersampling sample of a class", (x: Double) => x > 1)
  def setThreshold(value: Double): this.type = set(threshold, value)

  final val dependentColName = new Param[String](this, "dependentColName", "The column that " +
    "provide label values")
  def setDependentColName(value: String): this.type = set(dependentColName, value)

  final val primaryClass = new DoubleParam(this, "primaryClass", "primary class that to under " +
    "sampling")
  def setPrimaryClass(value: Double): this.type = set(primaryClass, value)
}


class OverSampling(override val uid: String) extends Transformer with OverSamplingParams {
  def this() = this(Identifiable.randomUID("OverSampling"))

  /**
    * Transforms the input dataset.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val labelCountPair = dataset.groupBy($(dependentColName)).count().collect()

    val primaryClassCount = labelCountPair
      .filter{ case Row(label: Double, count: Long) => label == ${primaryClass}}
          .map(x => x.get(1)).headOption.getOrElse(-1L).asInstanceOf[Long]

    if (primaryClassCount == -1) throw new Exception("The label is not exist")

    val res = labelCountPair.zipWithIndex
      .map {
          case (Row(label: Double, count: Long), index: Int) =>
            val ratio = primaryClassCount / count.toDouble

            /**
              * if ratio < threshold, only return samples of this label,
              * otherwise we sample the data from the samples of this label.
              *
              * The desired number of samples is : num = primaryClassCount * threshold
              * so the fraction of sample method is: num / count = ratio / threshold.
              * Because fraction > 1, the value of 'withReplacement' parameter must be true
              */
            val df = if (ratio < ${threshold}) {
              dataset.filter(col($(dependentColName)) === label)
            } else {
              val desiredFraction = ratio / ${threshold}
              dataset.filter(col($(dependentColName)) === label)
                .sample(withReplacement = true, desiredFraction)
            }
            df.toDF()
     }.reduce(_ union _)

    res
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  /**
    * :: DeveloperApi ::
    *
    * Check transform validity and derive the output schema from the input schema.
    *
    * Typical implementation should first conduct verification on schema change and parameter
    * validity, including complex parameter interaction checks.
    */
  override def transformSchema(schema: StructType): StructType = {
    schema
  }
}
