package org.apache.spark.ml.timeseries.models

import io.transwarp.discover.timeseries.params.TimeSeriesParams
import org.apache.commons.math3.random.RandomGenerator
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

/**
  * Created by endy on 16-12-22.
  */

class ARGARCH(override val uid: String)  extends Estimator[ARGARCHModel] with TimeSeriesParams {
  setDefault(timeCol -> "time",
    timeSeriesCol -> "timeseries")

  def this() = this(Identifiable.randomUID("ARGARCH"))
  /**
    * Fits a model to the input data.
    */
  override def fit(dataset: Dataset[_]): ARGARCHModel = {
    val data = dataset.select(${timeCol}, ${timeSeriesCol}).rdd.map {
      case Row(time: String, value: Double) => (time, value)
    }.sortByKey().collect()

    val dataVector = Vectors.dense(data.map(x => x._2))

    val arModel = new Autoregression().fit(dataset)
    val residuals = arModel.removeTimeDependentEffects(dataVector)
    val dataFrame = generateDf(dataset.sparkSession, residuals.toArray)
    val garchModel = new GARCH().fit(dataFrame)

    new ARGARCHModel(arModel.c, arModel.coefficients(0), garchModel.omega, garchModel.alpha,
      garchModel.beta)
  }

  override def copy(extra: ParamMap): Estimator[ARGARCHModel] = defaultCopy(extra)

  /**
    * :: DeveloperApi ::
    *
    * Check transform validity and derive the output schema from the input schema.
    *
    * Typical implementation should first conduct verification on schema change and parameter
    * validity, including complex parameter interaction checks.
    */
  override def transformSchema(schema: StructType): StructType = schema

  private def generateDf(sparkSession: SparkSession, array: Array[Double]): DataFrame = {
    val schema = StructType(Array(StructField(${timeCol}, StringType), StructField(${timeSeriesCol},
      DoubleType)))

    val rdd = sparkSession.sparkContext.parallelize(
      array.zipWithIndex.map(x => Row(x._2.formatted("%010d"), x._1)))

    sparkSession.createDataFrame(rdd, schema)
  }
}

class ARGARCHModel(override val uid: String, val c: Double, val phi: Double, val omega: Double,
                   val alpha: Double, val beta: Double) extends
                Model[ARGARCHModel] with TimeSeriesParams {

  def this(c: Double, phi: Double, omega: Double, alpha: Double, beta: Double) =
      this(Identifiable.randomUID("ARGARCHModel"), c, phi, omega, alpha, beta)

  override def copy(extra: ParamMap): ARGARCHModel = defaultCopy(extra)

  /**
    * Transforms the input dataset.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val data = dataset.select(${timeCol}, ${timeSeriesCol}).rdd.map {
      case Row(time: String, value: Double) => (time, value)
    }.sortByKey().collect()

    val dataVector = Vectors.dense(data.map(x => x._2))

    val dest = addTimeDependentEffects(dataVector)

    val resRDD = dataset.sparkSession.sparkContext.parallelize(dest.toArray.map(x => Row(x)))

    val structType = transformSchema(dataset.schema)

    dataset.sparkSession.createDataFrame(resRDD, structType)
  }

  /**
    * :: DeveloperApi ::
    *
    * Check transform validity and derive the output schema from the input schema.
    *
    * Typical implementation should first conduct verification on schema change and parameter
    * validity, including complex parameter interaction checks.
    */
  override def transformSchema(schema: StructType): StructType = {
    StructType(Array(StructField("ARGARCH", DoubleType)))
  }

  def removeTimeDependentEffects(ts: Vector): Vector = {
    val destArr = new Array[Double](ts.size)
    var prevEta = ts(0) - c
    var prevVariance = omega / (1.0 - alpha - beta)
    destArr(0) = prevEta / math.sqrt(prevVariance)
    for (i <- 1 until ts.size) {
      val variance = omega + alpha * prevEta * prevEta + beta * prevVariance
      val eta = ts(i) - c - phi * ts(i - 1)
      destArr(i) = eta / math.sqrt(variance)

      prevEta = eta
      prevVariance = variance
    }
    new DenseVector(destArr)
  }

  def addTimeDependentEffects(ts: Vector): Vector = {
    val destArr = new Array[Double](ts.size)
    var prevVariance = omega / (1.0 - alpha - beta)
    var prevEta = ts(0) * math.sqrt(prevVariance)
    destArr(0) = c + prevEta
    for (i <- 1 until ts.size) {
      val variance = omega + alpha * prevEta * prevEta + beta * prevVariance
      val standardizedEta = ts(i)
      val eta = standardizedEta * math.sqrt(variance)
      destArr(i) = c + phi * destArr(i - 1) + eta

      prevEta = eta
      prevVariance = variance
    }
    new DenseVector(destArr)
  }

  private def sampleWithVariances(n: Int, rand: RandomGenerator): (Array[Double], Array[Double]) = {
    val ts = new Array[Double](n)
    val variances = new Array[Double](n)
    variances(0) = omega / (1 - alpha - beta)
    var eta = math.sqrt(variances(0)) * rand.nextGaussian()
    for (i <- 1 until n) {
      variances(i) = omega + beta * variances(i-1) + alpha * eta * eta
      eta = math.sqrt(variances(i)) * rand.nextGaussian()
      ts(i) = c + phi * ts(i - 1) + eta
    }

    (ts, variances)
  }

  /**
    * Samples a random time series of a given length with the properties of the model.
    *
    * @param n The length of the time series to sample.
    * @param rand The random generator used to generate the observations.
    * @return The samples time series.
    */
  def sample(n: Int, rand: RandomGenerator): Array[Double] = sampleWithVariances(n, rand)._1
}
