package org.apache.spark.ml.timeseries.models

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.timeseries.{Lag, MatrixUtil}
import org.apache.spark.ml.timeseries.params.TimeSeriesParams
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
  * Created by endy on 16-12-16.
  */

trait AutoregressionParams extends TimeSeriesParams {

  final val maxLag = new Param[Int](this, "maxLag", "max lag")
  def setMaxLag(value: Int): this.type = set(maxLag, value)

  final val noIntercept = new Param[Boolean](this, "noIntercept", "no intercept")
  def setNoIntercept(value: Boolean): this.type = set(noIntercept, value)
}


class Autoregression(override val uid: String)
  extends Estimator[ARModel] with AutoregressionParams{

  def this() = this(Identifiable.randomUID("Autoregression"))

  setDefault(noIntercept -> false, maxLag -> 1, timeCol -> "time",
    timeSeriesCol -> "timeseries")
  /**
    * Fits a model to the input data.
    */
  override def fit(dataset: Dataset[_]): ARModel = {
    val data = dataset.select(${timeCol}, ${timeSeriesCol}).rdd.map {
      case Row(time: String, value: Double) => (time, value)
    }.sortByKey().collect()

    val dataVector = Vectors.dense(data.map(x => x._2))

    // Make left hand side
    val Y = MatrixUtil.toBreeze(dataVector)(${maxLag} until dataVector.size)
    // Make lagged right hand side
    val X = Lag.lagMatTrimBoth(dataVector, ${maxLag})

    val regression = new OLSMultipleLinearRegression()
    regression.setNoIntercept(${noIntercept}) // drop intercept in regression
    regression.newSampleData(Y.toArray, MatrixUtil.matToRowArrs(X))
    val params = regression.estimateRegressionParameters()
    val (c, coeffs) = if (${noIntercept}) (0.0, params) else (params.head, params.tail)

    new ARModel(c, coeffs)
      .setTimeCol(${timeCol})
      .setTimeSeriesCol(${timeSeriesCol})
  }

  override def copy(extra: ParamMap): Estimator[ARModel] = defaultCopy(extra)

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

class ARModel(override val uid: String, val c: Double, val coefficients: Array[Double]) extends
  Model[ARModel] with AutoregressionParams {

  def this(c: Double, coefficients: Array[Double]) = this(Identifiable.randomUID("ARModel"), c,
    coefficients)

  /**
    * Transforms the input dataset.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val data = dataset.select(${timeCol}, ${timeSeriesCol}).rdd.map {
      case Row(time: String, value: Double) => (time, value)
    }.sortByKey().collect()
      .map(x => x._2)

    val dataVector = Vectors.dense(data)

    val dest = addTimeDependentEffects(dataVector)

    val resRDD = dataset.sparkSession.sparkContext.parallelize(dest.toArray.map(x => Row(x)))

    val structType = transformSchema(dataset.schema)

    dataset.sparkSession.createDataFrame(resRDD, structType)
  }

  def removeTimeDependentEffects(ts: Vector): Vector = {
    val dest = new Array[Double](ts.size)
    var i = 0
    while (i < ts.size) {
      dest(i) = ts(i) - c
      var j = 0
      while (j < coefficients.length && i - j - 1 >= 0) {
        dest(i) -= ts(i - j - 1) * coefficients(j)
        j += 1
      }
      i += 1
    }
    new DenseVector(dest)
  }

  def addTimeDependentEffects(ts: Vector): Vector = {
    val dest = new Array[Double](ts.size)
    var i = 0
    while (i < ts.size) {
      dest(i) = c + ts(i)
      var j = 0
      while (j < coefficients.length && i - j - 1 >= 0) {
        dest(i) += dest(i - j - 1) * coefficients(j)
        j += 1
      }
      i += 1
    }
    new DenseVector(dest)
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
    StructType(Array(StructField("Autoregression", DoubleType)))

  }

  override def copy(extra: ParamMap): ARModel = defaultCopy(extra)

}
