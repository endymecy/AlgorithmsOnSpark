package org.apache.spark.ml.timeseries.models

import org.apache.commons.math3.random.{MersenneTwister, RandomGenerator}
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Dataset, Row}

/**
  * Created by endy on 16-12-19.
  */
class AutoregressionSuite extends SparkFunSuite with MLlibTestSparkContext
  with DefaultReadWriteTest {

  @transient var dataSet: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  test("fit AR(1) model") {
    val ts = sample(5000, new MersenneTwister(10L), 1.5, Array(.2))

    val fittedModel = new Autoregression()
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .setMaxLag(1)
      .setNoIntercept(false)
      .fit(ts)

    assert(fittedModel.coefficients.length == 1)
    assert(math.abs(fittedModel.c - 1.5) < .07)
    assert(math.abs(fittedModel.coefficients(0) - .2) < .03)
  }

  test("fit AR(2) model") {

    val ts = sample(5000, new MersenneTwister(10L), 1.5, Array(.2, .3))
    val fittedModel = new Autoregression()
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .setMaxLag(2)
      .setNoIntercept(false)
      .fit(ts)

    assert(fittedModel.coefficients.length == 2)
    assert(math.abs(fittedModel.c - 1.5) < .15)
    assert(math.abs(fittedModel.coefficients(0) - .2) < .03)
    assert(math.abs(fittedModel.coefficients(1) - .3) < .03)
  }

  def sample(n: Int, rand: RandomGenerator, c: Double, coefficients: Array[Double]): Dataset[_] = {
    val vec = new DenseVector(Array.fill[Double](n)(rand.nextGaussian()))
    val res = new ARModel(c, coefficients).addTimeDependentEffects(vec).toArray
        .zipWithIndex

    val schema = StructType(Array(StructField("time", StringType), StructField("timeseries",
      DoubleType)))

    val rdd = sc.parallelize(res.map(x => Row(x._2.formatted("%05d"), x._1)))

    spark.createDataFrame(rdd, schema)
  }
}
