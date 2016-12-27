package org.apache.spark.ml.timeseries.models

import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.timeseries.MatrixUtil
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, _}

/**
  * Created by endy on 16-12-22.
  */
class ARGARCHSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest{
  test("fit model") {
    val omega = 0.2
    val alpha = 0.3
    val beta = 0.5
    val genModel = new ARGARCHModel(0.0, 0.0, alpha, beta, omega)
    val rand = new MersenneTwister(5L)
    val n = 10000

    val ts = genModel.sample(n, rand)
    val data = genDf(ts)

    val model = new GARCH().fit(data)
    assert(model.omega - omega < .1) // TODO: we should be able to be more accurate
    assert(model.alpha - alpha < .02)
    assert(model.beta - beta < .02)
  }


  test("fit model 2") {
    val arr = Array[Double](0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1,
      0.0, -0.01, 0.00, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.01, 0.00, -0.1)
    val ts = genDf(arr)

    val model = new ARGARCH().fit(ts)

    assert(model.alpha ~== -0.106 absTol 0.001)
    assert(model.beta ~== -1.012 absTol 0.001)
    assert(model.omega ~== 0.190 absTol 0.01)
    assert(model.c ~== -0.0355 absTol 0.01)
    assert(model.phi ~== -0.339 absTol 0.01)
  }

  test("standardize and filter") {
    val model = new ARGARCHModel(40.0, .4, .2, .3, .4)
    val rand = new MersenneTwister(5L)
    val n = 10000

    val ts = new DenseVector(model.sample(n, rand))

    // de-heteroskedasticize
    val standardized = model.removeTimeDependentEffects(ts)
    // heteroskedasticize
    val filtered = model.addTimeDependentEffects(standardized)

    assert((MatrixUtil.toBreeze(filtered) - MatrixUtil.toBreeze(ts)).toArray.forall(math.abs(_) <
      .001))
  }

  def genDf(array: Array[Double]): DataFrame = {
    val schema = StructType(Array(StructField("time", StringType), StructField("timeseries",
      DoubleType)))

    val rdd = spark.sparkContext.parallelize(
      array.zipWithIndex.map(x => Row(x._2.formatted("%010d"), x._1)))

    spark.createDataFrame(rdd, schema)
  }
}
