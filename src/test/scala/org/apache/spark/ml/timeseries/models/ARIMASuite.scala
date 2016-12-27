package org.apache.spark.ml.timeseries.models

import org.apache.commons.math3.random.{MersenneTwister, RandomGenerator}
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.timeseries.UnivariateTimeSeries
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}


/**
  * Created by endy on 16-12-20.
  */
class ARIMASuite extends SparkFunSuite with MLlibTestSparkContext
  with DefaultReadWriteTest {

  @transient var dataSet: Dataset[_] = _
  test("compare with R") {
    // > R.Version()$version.string
    // [1] "R version 3.2.0 (2015-04-16)"
    // > set.seed(456)
    // y <- arima.sim(n=250,list(ar=0.3,ma=0.7),mean = 5)
    // write.table(y, file = "resources/R_ARIMA_DataSet1.csv", row.names = FALSE, col.names = FALSE)
    val dataFile = getClass.getResource("/timeseries/R_ARIMA_DataSet1.csv").toString

    val rawData = sc.textFile(dataFile).map(line => line.toDouble)
      .collect().zipWithIndex

    val schema = StructType(Array(StructField("time", StringType), StructField("timeseries",
      DoubleType)))

    val rdd = sc.parallelize(rawData.map(x => Row(x._2.formatted("%05d"), x._1)))
    val dataset = spark.createDataFrame(rdd, schema)

    val model = new ARIMA()
      .setP(1)
      .setD(0)
      .setQ(1)
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .fit(dataset)

    val Array(c, ar, ma) = model.coefficient
    assert(ar ~== 0.3 absTol  0.05)
    assert(ma ~== 0.7 absTol  0.05)
  }

  test("Data sampled from a given model should result in similar model if fit") {
    val rand = new MersenneTwister(10L)
    val model = new ARIMAModel(2, 1, 2, Array(8.2, 0.2, 0.5, 0.3, 0.1))
    val (_, sampled) = sample(1000, rand, model)

    val newModel = new ARIMA()
      .setP(2)
      .setD(1)
      .setQ(2)
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .fit(sampled)

    val Array(c, ar1, ar2, ma1, ma2) = model.coefficient
    val Array(cTest, ar1Test, ar2Test, ma1Test, ma2Test) = newModel.coefficient

    // intercept is given more leeway
    assert(c ~== cTest absTol 1)
    assert(ar1Test ~== ar1 absTol 0.1)
    assert(ma1Test ~== ma1 absTol 0.1)
    assert(ar2Test ~== ar2 absTol 0.1)
    assert(ma2Test ~== ma2 absTol 0.1)
  }

  test("Fitting CSS with BOBYQA and conjugate gradient descent should be fairly similar") {
    val rand = new MersenneTwister(10L)
    val model = new ARIMAModel(2, 1, 2, Array(8.2, 0.2, 0.5, 0.3, 0.1))
    val (_, sampled) = sample(1000, rand, model)

    val fitWithBOBYQA = new ARIMA()
        .setP(2)
        .setD(1)
        .setQ(2)
        .setTimeCol("time")
        .setTimeSeriesCol("timeseries")
        .setMethod("css-bobyqa")
        .fit(sampled)

    val fitWithCGD = new ARIMA()
      .setP(2)
      .setD(1)
      .setQ(2)
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .setMethod("css-cgd")
      .fit(sampled)

    val Array(c, ar1, ar2, ma1, ma2) = fitWithBOBYQA.coefficient
    val Array(cCGD, ar1CGD, ar2CGD, ma1CGD, ma2CGD) = fitWithCGD.coefficient

    // give more leeway for intercept
    assert(cCGD ~== c absTol 1)
    assert(ar1CGD ~== ar1 absTol 0.1)
    assert(ar2CGD ~== ar2 absTol 0.1)
    assert(ma1CGD ~== ma1 absTol 0.1)
    assert(ma2CGD ~== ma2 absTol 0.1)
  }

  test("Fitting ARIMA(p, d, q) should be the same as fitting a d-order differenced ARMA(p, q)") {
    val rand = new MersenneTwister(10L)
    val model = new ARIMAModel(1, 1, 2, Array(0.3, 0.7, 0.1), hasIntercept = false)
    val (vec, sampled) = sample(1000, rand, model)

    val arimaModel = new ARIMA()
      .setP(1)
      .setD(1)
      .setQ(2)
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .setIncludeIntercept(false)
      .fit(sampled)


    val differenceSample = UnivariateTimeSeries.differencesOfOrderD(vec, 1).toArray.drop(1)

    val dataFrame = genDf(differenceSample)

    val armaModel = new ARIMA()
      .setP(1)
      .setD(0)
      .setQ(2)
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .setIncludeIntercept(false)
      .fit(dataFrame)

    val Array(refAR, refMA1, refMA2) = model.coefficient
    val Array(iAR, iMA1, iMA2) = arimaModel.coefficient
    val Array(ar, ma1, ma2) = armaModel.coefficient

    // ARIMA model should match parameters used to sample, to some extent
    assert(iAR ~== refAR absTol 0.05)
    assert(iMA1 ~== refMA1 absTol 0.05)
    assert(iMA2 ~== refMA2 absTol 0.05)

    // ARMA model parameters of differenced sample should be equal to ARIMA model parameters
    assert(ar == iAR)
    assert(ma1 == iMA1)
    assert(ma2 == iMA2)
  }

  test("Fitting ARIMA(0, 0, 0) with intercept term results in model with average as parameter") {
    val rand = new MersenneTwister(10L)
    val (vec, sampled) = sample(100, rand)

    val model = new ARIMA()
      .setP(0)
      .setD(0)
      .setQ(0)
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .fit(sampled)

    val mean = vec.toArray.sum / vec.size

    assert(model.coefficient(0) ~== mean absTol 1e-4)
  }

  test("Fitting ARIMA(0, 0, 0) with intercept term results in model with average as the forecast") {
    val rand = new MersenneTwister(10L)
    val (vec, sampled) = sample(100, rand)
    val model = new ARIMA()
      .setP(0)
      .setD(0)
      .setQ(0)
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .fit(sampled)

    val mean = vec.toArray.sum / vec.size

    assert(model.coefficient(0) ~== mean absTol 1e-4)
    val forecast = model
      .setNFuture(10).transform(sampled).collect()
      .map{case Row(s: Double) => s}

    for(i <- 100 until 110) {
      assert(forecast(i) ~== mean absTol 1e-4)
    }
  }

  test("Fitting an integrated time series of order 3") {
    // > set.seed(10)
    // > vals <- arima.sim(list(ma = c(0.2), order = c(0, 3, 1)), 200)
    // > arima(order = c(0, 3, 1), vals, method = "CSS")
    //
    // Call:
    //  arima(x = vals, order = c(0, 3, 1), method = "CSS")
    //
    //  Coefficients:
    //   ma1
    //  0.2523
    //  s.e.  0.0623
    //
    //  sigma^2 estimated as 0.9218:  part log likelihood = -275.65
    // > write.table(y, file = "resources/R_ARIMA_DataSet2.csv", row.names = FALSE, col.names =
    // FALSE)
    val dataFile = getClass.getResource("/timeseries/R_ARIMA_DataSet2.csv").toString
    val rawData = sc.textFile(dataFile).map(line => line.toDouble)
      .collect().zipWithIndex

    val schema = StructType(Array(StructField("time", StringType), StructField("timeseries",
      DoubleType)))

    val rdd = sc.parallelize(rawData.map(x => Row(x._2.formatted("%05d"), x._1)))
    val dataset = spark.createDataFrame(rdd, schema)
    val model = new ARIMA()
      .setP(0)
      .setD(3)
      .setQ(1)
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .fit(dataset)

    val Array(c, ma) = model.coefficient
    assert(ma ~== 0.2 absTol 0.05)
  }
  /**
    * Sample a series of size n assuming an ARIMA(p, d, q) process.
    *
    * @param n size of sample
    * @return series reflecting ARIMA(p, d, q) process
    */
  def sample(n: Int, rand: RandomGenerator, model: ARIMAModel): (Vector, DataFrame) = {
    val vec = new DenseVector(Array.fill[Double](n)(rand.nextGaussian))
    val res = model.addTimeDependentEffects(vec, vec).toArray

    val schema = StructType(Array(StructField("time", StringType), StructField("timeseries",
      DoubleType)))

    val rdd = sc.parallelize(res.zipWithIndex.map(x => Row(x._2.formatted("%05d"), x._1)))

    (Vectors.dense(res), spark.createDataFrame(rdd, schema))
  }

  /**
    * Sample a series of size n assuming an ARIMA(p, d, q) process.
    *
    * @param n size of sample
    * @return series reflecting ARIMA(p, d, q) process
    */
  def sample(n: Int, rand: RandomGenerator): (Vector, DataFrame) = {
    val vec = new DenseVector(Array.fill[Double](n)(rand.nextGaussian)).toArray

    val schema = StructType(Array(StructField("time", StringType), StructField("timeseries",
      DoubleType)))

    val rdd = sc.parallelize(vec.zipWithIndex.map(x => Row(x._2.formatted("%05d"), x._1)))

    (Vectors.dense(vec), spark.createDataFrame(rdd, schema))
  }

  def genDf(array: Array[Double]): DataFrame = {
    val schema = StructType(Array(StructField("time", StringType), StructField("timeseries",
      DoubleType)))

    val rdd = spark.sparkContext.parallelize(
      array.zipWithIndex.map(x => Row(x._2.formatted("%010d"), x._1)))

    spark.createDataFrame(rdd, schema)
  }

}
