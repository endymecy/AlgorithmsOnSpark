package org.apache.spark.ml.timeseries.models

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{Dataset, Row}

class EWMASuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest{
  @transient var dataSet: Dataset[_] = _
  @transient var dataSet1: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    val schema = StructType(Array(StructField("time", StringType), StructField("timeseries",
      DoubleType)))

    val smoothed = Array(
      Array("201512", 7.0), Array("201601", 8.0), Array("201602", 9.0),
      Array("201509", 4.0), Array("201510", 5.0), Array("201511", 6.0),
      Array("201506", 1.0), Array("201507", 2.0), Array("201508", 3.0),
      Array("201603", 10.0))

    val orig1 = sc.parallelize(smoothed.map(x => Row(x: _*)))
    dataSet = spark.createDataFrame(orig1, schema)

    val oil = Array(
      Array("201506", 446.7), Array("201507", 454.5), Array("201508", 455.7),
      Array("201512", 425.3), Array("201601", 485.1), Array("201602", 506.0),
      Array("201509", 423.6), Array("201510", 456.3), Array("201511", 440.6),
      Array("201603", 526.8), Array("201604", 514.3), Array("201605", 494.2))

    val orig2 = sc.parallelize(oil.map(x => Row(x: _*)))
    dataSet1 = spark.createDataFrame(orig2, schema)
  }


  test("add time dependent effects") {

    val m1 = new EWMAModel(0.2).setTimeCol("time").setTimeSeriesCol("timeseries")
    val res = m1.transform(dataSet).collect().map{case Row(x: Double) => x}

    assert(res(0) == 1.0)
    assert(res(1) ~== 1.2 absTol 10E-5)
  }

  test("fitting EWMA model") {
    val model = new EWMA()
      .setTimeCol("time")
      .setTimeSeriesCol("timeseries")
      .setMaxIter(10000)
      .setMaxEval(10000)
      .setInitPoint(.94)
      .fit(dataSet1)

    assert(model.smoothing ~== 0.89 absTol 0.01) // approximately 0.89
  }

}
