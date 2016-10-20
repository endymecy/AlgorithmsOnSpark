package org.enme.knn_is

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}

import scala.util.Random

class KNN_ISSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {
  @transient var dataset: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    dataset = spark.createDataFrame(KNN_ISSuite.generateKnnInput(1.0, 1.0,
      nPoints = 1000, seed = 42))
  }

  test("knn: default params") {
    val knn_is = new KNN_ISClassifier()
    assert(knn_is.getLabelCol === "label")
    assert(knn_is.getFeaturesCol === "features")
    assert(knn_is.getPredictionCol === "prediction")
    assert(knn_is.getK == 1)
    assert(knn_is.getDistanceType == 1)
    assert(knn_is.getNumSamplesTest == 1)
    assert(knn_is.getNumClass == 1)
    assert(knn_is.getNumIter == 1)
    assert(knn_is.getInc == 0)
    assert(knn_is.getSubdel == 0)
    assert(knn_is.getTopdel == 0)
  }

  test("train"){
    val knn_is = new KNN_ISClassifier()
    knn_is.fit(dataset)
  }

  test("transform: one iterationNum"){
    val knn_is = new KNN_ISClassifier()
      .setNumClass(2)
      .setNumSamplesTest(dataset.count().toInt)
      .setK(5)

    val model = knn_is.fit(dataset)

    val results = model.transform(dataset)
    assert(results.count() == dataset.count())

    val source = dataset.select("label").rdd.map{case Row(x: Double) => x}
    val res = results.select("prediction").rdd.map{case Row(x: Double) => x}

    val predictions = source.zip(res.asInstanceOf[RDD[Double]])
    val metrics = new MulticlassMetrics(predictions)
    val precision = metrics.accuracy
    assert(precision == 0.64)
  }

  test("transform: more than one iterationNum"){
    val knn_is = new KNN_ISClassifier()
      .setNumClass(2)
      .setNumSamplesTest(dataset.count().toInt)
      .setNumIter(3)
      .setK(5)

    val model = knn_is.fit(dataset)

    val results = model.transform(dataset)
    assert(results.count() == dataset.count())

    val source = dataset.select("label")
      .rdd.map{case Row(x: Double) => x}.repartition(1)
    val res = results.select("prediction")
      .rdd.map{case Row(x: Double) => x}.repartition(1)

    val predictions = source.zip(res.asInstanceOf[RDD[Double]])
    val metrics = new MulticlassMetrics(predictions)
    val precision = metrics.accuracy
    assert(precision == 0.648)
  }
}

object KNN_ISSuite {
  def generateKnnInput(offset: Double,
                       scale: Double,
                       nPoints: Int,
                       seed: Int): Seq[LabeledPoint] = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val y = (0 until nPoints).map { i =>
      val p = 1.0 / (1.0 + math.exp(-(offset + scale * x1(i))))
      if (rnd.nextDouble() < p) 1.0 else 0.0
    }

    val testData = (0 until nPoints).map(i => LabeledPoint(y(i), Vectors.dense(Array(x1(i)))))
    testData
  }
}
