package org.enme.dbscan

import java.net.URI

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext

import scala.io.Source

case class TestRow(features: Vector)
class DBSCANSuit extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {
  private val dataFile = "labeled_data.csv"

  private val corresponding = Map(3 -> 2d, 2 -> 1d, 1 -> 3d, 0 -> 0d)

  test("local dbscan"){
    val labeled: Map[DBSCANPoint, Double] =
      new LocalDBSCAN(eps = 0.3F, minPoints = 10)
        .fit(getRawData(dataFile))
        .map(l => (l, l.cluster.toDouble))
        .toMap

    val expected: Map[DBSCANPoint, Double] = getExpectedData(dataFile).toMap

    labeled.foreach {
      case (key, value) => {
        val t = expected(key)
        assert(t == value)
      }
    }
    assert(labeled == expected)
  }

  test("dbscan") {
    val data = sc.textFile(getFile(dataFile).toString)

    val parsedData = data.map(s => Vectors.dense(s.split(',')
      .map(_.toDouble))).map(v => new TestRow(v))

    val dataset = spark.sqlContext.createDataFrame(parsedData)

    val dbscan = new DBSCAN()
      .setEps(0.3F)
      .setMinPoints(10)
      .setMaxPointsPerPartition(250)

    val model = dbscan.fit(dataset)

    val res = model.transform(dataset)

    val clustered = res.rdd.map{
      x => (DBSCANPoint(x.get(0).asInstanceOf[Vector]), x.get(1).asInstanceOf[Int])
    }.collectAsMap().mapValues(x => corresponding(x))

    val expected = getExpectedData(dataFile).toMap

    assert(expected === clustered)
  }

  def getFile(fileName: String): URI = {
    getClass.getClassLoader.getResource(fileName).toURI
  }

  def getExpectedData(file: String): Iterator[(DBSCANPoint, Double)] = {
    Source.fromFile(getFile(file)).getLines().map(s => {
      val vector = Vectors.dense(s.split(",").map(_.toDouble))
      val point = DBSCANPoint(vector)
      (point, vector(2))
    })
  }

  def getRawData(file: String): Iterable[(DBSCANPoint)] = {
    Source.fromFile(getFile(file)).getLines().map(s => {
      DBSCANPoint(Vectors.dense(s.split(",").map(_.toDouble)))
    }).toIterable
  }
}
