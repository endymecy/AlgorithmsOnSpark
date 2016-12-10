package org.enme.knn_is

import org.apache.spark.ml.classification.{ClassificationModel, Classifier}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.param._
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.types.DoubleType

import scala.collection.mutable.ArrayBuffer

trait KNN_ISParams extends PredictorParams {

  final val supportedDistanceTypes: Array[String] = Array("euclidean", "manhattan")

  // Obligatory parameters
  final val K: IntParam = new IntParam(this, "K", "Number of neighbors.", ParamValidators.gtEq(1))
  def setK(value: Int): this.type = set(K, value)
  setDefault (K -> 1)

  final val distanceType = new Param[String](this, "distanceType",
    "Distance Type that supported in this algorithm.",
    (value: String) => supportedDistanceTypes.contains(value))

  def setDistanceType(value: String): this.type = set(distanceType, value)
  setDefault (distanceType -> "manhattan")

  final var concurrency: IntParam = new IntParam(this, "concurrency",
    "Number of concurrency to classify the test.")
  def setConcurrency(value: Int): this.type = set(concurrency, value)
  setDefault (concurrency -> 1)


  // Setting Iterative MapReduce
  final var inc: LongParam = new LongParam(this, "inc", "Increment used to the iterative behavior.")
  def setInc(value: Long): this.type = set(inc, value)

  final var subdel: LongParam = new LongParam(this, "subdel",
    "Sub-delimiter used to the iterative behavior.")
  def setSubdel(value: Long): this.type = set(subdel, value)

  final var topdel: LongParam = new LongParam(this, "topdel",
    "Top-delimiter used to the iterative behavior.")
  def setTopdel(value: Long): this.type = set(topdel, value)
}

/**
  * This class implement of KNN_IS
  *
  * @see {KNN-IS:An Iterative Saprk-based design of k-Nearest Neighbors classifier for big data.}
  *
  * @param uid
  */

class KNN_ISClassifier(override val uid: String)
  extends Classifier[Vector, KNN_ISClassifier, KNN_ISClassificationModel] with KNN_ISParams {

  def this() = this(Identifiable.randomUID("kNN_IS"))

  /**
    * Initial setting necessary.
    * Auto-set the number of iterations and load the data sets and parameters.
    *
    * @return Instance of this class. *this*
    */
  override protected def train(dataset: Dataset[_]): KNN_ISClassificationModel = {
    val train: RDD[LabeledPoint] = extractLabeledPoints(dataset)

    val numSamplesTest = dataset.count()

    // Setting Iterative MapReduce
    this.setInc(numSamplesTest / $(concurrency))
    this.setSubdel(0)
    this.setTopdel($(inc))
    // If only one partition
    if ($(concurrency) == 1) {
      this.setTopdel(numSamplesTest + 1)
    }

    val numClasses: Int = getNumClasses(dataset)

    val knn = kNN_ISClassificationModel(train, $(K), $(distanceType), numSamplesTest,
      numClasses, $(inc), $(subdel), $(topdel), $(concurrency), this)

    knn
  }

  override def copy(extra: ParamMap): KNN_ISClassifier = defaultCopy(extra)
}

object kNN_ISClassifier extends DefaultParamsReadable[KNN_ISClassifier] {
  override def load(path: String): KNN_ISClassifier = super.load(path)
}

class KNN_ISClassificationModel (override val uid: String, val train: RDD[LabeledPoint],
                                 val k: Int, val distanceType: String, val numSamplesTest: Long, val numClass: Int,
                                 var inc: Long, var subdel: Long, var topdel: Long, val numIter: Int)
  extends ClassificationModel[Vector, KNN_ISClassificationModel] with Serializable {

  override val numClasses: Int = numClass

  def this(train: RDD[LabeledPoint], k: Int, distanceType: String, numSamplesTest: Long,
           numClass: Int, inc: Long, subdel: Long, topdel: Long, numIter: Int) =
    this(Identifiable.randomUID("kNN_IS"), train, k, distanceType, numSamplesTest,
      numClass, inc, subdel, topdel, numIter)


  /**
    * Calculate the K nearest neighbor from the test set over the train set.
    *
    * @param iter Iterator of each split of the training set.
    * @param testSet The test set in a broadcasting
    * @param subdel Int needed for take order when iterative version is running
    * @return K Nearest Neighbors for this split
    */
  def knn[T](iter: Iterator[LabeledPoint], testSet: Broadcast[Array[LabeledPoint]], subdel: Long):
  Iterator[(Long, Array[Array[Double]])] = {
    // Initialization
    val train = new ArrayBuffer[LabeledPoint]
    val size = testSet.value.length

    // Distance MANHATTAN or EUCLIDEAN
    val dist: Distance.Value = distanceType match {
      case "manhattan" => Distance.Manhattan
      case "euclidean" => Distance.Euclidean
      case _ => throw new Exception("Don't support this type")
    }
    // Join the train set
    while (iter.hasNext) train.append(iter.next)

    val knnMemb = new KNN(train, k, dist, numClasses)

    var auxSubDel = subdel
    val result = new Array[(Long, Array[Array[Double]])](size)

    for (i <- 0 until size) {
      result(i) = (auxSubDel, knnMemb.neighbors(testSet.value(i).features))
      auxSubDel = auxSubDel + 1
    }

    result.iterator
  }

  /**
    * Join the result of the map taking the nearest neighbors.
    *
    * @param mapOut1 A element of the RDD to join
    * @param mapOut2 Another element of the RDD to join
    * @return Combine of both element with the nearest neighbors
    */
  def combine(mapOut1: Array[Array[Double]], mapOut2: Array[Array[Double]]):
  Array[Array[Double]] = {
    var itOut1 = 0
    var itOut2 = 0
    val out: Array[Array[Double]] = new Array[Array[Double]](k)

    var i = 0
    while (i < k) {
      out(i) = new Array[Double](2)
      // Update the matrix taking the k nearest neighbors
      if (mapOut1(itOut1)(0) <= mapOut2(itOut2)(0)) {
        out(i)(0) = mapOut1(itOut1)(0)
        out(i)(1) = mapOut1(itOut1)(1)
        if (mapOut1(itOut1)(0) == mapOut2(itOut2)(0)) {
          i += 1
          if (i < k) {
            out(i) = new Array[Double](2)
            out(i)(0) = mapOut2(itOut2)(0)
            out(i)(1) = mapOut2(itOut2)(1)
            itOut2 = itOut2 + 1
          }
        }
        itOut1 = itOut1 + 1
      } else {
        out(i)(0) = mapOut2(itOut2)(0)
        out(i)(1) = mapOut2(itOut2)(1)
        itOut2 = itOut2 + 1
      }
      i += 1
    }
    out
  }

  /**
    * Calculate majority vote and return the predicted and right class for each instance.
    *
    * @param sample Real instance of the test set and his nearest neighbors
    * @return predicted and right class.
    */
  def calculatePredictedRightClassesFuzzy(
                                           sample: (Long, (LabeledPoint, Array[Array[Double]]))): (LabeledPoint, Double) = {

    val predictedNeigh = sample._2._2
    val labeledPoint = sample._2._1

    val auxClas = new Array[Int](numClasses)
    var clas = 0
    var numVotes = 0
    for (j <- 0 until k) {
      auxClas(predictedNeigh(j)(1).toInt) = auxClas(predictedNeigh(j)(1).toInt) + 1
      if (auxClas(predictedNeigh(j)(1).toInt) > numVotes) {
        clas = predictedNeigh(j)(1).toInt
        numVotes = auxClas(predictedNeigh(j)(1).toInt)
      }
    }

    (labeledPoint, clas.toDouble)
  }

  /**
    * Predict. kNN
    *
    * @return RDD[(Double, Double)]. First column Predicted class. Second Column Right class.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {

    import dataset.sparkSession.implicits._

    val test = { dataset.select($(labelCol), $(featuresCol))
      .map { case Row(label: Double, features: Vector) => LabeledPoint(label, features) } }

    val testWithKey = test.rdd.zipWithIndex()
      .map { line => (line._2, line._1) }
      .sortByKey()
      .cache()

    var testBroadcast: Broadcast[Array[LabeledPoint]] = null
    var output: RDD[(LabeledPoint, Double)] = null

    for (i <- 0 until numIter) {
      // Taking the iterative initial time.
      if (numIter == 1){
        testBroadcast = test.sparkSession.sparkContext.broadcast(test.collect())
      }
      else {
        testBroadcast = testWithKey.sparkContext.broadcast(
          testWithKey.filterByRange(subdel, topdel).map(line => line._2).collect())
      }

      if (output == null) {
        output = testWithKey.join(train
          .mapPartitions(split => knn(split, testBroadcast, subdel))
          .reduceByKey(combine))
          .map(sample => calculatePredictedRightClassesFuzzy(sample))
          .cache()
      } else {
        output = output.union(testWithKey
          .join(train.mapPartitions(split => knn(split, testBroadcast, subdel))
            .reduceByKey(combine))
          .map(sample => calculatePredictedRightClassesFuzzy(sample)))
          .cache()
      }
      output.first()
      // Update the pairs of delimiters
      subdel = topdel + 1
      topdel = topdel + inc + 1
      testBroadcast.destroy()
    }

    val resRDD = output.map{
      case (point: LabeledPoint, pre: Double) =>
        Row.fromSeq(Array(point.label, point.features, pre))
    }

    val schema = dataset.schema.add(${predictionCol}, DoubleType)

    val res = dataset.sparkSession.createDataFrame(resRDD, schema)
    res
  }

  override protected def predict(features: Vector): Double = {
    0.0
  }

  override protected def predictRaw(features: Vector): Vector = {
    null
  }

  override def copy(extra: ParamMap): KNN_ISClassificationModel = {
    copyValues(new KNN_ISClassificationModel(train: RDD[LabeledPoint], k: Int, distanceType: String,
      numSamplesTest: Long, numClass: Int, inc: Long, subdel: Long, topdel: Long, numIter: Int))
  }

  override def toString: String = {
    s"kNN_ISClassificationModel (uid=$uid) with $k nearest neighbor(s)"
  }
}

private object kNN_ISClassificationModel {

  def apply(train: RDD[LabeledPoint], k: Int, distanceType: String, numSamplesTest: Long,
            numClass: Int, inc: Long, subdel: Long, topdel: Long, numIter: Int,
            parent: KNN_ISClassifier): KNN_ISClassificationModel = {

    val uid = if (parent != null) parent.uid else Identifiable.randomUID("kNN_IS")

    new KNN_ISClassificationModel(uid, train, k, distanceType, numSamplesTest, numClass,
      inc: Long, subdel: Long, topdel: Long, numIter: Int)
  }
}


