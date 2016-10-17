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

import scala.collection.mutable.ArrayBuffer

trait KNN_ISParams extends PredictorParams {
  // Obligatory parameters
  final val K: IntParam = new IntParam(this, "K", "Number of neighbors.", ParamValidators.gtEq(1))
  final def getK: Int = $(K)

  final val distanceType: IntParam = new IntParam(this, "distanceType",
    "Distance Type: MANHATTAN = 1 ; EUCLIDEAN = 2 ; HVDM = 3.", ParamValidators.gtEq(1))
  final def getDistanceType: Int = $(distanceType)

  final var numIter: IntParam = new IntParam(this, "numIterations",
    "Number of Iteration to classify the test.")
  final def getNumIter: Int = $(numIter)

  final val numSamplesTest: IntParam = new IntParam(this, "numSamplesTest",
    "Number of instance in the test set.", ParamValidators.gtEq(1))
  final def getNumSamplesTest: Int = $(numSamplesTest)

  // Count the samples of each data set and the number of classes.
  // Array with the right classes of the test set.

  final var numClass: IntParam = new IntParam(this, "numClass",
    "Number of classes.", ParamValidators.gtEq(1))
  final def getNumClass: Int = $(numClass)

  // Setting Iterative MapReduce
  final var inc: IntParam = new IntParam(this, "inc", "Increment used to the iterative behavior.")
  final def getInc: Int = $(inc)
  final var subdel: IntParam = new IntParam(this, "subdel",
    "Sub-delimiter used to the iterative behavior.")
  final def getSubdel: Int = $(subdel)
  final var topdel: IntParam = new IntParam(this, "topdel",
    "Top-delimiter used to the iterative behavior.")
  final def getTopdel: Int = $(topdel)
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

  def setK(value: Int): this.type = set(K, value)
  setDefault (K -> 1)
  def setDistanceType(value: Int): this.type = set(distanceType, value)
  setDefault (distanceType -> 1)
  def setNumSamplesTest(value: Int): this.type = set(numSamplesTest, value)
  setDefault (numSamplesTest -> 1)
  def setNumClass(value: Int): this.type = set(numClass, value)
  setDefault (numClass -> 1)

  def setInc(value: Int): this.type = set(inc, value)
  setDefault (inc -> 0)
  def setTopdel(value: Int): this.type = set(topdel, value)
  setDefault (topdel -> 0)
  def setSubdel(value: Int): this.type = set(subdel, value)
  setDefault (subdel -> 0)
  def setNumIter(value: Int): this.type = set(numIter, value)
  setDefault (numIter -> 1)

  /**
    * Initial setting necessary.
    * Auto-set the number of iterations and load the data sets and parameters.
    *
    * @return Instance of this class. *this*
    */
  override protected def train(dataset: Dataset[_]): KNN_ISClassificationModel = {
    val train: RDD[LabeledPoint] = extractLabeledPoints(dataset)

    // Setting Iterative MapReduce
    this.setInc($(numSamplesTest) / $(numIter))
    this.setSubdel(0)
    this.setTopdel($(inc))
    // If only one partition
    if ($(numIter) == 1) {
      this.setTopdel($(numSamplesTest) + 1)
    }

    val knn = kNN_ISClassificationModel(train, $(K), $(distanceType), $(numSamplesTest),
      $(numClass), $(inc), $(subdel), $(topdel), $(numIter), this)

    knn
  }

  override def copy(extra: ParamMap): KNN_ISClassifier = defaultCopy(extra)
}

object kNN_ISClassifier extends DefaultParamsReadable[KNN_ISClassifier] {
  override def load(path: String): KNN_ISClassifier = super.load(path)
}

class KNN_ISClassificationModel (override val uid: String, val train: RDD[LabeledPoint],
        val k: Int, val distanceType: Int, val numSamplesTest: Int, val numClass: Int,
        var inc: Int, var subdel: Int, var topdel: Int, val numIter: Int)
  extends ClassificationModel[Vector, KNN_ISClassificationModel] with Serializable {

  override val numClasses: Int = numClass

  def this(train: RDD[LabeledPoint], k: Int, distanceType: Int, numSamplesTest: Int, numClass: Int,
           inc: Int, subdel: Int, topdel: Int, numIter: Int) =
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
  def knn[T](iter: Iterator[LabeledPoint], testSet: Broadcast[Array[LabeledPoint]], subdel: Int):
      Iterator[(Int, Array[Array[Double]])] = {
    // Initialization
    val train = new ArrayBuffer[LabeledPoint]
    val size = testSet.value.length

    var dist: Distance.Value = null
    // Distance MANHATTAN or EUCLIDEAN
    if(distanceType == 1) dist = Distance.Manhattan else dist = Distance.Euclidean
    // Join the train set
    while (iter.hasNext) train.append(iter.next)

    val knnMemb = new KNN(train, k, dist, numClass)

    var auxSubDel = subdel
    val result = new Array[(Int, Array[Array[Double]])](size)

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
               sample: (Int, (LabeledPoint, Array[Array[Double]]))): Double = {

    val predictedNeigh = sample._2._2

    val auxClas = new Array[Int](numClass)
    var clas = 0
    var numVotes = 0
    for (j <- 0 until k) {
      auxClas(predictedNeigh(j)(1).toInt) = auxClas(predictedNeigh(j)(1).toInt) + 1
      if (auxClas(predictedNeigh(j)(1).toInt) > numVotes) {
        clas = predictedNeigh(j)(1).toInt
        numVotes = auxClas(predictedNeigh(j)(1).toInt)
      }
    }

    clas.toDouble
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
      .map { line => (line._2.toInt, line._1) }
      .sortByKey()
      .cache()

    var testBroadcast: Broadcast[Array[LabeledPoint]] = null
    var output: RDD[Double] = null

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
      output.count()
      // Update the pairs of delimiters
      subdel = topdel + 1
      topdel = topdel + inc + 1
      testBroadcast.destroy()
    }

    val res = output.toDF().withColumnRenamed("value", ${predictionCol})
    res
  }

  override protected def predict(features: Vector): Double = {
    0.0
  }

  override protected def predictRaw(features: Vector): Vector = {
    null
  }

  override def copy(extra: ParamMap): KNN_ISClassificationModel = {
    copyValues(new KNN_ISClassificationModel(train: RDD[LabeledPoint], k: Int, distanceType: Int,
      numSamplesTest: Int, numClass: Int, inc: Int, subdel: Int, topdel: Int, numIter: Int))
  }

  override def toString: String = {
    s"kNN_ISClassificationModel (uid=$uid) with $k nearest neighbor(s)"
  }
}

private object kNN_ISClassificationModel {

  def apply(train: RDD[LabeledPoint], k: Int, distanceType: Int, numSamplesTest: Int, numClass: Int,
            inc: Int, subdel: Int, topdel: Int, numIter: Int, parent: KNN_ISClassifier):
      KNN_ISClassificationModel = {

    val uid = if (parent != null) parent.uid else Identifiable.randomUID("kNN_IS")

    new KNN_ISClassificationModel(uid, train, k, distanceType, numSamplesTest, numClass,
      inc: Int, subdel: Int, topdel: Int, numIter: Int)
  }
}

