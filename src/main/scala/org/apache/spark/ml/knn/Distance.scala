package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.Vector

object Distance extends Enumeration {

  val Euclidean, Manhattan = Value

  /**
    * Computes the (Manhattan or Euclidean) distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @param distanceType type of the distance used (Distance.Euclidean or Distance.Manhattan)
    * @return Distance
    */
  def apply(x: Vector, y: Vector, distanceType: Distance.Value): Double = {
    distanceType match {
      case Euclidean => euclidean(x, y)
      case Manhattan => manhattan(x, y)
      case _ => euclidean(x, y)
    }
  }

  /**
    * Computes the Euclidean distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @return Euclidean distance
    */
  private def euclidean(x: Vector, y: Vector): Double = {
    var sum = 0.0
    val size = x.size

    for (i <- 0 until size) sum += (x(i) - y(i)) * (x(i) - y(i))

    Math.sqrt(sum)
  }

  /**
    * Computes the Manhattan distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @return Manhattan distance
    */
  private def manhattan(x: Vector, y: Vector): Double = {
    var sum = 0.0
    val size = x.size

    for (i <- 0 until size) sum += Math.abs(x(i) - y(i))

    sum
  }
}
