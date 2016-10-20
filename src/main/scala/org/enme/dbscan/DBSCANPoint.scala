package org.enme.dbscan

import org.apache.spark.ml.linalg.{Vector, Vectors}

case class DBSCANPoint(vector: Vector) {

  def x: Double = vector(0)
  def y: Double = vector(1)

  def distanceSquared(other: DBSCANPoint): Double = {
      Vectors.sqdist(vector, other.vector)
  }
}
