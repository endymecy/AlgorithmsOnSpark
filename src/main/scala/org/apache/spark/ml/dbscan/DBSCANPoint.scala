package org.apache.spark.ml.dbscan

import org.apache.spark.ml.linalg.Vector

case class DBSCANPoint(val vector: Vector) {

  def x: Double = vector(0)
  def y: Double = vector(1)

  def distanceSquared(other: DBSCANPoint): Double = {
    val dx = other.x - x
    val dy = other.y - y
    (dx * dx) + (dy * dy)
  }
}
