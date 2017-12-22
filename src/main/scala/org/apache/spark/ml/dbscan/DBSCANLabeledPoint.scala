package org.apache.spark.ml.dbscan

import org.apache.spark.ml.linalg.Vector

/**
  * Companion constants for labeled points
  */
object DBSCANLabeledPoint {

  val Unknown = 0

  object Flag extends Enumeration {
    type Flag = Value
    val Border, Core, Noise, NotFlagged = Value
  }

}

class DBSCANLabeledPoint(vector: Vector) extends DBSCANPoint(vector) {

  def this(point: DBSCANPoint) = this(point.vector)

  var flag = DBSCANLabeledPoint.Flag.NotFlagged
  var cluster = DBSCANLabeledPoint.Unknown
  var visited = false

  override def toString(): String = {
    s"$vector,$cluster,$flag"
  }

}

