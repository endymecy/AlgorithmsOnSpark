package org.apache.spark.ml.util

import scala.reflect.ClassTag

import org.apache.spark.HashPartitioner
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.storage.StorageLevel

/**
  * Degree-Based Hashing, the paper:
  * Distributed Power-law Graph Computing: Theoretical and Empirical Analysis
  */
class DBHPartitioner(val partitions: Int, val threshold: Int = 0)
  extends HashPartitioner(partitions) {
  /**
    * Default DBH doesn't consider the situation where both the degree of src and
    * dst vertices are both small than a given threshold value
    */
  def getKey(et: EdgeTriplet[Int, _]): Long = {
    val srcId = et.srcId
    val dstId = et.dstId
    val srcDeg = et.srcAttr
    val dstDeg = et.dstAttr
    val maxDeg = math.max(srcDeg, dstDeg)
    val minDegId = if (maxDeg == srcDeg) dstId else srcId
    val maxDegId = if (maxDeg == srcDeg) srcId else dstId
    if (maxDeg < threshold) {
      maxDegId
    } else {
      minDegId
    }
  }

  override def equals(other: Any): Boolean = other match {
    case dbh: DBHPartitioner =>
      dbh.numPartitions == numPartitions
    case _ =>
      false
  }
}

object DBHPartitioner {
  def partitionByDBH[VD: ClassTag, ED: ClassTag](input: Graph[VD, ED],
                                                 storageLevel: StorageLevel): Graph[VD, ED] = {
    val edges = input.edges
    val conf = edges.context.getConf
    val numPartitions = conf.getInt("", edges.partitions.length)
    val dbh = new DBHPartitioner(numPartitions, 0)
    val degGraph = GraphImpl(input.degrees, edges)
    val newEdges = degGraph.triplets.mapPartitions(_.map(et =>
      (dbh.getKey(et), Edge(et.srcId, et.dstId, et.attr))
    )).partitionBy(dbh).map(_._2)
    GraphImpl(input.vertices, newEdges, null.asInstanceOf[VD], storageLevel, storageLevel)
  }
}
