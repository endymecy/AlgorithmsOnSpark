package org.enme.dbscan

import archery.{Box, Entry, Point, RTree}

import scala.collection.mutable
import scala.language.implicitConversions

/**
 * An implementation of DBSCAN using an R-Tree
 */
class LocalDBSCAN(eps: Float, minPoints: Int){

  implicit def float2Double(x: Double): Float = x.toFloat

  val minDistanceSquared = eps * eps

  def fit(points: Iterable[DBSCANPoint]): Iterable[DBSCANLabeledPoint] = {

    val tree = points.foldLeft(RTree[DBSCANLabeledPoint]())(
      (tempTree, p) => tempTree.insert(Entry(Point(p.x, p.y), new DBSCANLabeledPoint(p))))

    var cluster = DBSCANLabeledPoint.Unknown

    tree.entries.foreach(entry => {
      val point = entry.value

      if (!point.visited) {
        point.visited = true

        val neighbors = tree.search(toBoundingBox(point), inRange(point))

        if (neighbors.size < minPoints) {
          point.flag = Flag.Noise
        } else {
          cluster += 1
          expandCluster(point, neighbors, tree, cluster)
        }
      }
    })

    tree.entries.map(_.value).toIterable
  }

  private def expandCluster(
    point: DBSCANLabeledPoint,
    neighbors: Seq[Entry[DBSCANLabeledPoint]],
    tree: RTree[DBSCANLabeledPoint],
    cluster: Int): Unit = {

    point.flag = Flag.Core
    point.cluster = cluster

    val left = mutable.Queue(neighbors)

    while (left.nonEmpty) {
      left.dequeue().foreach(neighborEntry => {

        val neighbor = neighborEntry.value

        if (!neighbor.visited) {
          neighbor.visited = true
          neighbor.cluster = cluster

          val neighborNeighbors = tree.search(toBoundingBox(neighbor), inRange(neighbor))

          if (neighborNeighbors.size >= minPoints) {
            neighbor.flag = Flag.Core
            left.enqueue(neighborNeighbors)
          } else {
            neighbor.flag = Flag.Border
          }
        } else if (neighbor.flag == Flag.Noise){
          neighbor.cluster = cluster
          neighbor.flag = Flag.Border
        }
      })
    }
  }

  private def inRange(point: DBSCANPoint)(entry: Entry[DBSCANLabeledPoint]): Boolean = {
    entry.value.distanceSquared(point) <= minDistanceSquared
  }

  private def toBoundingBox(point: DBSCANPoint): Box = {
    Box(point.x - eps, point.y - eps, point.x + eps, point.y + eps)
  }

}
