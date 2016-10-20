package org.enme.dbscan

import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.annotation.tailrec


trait DBSCANParams  extends Params with HasFeaturesCol with HasPredictionCol{
  /**
    * The maximum distance between two points. Must be > 0.
    * @group param
    */
  final val eps = new FloatParam(this, "eps",
    "The maximum distance between two points for them to be considered as part of the same region",
    ParamValidators.gt(0F))
  def getEps: Float = ${eps}

  /**
    * the minimum number of points required to form a dense region. Must be > 1.
    * @group param
    */
  final val minPoints = new IntParam(this, "minPoints",
    "The minimum number of points required to form a dense region", ParamValidators.gt(1))
  def getMinPoints: Int = ${minPoints}

  /**
    * the largest number of points in a single partition. Must be > 1.
    * @group param
    */
  final val maxPointsPerPartition = new LongParam(this, "maxPointsPerPartition",
    "The largest number of points in a single partition", ParamValidators.gt(1L))
  def getMaxPointsPerPartiton: Long = ${maxPointsPerPartition}

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.appendColumn(schema, $(predictionCol), IntegerType)
  }
}

class DBSCANModel(override val uid: String, val partitions: List[(Int, DBSCANRectangle)],
                  val labeledPartitionedPoints: RDD[(Int, DBSCANLabeledPoint)]
                 ) extends Model[DBSCANModel] with DBSCANParams {

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def copy(extra: ParamMap): DBSCANModel = {
    val copied = new DBSCANModel(uid, partitions, labeledPartitionedPoints)
    copyValues(copied, extra)
  }

  /**
    * Transforms the input dataset.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val test = dataset.select(col(${featuresCol})).rdd.map{
      case Row(v: Vector) => v
    }.collect()

    val resRDD = labeledPartitionedPoints.values
      .filter(p => test.contains(p.vector))
      .map(p => (p.vector, p.cluster))
      .collect()

    dataset.sparkSession.sqlContext.createDataFrame(resRDD)
      .withColumnRenamed("_1", ${featuresCol})
      .withColumnRenamed("_2", ${predictionCol})
  }

  def predict(features: Vector): Int = {
    labeledPartitionedPoints.values
      .filter(p => p.vector.equals(features))
      .map(p => p.cluster).first()
  }
  /**
    * :: DeveloperApi ::
    *
    * Check transform validity and derive the output schema from the input schema.
    *
    * Typical implementation should first conduct verification on schema change and parameter
    * validity, including complex parameter interaction checks.
    */
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

/**
 *  A parallel implementation of DBSCAN clustering. The implementation will split the data space
 *  into a number of partitions, making a best effort to keep the number of points in each
 *  partition under `maxPointsPerPartition`. After partitioned, traditional DBSCAN
 *  clustering will be run in parallel for each partition and finally the results
 *  of each partition will be merged to identify global clusters.
 *
 *  This is an iterative algorithm that will make multiple passes over the data,
 *  any given RDDs should be cached by the user.
 */
class DBSCAN(override val uid: String)
  extends Estimator[DBSCANModel] with DBSCANParams{

  setDefault(eps -> 0.3f, minPoints -> 10, maxPointsPerPartition -> 250)

  type Margins = (DBSCANRectangle, DBSCANRectangle, DBSCANRectangle)
  type ClusterId = (Int, Int)

  val minimumRectangleSize: Double = ${eps} * 2

  def this() = this(Identifiable.randomUID("DBSCAN"))

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setEps(value: Float): this.type = set(eps, value)

  def setMinPoints(value: Int): this.type = set(minPoints, value)

  def setMaxPointsPerPartition(value: Long): this.type = set(maxPointsPerPartition, value)

  override def fit(dataset: Dataset[_]): DBSCANModel = {
    transformSchema(dataset.schema, logging = true)
    val vectors: RDD[Vector] = dataset.select(col($(featuresCol))).rdd.map{
      case Row(point: Vector) => point
    }
    train(vectors)
  }

  def train(vectors: RDD[Vector]): DBSCANModel = {
    // generate the smallest rectangles that split the space
    // and count how many points are contained in each one of them
    val minimumRectanglesWithCount = vectors
      .map(toMinimumBoundingRectangle).map((_, 1)).aggregateByKey(0)(_ + _, _ + _).collect().toSet

    // find the best partitions for the data space
    val localPartitions = new SplitPartitioner(${maxPointsPerPartition}, minimumRectangleSize)
        .findPartitions(minimumRectanglesWithCount)

    // grow partitions to include eps
    val localMargins = localPartitions.map({ case (p, _) =>
      (p.shrink(${eps}.toDouble), p, p.shrink(-${eps})) }).zipWithIndex

    val margins = vectors.context.broadcast(localMargins)

    // assign each point to its proper partition
    val duplicated = for {
      point <- vectors.map(DBSCANPoint)
      ((inner, main, outer), id) <- margins.value
      if outer.contains(point)
    } yield (id, point)

    val numOfPartitions = localPartitions.size

    // perform local dbscan
    val clustered = duplicated.groupByKey(numOfPartitions).flatMapValues(points =>
          new LocalDBSCAN(${eps}, ${minPoints}).fit(points))
        .cache()

    // find all candidate points for merging clusters and group them
    val mergePoints = clustered.flatMap({
          case (partition, point) =>
            margins.value.filter({
                case ((inner, main, _), _) => main.contains(point) && !inner.almostContains(point)
              })
              .map({
                case (_, newPartition) => (newPartition, (partition, point))
              })
        })
        .groupByKey()

    // find all clusters with aliases from merging candidates
    val adjacencies = mergePoints.flatMapValues(findAdjacencies).values.collect()

    // generated adjacency graph
    val adjacencyGraph = adjacencies.foldLeft(DBSCANGraph[ClusterId]()) {
      case (graph, (from, to)) => graph.connect(from, to)
    }

    // find all cluster ids
    val localClusterIds = clustered.filter({ case (_, point) => point.flag != Flag.Noise })
        .mapValues(_.cluster).distinct().collect().toList

    // assign a global Id to all clusters, where connected clusters get the same id
    val (_, clusterIdToGlobalId) = localClusterIds.foldLeft((0, Map[ClusterId, Int]())) {
      case ((id, map), clusterId) => {
        map.get(clusterId) match {
          case None => {
            val nextId = id + 1
            val connectedClusters = adjacencyGraph.getConnected(clusterId) + clusterId
            val toAdd = connectedClusters.map((_, nextId)).toMap
            (nextId, map ++ toAdd)
          }
          case Some(x) =>
            (id, map)
        }
      }
    }

    val clusterIds = vectors.context.broadcast(clusterIdToGlobalId)

    // relabel non-duplicated points
    val labeledInner = clustered
        .filter(isInnerPoint(_, margins.value))
        .map {
          case (partition, point) => {
            if (point.flag != Flag.Noise) {
              point.cluster = clusterIds.value((partition, point.cluster))
            }
            (partition, point)
          }
        }

    // de-duplicate and label merge points
    val labeledOuter =
      mergePoints.flatMapValues(partition => {
        partition.foldLeft(Map[DBSCANPoint, DBSCANLabeledPoint]())({
          case (all, (partition1, point)) =>
            if (point.flag != Flag.Noise) {
              point.cluster = clusterIds.value((partition1, point.cluster))
            }

            all.get(point) match {
              case None => all + (point -> point)
              case Some(prev) => {
                // override previous entry unless new entry is noise
                if (point.flag != Flag.Noise) {
                  prev.flag = point.flag
                  prev.cluster = point.cluster
                }
                all
              }
            }

        }).values
      })

    val finalPartitions = localMargins.map {
      case ((_, p, _), index) => (index, p)
    }

    val partitionedPoints = labeledInner.union(labeledOuter)

    new DBSCANModel(uid, finalPartitions, partitionedPoints)
  }

  private def isInnerPoint(entry: (Int, DBSCANLabeledPoint),
    margins: List[(Margins, Int)]): Boolean = {
    entry match {
      case (partition, point) =>
        val ((inner, _, _), _) = margins.filter({
          case (_, id) => id == partition
        }).head

        inner.almostContains(point)
    }
  }

  private def findAdjacencies(
              partition: Iterable[(Int, DBSCANLabeledPoint)]): Set[((Int, Int), (Int, Int))] = {

    val zero = (Map[DBSCANPoint, ClusterId](), Set[(ClusterId, ClusterId)]())

    val (_ , adjacencies) = partition.foldLeft(zero)({
      case ((seen1, adjacencies1), (partition1, point)) =>
        // noise points are not relevant for adjacencies
        if (point.flag == Flag.Noise) {
          (seen1, adjacencies1)
        } else {
          val clusterId = (partition1, point.cluster)

          seen1.get(point) match {
            case None => (seen1 + (point -> clusterId), adjacencies1)
            case Some(prevClusterId) => (seen1, adjacencies1 + ((prevClusterId, clusterId)))
          }
        }
    })

    adjacencies
  }

  def toMinimumBoundingRectangle(vector: Vector): DBSCANRectangle = {
    val point = DBSCANPoint(vector)
    val x = corner(point.x)
    val y = corner(point.y)
    DBSCANRectangle(x, y, x + minimumRectangleSize, y + minimumRectangleSize)
  }

  def corner(p: Double): Double =
    (shiftIfNegative(p) / minimumRectangleSize).intValue * minimumRectangleSize

  def shiftIfNegative(p: Double): Double =
    if (p < 0) p - minimumRectangleSize else p

  override def copy(extra: ParamMap): Estimator[DBSCANModel] = defaultCopy(extra)

  /**
    * :: DeveloperApi ::
    *
    * Check transform validity and derive the output schema from the input schema.
    *
    * Typical implementation should first conduct verification on schema change and parameter
    * validity, including complex parameter interaction checks.
    */
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

// split partition
class SplitPartitioner(maxPointsPerPartition: Long, minimumRectangleSize: Double){

  type RectangleWithCount = (DBSCANRectangle, Int)

  def findPartitions(toSplit: Set[RectangleWithCount]): List[RectangleWithCount] = {
    val boundingRectangle = findBoundingRectangle(toSplit)
    def pointsIn = pointsInRectangle(toSplit, _: DBSCANRectangle)
    val toPartition = List((boundingRectangle, pointsIn(boundingRectangle)))
    val partitioned = List[RectangleWithCount]()
    val partitions = partition(toPartition, partitioned, pointsIn)
    // remove empty partitions
    partitions.filter({ case (partition, count) => count > 0 })
  }

  @tailrec
  private def partition(remaining: List[RectangleWithCount], partitioned: List[RectangleWithCount],
                        pointsIn: (DBSCANRectangle) => Int): List[RectangleWithCount] = {

    remaining match {
      case (rectangle, count) :: rest =>
        if (count > maxPointsPerPartition) {
          if (canBeSplit(rectangle)) {
            def cost = (r: DBSCANRectangle) => ((pointsIn(rectangle) / 2) - pointsIn(r)).abs
            val (split1, split2) = split(rectangle, cost)
            val s1 = (split1, pointsIn(split1))
            val s2 = (split2, pointsIn(split2))
            partition(s1 :: s2 :: rest, partitioned, pointsIn)
          } else {
            partition(rest, (rectangle, count) :: partitioned, pointsIn)
          }
        } else {
          partition(rest, (rectangle, count) :: partitioned, pointsIn)
        }
      case Nil => partitioned
    }
  }

  def split(rectangle: DBSCANRectangle,
            cost: (DBSCANRectangle) => Int): (DBSCANRectangle, DBSCANRectangle) = {

    val smallestSplit =
      findPossibleSplits(rectangle).reduceLeft {
        (smallest, current) => if (cost(current) < cost(smallest)) current else smallest
      }
    (smallestSplit, complement(smallestSplit, rectangle))
  }

  /**
    * Returns the box that covers the space inside boundary that is not covered by box
    */
  private def complement(box: DBSCANRectangle, boundary: DBSCANRectangle): DBSCANRectangle =
  if (box.x == boundary.x && box.y == boundary.y) {
    if (boundary.x2 >= box.x2 && boundary.y2 >= box.y2) {
      if (box.y2 == boundary.y2) {
        DBSCANRectangle(box.x2, box.y, boundary.x2, boundary.y2)
      } else if (box.x2 == boundary.x2) {
        DBSCANRectangle(box.x, box.y2, boundary.x2, boundary.y2)
      } else {
        throw new IllegalArgumentException("rectangle is not a proper sub-rectangle")
      }
    } else {
      throw new IllegalArgumentException("rectangle is smaller than boundary")
    }
  } else {
    throw new IllegalArgumentException("unequal rectangle")
  }

  /**
    * Returns all the possible ways in which the given box can be split
    */
  private def findPossibleSplits(box: DBSCANRectangle): Set[DBSCANRectangle] = {

    val xSplits = (box.x + minimumRectangleSize) until box.x2 by minimumRectangleSize
    val ySplits = (box.y + minimumRectangleSize) until box.y2 by minimumRectangleSize

    val splits =
      xSplits.map(x => DBSCANRectangle(box.x, box.y, x, box.y2)) ++
        ySplits.map(y => DBSCANRectangle(box.x, box.y, box.x2, y))

    splits.toSet
  }

  /**
    * Returns true if the given rectangle can be split into at least two rectangles of minimum size
    */
  private def canBeSplit(box: DBSCANRectangle): Boolean = {
    box.x2 - box.x > minimumRectangleSize * 2 || box.y2 - box.y > minimumRectangleSize * 2
  }

  def pointsInRectangle(space: Set[RectangleWithCount], rectangle: DBSCANRectangle): Int = {
    space.view.filter({ case (current, _) => rectangle.contains(current) }).foldLeft(0) {
      case (total, (_, count)) => total + count
    }
  }

  def findBoundingRectangle(rectanglesWithCount: Set[RectangleWithCount]): DBSCANRectangle = {

    val invertedRectangle =
      DBSCANRectangle(Double.MaxValue, Double.MaxValue, Double.MinValue, Double.MinValue)

    rectanglesWithCount.foldLeft(invertedRectangle) {
      case (bounding, (c, _)) =>
        DBSCANRectangle(bounding.x.min(c.x), bounding.y.min(c.y),
          bounding.x2.max(c.x2), bounding.y2.max(c.y2))
    }

  }

}



