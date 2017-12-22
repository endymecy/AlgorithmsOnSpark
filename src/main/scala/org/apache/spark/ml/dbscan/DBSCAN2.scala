package org.apache.spark.ml.dbscan

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.sql.functions.{col, udf}

/**
  * Created by endy on 17-12-5.
  */

trait DBSCANParams extends Params with HasFeaturesCol with HasPredictionCol{
  final val eps = new DoubleParam(this, "eps", "the maximum distance between two points" +
    " for them to be considered as part of the same region")
  def getEps: Double = ${eps}

  final val minPoints = new IntParam(this, "minPoints", "the minimum number of" +
    " points required to form a dense region")
  def getMinPoints: Int = ${minPoints}

  final val maxPointsPerPartition = new IntParam(this, "maxPointsPerPartition",
    "the largest number of points in a single partition")

  def getMaxPointsPerPartition: Int = ${maxPointsPerPartition}

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, ${featuresCol}, new VectorUDT)
    SchemaUtils.appendColumn(schema, ${predictionCol}, IntegerType)
  }
}

class DBSCAN2(override val uid: String) extends Estimator[DBSCAN2Model] with DBSCANParams{

  setDefault(eps -> 0.3, minPoints -> 10, maxPointsPerPartition -> 250)

  def this() = this(Identifiable.randomUID("dbscan"))

  def setEps(value: Double): this.type = set(eps, value)

  def setMinPoints(value: Int): this.type = set(minPoints, value)

  def setMaxPointsPerPartition(value: Int): this.type = set(maxPointsPerPartition, value)

  override def fit(dataset: Dataset[_]): DBSCAN2Model = {
    val instances: RDD[Vector] = dataset.select(col(${featuresCol})).rdd.map {
      case Row(point: Vector) => point
    }

    val dbscan = DBSCAN.train(instances, ${eps}, ${minPoints}, ${maxPointsPerPartition})

    new DBSCAN2Model(uid, dbscan)
  }

  override def copy(extra: ParamMap): Estimator[DBSCAN2Model] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

class DBSCAN2Model(override val uid: String, val model: DBSCAN) extends
  Model[DBSCAN2Model] with DBSCANParams{

  override def copy(extra: ParamMap): DBSCAN2Model = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val clustered = model.labeledPoints
      .map(p => (p.vector(0), p.vector(1), p.vector, p.cluster))

    dataset.sparkSession.createDataFrame(clustered)
      .toDF(dataset.schema.fieldNames(0),
        dataset.schema.fieldNames(1),
        ${featuresCol}, ${predictionCol})
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

