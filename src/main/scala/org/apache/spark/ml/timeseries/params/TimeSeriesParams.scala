package org.apache.spark.ml.timeseries.params

import org.apache.spark.ml.param.{Param, Params}

/**
  * Created by endy on 16-12-22.
  */
trait TimeSeriesParams extends Params {
  final val timeCol = new Param[String](this, "timeCol",
    "The column that stored time value")
  def setTimeCol(value: String): this.type = set(timeCol, value)

  final val timeSeriesCol = new Param[String](this, "timeSeriesCol",
    "The column that stored time series value")
  def setTimeSeriesCol(value: String): this.type = set(timeSeriesCol, value)
}
