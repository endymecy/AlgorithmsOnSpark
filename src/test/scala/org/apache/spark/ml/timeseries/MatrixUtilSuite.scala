package org.apache.spark.ml.timeseries

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.{Matrices, Vectors}
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext

/**
  * Created by endy on 16-12-21.
  */
class MatrixUtilSuite extends SparkFunSuite with MLlibTestSparkContext
  with DefaultReadWriteTest {
  test("modifying toBreeze version modifies original tensor") {
    val vec = Vectors.dense(1.0, 2.0, 3.0)
    val breezeVec = MatrixUtil.toBreeze(vec)
    breezeVec(1) = 4.0
    assert(vec(1) == 4.0)

    val mat = Matrices.zeros(3, 4)
    val breezeMat = MatrixUtil.toBreeze(mat)
    breezeMat(0, 1) = 2.0
    assert(mat(0, 1) == 2.0)
  }
}
