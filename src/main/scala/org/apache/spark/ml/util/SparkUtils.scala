package org.apache.spark.ml.util

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import breeze.storage.Zero
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}

import scala.language.implicitConversions
import scala.reflect.ClassTag


object SparkUtils {
  implicit def toBreeze(sv: SV): BV[Double] = {
    sv match {
      case SDV(data) =>
        new BDV(data)
      case SSV(size, indices, values) =>
        new BSV(indices, values, size)
    }
  }

  implicit def fromBreeze(breezeVector: BV[Double]): SV = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SDV(v.data)
        } else {
          new SDV(v.toArray) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SSV(v.length, v.index, v.data)
        } else {
          new SSV(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  def toBreezeConv[T: ClassTag](sv: SV)(implicit num: Numeric[T]): BV[T] = {
    val zero = num.zero
    implicit val conv: Array[Double] => Array[T] = (data) => {
      data.map(ele => (zero match {
        case zero: Double => ele
        case zero: Float => ele.toFloat
        case zero: Int => ele.toInt
        case zero: Long => ele.toLong
      }).asInstanceOf[T]).array
    }
    sv match {
      case SDV(data) =>
        new BDV[T](data)
      case SSV(size, indices, values) =>
        new BSV[T](indices, values, size)(Zero[T](zero))
    }
  }

  def fromBreezeConv[T: ClassTag](breezeVector: BV[T])(implicit num: Numeric[T]): SV = {
    implicit val conv: Array[T] => Array[Double] = (data) => {
      data.map(num.toDouble).array
    }
    breezeVector match {
      case v: BDV[T] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SDV(v.data)
        } else {
          new SDV(v.toArray) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[T] =>
        if (v.index.length == v.used) {
          new SSV(v.length, v.index, v.data)
        } else {
          new SSV(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[T] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  def getFileSystem(conf: SparkConf, path: Path): FileSystem = {
    val hadoopConf = SparkHadoopUtil.get.newConfiguration(conf)
    if (sys.env.contains("HADOOP_CONF_DIR") || sys.env.contains("YARN_CONF_DIR")) {
      val hdfsConfPath = if (sys.env.get("HADOOP_CONF_DIR").isDefined) {
        sys.env.get("HADOOP_CONF_DIR").get + "/core-site.xml"
      } else {
        sys.env.get("YARN_CONF_DIR").get + "/core-site.xml"
      }
      hadoopConf.addResource(new Path(hdfsConfPath))
    }
    path.getFileSystem(hadoopConf)
  }

  def deleteChkptDirs(conf: SparkConf, dirs: Array[String]): Unit = {
    val fs = getFileSystem(conf, new Path(dirs(0)))
    dirs.foreach(dir => {
      fs.delete(new Path(dir), true)
    })
  }
}
