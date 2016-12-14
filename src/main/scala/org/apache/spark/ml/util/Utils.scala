package org.apache.spark.ml.util

import java.util.Random

object Utils {
  val random = new Random()
  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }
}