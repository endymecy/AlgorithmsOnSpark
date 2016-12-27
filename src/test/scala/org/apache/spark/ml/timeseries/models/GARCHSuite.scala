package org.apache.spark.ml.timeseries.models

import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext

/**
  * Created by endy on 16-12-22.
  */
class GARCHSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest{

  test("GARCH log likelihood") {
    val model = new GARCHModel(.2, .3, .4)
    val rand = new MersenneTwister(5L)
    val n = 10000

    val ts = new DenseVector(model.sample(n, rand))
    val logLikelihoodWithRightModel = model.logLikelihood(ts)

    val logLikelihoodWithWrongModel1 = new GARCHModel(.3, .4, .5).logLikelihood(ts)
    val logLikelihoodWithWrongModel2 = new GARCHModel(.25, .35, .45).logLikelihood(ts)
    val logLikelihoodWithWrongModel3 = new GARCHModel(.1, .2, .3).logLikelihood(ts)

    assert(logLikelihoodWithRightModel > logLikelihoodWithWrongModel1)
    assert(logLikelihoodWithRightModel > logLikelihoodWithWrongModel2)
    assert(logLikelihoodWithRightModel > logLikelihoodWithWrongModel3)
    assert(logLikelihoodWithWrongModel2 > logLikelihoodWithWrongModel1)
  }

  test("gradient") {
    val alpha = 0.3
    val beta = 0.4
    val omega = 0.2
    val genModel = new GARCHModel(omega, alpha, beta)
    val rand = new MersenneTwister(5L)
    val n = 10000

    val ts = new DenseVector(genModel.sample(n, rand))

    val gradient1 = new GARCHModel(omega + .1, alpha + .05, beta + .1).gradient(ts)
    assert(gradient1.forall(_ < 0.0))
    val gradient2 = new GARCHModel(omega - .1, alpha - .05, beta - .1).gradient(ts)
    assert(gradient2.forall(_ > 0.0))
  }
}
