/*
 * Copyright © 2017 TU Berlin (emma@dima.tu-berlin.de)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.emmalanguage
package lib.ml.regression

import lib.BaseLibSpec
import api.DataBag
import lib.linalg._
import lib.ml._
import lib.ml.optimization.objectives.squaredLoss
import lib.ml.optimization.solvers.SGD
import lib.util.TestUtil
import test.util.materializeResource
import test.util.tempPath

class LinRegSpec extends BaseLibSpec {

  val path = "/ml/regression/winequality"
  val temp = tempPath(path)

  override def tempPaths = Seq(path)

  override def resources = for {
    file <- Seq("winequality-red.csv")
  } yield () => materializeResource(s"$path/$file"): Unit

  // hyper-parameter
  val lr = 0.00001 // learning-rate
  val maxIter = 150 // maximum number of iterations through the whole data-set
  val miniBatchFrac = 0.001 // percentage of data to use for each minibatch update
  val convergenceTolerance = 1e-5 // size of l2norm(oldWeights - newWeights) before we stop

  val loss   = squaredLoss.loss _
  val grad   = squaredLoss.gradient _
  val solver = SGD.apply(lr, maxIter, miniBatchFrac, convergenceTolerance)(loss, grad)(_, _)

  "Linear Regression" should "minimize the training loss" in {
    val act = run(s"$temp/winequality-red.csv", solver, loss, grad)

    act._2.last should be < 0.01
  }

  it should "compute the correct weights" in {
    val data = for ((line, index) <- DataBag.readText(s"$temp/winequality-red.csv").zipWithIndex() if index > 0) yield {
      val record = line.split(";").map(_.toDouble)
      val label = record.head
      val dVector = dense(record.slice(1, record.length))
      LDPoint(index, dVector, label)
    }

    val seq = data.map(x => (x.pos.values, x.label)).collect()
    val exp = TestUtil.solve(seq)
    val act = run(s"$temp/winequality-red.csv", solver, loss, grad)

    println("breeze weights: " + exp.mkString(", "))
    println("our weights:    " + act._1.values.mkString(", "))

    val diff = TestUtil.normL2(exp.zip(act._1.values).map(v => v._1 - v._2))

    println("Diff: " + diff)
    println("breeze mse: " + TestUtil.mse(seq, exp))
    println("our mse:    " + TestUtil.mse(TestUtil.prependBias(seq), act._1.values))

    diff should be < 30.0
  }

  it should "converge" in {
    val act = run(s"$temp/winequality-red.csv", solver, loss, grad)

    act._2.length should be < maxIter
  }

  def run(
    input: String,
    solver: (DataBag[LDPoint[Long, Double]], DVector) => (DVector, Array[Double]),
    lossFunction: (LDPoint[Long, Double], DVector) => Double,
    gradientFunction: (LDPoint[Long, Double], DVector) => DVector): (DVector, Array[Double]) = {

    val data = for ((line, index) <- DataBag.readText(input).zipWithIndex() if index > 0) yield {
      val record = line.split(";").map(_.toDouble)
      val label = record.head
      val dVector = dense(record.slice(1, record.length))
      LDPoint(index, dVector, label)
    }

    val result = LinReg.train(data, solver)

    result
  }

}
