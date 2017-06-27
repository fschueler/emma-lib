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
import org.emmalanguage.lib.util.TestUtil
import test.util.materializeResource
import test.util.tempPath

class LinRegSpec extends BaseLibSpec {

  val path = "/ml/regression/winequality"
  val temp = tempPath(path)

  override def tempPaths = Seq(path)

  override def resources = for {
    file <- Seq("winequality-red.csv")
  } yield () => materializeResource(s"$path/$file"): Unit

  "Linear Regression" should "fit a linear function correctly" in {
    val loss   = squaredLoss.loss _
    val grad   = squaredLoss.gradient _

    val miniBatchSize = 10
    val lr = 0.5
    val maxIter = 10000
    val convergenceTolerance = 1e-5

    val solver = SGD.apply(lr, maxIter, miniBatchSize, convergenceTolerance)(loss, grad)(_, _)

    val a = 1.0
    val b = 7.0 // bias

    val data = for ((x, i) <- (-5.0 to 5.0 by 0.5).zipWithIndex) yield LDPoint(i.toLong, dense(Array(x)), a * x + b)

    val breezeData = data.map(ldp => (Array(1.0, ldp.pos.values(0)), ldp.label))
    val exp = TestUtil.solve(breezeData)

    val (weights, losses) = LinReg.train(DataBag(data), solver)
    
    // compare squared error (exp - act)^2
    exp.zip(weights.values).map(v => (v._1 - v._2) * (v._1 - v._2)).sum should be < 1e-5
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
