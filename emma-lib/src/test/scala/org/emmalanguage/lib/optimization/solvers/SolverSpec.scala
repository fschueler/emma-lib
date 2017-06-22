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
package lib.optimization.solvers

import lib.BaseLibSpec
import lib.linalg._
import lib.ml.LDPoint
import lib.ml.optimization.solvers.SGD
import lib.ml.optimization.objectives.squaredLoss
import api.DataBag
import lib.util.TestUtil

import breeze.linalg._
//import breeze.numerics._

import scala.util.Random

class SolverSpec extends BaseLibSpec {
  val learningRate  = 0.1
  val maxIterations = 100
  val fraction      = 0.2
  val tolerance     = 1e-6
  val prng          = new Random()

  private def noise = prng.nextGaussian()

  val instances1: Seq[(Array[Double], Double)] = Seq((Array(5.0, 4.0), 1.0), (Array(3.0, 6.0), 2.0))
  val instances2: Seq[(Array[Double], Double)] = Seq((Array(2.5, 4.0), -10.0), (Array(6.0, 3.0), 6.0))
  val solution1  = Array(1.0 / 3.0, -1.0 / 6.0)
  val solution2  = Array(3.0, -4.0)

  private def solve(instances: Seq[(Array[Double], Double)]): Array[Double] = {
    val Xv: Seq[DenseMatrix[Double]] = instances.map { x =>
      val d = x._1.length
      val vec = Array.fill(d + 1)(0.0) // add bias
      vec(0) = 1.0
      var i = 1
      while (i < d) {
        vec(i) = x._1(i-1)
        i += 1
      }
      DenseMatrix(vec)
    }

    val X: DenseMatrix[Double] = DenseMatrix.vertcat(Xv:_*)
    val Y: DenseVector[Double] = DenseVector(instances.map(x => x._2).toArray)

    val A: DenseMatrix[Double] = X.t * X
    val b: DenseVector[Double] = X.t * Y
    val w: DenseVector[Double] = A \ b

    w.valuesIterator.toArray
  }

  "SGD solver" should "compute the correct weights" in {
    val Xy = DataBag(for (i <- instances1.indices) yield
      LDPoint(i.toLong, dense(instances1(i)._1), instances1(i)._2))
    val w  = dense(Array.fill(2)(0.0))

    val exp = solution1 // solve(instances1)

    println("Solution using breeze: " + exp.mkString(", "))
    println("Solution by hand     : " + solution1.mkString(", "))

    val act = SGD(
      learningRate,
      maxIterations,
      fraction,
      tolerance
    )(
      Xy,
      w,
      squaredLoss.loss,
      squaredLoss.gradient
    )

    println("Solution using SGD: " + act._1.values.mkString(", "))
    println("loss: " + act._2.mkString(", "))

    TestUtil.normL2(exp.zip(act._1.values).map(v => v._1 - v._2)) should be < tolerance
  }

  it should "converge" in pending

  // TODO add tests
}
