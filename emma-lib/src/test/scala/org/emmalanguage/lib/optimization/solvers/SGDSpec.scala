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

import scala.util.Random

class SGDSpec extends BaseLibSpec {
  val learningRate  = 0.1
  val maxIterations = 1000
  val miniBatchSize = 10
  val tolerance     = 1e-6
  val prng          = new Random()
  val N             = 100

  private def noise = 0.0 // prng.nextGaussian() / 10.0

  val equation1: Seq[(Array[Double], Double)] = Seq((Array(5.0, 4.0), 1.0), (Array(3.0, 6.0), 2.0))
  val solution1  = Array(-1.0 / 6.0, 1.0 / 3.0)

  val instances1: Seq[(Array[Double], Double)] = for (i <- 1 to N) yield {
    val idx  = prng.nextInt(2) // 0 or 1
    val inst = equation1(idx)

    (inst._1.map(x => x + noise), inst._2)
  }

  val equation2: Seq[(Array[Double], Double)] = Seq((Array(2.5, 4.0), -10.0), (Array(6.0, 3.0), 6.0))
  val solution2  = Array(3.0, -4.0)

  val instances2: Seq[(Array[Double], Double)] = for (i <- 1 to N) yield {
    val idx  = prng.nextInt(2) // 0 or 1
    val inst = equation2(idx)

    (inst._1.map(x => x + noise), inst._2)
  }

  "SGD solver" should "compute the correct weights for problem 1" in {
    val Xy = DataBag(for (i <- instances1.indices) yield
      LDPoint(i.toLong, dense(instances1(i)._1), instances1(i)._2))
    val w  = dense(Array.fill(2)(0.0))

    val exp = TestUtil.solve(instances1)

        println("Solution using breeze: " + exp.mkString(", "))
        println("Solution by hand     : " + solution1.mkString(", "))

    val act = SGD(
      learningRate,
      maxIterations,
      miniBatchSize,
      tolerance
    )(
      squaredLoss.loss,
      squaredLoss.gradient
    )(
      Xy,
      w
    )

        println("Solution using SGD: " + act._1.values.mkString(", "))
        println("loss: " + act._2.mkString(", "))

    TestUtil.normL2(exp.zip(act._1.values).map(v => v._1 - v._2)) should be < 1e-3
  }

  it should "compute the correct weights for problem 2" in {
    val Xy = DataBag(for (i <- instances2.indices) yield
      LDPoint(i.toLong, dense(instances2(i)._1), instances2(i)._2))
    val w  = dense(Array.fill(2)(0.0))

    val exp = TestUtil.solve(instances2)

        println("Solution using breeze: " + exp.mkString(", "))
        println("Solution by hand     : " + solution2.mkString(", "))

    val act = SGD(
      learningRate,
      maxIterations,
      miniBatchSize,
      tolerance
    )(
      squaredLoss.loss,
      squaredLoss.gradient
    )(
      Xy,
      w
    )

        println("Solution using SGD: " + act._1.values.mkString(", "))
        println("loss: " + act._2.mkString(", "))

    TestUtil.normL2(exp.zip(act._1.values).map(v => v._1 - v._2)) should be < 1e-3
  }
}
