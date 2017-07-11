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
package lib.ml.optimization.solver

import lib.linalg._
import lib.ml.LDPoint
import lib.ml.optimization.loss.squared
import api.DataBag
import lib.util.TestUtil

import scala.util.Random

class SGDSpec extends lib.BaseLibSpec {
  val prng          = new Random()
  val N             = 100

  val equation1: Seq[(Array[Double], Double)] = Seq((Array(5.0, 4.0), 1.0), (Array(3.0, 6.0), 2.0))
  val solution1  = Array(-1.0 / 6.0, 1.0 / 3.0)

  val instances1: Seq[(Array[Double], Double)] = for (i <- 1 to N) yield {
    val idx  = prng.nextInt(2) // 0 or 1
    val inst = equation1(idx)

    (inst._1, inst._2)
  }

  val equation2: Seq[(Array[Double], Double)] = Seq((Array(2.5, 4.0), -10.0), (Array(6.0, 3.0), 6.0))
  val solution2  = Array(3.0, -4.0)

  val instances2: Seq[(Array[Double], Double)] = for (i <- 1 to N) yield {
    val idx  = prng.nextInt(2) // 0 or 1
    val inst = equation2(idx)

    (inst._1, inst._2)
  }

  val learningRate  = 0.2
  val maxIterations = 1000
  val miniBatchSize = 10
  val tolerance     = 1e-6

  "SGD solver" should "compute the correct weights for problem 1" in {
    val exp = TestUtil.solve(instances1)
    val act = run(instances1)

    TestUtil.normL2(exp.zip(act._1.values).map(v => v._1 - v._2)) should be < 1e-3
  }

  it should "compute the correct weights for problem 2" in {
    val exp = TestUtil.solve(instances2)
    val act = run(instances2)

    TestUtil.normL2(exp.zip(act._1.values).map(v => v._1 - v._2)) should be < 1e-3
  }

  def run(instances: Seq[(Array[Double], Double)]): (DVector, Array[Double]) = {
    val Xy = DataBag(for ((x, i) <- instances.zipWithIndex) yield
      LDPoint(i.toLong, dense(x._1), x._2))
    val w  = dense(Array.fill(2)(0.0))

    sgd(
      learningRate,
      maxIterations,
      miniBatchSize,
      tolerance
    )(squared)(Xy, w)
  }
}
