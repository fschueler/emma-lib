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
package lib.ml.optimization.loss

import lib.BaseLibSpec
import lib.linalg._
import lib.ml.LDPoint
import lib.util.TestUtil

import scala.util.Random

class ObjectiveSpec extends BaseLibSpec {
  val tolerance = 1e-6

  val d = 10 // dimensionality
  val N = 20 // number of instances
  val prng = new Random()

  private def sqLoss(weights: Array[Double], instance: Array[Double], target: Double): Double = {
    var sum = 0.0
    var i = 0
    while (i < d) {
      sum += weights(i) * instance(i)
      i += 1
    }
    val residual = target - sum
    residual * residual
  }

  private def sqGradient(weights: Array[Double], instance: Array[Double], target: Double): Array[Double] = {
    // calculate wTx
    var sum = 0.0
    var i = 0
    while (i < d) {
      sum += weights(i) * instance(i)
      i += 1
    }

    // calculate (t - wTx)
    val residual = target - sum

    // calculate (t - wTx)*x
    val gradient = Array.fill[Double](d)(0.0)
    i = 0
    while (i < d) {
      gradient(i) = residual * instance(i)
      i += 1
    }
    gradient
  }

  "squared loss objective" should "calculate correct losses" in {
    val w = Array.fill(d)(1.0) // weight vector
    val x = Array.fill(d)(prng.nextDouble()) // instance vector

    val target = 5.0 // label

    val exp = sqLoss(w, x, target)
    val act = squared(LDPoint(1L, dense(x), target), dense(w))

    math.abs(act - exp) should be < tolerance
  }

  it should "calculate correct gradients" in {
    val w = Array.fill(d)(1.0) // weight vector
    val x = Array.fill(d)(prng.nextDouble()) // instance vector

    val target = 5.0 // label

    val exp = sqGradient(w, x, target)
    val act = squared.gradient(LDPoint(1L, dense(x), target), dense(w))

    TestUtil.normL2(exp.zip(act.values).map(v => v._1 - v._2)) should be < tolerance
  }

  it should "update weights correctly" in pending

  "squared loss objective with l2 regularization" should "calculate correct losses" in pending

  it should "calculate correct gradients" in pending

  it should "update weights correctly" in pending

  // TODO add tests
}
