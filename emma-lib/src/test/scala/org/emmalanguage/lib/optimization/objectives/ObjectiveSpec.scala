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
package lib.optimization.objectives

import lib.BaseLibSpec
import lib.linalg._
import lib.ml.optimization.objectives._
import org.emmalanguage.lib.ml.LDPoint

import scala.util.Random

class ObjectiveSpec extends BaseLibSpec {

  val d = 10 // dimensionality
  val N = 20 // number of instances
  val prng = new Random()

  "squared loss objective" should "calculate correct losses" in {
    val w = Array.fill(d)(1.0) // weight vector
    val x = Array.fill(d)(prng.nextDouble()) // instance vector

    val target = 5.0 // label

    var sum = 0.0
    var i = 0
    while (i < d) {
      sum += w(i) * x(i)
      i += 1
    }
    val exp = (1.0 / 2.0) * ((sum - target) * (sum - target))

    val act = squaredLoss.loss(LDPoint(1L, dense(x), target), dense(w))

    act shouldEqual exp
  }

  it should "calculate correct gradients" in {
    val w = Array.fill(d)(1.0) // weight vector
    val x = Array.fill(d)(prng.nextDouble()) // instance vector

    val target = 5.0 // label


  }

  it should "update weights correctly" in pending

  "squared loss objective with l2 regularization" should "calculate correct losses" in pending

  it should "calculate correct gradients" in pending

  it should "update weights correctly" in pending

  // TODO add tests
}
