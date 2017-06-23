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
package lib.util

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

object TestUtil {

  def normL2(vec: Array[Double]): Double = {
    val d = vec.length
    var i = 0
    var sum = 0.0
    while (i < d) {
      val xi = vec(i)
      sum += xi * xi
      i += 1
    }
    math.sqrt(sum)
  }

  def solve(instances: Seq[(Array[Double], Double)]): Array[Double] = {
    val Xv: Seq[DenseMatrix[Double]] = instances.map(x =>DenseMatrix(x._1))

    val X: DenseMatrix[Double] = DenseMatrix.vertcat(Xv:_*)
    val Y: DenseVector[Double] = DenseVector(instances.map(x => x._2).toArray)

    val A: DenseMatrix[Double] = X.t * X
    val b: DenseVector[Double] = X.t * Y
    val w: DenseVector[Double] = A \ b

    w.valuesIterator.toArray
  }

  def mse(instances: Seq[(Array[Double], Double)], weights: Array[Double]): Double = {
    val residuals = for (x <- instances) yield {
      require(x._1.length == weights.length,
        s"instance-dimensions and weight-dimension are different: ${x._1.length}, ${weights.length}")
      var N = weights.length
      var sum = 0.0
      var i = 0
      while (i < N) {
        sum += weights(i) * x._1(i)
        i += 1
      }
      sum - x._2
    }

    val squares = residuals.map(x => x * x)

    squares.sum / instances.length
  }

  def prependBias(instances: Seq[(Array[Double], Double)]): Seq[(Array[Double], Double)] = {
    for (instance <- instances) yield {
      val N = instance._1.length
      val ni = Array.fill(N + 1)(0.0)
      ni(0) = 1.0
      var i = 1
      while (i < N) {
        ni(i) = instance._1(i - 1)
        i += 1
      }
      (ni, instance._2)
    }
  }
}
