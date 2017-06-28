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
package lib.ml.optimization.solvers

import api._
import lib.ml.LDPoint
import lib.linalg._
import lib.stats.stat

import scala.collection.mutable.ArrayBuffer

@emma.lib
object SGD {

  def apply(
    learningRate      : Double,
    maxIterations     : Int,
    miniBatchSize     : Int,
    tolerance         : Double
  )(
    objectiveLoss     : (LDPoint[Long, Double], DVector) => Double,
    objectiveGradient : (LDPoint[Long, Double], DVector) => DVector
  )(
    instances         : DataBag[LDPoint[Long, Double]],
    initialWeights    : DVector
  ): (DVector, Array[Double]) = {

    val numInstances = instances.size

    if (numInstances == 0) {
      return (initialWeights, Array())
    }

    require(tolerance > 0, "Tolearnce must be > 0.")
    // TODO add prerequesite conditions

    // initialize weights
    var weights     = initialWeights
    val numFeatures = weights.size

    // initialize the loss history
    val stochasticLossHistory = new ArrayBuffer[Double](maxIterations)

    var converged = false
    var iter = 1

    // start the actual solving iterations
    while (!converged && iter <= maxIterations) {
      // sample a subset of the data
      val batch = DataBag(instances.sample(miniBatchSize, 42 + iter))

      // compute subgradients and losses for each instance in the batch
      val lossesAndGradients = for (x <- batch) yield {
        val l = objectiveLoss(x, weights)
        val g = objectiveGradient(x, weights)
        (l, g)
      }

      // sum the partial losses and gradients
      val loss = lossesAndGradients.map(_._1).sum / miniBatchSize.toDouble
      val grad = stat.mean(numFeatures)(lossesAndGradients.map(_._2))

      // compute learning rate for this iteration
      val lr = learningRate / math.sqrt(iter)

      // perform weight update against the direction of the gradient proportional to size of lr
      val newWeights = weights - (grad * lr)

      // compute convergence criterion
      val diff = normL2(weights - newWeights)
      converged = diff < tolerance * Math.max(normL2(weights), 1.0)

      // update weights
      weights = newWeights
      // append loss for this iteration
      stochasticLossHistory.append(loss)
      iter += 1
    }

    (weights, stochasticLossHistory.toArray)
  }
}
