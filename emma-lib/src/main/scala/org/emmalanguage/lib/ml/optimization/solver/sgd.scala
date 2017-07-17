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

import api._
import api.Meta.Projections._
import lib.linalg._
import lib.ml.LDPoint
import lib.ml.optimization.error.ErrorFun
import lib.ml.optimization.regularization.Regularization
import lib.ml.optimization.regularization.noRegularization

import scala.collection.mutable.ArrayBuffer

@emma.lib
object sgd {

  def apply[EF <: ErrorFun, ID: Meta](
    learningRate      : Double,
    maxIterations     : Int,
    miniBatchSize     : Int,
    tolerance         : Double,
    lambda            : Double = 0.0
  )(
    errorfunc          : EF,
    reg               : Regularization = noRegularization
  )(
    instances         : DataBag[LDPoint[ID, Double]],
    initialWeights    : DVector
  ): (DVector, Array[Double]) = {

    val numInstances = instances.size

    require(numInstances > 0, "Number of instances must be > 0.")
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

      // sum the partial losses and gradients
      val loss = errorfunc.loss(weights, batch) + lambda * reg.loss(weights)
      val grad = errorfunc.gradient(weights, batch) + lambda * reg.gradient(weights)

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
