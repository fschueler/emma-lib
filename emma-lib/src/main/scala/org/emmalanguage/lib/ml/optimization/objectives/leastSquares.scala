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
package lib.ml.optimization.objectives

import api._
import lib.linalg._
import lib.ml.LDPoint

/**
 * Represents a sum of squares objective function of the form
 *
 * E(w) = 1/2 *(Y − Xw)T(Y − Xw)
 *
 * where
 *      w: Weights
 *      X: Instances
 *      Y: Labels
 */
@emma.lib
object leastSquares {

  /**
   * Compute the least squares solution loss function.
   *
   * @param instance The instance that is evaluated.
   * @param weights The weights that are used to evaluate the loss.
   * @return The loss as measured by the least squares solution.
   */
  def loss(instance: LDPoint[Long, Double], weights: DVector): Double = {
    val diff = BLAS.dot(instance.pos, weights) - instance.label
    (diff * diff) * (1.0 / 2.0)
  }

  /**
   * Compute the gradient of the least squares solution loss function.
   *
   * @param instance The instance for which the gradient is computed.
   * @param weights The weights for which the gradient is computed.
   * @return The computed gradient.
   */
  def gradient(instance: LDPoint[Long, Double], weights: DVector): DVector = {
    val diff = BLAS.dot(instance.pos, weights) - instance.label
    val gradient = dense(instance.pos.values.clone())
    BLAS.scal(diff, gradient)
    gradient
  }

  /**
   * Compute loss and gradient for the least squares solution loss function.
   *
   * @param instance The instance for which the values are computed.
   * @param weights The weights for which the values are computed.
   * @return A tuple containing the loss and its gradient with respect to the instance and weights.
   */
  def LossWithGradient(instance: LDPoint[Long, Double], weights: DVector): (Double, DVector) = {
    val diff = BLAS.dot(instance.pos, weights) - instance.label
    val l = (diff * diff) * (1.0 / 2.0)
    val gradient = dense(instance.pos.values.clone())
    BLAS.scal(diff, gradient)
    (l, gradient)
  }


  /**
   * Perform a weight update step for this objective function.
   *
   * @param weights The old weights that will be updated.
   * @param gradient The gradient that will be used to update the weights.
   * @param learningRate The size of the update.
   * @return The new weights and 0.0 for the historical regularization parameter because we are not applying any
   *         regularization here.
   */
  def update(weights: DVector, gradient: DVector, learningRate: Double): (DVector, Double) = {
    val newWeights = weights.copy
    BLAS.axpy(-learningRate, gradient, newWeights)
    (newWeights, 0.0)
  }
}
