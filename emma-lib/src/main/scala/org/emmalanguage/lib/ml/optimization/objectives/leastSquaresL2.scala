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
import lib.ml._

/**
 * Represents a sum of squares objective function with L2 regularization of the form
 *
 * E(w) = 1/2 * (Y − Xw)T(Y − Xw) + λ/2 * wTw
 *
 * where
 *      w: Weights
 *      X: Instances
 *      Y: Labels
 *      λ: Regularization Parameter
 */
@emma.lib
object leastSquaresL2 {

  /**
   * Compute the least squares solution loss function.
   *
   * @param instance The instance that is evaluated.
   * @param weights The weights that are used to evaluate the loss.
   * @return The loss as measured by the least squares solution.
   */
  def loss(instance: LDPoint[Long, Double], weights: DVector): Double = ???

  /**
   * Compute the gradient of the least squares solution loss function.
   *
   * @param instance The instance for which the gradient is computed.
   * @param weights The weights for which the gradient is computed.
   * @return The computed gradient.
   */
  def gradient(instance: LDPoint[Long, Double], weights: DVector): DVector = ???

  /**
   * Compute loss and gradient for the least squares solution loss function.
   *
   * @param instance The instance for which the values are computed.
   * @param weights The weights for which the values are computed.
   * @return A tuple containing the loss and its gradient with respect to the instance and weights.
   */
  def LossWithGradient(instance: LDPoint[Long, Double], weights: DVector): (Double, DVector) = ???

  /**
   * Perform a weight update step for this objective function.
   *
   * @param weights The old weights that will be updated.
   * @param gradient The gradient that will be used to update the weights.
   * @param learningRate The size of the update.
   * @return The new weights and information about the regularization parameter for further regulatization updates.
   */
  def update(weights: DVector, gradient: DVector, learningRate: Double): (DVector, Double) = ???

}
