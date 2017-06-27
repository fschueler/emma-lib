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
 * E(w) = (wx - y)**2
 *
 * where
 *      w: Weights
 *      X: Instances
 *      Y: Labels
 */
@emma.lib
object squaredLoss {

  /**
   * Compute the squared loss objective function.
   *
   * @param instance The instance that is evaluated.
   * @param weights The weights that are used to evaluate the loss.
   * @return The loss as measured by the least squares solution.
   */
  def loss(instance: LDPoint[Long, Double], weights: DVector): Double = {
    val residual = BLAS.dot(weights, instance.pos) - instance.label
    residual * residual
  }

  /**
   * Compute the gradient of the squared loss objective function.
   *
   * dE(w) = (wx - y)x
   *
   * @param instance The instance for which the gradient is computed.
   * @param weights The weights for which the gradient is computed.
   * @return The computed gradient.
   */
  def gradient(instance: LDPoint[Long, Double], weights: DVector): DVector = {
    val residual = BLAS.dot(weights, instance.pos) - instance.label
    val gradient = instance.pos.copy
    BLAS.scal(residual, gradient)
    gradient
  }

  // TODO implement a lossWithGradient function that only computes the residuals once
}
