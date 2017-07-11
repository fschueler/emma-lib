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

import lib.linalg._
import lib.ml.LDPoint

/**
 * Represents a sum of squares loss function of the form
 *
 * E(w) = (wx - y)**2
 *
 * where
 *      w: Weights
 * x: Instance features
 * y: Instance label
 */
object squared extends Loss {

  /**
   * Compute the squared loss loss function.
   *
   * @param x The instance that is evaluated.
   * @param w The weights that are used to evaluate the loss.
   * @return The loss as measured by the least squares solution.
   */
  def apply[ID](x: LDPoint[ID, Double], w: DVector): Double = {
    val residual = BLAS.dot(w, x.pos) - x.label
    residual * residual
  }

  /**
   * Compute the gradient of the squared loss loss function.
   * {{{
   * dE(w) = (wx - y)x
   * }}}
   *
   * @param x The instance for which the gradient is computed.
   * @param w The weights for which the gradient is computed.
   * @return The computed gradient.
   */
  def gradient[ID](x: LDPoint[ID, Double], w: DVector): DVector = {
    val residual = BLAS.dot(w, x.pos) - x.label
    val gradient = x.pos.copy
    BLAS.scal(residual, gradient)
    gradient
  }

  // TODO implement a lossWithGradient function that only computes the residuals once
}
