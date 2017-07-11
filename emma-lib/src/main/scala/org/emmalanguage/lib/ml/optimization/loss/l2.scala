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
import lib.ml._

/**
 * Represents a squared loss objective function with L2 regularization of the form
 *
 * E(w) = (wx - y)**2 + λ/2 * wTw
 *
 * where
 *      w: Weights
 *      X: Instances
 *      Y: Labels
 *      λ: Regularization Parameter
 */
object l2 {

  /**
   * Compute the loss of the squared loss objective function with L2 regulatization.
   *
   * @param x The instance that is evaluated.
   * @param w The weights that are used to evaluate the loss.
   * @return The loss as measured by the least squares solution.
   */
  def loss(x: LDPoint[Long, Double], w: DVector): Double = ???

  /**
   * Compute the gradient of the squared loss objective function with L2 regulatization.
   *
   * @param x The instance for which the gradient is computed.
   * @param w The weights for which the gradient is computed.
   * @return The computed gradient.
   */
  def gradient(x: LDPoint[Long, Double], w: DVector): DVector = ???

  // TODO implement a lossWithGradient function that only computes the residuals once

}
