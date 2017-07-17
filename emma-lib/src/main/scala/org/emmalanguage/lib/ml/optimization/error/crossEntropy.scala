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
package lib.ml.optimization.error

import api.DataBag
import lib.linalg.DVector
import lib.ml.LDPoint
import lib.ml.optimization.loss.crossentropy
import lib.stats.stat

object crossEntropy extends ErrorFun {
  def loss[ID](
    weights: DVector,
    instances: DataBag[LDPoint[ID, Double]]
  ): Double = -1 * instances.map(x => crossentropy(x, weights)).sum

  def gradient[ID](
    weights: DVector,
    instances: DataBag[LDPoint[ID, Double]]
  ): DVector = stat.sum(weights.size)(instances.map(x => crossentropy.gradient(x, weights)))
}
