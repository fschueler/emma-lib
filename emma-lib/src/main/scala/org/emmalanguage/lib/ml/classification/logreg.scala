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
package lib.ml.classification

import api.DataBag
import lib.linalg.DVector
import lib.linalg.dense
import lib.ml.LDPoint

object logreg {
  type Instance = LDPoint[Long, Double]

  def train(
    instances: DataBag[Instance],
    solver   : (DataBag[LDPoint[Long, Double]], DVector) => (DVector, Array[Double])): (DVector, Array[Double]) = {

    // extract the number of features
    val numFeatures = instances.sample(1)(0).pos.size

    // initialize weights with bias
    val W = dense(Array.fill[Double](numFeatures)(0.0))

    val (solution, losses) = solver(
      instances,
      W
    )

    (solution, losses)
  }
}
