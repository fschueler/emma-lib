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
package lib.ml.regression

import api._
import lib.ml._
import lib.linalg._
import lib.ml.optimization.error.ErrorFun
import api.Meta.Projections._

@emma.lib
object linreg {

  def train[ID: Meta](
    instances: DataBag[LDPoint[ID, Double]],
    solver   : (DataBag[LDPoint[ID, Double]], DVector) => (DVector, Array[Double])): (DVector, Array[Double]) = {

    // extract the number of features
    val numFeatures = instances.sample(1)(0).pos.size

    // prepend bias feature column
    val X = addBias(instances)

    // initialize weights with bias
    val W = dense(Array.fill[Double](numFeatures + 1)(0.0))

    val (solution, losses) = solver(
      X,
      W
    )

    (solution, losses)
  }

  def predict[EF <: ErrorFun, ID: Meta](
    model: DVector,
    errorFun: EF
  )(
    instances: DataBag[LDPoint[ID, Double]]
  ): Double = {

    errorFun.loss(model, addBias(instances))
  }

  def addBias[ID: Meta](data: DataBag[LDPoint[ID, Double]]): DataBag[LDPoint[ID, Double]] = {
    data.map(x => x.copy(x.id, dense(Array(1.0) ++ x.pos.values), x.label))
  }
}
