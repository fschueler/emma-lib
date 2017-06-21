/*
 * Copyright © 2014 TU Berlin (emma@dima.tu-berlin.de)
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

@emma.lib
object LinReg {
  type Instance = LDPoint[Long, Double]

  def train(
    learningRate      : Double,
    maxIterations     : Int,
    regParam          : Double,
    fraction          : Double,
    tolerance         : Double
  )(
    instances: DataBag[Instance],
    solver            :  Any => Any => (DVector, Double),
    objectiveLoss     : (LDPoint[Long, Double], DVector) => Double,
    objectiveGradient : (LDPoint[Long, Double], DVector) => DVector,
  ): (DVector, Double) = {

    // extract the number of features
    val numFeatures = instances.sample(1)(0).pos.size

    // prepend bias feature column
    val X = for (x <- instances) yield {
      val inputValues = x.pos.values
      val outputValues = Array.ofDim[Double](numFeatures + 1)
      System.arraycopy(inputValues, 1, outputValues, 1, numFeatures)
      outputValues(0) = 1.0
      dense(outputValues)
    }

    // initialize weights with bias
    val W = dense(Array.fill[Double](numFeatures + 1)(0.0))

    val (solution, losses) = solver(
      learningRate,
      maxIterations,
      regParam,
      fraction,
      tolerance
    )(
      instances,
      W,
      objectiveLoss,
      objectiveGradient
    )

    (solution, losses)
  }

}
