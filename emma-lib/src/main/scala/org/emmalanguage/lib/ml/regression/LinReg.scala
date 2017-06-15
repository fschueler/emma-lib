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

import breeze.linalg._

@emma.lib
object LinReg {
  type Instance = LDPoint[Long, Double]

  def train(
           regParam: Double = 0.0, maxIter: Int = 100, tolerance: Double = 1E-6 // hyper-parameters
           )(
           dataset: DataBag[Instance]// data-parameters
  ): DataBag[DVector] = {
    // extract the number of features
    val numFeatures = dataset.sample(1)(0).pos.dat.size





  }

}
