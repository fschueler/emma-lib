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
package lib.ml.optimization.solver

import api._
import api.Meta.Projections._
import lib.linalg.DVector
import lib.linalg.dense
import lib.ml.LDPoint
import lib.ml.optimization.loss.squared

class SGDFlinkSpec extends SGDSpec with FlinkAware {
  override def run(instances: Seq[(Array[Double], Double)]): (DVector, Array[Double]) = {
    withDefaultFlinkEnv(implicit spark => emma.onFlink {
      val Xy = DataBag(for ((x, i) <- instances.zipWithIndex) yield
        LDPoint(i.toLong, dense(x._1), x._2))
      val w = dense(Array.fill(2)(0.0))

      sgd(
        learningRate,
        maxIterations,
        miniBatchSize,
        tolerance
      )(squared)(Xy, w)
    })
  }
}
