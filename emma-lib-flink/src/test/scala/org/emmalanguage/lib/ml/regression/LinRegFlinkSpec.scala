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
import api.Meta.Projections._
import lib.linalg.DVector
import lib.linalg.dense
import lib.ml.LDPoint
import lib.ml.optimization.solver.sgd
import lib.ml.optimization.loss.squared

class LinRegFlinkSpec extends LinRegSpec with FlinkAware {
  override def run(instances: Seq[(Array[Double], Double)]): (DVector, Array[Double]) = {
    withDefaultFlinkEnv(implicit spark => emma.onFlink {
      val data = DataBag(for ((x, i) <- instances.zipWithIndex) yield LDPoint(i.toLong, dense(x._1.drop(1)), x._2))
      val solver = sgd[Long](lr, maxIter, miniBatchSize, convergenceTolerance)(squared)(_, _)

      linreg.train(data, solver)
    })
  }
}
