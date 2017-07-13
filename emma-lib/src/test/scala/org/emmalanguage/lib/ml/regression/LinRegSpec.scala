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

import api.DataBag
import lib.linalg._
import lib.ml._
import lib.ml.optimization.solver.sgd
import lib.util.TestUtil
import lib.ml.optimization.error.MSE
import lib.ml.optimization.regularization.l2

class LinRegSpec extends lib.BaseLibSpec {
  val miniBatchSize = 10
  val lr = 0.5
  val maxIter = 10000
  val convergenceTolerance = 1e-5

  "Linear Regression" should "fit a linear function correctly" in {
    val a = 1.0
    val b = 7.0 // bias
    val _from = -5.0
    val _to   =  5.0
    val _by   =  0.5

    val instances = for ((x, i) <- (_from to _to by _by).zipWithIndex) yield (Array(1.0, x), a * x + b)
    val exp = TestUtil.solve(instances)

    val act = run(instances)
    
    // compare squared error (exp - act)^2
    exp.zip(act._1.values).map(v => (v._1 - v._2) * (v._1 - v._2)).sum should be < 1e-5
  }

  def run(instances: Seq[(Array[Double], Double)]): (DVector, Array[Double]) = {
    val data = DataBag(for ((x, i) <- instances.zipWithIndex) yield LDPoint(i.toLong, dense(x._1.drop(1)), x._2))
    val solver = sgd[Long](lr, maxIter, miniBatchSize, convergenceTolerance)(MSE, l2)(_, _)

    linreg.train(data, solver)
  }
}
