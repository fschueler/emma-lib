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
import lib.ml.optimization.solver.sgd
import lib.ml.optimization.error.crossEntropy
import org.emmalanguage.lib.linalg.BLAS

class LogRegSpec extends lib.BaseLibSpec {
  val miniBatchSize = 10
  val lr = 0.5
  val maxIter = 10000
  val convergenceTolerance = 1e-5

  "Logistic Regression" should "seperate two classes correctly" in {
    val a = 1.0
    val b = 7.0 // bias
    val _from = -5.0
    val _to   =  5.0
    val _by   =  0.5

    val instances = for ((x, i) <- (_from to _to by _by).zipWithIndex) yield
                          (Array(x), if (a * x > 0.0) 1.0 else -1.0)

    val act = run(instances)

    val res = instances.map(x => if (BLAS.dot(act._1, dense(x._1)) > 0.0) (1.0, x._2) else (-1.0, x._2))
                       .map(x => if (x._1 == x._2) 0.0 else 1.0)
                       .sum

    res shouldBe 0.0
  }

  def run(instances: Seq[(Array[Double], Double)]): (DVector, Array[Double]) = {
    val data = DataBag(for ((x, i) <- instances.zipWithIndex) yield LDPoint(i.toLong, dense(x._1), x._2))
    val solver = sgd[Long](lr, maxIter, miniBatchSize, convergenceTolerance)(crossEntropy)(_, _)

    logreg.train(data, solver)
  }
}
