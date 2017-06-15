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
package lib.ml.stats

import breeze.linalg.DenseVector
import org.emmalanguage.lib.ml.{DVector, WLDPoint}

case class Summary(
                    count : Long,
                    wSum  : Double,
                    wwSum : Double,
                    bSum  : Double,
                    bbSum : Double,
                    aSum  : DVector,
                    abSum : DVector,
                    aaSum : DVector
                  ) {

  /* Weighted mean of features. */
  lazy val aBar: DVector = ???

  /* Weighted mean of labels. */
  lazy val bBar: Double = bSum / wSum

  /* Weighted mean of squared labels. */
  lazy val bbBar: Double = bbSum / wSum

  /* Weighted population standard deviation of labels. */
  lazy val bStd: Double = math.sqrt(bbSum / wSum - bBar * bBar)

  /* Weighted mean of (label * features) */
  lazy val abBar: DVector = ???

  /* Weighted mean of (features * features^T^). */
  lazy val aaBar: DVector = ???

  /* Weighted pipulation standard deviation of features. */
  lazy val aStd: DVector = ???

  /* Weighted population variance of features */
  lazy val aVar: DVector = ???
}

object Summary {

  def apply[ID](instance: WLDPoint[ID, Double]): Summary = {
    val i = instance
    Summary(
      1L,
      i.w,
      i.w * i.w,
      i.w * i.label,
      i.w * i.label * i.label,
      i.w * DenseVector[Double](i.pos.dat),
      i.w * i.label * DenseVector(i.pos.dat),
      i.w * DenseVector(i.pos * i.pos.t)
    )
  }

  def empty(k: Int): Summary =
    Summary(
      0L,
      0.0,
      0.0,
      0.0,
      0.0,
      DenseVector.zeros[Double](k),
      DenseVector.zeros[Double](k),
      DenseVector.zeros[Double](k * (k + 1) / 2)
    )

  // TODO make numerically stable
  def combine(s1: Summary, s2: Summary): Summary =
    Summary(
      s1.count + s2.count,
      s1.wSum  + s2.wSum,
      s1.wwSum + s2.wwSum,
      s1.bSum  + s2.bSum,
      s1.bbSum + s2.bbSum,
      s1.aSum  + s2.aSum,
      s1.abSum + s2.abSum,
      s1.aaSum + s2.aaSum
    )
  // FIXME: make sure these aggregations are correct
  // original code in org.apache.spark.ml.optim.WeightedLeastSquares.Aggregator
}
