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
package lib.ml

import api._
import lib.linalg._

/** Point with identity and a dense vector position. */
case class DPoint[ID](@emma.pk id: ID, pos: DVector)

/** Point with identity and a sparse vector position. */
case class SPoint[ID](@emma.pk id: ID, pos: SVector)

/** Point with identity, a dense vector position, and a label. */
case class LDPoint[ID, L](@emma.pk id: ID, pos: DVector, label: L)

/** Point with identity, a dense vector position, and a label. */
case class LSPoint[ID, L](@emma.pk id: ID, pos: SVector, label: L)

/** Point with identity, weight, dense vector position, and label */
case class WLDPoint[ID, L](@emma.pk id: ID, w: Double, pos: DVector, label: L)

/** Point with identity, weight, sparse vector position, and label */
case class WLSPoint[ID, L](@emma.pk id: ID, w: Double, pos: SVector, label: L)

/** Features point. */
case class FPoint[ID, F](@emma.pk id: ID, features: F)
