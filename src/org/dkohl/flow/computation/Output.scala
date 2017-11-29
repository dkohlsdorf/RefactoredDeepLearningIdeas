package org.dkohl.flow.computation

import org.dkohl.flow._

trait Output extends Computation {

  def cost(p: Mat, y: Mat): Float

  def loss(truth: Mat, predicted: Mat): Mat

}
