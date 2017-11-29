package org.dkohl.flow.optimizers

import org.dkohl.flow._

trait Optimizer {

  def dup(): Optimizer

  def takeStep(weights: TensorStore, gradients: Map[NodeIdentifier, Mat])

}
