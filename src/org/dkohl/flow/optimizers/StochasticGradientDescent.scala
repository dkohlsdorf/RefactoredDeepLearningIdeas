package org.dkohl.flow.optimizers
import org.dkohl.flow._

class StochasticGradientDescent(val rate: Float) extends Serializable with Optimizer {

  override def dup(): Optimizer = new StochasticGradientDescent(rate)

  override def takeStep(weights: TensorStore, gradients: Map[NodeIdentifier, Mat]): Unit = {
    for ((id, d) <- gradients) {
      weights(id) = weights(id).add(d.mul(rate))
    }
  }

}
