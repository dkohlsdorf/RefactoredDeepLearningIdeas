package org.dkohl.flow.computation

import org.dkohl.flow._

class Addition extends Computation {

  override def fwd(node: NodeIdentifier, parents: Graph, activations: TensorStore): Mat = {
    val x = activations(parents(node)(0))
    val y = activations(parents(node)(1))
    x.add(y)
  }

  override def bwd(node: NodeIdentifier, parents: Adjacency, gradient: Mat, activations: TensorStore): Mat = gradient

}

object Addition {

  final val Name = "add"

  def apply(): Addition = new Addition()

}
