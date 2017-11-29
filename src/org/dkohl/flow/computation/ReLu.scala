package org.dkohl.flow.computation

import org.dkohl.flow._

class ReLu extends Computation {

  override def fwd(node: NodeIdentifier, parents: Graph, activations: TensorStore): Mat = {
    val x = activations(parents(node)(0)).dup()
    for(i <- 0 until x.length) x.put(i, Math.max(x.get(i), 0.0f))
    x
  }

  override def bwd(node: NodeIdentifier, parents: Adjacency, gradient: Mat, activations: TensorStore): Mat = {
    val x = activations(parents(0)).dup()
    for(i <- 0 until x.length) if(x.get(i) > 0) x.put(i, gradient.get(i)) else x.put(i, 0.0f)
    x
  }

}

object ReLu {

  final val Name = "relu"

  def apply(): ReLu = new ReLu()

}
