package org.dkohl.flow.computation

import org.dkohl.flow._

class Multiplication with Computation {

  final val Name = "mul"

  override def fwd(node: NodeIdentifier, parents: Graph, activations: TensorStore): Mat = {
    val x = activations(parents(node)(0))
    val y = activations(parents(node)(1))
    x.mmul(y)
  }

  override def bwd(node: NodeIdentifier, parents: Adjacency, gradient: Mat, activations: TensorStore): Mat = {
    if (node == parents(0)) gradient.mmul(activations(parents(1)).transpose())
    else activations(parents(0)).transpose().mmul(gradient)
  }

}

object Multiplication {

  final val Name = "mul"

  def apply(): Multiplication = new Multiplication()

}
