package org.dkohl.flow.computation

import org.dkohl.flow._

trait Computation {

  def fwd(node: NodeIdentifier, parents: Graph, activations: TensorStore): Mat

  def bwd(node: NodeIdentifier, parents: Adjacency, gradient: Mat, activations: TensorStore): Mat

}
