package org.dkohl.flow.computation

import org.dkohl.flow._

class SoftMaxOutput extends Output {

  import Numerics._

  override def fwd(node: NodeIdentifier, parents: Graph, activations: TensorStore): Mat = {
    val x = activations(parents(node)(0)).dup()
    val scaler = (for(i <- 0 until x.length) yield x.get(i)).reduce(logSum)
    for(i <- 0 until x.length) x.put(i, Math.exp(x.get(i) - scaler).toFloat)
    x
  }

  override def bwd(node: NodeIdentifier, parents: Adjacency, gradient: Mat, activations: TensorStore): Mat = {
    return gradient
  }

  override def cost(p: Mat, y: Mat): Float = {
    val predicted = p.dup()
    for(i <- 0 until predicted.length) predicted.put(i, Math.log(predicted.get(i)).toFloat)
    -1.0f * y.mul(predicted).sum()
  }

  override def loss(truth: Mat, predicted: Mat): Mat = {
    truth.sub(predicted)
  }

}

object SoftMaxOutput {

  final val Name = "softmax"

  def apply(): SoftMaxOutput = new SoftMaxOutput()

}
