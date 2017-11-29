package org.dkohl.flow

import org.dkohl.flow.computation.Output
import org.dkohl.flow.optimizers.Optimizer

/**
  * A Neural Network
  *
  * References:
  *   [1] Goodfellow, Beningo: "Deep Learning", 2016, MIT-Press, http://www.deeplearningbook.org/contents/mlp.html
  *
  * @param children Maps each node id to its' childrens' node ids
  * @param parents Maps each node id to its' parents' node ids
  * @param computation Maps each node id to it's computation
  * @param weights Holds the weight and bias matrices
  * @param optimizer The optimizer to use
  */
class NeuralNetwork(val children: Graph, val parents: Graph, val computation: Instructions, val weights: TensorStore, val optimizer: Optimizer) {

  /**
    * Run forward pass to output node
    *
    * @param node output node id
    * @param feedDict input dict
    * @return f(input)
    */
  def predict(node: NodeIdentifier, feedDict: TensorStore): Mat = {
    val activations = weights ++ feedDict
    fwd(node, activations)
    activations(node)
  }

  /**
    * One fwd/bwd step and param update
    *
    * @param node output node id
    * @param truth truth node id
    * @param feedDict input dict
    * @return L(f(x), y)
    */
  def fit(node: NodeIdentifier, truth: NodeIdentifier, feedDict: TensorStore): Float = {
    val (loss, delta) = gradients(node, truth, feedDict)
    update(delta)
    loss
  }

  private def gradients(node: NodeIdentifier, truth: NodeIdentifier, feedDict: TensorStore): (Float, Map[NodeIdentifier, Mat]) = {
    assert(computation(node).isInstanceOf[Output])
    assert(weights.size > 0 && feedDict.size > 0)

    val activations = weights ++ feedDict
    fwd(node, activations)

    val out = computation(node).asInstanceOf[Output]
    val loss = out.cost(activations(node), activations(truth))
    val gradient = out.loss(activations(truth), activations(node))

    val gradients = scala.collection.mutable.Map.empty[NodeIdentifier, Mat]
    gradients(node) = gradient

    val delta = (for((weightId, weight) <- weights) yield {
      weightId -> bwd(weightId, gradients, activations)
    }).toMap
    return (loss, delta)
  }

  private def update(gradients: Map[NodeIdentifier, Mat]): Unit = {
    optimizer.takeStep(weights, gradients)
  }

  private def fwd(node: NodeIdentifier, activations: TensorStore): Unit = {
    if(!activations.contains(node)) {
      for (p <- parents(node)) {
        fwd(p, activations)
      }
      val a = computation(node).fwd(node, parents, activations)
      activations(node) = a
    }
  }

  private def bwd(
                   node: NodeIdentifier,
                   gradients: TensorStore,
                   activations: TensorStore
                 ): Mat = {
    if(gradients.contains(node)) {
      gradients(node)
    }
    else {
      val gradient = for (c <- children(node)) yield {
        val gradient = bwd(c, gradients, activations)
        computation(c).bwd(node, parents(c), gradient, activations)
      }
      val weightShared = gradient.reduce((a,b) => a.add(b))
      gradients(node) = weightShared
      weightShared
    }
  }

}

