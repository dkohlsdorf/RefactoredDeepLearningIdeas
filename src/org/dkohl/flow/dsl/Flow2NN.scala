package org.dkohl.flow.dsl

import org.dkohl.flow.{NeuralNetwork, _}
import org.dkohl.flow.computation._
import org.dkohl.flow.optimizers.Optimizer
import org.jblas.FloatMatrix

object Flow2NN {

  def toNN(computeTree: FlowNode, optimizer: Optimizer): NeuralNetwork = {
    val c = children(computeTree).map {case(k, v) => k -> v.distinct}
    val p = parents(computeTree).map {case(k, v) => k -> v.distinct}
    val i = computation(c)
    val w = weights(p)
    new NeuralNetwork(c,p,i,w, optimizer)
  }

  def id(computeNode: FlowNode): NodeIdentifier = new NodeIdentifier(computeNode.name, computeNode.computation, computeNode.shape)

  private def parents(computeTree: FlowNode): Graph = {
    if (computeTree.isLeaf) Map(id(computeTree) -> computeTree.parents.map(id))
    else {
      Map(id(computeTree) -> computeTree.parents.map(id)) ++ computeTree.parents.flatMap(child => parents(child))
    }
  }

  private def children(computeTree: FlowNode): Graph = {
    val children = scala.collection.mutable.Map.empty[NodeIdentifier, Vector[NodeIdentifier]]
    val open = scala.collection.mutable.Queue.empty[FlowNode]
    open.enqueue(computeTree)
    while (open.nonEmpty) {
      val c = open.dequeue()
      val child = id(c)
      for(p <- c.parents) {
        val parent = id(p)
        if (children.contains(parent)) children(parent) = children(parent) ++ Vector(child)
        else children(parent) = Vector(child)
        open.enqueue(p)
      }
    }
    children.toMap
  }

  private def computation(graph: Graph): Map[NodeIdentifier, Computation] = {
    val children = graph.values.flatten.toSet ++ graph.keySet
    children.collect {
      case node if node.computation.equals(Addition.Name) => node -> Addition()
      case node if node.computation.equals(ReLu.Name) => node -> ReLu()
      case node if node.computation.equals(Multiplication.Name) => node -> Multiplication()
      case node if node.computation.equals(SoftMaxOutput.Name) => node -> SoftMaxOutput()
    }.toMap
  }

  private def weights(parents: Graph): TensorStore = {
    val w = parents.keys.collect {
      case node if node.computation.equals(Flow.Variable) => {
        if (node.shape.get._3 == Zeros) {
          node -> FloatMatrix.zeros(node.shape.get._1, node.shape.get._2)
        } else {
          val variance = 2.0f / node.shape.get._1
          node -> FloatMatrix.randn(node.shape.get._1, node.shape.get._2).mul(Math.sqrt(variance).toFloat)
        }
      }
    }
    scala.collection.mutable.Map(w.toSeq: _*)
  }

}
