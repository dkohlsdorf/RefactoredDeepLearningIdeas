package org.dkohl

import org.dkohl.computation.Computation
import org.jblas.FloatMatrix

package object flow {

  final val Zeros  = 0
  final val Xavier = 1

  type Adjacency = Vector[NodeIdentifier]
  type Graph = Map[NodeIdentifier, Adjacency]

  type Instructions = Map[NodeIdentifier, Computation]
  type Mat = FloatMatrix
  type TensorStore = scala.collection.mutable.Map[NodeIdentifier, Mat]

  class NodeIdentifier(val name: String, val computation: String, val shape: Option[(Int, Int, Int)]) {
    override def equals(obj: scala.Any): Boolean = {
      if(obj.isInstanceOf[NodeIdentifier]) toString().equals(obj.asInstanceOf[NodeIdentifier].toString())
      else false
    }
    override def hashCode(): Int = toString().hashCode()
    override def toString(): String = name + "_" + computation
  }

}
