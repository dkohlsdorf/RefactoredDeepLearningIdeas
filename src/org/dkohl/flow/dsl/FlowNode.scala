package org.dkohl.flow.dsl

class FlowNode(val name: String, val computation: String, val parents: Vector[FlowNode], val shape: Option[(Int, Int, Int)]) extends Serializable {

  def isLeaf = parents.isEmpty

  override def toString(): String = name + " : " + computation + "()"

}
