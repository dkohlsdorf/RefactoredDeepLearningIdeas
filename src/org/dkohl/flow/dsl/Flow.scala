package org.dkohl.flow.dsl

import org.dkohl.flow.computation._

object Flow {

  final val Variable = "var"

  final val PlaceHolder = "placeholder"

  def add(name: String, x: FlowNode, y: FlowNode): FlowNode =
    new FlowNode(name, Addition.Name, Vector(x,y), None)

  def mul(name: String, x: FlowNode, y: FlowNode): FlowNode =
    new FlowNode(name, Multiplication.Name, Vector(x,y), None)

  def relu(name: String, x: FlowNode): FlowNode =
    new FlowNode(name, ReLu.Name, Vector(x), None)

  def softmax(name: String, x: FlowNode): FlowNode =
    new FlowNode(name, SoftMaxOutput.Name, Vector(x), None)

  def placeholder(name: String): FlowNode =
    new FlowNode(name, PlaceHolder, Vector.empty[FlowNode], None)

  def variable(name: String, shape: (Int, Int, Int)): FlowNode =
    new FlowNode(name, Variable, Vector.empty[FlowNode], Some(shape))

}

