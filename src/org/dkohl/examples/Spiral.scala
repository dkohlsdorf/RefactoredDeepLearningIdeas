package org.dkohl.examples

import org.dkohl.flow.optimizers.{Adam, StochasticGradientDescent}
import org.jblas.FloatMatrix

import scala.io.Source
import scala.util.Random

/**
  * Very easy non-linear classification example
  */
object Spiral {

  import org.dkohl.flow.dsl.Flow2NN._
  import org.dkohl.flow.dsl.Flow._
  import org.dkohl.flow.{Zeros, Xavier}

  def main(args: Array[String]): Unit = {
    val x = placeholder("input")
    val y = placeholder("truth")
    val w1 = variable("w1", (2, 25, Xavier))
    val b1 = variable("b1", (1, 25, Zeros))
    val z1 = add("add1", mul("dot1", x, w1), b1)
    val a1 = relu("relu", z1)

    val w11 = variable("w11", (25, 3, Xavier))
    val b11 = variable("b11", (1,  3, Zeros))
    val z11 = add("add11", mul("dot11", a1, w11), b11)
    val a11 = relu("relu11", z11)

    val w2 = variable("w2", (25, 3, Xavier))
    val b2 = variable("b2", (1,  3, Zeros))
    val z2 = add("res", add("add2", mul("dot2", a1, w2), b2), a11)
    val a2 = softmax("out", z2)

    val nn = toNN(a2, new Adam(0.01f, 0.9f, 0.999f, 10e-8f))

    var trained = nn
    val X = Random.shuffle(Source.fromFile("data/spiral.csv").getLines().toList)
    val train = X.slice(0, (X.size * 0.8).toInt)
    val test  = X.slice((X.size * 0.2).toInt, X.size)

    for (i <- 1 until 2500) {
      val (n, s) = train.foldLeft((nn, 0.0f)) { case ((nn, l), line) => {
        val cmp = line.trim.split(",").map(_.toFloat)
        val data = cmp.slice(0, 2)
        val inst = new FloatMatrix(data).reshape(1, 2)
        val truth = FloatMatrix.zeros(1, 3)
        truth.put((cmp(2).toInt), 1.0f)
        val feed = scala.collection.mutable.Map(id(x) -> inst, id(y) -> truth)
        val loss = nn.fit(id(a2), id(y), feed)
        nn -> (l + loss)
      }}
      println(s)
      trained = n
    }

    val conf = Array.ofDim[Int](3,3)
    val correct = test.map(line => {
      val cmp = line.trim.split(",").map(_.toFloat)
      val data = cmp.slice(0, 2)
      val inst = new FloatMatrix(data).reshape(1, 2)
      val feed = scala.collection.mutable.Map(id(x) -> inst)
      val prediction = trained.predict(id(a2), feed).argmax()
      conf(prediction)(cmp(2).toInt) += 1
      if(prediction == cmp(2).toInt) 1.0 else 0.0
    }).reduce(_ + _)


    println(correct / test.size)
    println(conf(0)(0) + " " + conf(0)(1) + " " + conf(0)(2))
    println(conf(1)(0) + " " + conf(1)(1) + " " + conf(1)(2))
    println(conf(2)(0) + " " + conf(2)(1) + " " + conf(2)(2))
  }
}
