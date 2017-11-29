package org.dkohl.examples

import org.dkohl.flow.computation.Numerics
import org.dkohl.flow.optimizers.Adam
import org.jblas.FloatMatrix

import scala.io.Source

/**
  * Classification Of Handwritten Digits using a Feed Forward Neural Network
  */
object Handwritten {

  import org.dkohl.flow._
  import org.dkohl.flow.dsl.Flow2NN._
  import org.dkohl.flow.dsl.Flow._
  import org.dkohl.flow.{Zeros, Xavier}
  import Numerics._

  def read(file: String): Iterator[(Int, Mat)] = Source.fromFile(file).getLines().map(line => {
    val cmp = line.split(",")
    val N = cmp.length
    val label = cmp(0).toInt
    val data = standardScore(new FloatMatrix(cmp.slice(1, N).map(_.toFloat)).reshape(1, N - 1))
    (label, data)
  })

  val Train = read("data/mnist_train.csv").toArray
  val Test = read("data/mnist_test.csv").toArray

  def main(args: Array[String]): Unit = {
    // Build Neural Network
    val x = placeholder("input")
    val y = placeholder("truth")
    val w1 = variable("w1", (784, 512, Xavier))
    val b1 = variable("b1", (  1, 512, Zeros))
    val z1 = add("1", mul("1", x, w1), b1)
    val a1 = relu("1", z1)
    val w2= variable("w2",  (512, 128, Xavier))
    val b2 = variable("b2", (  1, 128, Zeros))
    val z2 = add("2", mul("2", a1, w2), b2)
    val a2 = relu("2", z2)
    val w3 = variable("w3",  (128, 128, Xavier))
    val b3 = variable("b3", (  1, 128, Zeros))
    val z3 = add("3", mul("3", a2, w3), b3)
    val a3 = relu("3", z3)
    val w4= variable("w4",  (128, 128, Xavier))
    val b4 = variable("b4", (  1, 128, Zeros))
    val z4 = add("1->4", add("4", mul("4", a3, w4), b4), a2) // Skip connection in order to test weight sharing code
    val a4 = relu("4", z4)
    val w5= variable("w5",  (128, 10, Xavier))
    val b5 = variable("b5", (  1, 10, Zeros))
    val z5 = add("5", mul("5", a4, w5), b5)
    val a5 = softmax("out", z5)
    var net = toNN(a5, new Adam(0.001f, 0.9f, 0.999f, 10e-8f))

    for (epoch <- 0 until 10) {
      val start = System.currentTimeMillis()
      val (trained, loss) = Train.zipWithIndex.foldLeft((net, 0.0f)) { case ((nn, l), ((label, inst), i)) => {
        val truth = FloatMatrix.zeros(1, 10)
        truth.put(label, 1.0f)
        val feed = scala.collection.mutable.Map(id(x) -> inst, id(y) -> truth)
        val loss = nn.fit(id(a5), id(y), feed)
        nn -> (l + loss)
      }}
      net = trained
      val stop = System.currentTimeMillis()
      val correct = Test.map { case (label, inst) => {
        val feed = scala.collection.mutable.Map(id(x) -> inst)
        val prediction = trained.predict(id(a5), feed).argmax()
        if(prediction == label.toInt) 1.0 else 0.0
      }}.reduce(_ + _)
      println("Accuracy: " + correct / Test.size + " Took: " + (stop - start) + " [ms]")
    }

    val conf = Array.ofDim[Int](10,10)
    val correct = Test.map{ case (label, inst) => {
      val feed = scala.collection.mutable.Map(id(x) -> inst)
      val prediction = net.predict(id(a5), feed).argmax()
      conf(prediction)(label) += 1
      if(prediction == label.toInt) 1.0 else 0.0
    }}.reduce(_ + _)
    println(correct / Test.size)
    conf.foreach(row => println(row.mkString(", ")))
  }

}
