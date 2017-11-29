package org.dkohl.flow.optimizers
import org.dkohl.flow.{Mat, NodeIdentifier, TensorStore}

class Adam(val baseRate: Float, val decayFirstOrder: Float, val decaySecondOrder: Float, val scaler: Float) extends Serializable with Optimizer {

  private val means = scala.collection.mutable.Map.empty[NodeIdentifier, Mat]
  private val variances = scala.collection.mutable.Map.empty[NodeIdentifier, Mat]
  private var t = 0

  override def dup(): Optimizer = new Adam(baseRate, decayFirstOrder, decaySecondOrder, scaler)

  def manageHistory(gradients: Map[NodeIdentifier, Mat]): Unit = {
    for ((id, g) <- gradients) {
      if (means.contains(id)) {
        val scaleGradientMean     = g.mul(1.0f - decaySecondOrder)
        val squareGradient        = g.mul(g)
        val scaleGradientVariance = squareGradient.mul(1.0f - decaySecondOrder)
        val scaleLastMean         = means(id).mul(decayFirstOrder)
        val scaleLastVariance     = variances(id).mul(decaySecondOrder)
        means(id) = scaleLastMean.add(scaleGradientMean)
        variances(id) = scaleLastVariance.add(scaleGradientVariance)
      } else {
        means(id) = g
        variances(id) = g.mul(g)
      }
    }
    t += 1
  }

  override def takeStep(weights: TensorStore, gradients: Map[NodeIdentifier, Mat]): Unit = {
    manageHistory(gradients)
    for (id <- gradients.keySet) {
      val correctedMean  = means(id).div(1.0f - Math.pow(decayFirstOrder, t).toFloat)
      val gradientScaler = variances(id).div(1.0f - Math.pow(decaySecondOrder, t).toFloat)
      for(i <- 0 until gradientScaler.length) gradientScaler.put(i, baseRate / Math.sqrt(gradientScaler.get(i) + scaler).toFloat)
      val gradient = gradientScaler.mul(correctedMean)
      weights(id) = weights(id).add(gradient)
    }
  }

}
