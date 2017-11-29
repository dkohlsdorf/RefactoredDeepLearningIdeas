package org.dkohl.flow.computation

import org.dkohl.flow._

object Numerics {

  final val LOG_ZERO = Float.NegativeInfinity

  def standardScore(x: Mat): Mat = {
    val norm = x.dup()
    val mu = x.mean()
    val variance = (for(i <- 0 until norm.length) yield Math.pow(mu - norm.get(i), 2)).sum / norm.length
    val std = Math.sqrt(variance).toFloat
    for(i <- 0 until norm.length) norm.put(i, (norm.get(i) - mu) / (std + 0.0001f))
    norm
  }

  def logSum(logX: Float, logY: Float): Float = {
    if (logX == LOG_ZERO || logY == LOG_ZERO) {
      if (logX == LOG_ZERO) {
        logY
      } else {
        logX
      }
    } else {
      if (logX > logY) {
        logX + Math.log(1.0f + Math.exp(logY - logX)).toFloat
      } else {
        logY + Math.log(1.0f + Math.exp(logX - logY)).toFloat
      }
    }
  }



}
