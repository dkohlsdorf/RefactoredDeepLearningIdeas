# Refactored Deep Learning Ideas

+ examples: standard examples including the spiral dataset and mnist
+ flow/dsl: setup of the neural computation graph
+ flow/computation: all possible computations:
  - add
  - mul
  - relu
  - softmax
+ flow/optimization: sgd and adam
+ putting all together in neural net class

You will need jblas

+ Accuracy is 97.4% on MNIST
+ Confusion

|label|0|1|2|3|4|5|6|7|8|9|
|-|-|-|-|-|-|-|-|-|-|-|
|0|969|0|3|0|0|2|8|1|1|0|
|1|1|1129|3|0|0|0|3|8|2|3|
|2|1|1|1006|3|5|0|2|9|4|1|
|3|0|1|6|997|1|13|1|5|19|7|
|4|0|0|2|2|960|2|2|4|6|7|
|5|1|0|0|1|0|864|11|0|7|1|
|6|3|1|2|0|2|3|930|0|1|1|
|7|0|1|7|3|0|1|0|981|2|1|
|8|1|2|2|2|0|4|1|1|924|1|
|9|4|0|1|2|14|3|0|19|8|987|
