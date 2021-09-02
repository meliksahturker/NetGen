# NetGen
Network Generator framework that simulates daily interactions in an urban town

This implementation is based on NetworkX.

Example usage:

  net = Network()
  net.build_layer_one(1000, 2.52, 2, 0.5)
  net.build_layer_two(0.21, 10, 5, 0, 1000, 0.4)
  net.build_layer_three(0.33, 10, 5, 0, 1000, 0.3)
  net.build_layer_four(0.25, 20, 2, 0, 1000, 3, 0.2)
  net.build_layer_five(10, 5, 0, 1000, 0.1)
  net.build_layer_six(0.10, 100, 20, 0, 1000, 0.05)
  net.build_layer_seven(50, 20, 0, 1000, 0.01)
