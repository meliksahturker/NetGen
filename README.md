# NetGen
Network Generator framework for epidemics, that simulates daily interactions in an urban town

This is the implementation of **Multi-layer network approach in modeling epidemics in an urban town** which can be found on https://arxiv.org/abs/2109.02272

This implementation is based on NetworkX.

Example usage:

    from NetworkGenerator import Network

    net = Network()
    net.build_layer_one(1000, 3.9, 1.2, 1.7, 0.5)
    net.build_layer_two(0.21, 10, 5, 0, 1000, 0.4)
    net.build_layer_three(0.33, 10, 5, 0, 1000, 0.3)
    net.build_layer_four(0.25, 20, 2, 0, 1000, 3, 0.2)
    net.build_layer_five(10, 5, 0, 1000, 0.1)
    net.build_layer_six(0.10, 100, 20, 0, 1000, 0.05)
    net.build_layer_seven(50, 20, 0, 1000, 0.01)


# Citation
If you use this, cite as below:

    @article{turker2023multi,
      title={Multi-layer network approach in modeling epidemics in an urban town},
      author={Turker, Meliksah and Bingol, Haluk O},
      journal={The European Physical Journal B},
      volume={96},
      number={2},
      pages={16},
      year={2023},
      publisher={Springer}
    }
