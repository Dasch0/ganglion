// ndarray imported as default since linalg is core to the project
use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

struct LayerDense {
  pub weights: Array2<f64>,
  pub biases: Array1<f64>
}

impl LayerDense {
  fn new(n_inputs: Ix, n_neurons: Ix) -> LayerDense {
    LayerDense {
      weights: Array::random((n_inputs, n_neurons), Uniform::new(0., 1.)),
      biases: Array1::<f64>::zeros(n_neurons)
    }
  }
  fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
    inputs.dot(&self.weights) + &self.biases
  }
}

fn generate_spiral_data() -> (Array2<f64>, Array1<u8>) {
    let n = 100; // points per class
    let d = 2; // dimensionality
    let k = 3; // number of classes

    let mut data = Array2::<f64>::zeros((n*k, d));
    let mut labels = Array1::<u8>::zeros(n*k);

    for i in 0..k {
        let ix = Array::range((n*i) as f64, (n*(i+1)) as f64, 1.0);
        let r = Array::linspace(0.0, 1.0, n);
        let t = Array::linspace((i*4) as f64, ((i+1)*4) as f64, n)
            + Array::random(n, Uniform::new(0.0, 0.2));
        `
        data[ix] = r * t.sin();
        data[ix] = r * t.cos();
        labels[ix] = j;
    }
    (data, labels)
}

fn main() {
    let (data, labels) = generate_spiral_data();
    let inputs = arr2(&[[1.0, 2.0, 3.0, 2.5],
                     [2.0, 5.0, -1.0, 2.0],
                     [-1.5, 2.7, 3.3, -0.8]]);
    println!("inputs:\n{:#.2}\n", inputs);

    let weights = arr2(&[[0.2, 0.8, -0.5, 1.0],
                      [0.5, -0.91, 0.26, -0.5],
                      [-0.26, -0.27, 0.17, 0.87]]);
  println!("weights:\n{:#.2}\n", weights);
  
  let biases = arr1(&[2.0, 3.0, 0.5]);
  println!("biases:\n{:#.2}\n", biases);

  let layer_outputs = inputs.dot(&weights.reversed_axes()) + biases;

  println!("layer outputs:\n{:#.2}\n", layer_outputs); 
}
