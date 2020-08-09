
// ND array imported as default since linalg is core to the project
// All other modules follow the practice of explicit namespaces
use ndarray::*;

fn main() {
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
