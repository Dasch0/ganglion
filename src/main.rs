use ndarray::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

struct LayerDense {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

impl LayerDense {
    fn new(n_inputs: Ix, n_neurons: Ix) -> LayerDense {
        LayerDense {
            weights: Array::random((n_inputs, n_neurons), Uniform::new(0., 1.)),
            biases: Array1::<f64>::zeros(n_neurons),
        }
    }
    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        inputs.dot(&self.weights) + &self.biases
    }
}

/// Apply relu activation function to the input, elementwise
fn apply_relu(inputs: &mut Array2<f64>) {
    inputs
        .iter_mut()
        .for_each(|x| *x = *x * ((*x > 0.0) as u64 as f64));
}

// TODO: The current implementation of softmax is naive and very slow.
// Should be possible to speed this up quite a bit
fn apply_softmax(inputs: &mut Array2<f64>) {
    // algorithm from nnfs.io chapter 4
    let e = 2.71828182846f64;

    // Step 1, find the max value in each row and subtract from each element in the row
    inputs.axis_iter_mut(Axis(0)).for_each(|row| {
        let max = *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        for x in row {
            *x = *x - max
        }
    });

    // Step 2, exponentiate each shifted element
    inputs.iter_mut().for_each(|x| *x = e.powf(*x));

    // Step 3, normalize each row
    inputs.axis_iter_mut(Axis(0)).for_each(|row| {
        let sum = row.sum();
        for x in row {
            *x = *x / sum
        }
    });
}

fn generate_spiral_data() -> (ndarray::Array2<f64>, ndarray::Array1<u8>) {
    let n = 100; // points per class
    let k = 3; // number of classes
    let d = 2;

    let mut data = Array2::<f64>::zeros((n * k, d));
    let mut labels = Array1::<u8>::zeros(n * k);
    for i in 0..k {
        let ix = Array::range((n * i) as f64, (n * (i + 1)) as f64, 1.0);
        let r = Array::linspace(0.0, 1.0, n);
        let t = Array::linspace((i * 4) as f64, ((i + 1) * 4) as f64, n)
            + Array::random(n, Uniform::new(0.0, 0.2));
        // TODO: Cleanup/optimization
        for j in 0..n {
            // Get slice for each row of the matrix, and assign x & y vals
            let mut row = data.slice_mut(s![ix[j] as usize, ..]);
            row[0] = r[j] * t[j].sin();
            row[1] = r[j] * t[j].cos();
            labels[ix[i] as usize] = i as u8;
        }
    }
    (data, labels)
}

/// Helper method to convert an ndarray to a plottable vec of tuples
/// Plotlib is fairly restrictive on what data types may be plotted.
///
/// Notes:
///     This accepts any 2D array, and assumes the first two elems
///     of each row correspond to the x and y values of a point.
///
///     This method is not performant. A per element copy is performed as
///     tuple memory layout cannot be guaranteed

fn to_plottable(data: &Array2<f64>) -> Vec<(f64, f64)> {
    data.axis_iter(Axis(0))
        .map(|row| (row[0], row[1]))
        .collect()
}

fn main() {
    // generate and plot training data
    let (data, _labels) = generate_spiral_data();
    let plot_data = to_plottable(&data);
    let s1 = Plot::new(plot_data).point_style(PointStyle::new().marker(PointMarker::Circle));
    let v = ContinuousView::new()
        .add(s1)
        .x_range(-1., 1.)
        .y_range(-1., 1.)
        .x_label("X")
        .y_label("Y");
    println!("{}", Page::single(&v).dimensions(80, 30).to_text().unwrap());

    // create simple two layer network
    let dense_layer_0 = LayerDense::new(2, 3);
    let dense_layer_1 = LayerDense::new(3, 3);

    // run inference on network
    let mut output_layer_0 = dense_layer_0.forward(&data);
    apply_relu(&mut output_layer_0);
    let mut output_layer_1 = dense_layer_1.forward(&output_layer_0);
    apply_softmax(&mut output_layer_1);

    // print the first few outputs
    println!("{:#?}", output_layer_1.slice(s![..5, ..]));
}
