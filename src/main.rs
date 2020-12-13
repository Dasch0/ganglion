use ndarray::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

/// Calculates the cross entropy loss of a prediction
/// Based on an array of categorical labels
fn calculate_cross_entropy_loss(
    prediction: &Array2<f64>,
    truth: &Array1<u8>
) -> f64
{
    // clip prediction values of 0 or 1
    // TODO: Remove extra allocations, or preallocate
    let clipped_prediction = prediction.mapv(|v| v.max(1e-7).min(1.0 - 1e-7));

    // TODO: Remove extra allocations, or preallocate 
    let confidences: Array1<f64> = clipped_prediction
        .axis_iter(Axis(0))
        .zip(truth)
        .map(|(row, label)| {
            row[*label as usize]
        }).collect();

    // TODO: Remove extra allocations, or preallocate
    let sample_losses = confidences.mapv(|v|-v.ln());

    sample_losses.mean().unwrap()
}

/// Calculates the accuracy of a prediction
/// Based on an array of categorical labels
// TODO: Cleanup convoluted logic 
fn calculate_accuracy(
    prediction: &Array2<f64>,
    truth: &Array1<u8>,
) -> f64 {

    let classifications: Array1<usize> = prediction.axis_iter(Axis(0)).map(|row| {
        let (i, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        i
    }).collect();

    let sample_accuracy: Array1<f64> = classifications
        .iter()
        .zip(truth)
        .map(|(c, t)| (*c == *t as usize) as u64 as f64)
        .collect();
    sample_accuracy.mean().unwrap()
}


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

/// Apply relu activation function to each element of the input
fn apply_relu(inputs: &mut Array2<f64>) {
    inputs
        .iter_mut()
        .for_each(|x| *x = *x * ((*x > 0.0) as u64 as f64));
}


/// Apply softmax to each row of the input
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

/// Generate an input data set with three labels in a spiral pattern
/// Algorithm adapted from https://cs231n.github.io/neural-networks-case-study/
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
            labels[ix[j] as usize] = i as u8;
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
    let (data, labels) = generate_spiral_data();
    let plot_data = to_plottable(&data);
    let s1 = Plot::new(plot_data).point_style(PointStyle::new().marker(PointMarker::Circle));
    let v = ContinuousView::new()
        .add(s1)
        .x_range(-1., 1.)
        .y_range(-1., 1.)
        .x_label("X")
        .y_label("Y");
    println!("input data: \n{}", Page::single(&v).dimensions(80, 30).to_text().unwrap());

    // create simple two layer network
    let dense_layer_0 = LayerDense::new(2, 3);
    let dense_layer_1 = LayerDense::new(3, 3);

    // run inference on network
    let mut output_layer_0 = dense_layer_0.forward(&data);
    apply_relu(&mut output_layer_0);
    let mut output_layer_1 = dense_layer_1.forward(&output_layer_0);
    apply_softmax(&mut output_layer_1);

    // calculate loss
    let loss = calculate_cross_entropy_loss(&output_layer_1, &labels);
    println!("loss: {:#?}", loss);

    // calculate accuracy
    let acc = calculate_accuracy(&output_layer_1, &labels);
    println!("accuracy: {:#?}", acc);
}
