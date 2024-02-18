mod adam;
mod functions;
mod neural_net;

use crate::{
    functions::{activation_mish, loss_mse, metric_mae},
    neural_net::{shuffle_with_seed, Layer, NeuralNet},
};
use functions::activation_tanh;
use nalgebra::DVector;

fn main() {
    let mut net = NeuralNet::new(
        1,
        vec![
            Layer::linear(8, activation_mish),
            Layer::linear(8, activation_mish),
            Layer::linear(8, activation_mish),
            Layer::linear(1, activation_tanh),
        ],
    );

    let mut inputs: Vec<_> = (0..1000)
        .map(|i| DVector::from_element(1, i as f32 / 999.0))
        .collect();
    let mut targets: Vec<_> = inputs
        .iter()
        .map(|x| DVector::from_element(1, (x[0] * 6.282).sin()))
        .collect();

    shuffle_with_seed(&mut inputs, 0);
    shuffle_with_seed(&mut targets, 0);

    let train_len = (inputs.len() as f32 * 0.8) as usize;
    let train_inputs = &inputs[..train_len];
    let train_targets = &targets[..train_len];
    let val_inputs = &inputs[train_len..];
    let val_targets = &targets[train_len..];

    net.train(train_inputs, train_targets, loss_mse, 500);

    let mae = net.eval(val_inputs, val_targets, metric_mae);
    println!("MAE: {}", mae);

    let test_x = inputs[100].clone();
    let target_y = targets[100].clone();
    let result = net.forward(&test_x);

    println!(
        "val: f({}) = {}, target = {}",
        test_x[0], result, target_y[0]
    );
}
