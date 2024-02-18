use std::iter;

use nalgebra::{DMatrix, DVector};
use rand::{seq::SliceRandom, Rng, SeedableRng};

use crate::adam::Adam;

pub type ActivationFn = fn(v: f32, derivative: bool) -> f32;
pub type LossFn = fn(output: &DVector<f32>, target: &DVector<f32>, derivative: bool) -> f32;
pub type MetricFn = fn(output: &DVector<f32>, target: &DVector<f32>) -> f32;

pub struct Layer {
    biases: DVector<f32>,
    activation_fn: ActivationFn,
}

impl Layer {
    pub fn linear(num_neurons: usize, activation_fn: ActivationFn) -> Self {
        Self {
            biases: DVector::zeros(num_neurons),
            activation_fn,
        }
    }

    pub fn num_neurons(&self) -> usize {
        self.biases.len()
    }
}

struct LayerResult {
    a: DVector<f32>,
    de_df: DVector<f32>,
    df_dz: DVector<f32>,
}

struct Gradients {
    weights_grads: Vec<DMatrix<f32>>,
    bias_grads: Vec<DVector<f32>>,
    loss: f32,
}

impl Gradients {
    fn flatten(&self) -> DVector<f32> {
        let weights_grads: Vec<f32> = self
            .weights_grads
            .iter()
            .flat_map(|v| v.as_slice().iter().cloned())
            .chain(
                self.bias_grads
                    .iter()
                    .flat_map(|v| v.as_slice().iter().cloned()),
            )
            .collect();

        DVector::from_vec(weights_grads)
    }
}

pub struct NeuralNet {
    layers: Vec<Layer>,
    inter_weights: Vec<DMatrix<f32>>,
}

impl NeuralNet {
    pub fn new(num_inputs: usize, layers: Vec<Layer>) -> Self {
        let mut rng = rand::thread_rng();

        let inter_weights: Vec<_> = iter::once(num_inputs)
            .chain(layers.iter().map(|v| v.num_neurons()))
            .collect::<Vec<_>>()
            .windows(2)
            .map(|window| {
                DMatrix::from_fn(window[1], window[0], |_, _| Self::random_weight(&mut rng))
            })
            .collect();

        Self {
            layers,
            inter_weights,
        }
    }

    pub fn num_params(&self) -> usize {
        self.inter_weights.iter().map(|v| v.len()).sum::<usize>()
            + self.layers.iter().map(|v| v.biases.len()).sum::<usize>()
    }

    pub fn forward(&self, input: &DVector<f32>) -> DVector<f32> {
        let mut vals = input.clone();

        for (layer, curr_weights) in self.layers.iter().zip(self.inter_weights.iter()) {
            vals = (curr_weights * &vals + &layer.biases)
                .apply_into(|v| *v = (layer.activation_fn)(*v, false));
        }

        vals
    }

    fn random_weight(rng: &mut impl Rng) -> f32 {
        rng.gen_range::<f32, _>(-0.5..0.5)
    }

    fn calc_gradient(
        &self,
        input: &DVector<f32>,
        target: &DVector<f32>,
        loss_fn: LossFn,
    ) -> Gradients {
        let mut feed_results = vec![LayerResult {
            a: input.clone(),
            de_df: DVector::zeros(0),
            df_dz: DVector::zeros(0),
        }];

        for (layer, curr_weights) in self.layers.iter().zip(self.inter_weights.iter()) {
            let z = curr_weights * &feed_results.last().unwrap().a + &layer.biases;
            let a = z.map(|v| (layer.activation_fn)(v, false));

            let df_dz = z.map(|v| (layer.activation_fn)(v, true));

            feed_results.push(LayerResult {
                a,
                de_df: DVector::zeros(layer.num_neurons()),
                df_dz,
            });
        }

        let num_results = feed_results.len();
        let num_layers = feed_results.len() - 1;

        // skip calculation for the input layer
        for layer_idx in (0..num_layers).rev() {
            if layer_idx == num_layers - 1 {
                let curr_res = &mut feed_results[1 + layer_idx]; // skip input layer
                curr_res.de_df =
                    DVector::from_element(curr_res.a.nrows(), loss_fn(&curr_res.a, target, true));
            } else {
                let next_layer_res = &feed_results[1 + layer_idx + 1];

                let weights_dz_df = self.inter_weights[layer_idx + 1].transpose();
                let df_dz = &next_layer_res.df_dz;
                let de_df = weights_dz_df * df_dz.component_mul(&next_layer_res.de_df);

                let curr_res = &mut feed_results[1 + layer_idx];
                curr_res.de_df = de_df;
            }
        }

        let biases_grads: Vec<_> = feed_results
            .iter()
            .skip(1)
            .map(|res| res.df_dz.component_mul(&res.de_df))
            .collect();

        let inter_weights_grads: Vec<_> = (1..num_results)
            .map(|layer_idx| {
                let prev_res = &feed_results[layer_idx - 1];
                let res = &feed_results[layer_idx];

                res.df_dz.component_mul(&res.de_df) * prev_res.a.transpose()
            })
            .collect();

        let loss = loss_fn(&feed_results.last().unwrap().a, target, false);

        Gradients {
            weights_grads: inter_weights_grads,
            bias_grads: biases_grads,
            loss,
        }
    }

    fn apply_offsets(&mut self, all_offsets: &DVector<f32>) {
        // Decodes `Gradients::flatten`

        let mut slice = all_offsets.as_slice();

        for weights in &mut self.inter_weights {
            let len = weights.len();
            let offsets =
                DMatrix::from_column_slice(weights.nrows(), weights.ncols(), &slice[..len]);
            *weights += offsets;
            slice = &slice[len..];
        }

        for layer in &mut self.layers {
            let len = layer.biases.len();
            let offsets = DVector::from_column_slice(&slice[..len]);
            layer.biases += offsets;
            slice = &slice[len..];
        }

        assert!(slice.is_empty());
    }

    pub fn eval(
        &mut self,
        inputs: &[DVector<f32>],
        targets: &[DVector<f32>],
        metric_fn: MetricFn,
    ) -> f32 {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = self.forward(input);
            let loss = metric_fn(&output, target);
            total_loss += loss;
        }

        total_loss /= inputs.len() as f32;

        total_loss
    }

    pub fn train(
        &mut self,
        inputs: &[DVector<f32>],
        targets: &[DVector<f32>],
        loss_fn: LossFn,
        num_epochs: usize,
    ) {
        let mut inputs = inputs.to_vec();
        let mut targets = targets.to_vec();

        let mut adam = Adam::new(0.001, self.num_params());

        for i in 0..num_epochs {
            adam.reset();

            shuffle_with_seed(&mut inputs, i as u64);
            shuffle_with_seed(&mut targets, i as u64);

            let mut total_loss = 0.0;
            let num_iters = inputs.len();

            for (input, target) in inputs.iter().zip(&targets) {
                let grad = self.calc_gradient(input, target, loss_fn);

                let offsets = adam.do_step(&grad.flatten());
                self.apply_offsets(&offsets);

                total_loss += grad.loss;
            }

            println!("epoch {}, loss: {}", i, total_loss / num_iters as f32);
        }
    }
}

pub fn shuffle_with_seed<T>(v: &mut [T], seed: u64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    v.shuffle(&mut rng);
}
