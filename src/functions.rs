use nalgebra::DVector;

pub fn activation_tanh(x: f32, derivative: bool) -> f32 {
    if derivative {
        1.0 / x.cosh().powi(2)
    } else {
        x.tanh()
    }
}

// https://arxiv.org/abs/1908.08681
pub fn activation_mish(x: f32, derivative: bool) -> f32 {
    if derivative {
        let p = x.exp();
        let p2 = (x * 2.0).exp();
        let omega = 4.0 * (x + 1.0) + 4.0 * p2 + (3.0 * x).exp() + p * (4.0 * x + 6.0);
        let delta = 2.0 * p + p2 + 2.0;
        p * omega / delta.powi(2)
    } else {
        x * (1.0 + x.exp()).ln().tanh()
    }
}

pub fn loss_mse(output: &DVector<f32>, target: &DVector<f32>, derivative: bool) -> f32 {
    if derivative {
        (2.0 * (output - target)).mean()
    } else {
        (output - target).apply_into(|v| *v = v.powi(2)).mean()
    }
}

pub fn metric_mae(output: &DVector<f32>, target: &DVector<f32>) -> f32 {
    (output - target).abs().mean()
}
