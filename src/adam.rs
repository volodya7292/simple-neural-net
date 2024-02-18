use nalgebra::DVector;

// https://arxiv.org/abs/1412.6980
pub struct Adam {
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: f32,
    m: DVector<f32>,
    v: DVector<f32>,
}

impl Adam {
    pub fn new(step: f32, num_params: usize) -> Adam {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-07;

        Adam {
            m: DVector::zeros(num_params),
            v: DVector::zeros(num_params),
            beta1,
            beta2,
            eps,
            step,
        }
    }

    pub fn reset(&mut self) {
        self.m = DVector::zeros(self.m.nrows());
        self.v = DVector::zeros(self.v.nrows());
    }

    pub fn do_step(&mut self, grad: &DVector<f32>) -> DVector<f32> {
        self.m = self.beta1 * &self.m + (1.0 - self.beta1) * grad;
        self.v = self.beta2 * &self.v + (1.0 - self.beta2) * grad.map(|v| v.powi(2));

        let step_curr = self.step * (1.0 - self.beta2).sqrt() / (1.0 - self.beta1);

        -step_curr
            * self
                .m
                .component_div(&self.v.map(|v| v.sqrt()).add_scalar(self.eps))
    }
}