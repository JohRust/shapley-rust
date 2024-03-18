mod predictor{

    struct LinearRegression {
        weights: Vec<f32>,
        bias: f32,
    }

    impl LinearRegression {
        fn predict(&self, input: &Vec<f32>) -> f32 {
            let mut result = 0.0;
            for i in 0..input.len() {
                result += input[i] * self.weights[i];
            }
            result += self.bias;
            result
        }
    }
}
