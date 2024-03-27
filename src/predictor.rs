pub trait Predictor {
    fn predict(&self, input: &Vec<f32>) -> f32;
}
pub struct LinearRegression {
    weights: Vec<f32>,
    bias: f32,
}

impl Predictor for LinearRegression {
    fn predict(&self, input: &Vec<f32>) -> f32 {
        let mut result = 0.0;
        for i in 0..input.len() {
            result += input[i] * self.weights[i];
        }
        result += self.bias;
        result
    }
}

impl LinearRegression {
    pub fn new(weights: Vec<f32>, bias: f32) -> LinearRegression {
        LinearRegression { weights, bias }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_predict() {
        let predictor = LinearRegression {
            weights: vec![1.0, 2.0, 3.0],
            bias: 4.0,
        };
        let input = vec![1.0, 2.0, 3.0];
        let result = predictor.predict(&input);
        assert_eq!(result, 18.0);
    }
}
