fn replace_values(array: &Vec<f32>, mask: &Vec<bool>, new_values: &Vec<f32>) -> Vec<f32> {
    let mut result = Vec::new();
    for i in 0..array.len() {
        if mask[i] {
            result.push(array[i]);
        } else {
            result.push(new_values[i]);
        }
    }
    result
}

fn binomial_coefficient(n: u32, k: u32) -> u32 {
    if k > n - k {
        return binomial_coefficient(n, n - k);
    }
    let mut result = 1;
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }
    result
}

fn factorial(n: u32) -> u64 {
    if n > 20 {
        panic!("factorial overflow");
    }
    let mut result: u64 = 1;
    for i in 1..(n + 1) {
        result *= i as u64;
    }
    result
}

fn sample_from_data(data: &Vec<Vec<f32>>) -> Vec<f32> {
    use rand::Rng;
    let mut result = Vec::new();
    for i in 0..data[0].len() {
        let random_idx = rand::thread_rng().gen_range(0..data[0].len());
        result.push(data[random_idx][i]);
    }
    result
}

fn get_as_bool_vector(n: u32, length: usize) -> Vec<bool> {
    let mut result = Vec::new();
    for i in 0..length {
        result.push((n >> i) & 1 == 1);
    }
    result
}

fn shapley_frequency(n: u32, s: u32) -> f32 {
    let mut result = 0.0;
    if n-s <= 0{
        return 0.0;
    }
    1.0 / (binomial_coefficient(n, s) * (n-s)) as f32
}

fn get_shapley_values(input_data: Vec<f32>, model: &Predictor, background_data: &Vec<Vec<f32>>) -> Vec<f32> {
    let n = input_data.len();
    let mut shapley_values = vec![0.0; n];

    for i in 0..n {
        // There are 2^(n-1) subsets to iterate over
        let num_subsets: u32 = 1 << (n-1);
        for j in 0..num_subsets {
            let mask = get_as_bool_vector(j, n-1);
            mask.insert(i, false);
            //Count true values in mask
            let subset_size = mask.iter().filter(|&x| *x).count();
            let data_masked = input_data.clone();
            data_masked = replace_values(&data_masked, &mask, &sample_from_data(&background_data));
            let pred_without_i = model.predict(&data_masked);
            data_masked[i] = input_data[i];
            let pred_with_i = model.predict(&data_masked);
            shapley_values[i] += shapley_frequency(n as u32, subset_size as u32) * (pred_without_i - pred_with_i);
        }
    }
    shapley_values
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_replace_values() {
        let array = vec![1.0, 2.0, 3.0];
        let mask = vec![true, false, true];
        let new_values = vec![4.0, 5.0, 6.0];
        let result = replace_values(&array, &mask, &new_values);
        assert_eq!(result, vec![1.0, 5.0, 3.0]);
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 2), 10);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_sample_from_data() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = sample_from_data(&data);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_get_as_bool_vector() {
        let result = get_as_bool_vector(5, 3);
        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn test_shapley_frequency() {
        assert_eq!(shapley_frequency(5, 2), 0.1);
    }
}