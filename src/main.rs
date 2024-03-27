mod shapley;
mod predictor;

fn main() {
    let lin_reg = predictor::LinearRegression::new(vec![1.0, 2.0, 3.0], 4.0);
    let shapley_values = shapley::get_shapley_values(
        &vec![1.0, 2.0, 3.0], &lin_reg, &vec![vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]]);
    println!("Shapley Values: {:?}", shapley_values);
}
