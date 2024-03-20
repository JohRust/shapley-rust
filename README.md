# shapley
Calculate Shapley values, a game theoretic approach to fairly distribute resources among a set of contributors.

It can be used to calculate feature attribution values for any predictive AI model.
The implementation is similar to the ExactExplainer from the SHAP python package, as it calculates the exact Shapley values, but uses masking to keep the input feature size constant and samples from a given background distribution. The Shapley equation was slightly, so it is based on a binomial coefficient. This value can be calculated efficiently without risking integer overflows that occur with factorial numbers larger than 20.
main.rs shows a small example to calculate Shapley values for a Linear Regression model, shap.rs contains the Shapley value calculation and predictors.rs defines the Predictor trait and a simple Linear Regression struct that implements it.

# Building and running the project.
Its a simple crate. Use cargo run to build and run it, cargo test runs the unit tests.
