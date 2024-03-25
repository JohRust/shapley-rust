# shapley
Calculate Shapley values, a game theoretic approach to fairly distribute resources among a set of contributors.

It can be used to calculate feature attribution values for any predictive AI model.
The implementation is similar to the ExactExplainer from the SHAP python package, as it calculates the exact Shapley values, but uses masking to keep the input feature size constant and samples from a given background distribution. The Shapley equation was slightly, so it is based on a binomial coefficient. This value can be calculated efficiently without risking integer overflows that occur with factorial numbers larger than 20.
main.rs shows a small example to calculate Shapley values for a Linear Regression model, shap.rs contains the Shapley value calculation and predictors.rs defines the Predictor trait and a simple Linear Regression struct that implements it.

# Building and running the project.
Its a simple crate. Use cargo run to build and run it, cargo test runs the unit tests.

# Shapley Formula
To avoid the calculation of factorials, we reformulate the Shapley formula so it contains the binomial coefficient instead.
The binomial coefficient can be efficiently calculated using a running sum and the Shapley formula becomes quite easy to implement.

$$Sh_i(N,v) = \sum_{S \subseteq N\backslash \lbrace i \rbrace} \frac{(n-1-|S|)!\cdot|S|!}{n!} (v(S)-v(S\backslash \lbrace i \rbrace))$$

$$= \sum_{S \subseteq N\backslash \lbrace i \rbrace} \left(\frac{n!}{(n-1-|S|)!\cdot|S|!}\right)^{-1} (v(S)-v(S\backslash \lbrace i \rbrace))$$

$$= \sum_{S \subseteq N\backslash \lbrace i \rbrace} \left(\underbrace{\frac{n!}{(n-|S|)!\cdot|S|!}}_{\text{binomial coefficient}}(n-|S|)\right)^{-1} (v(S)-v(S\backslash \lbrace i \rbrace))$$

$$= \sum_{S \subseteq N\backslash \lbrace i \rbrace} \left( \binom{n}{|S|} (n-|S|)\right)^{-1} (v(S)-v(S\backslash \lbrace i \rbrace))$$

$$= \sum_{S \subseteq N\backslash \lbrace i \rbrace} \frac{1}{\binom{n}{|S|} (n-|S|)} (v(S)-v(S\backslash \lbrace i \rbrace))$$

with $n=|N|$ being the number of features, $S$ being a subset of features and $|S|$ being the number of features in the subset.

To iterate over all subsets, we simply count from 0 to $|N\backslash \lbrace i \rbrace| = n-1$ and convert the binary representation to a mask. E.g. 5 = 0b00000101 = {false, false, false, false, false, true, false, false}. S would be the 3rd last and last feature and $|S| = 2$.
