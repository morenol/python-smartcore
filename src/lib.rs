use pyo3::prelude::*;
use pyo3::wrap_pymodule;

mod naive_bayes;
mod utils;

#[pymodule]
fn naive_bayes(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<naive_bayes::PyGaussianNB>()?;

    Ok(())
}

#[pymodule]
fn smartcore(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(naive_bayes))?;
    Ok(())
}
