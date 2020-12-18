use crate::pyany_converter;
use ndarray::Array2;
use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::*;
use smartcore::naive_bayes::{GaussianNB, GaussianNBParameters};

#[pyclass(module = "smartcore.naive_bayes", name=GaussianNB)]
pub struct PyGaussianNB {
    inner: Option<GaussianNB<f64, Array2<f64>>>,
    parameters: GaussianNBParameters<f64>,
}

impl From<GaussianNB<f64, Array2<f64>>> for PyGaussianNB {
    fn from(inner: GaussianNB<f64, Array2<f64>>) -> Self {
        Self {
            inner: Some(inner),
            parameters: Default::default(),
        }
    }
}

impl From<PyGaussianNB> for GaussianNB<f64, Array2<f64>> {
    fn from(python_class: PyGaussianNB) -> Self {
        python_class.inner.unwrap()
    }
}

#[pymethods]
impl PyGaussianNB {
    #[new]
    #[args(kwargs = "**")]
    pub fn __new__(kwargs: Option<&PyDict>) -> PyResult<Self> {
        let inner = None;
        let mut priors = None;
        let parameters = if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "priors" => priors = Some(value.extract()?),
                    _ => println!("Ignored unknown kwarg: {}", key),
                }
            }
            GaussianNBParameters::new(priors)
        } else {
            Default::default()
        };

        Ok(Self { inner, parameters })
    }

    pub fn fit(&mut self, x: &PyAny, y: &PyAny) -> PyResult<()> {
        let x = pyany_converter!(x, PyArray2);
        let y = pyany_converter!(y, PyArray1);

        let gnb = GaussianNB::fit(&x, &y, self.parameters.clone()).unwrap();
        self.inner = Some(gnb);
        Ok(())
    }

    pub fn predict(&self, x: &PyAny) -> PyResult<Py<PyArray1<f64>>> {
        let x = pyany_converter!(x, PyArray2);

        let array = self.inner.as_ref().unwrap().predict(&x).unwrap();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(array.into_pyarray(py).to_owned())
    }

    pub fn to_json(&self) -> PyResult<String> {
        let gnb = self.inner.as_ref().unwrap();
        Ok(serde_json::to_string(&gnb).unwrap())
    }
}
