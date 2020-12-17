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
        let x = if let Ok(x) = x.extract::<&PyArray2<f64>>() {
            x.to_owned_array()
        } else if let Ok(x) = x.extract::<&PyArray2<i64>>() {
            x.cast::<f64>(false).unwrap().to_owned_array()
        } else {
            todo!("This should return an input error")
        };

        let y = if let Ok(y) = y.extract::<&PyArray1<f64>>() {
            y.to_owned_array()
        } else if let Ok(y) = y.extract::<&PyArray1<i64>>() {
            y.cast::<f64>(false).unwrap().to_owned_array()
        } else {
            todo!("This should return an input error")
        };

        let gnb = GaussianNB::<f64, Array2<f64>>::fit(&x, &y, self.parameters.clone()).unwrap();
        self.inner = Some(gnb);
        Ok(())
    }

    pub fn predict(&self, x: &PyAny) -> PyResult<Py<PyArray1<f64>>> {
        let x = if let Ok(x) = x.extract::<&PyArray2<f64>>() {
            x.to_owned_array()
        } else if let Ok(x) = x.extract::<&PyArray2<i32>>() {
            x.cast::<f64>(false).unwrap().to_owned_array()
        } else {
            todo!("This should return an input error")
        };

        let array = self.inner.as_ref().unwrap().predict(&x).unwrap();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(array.into_pyarray(py).to_owned())
    }
}
