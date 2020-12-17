/// macro used internally to convert from PyAny to ndarray::Array<f64>
#[macro_export]
macro_rules! pyany_converter {
    ($obj:expr, $T: tt) => {
        if let Ok(value) = $obj.extract::<&$T<f64>>() {
            value.to_owned_array()
        } else if let Ok(value) = $obj.extract::<&$T<i64>>() {
            value.cast::<f64>(false).unwrap().to_owned_array()
        } else {
            todo!("This should return an Error")
        }
    };
}
