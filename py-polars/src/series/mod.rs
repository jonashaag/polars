use crate::apply::series::{call_lambda_and_extract, ApplyLambda};
use crate::arrow_interop::to_rust::array_to_rust;
use crate::conversion::{slice_extract_wrapped, vec_extract_wrapped, Wrap};
use crate::dataframe::PyDataFrame;
use crate::error::PyPolarsErr;
use crate::prelude::*;
use crate::prelude::{ObjectValue, *};
use crate::py_modules::POLARS;
use crate::{apply_method_all_arrow_series2};
use crate::{arrow_interop, raise_err};
use ndarray::IntoDimension;
use numpy::npyffi::types::npy_intp;
use numpy::npyffi::{self, flags};
use numpy::{Element, PyArray1, ToNpyDims, PY_ARRAY_API};
use polars::export::arrow::array::Array;
use polars::prelude::*;
use polars_algo::hist;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::CustomIterTools;
use polars_core::utils::arrow::types::NativeType;
use polars_core::utils::flatten::flatten_series;
use polars_core::with_match_physical_numeric_polars_type;
use pyo3::Python;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::types::PyList;
use pyo3::types::PyTuple;
use std::{mem, ptr};


macro_rules! impl_arithmetic {
    ($name:ident, $type:ty, $operand:tt) => {
        //#[pymethods]
        impl PySeries {
            fn $name(&self, other: $type) -> PyResult<Self> {
                Ok((&self.series $operand other).into())
            }
        }
    };
}

// impl_arithmetic!(add_u8, u8, +);
// impl_arithmetic!(add_u16, u16, +);
// impl_arithmetic!(add_u32, u32, +);
// impl_arithmetic!(add_u64, u64, +);
// impl_arithmetic!(add_i8, i8, +);
// impl_arithmetic!(add_i16, i16, +);
// impl_arithmetic!(add_i32, i32, +);
// impl_arithmetic!(add_i64, i64, +);
// impl_arithmetic!(add_datetime, i64, +);
// impl_arithmetic!(add_duration, i64, +);
// impl_arithmetic!(add_f32, f32, +);
// impl_arithmetic!(add_f64, f64, +);
// impl_arithmetic!(sub_u8, u8, -);
// impl_arithmetic!(sub_u16, u16, -);
// impl_arithmetic!(sub_u32, u32, -);
// impl_arithmetic!(sub_u64, u64, -);
// impl_arithmetic!(sub_i8, i8, -);
// impl_arithmetic!(sub_i16, i16, -);
// impl_arithmetic!(sub_i32, i32, -);
// impl_arithmetic!(sub_i64, i64, -);
// impl_arithmetic!(sub_datetime, i64, -);
// impl_arithmetic!(sub_duration, i64, -);
// impl_arithmetic!(sub_f32, f32, -);
// impl_arithmetic!(sub_f64, f64, -);
// impl_arithmetic!(div_u8, u8, /);
// impl_arithmetic!(div_u16, u16, /);
// impl_arithmetic!(div_u32, u32, /);
// impl_arithmetic!(div_u64, u64, /);
// impl_arithmetic!(div_i8, i8, /);
// impl_arithmetic!(div_i16, i16, /);
// impl_arithmetic!(div_i32, i32, /);
// impl_arithmetic!(div_i64, i64, /);
// impl_arithmetic!(div_f32, f32, /);
// impl_arithmetic!(div_f64, f64, /);
// impl_arithmetic!(mul_u8, u8, *);
// impl_arithmetic!(mul_u16, u16, *);
// impl_arithmetic!(mul_u32, u32, *);
// impl_arithmetic!(mul_u64, u64, *);
// impl_arithmetic!(mul_i8, i8, *);
// impl_arithmetic!(mul_i16, i16, *);
// impl_arithmetic!(mul_i32, i32, *);
// impl_arithmetic!(mul_i64, i64, *);
// impl_arithmetic!(mul_f32, f32, *);
// impl_arithmetic!(mul_f64, f64, *);
// impl_arithmetic!(rem_u8, u8, %);
// impl_arithmetic!(rem_u16, u16, %);
// impl_arithmetic!(rem_u32, u32, %);
// impl_arithmetic!(rem_u64, u64, %);
// impl_arithmetic!(rem_i8, i8, %);
// impl_arithmetic!(rem_i16, i16, %);
// impl_arithmetic!(rem_i32, i32, %);
// impl_arithmetic!(rem_i64, i64, %);
// impl_arithmetic!(rem_f32, f32, %);
// impl_arithmetic!(rem_f64, f64, %);

macro_rules! impl_rhs_arithmetic {
    ($name:ident, $type:ty, $operand:ident) => {
        //#[pymethods]
        impl PySeries {
            fn $name(&self, other: $type) -> PyResult<Self> {
                Ok(other.$operand(&self.series).into())
            }
        }
    };
}

// impl_rhs_arithmetic!(add_u8_rhs, u8, add);
// impl_rhs_arithmetic!(add_u16_rhs, u16, add);
// impl_rhs_arithmetic!(add_u32_rhs, u32, add);
// impl_rhs_arithmetic!(add_u64_rhs, u64, add);
// impl_rhs_arithmetic!(add_i8_rhs, i8, add);
// impl_rhs_arithmetic!(add_i16_rhs, i16, add);
// impl_rhs_arithmetic!(add_i32_rhs, i32, add);
// impl_rhs_arithmetic!(add_i64_rhs, i64, add);
// impl_rhs_arithmetic!(add_f32_rhs, f32, add);
// impl_rhs_arithmetic!(add_f64_rhs, f64, add);
// impl_rhs_arithmetic!(sub_u8_rhs, u8, sub);
// impl_rhs_arithmetic!(sub_u16_rhs, u16, sub);
// impl_rhs_arithmetic!(sub_u32_rhs, u32, sub);
// impl_rhs_arithmetic!(sub_u64_rhs, u64, sub);
// impl_rhs_arithmetic!(sub_i8_rhs, i8, sub);
// impl_rhs_arithmetic!(sub_i16_rhs, i16, sub);
// impl_rhs_arithmetic!(sub_i32_rhs, i32, sub);
// impl_rhs_arithmetic!(sub_i64_rhs, i64, sub);
// impl_rhs_arithmetic!(sub_f32_rhs, f32, sub);
// impl_rhs_arithmetic!(sub_f64_rhs, f64, sub);
// impl_rhs_arithmetic!(div_u8_rhs, u8, div);
// impl_rhs_arithmetic!(div_u16_rhs, u16, div);
// impl_rhs_arithmetic!(div_u32_rhs, u32, div);
// impl_rhs_arithmetic!(div_u64_rhs, u64, div);
// impl_rhs_arithmetic!(div_i8_rhs, i8, div);
// impl_rhs_arithmetic!(div_i16_rhs, i16, div);
// impl_rhs_arithmetic!(div_i32_rhs, i32, div);
// impl_rhs_arithmetic!(div_i64_rhs, i64, div);
// impl_rhs_arithmetic!(div_f32_rhs, f32, div);
// impl_rhs_arithmetic!(div_f64_rhs, f64, div);
// impl_rhs_arithmetic!(mul_u8_rhs, u8, mul);
// impl_rhs_arithmetic!(mul_u16_rhs, u16, mul);
// impl_rhs_arithmetic!(mul_u32_rhs, u32, mul);
// impl_rhs_arithmetic!(mul_u64_rhs, u64, mul);
// impl_rhs_arithmetic!(mul_i8_rhs, i8, mul);
// impl_rhs_arithmetic!(mul_i16_rhs, i16, mul);
// impl_rhs_arithmetic!(mul_i32_rhs, i32, mul);
// impl_rhs_arithmetic!(mul_i64_rhs, i64, mul);
// impl_rhs_arithmetic!(mul_f32_rhs, f32, mul);
// impl_rhs_arithmetic!(mul_f64_rhs, f64, mul);
// impl_rhs_arithmetic!(rem_u8_rhs, u8, rem);
// impl_rhs_arithmetic!(rem_u16_rhs, u16, rem);
// impl_rhs_arithmetic!(rem_u32_rhs, u32, rem);
// impl_rhs_arithmetic!(rem_u64_rhs, u64, rem);
// impl_rhs_arithmetic!(rem_i8_rhs, i8, rem);
// impl_rhs_arithmetic!(rem_i16_rhs, i16, rem);
// impl_rhs_arithmetic!(rem_i32_rhs, i32, rem);
// impl_rhs_arithmetic!(rem_i64_rhs, i64, rem);
// impl_rhs_arithmetic!(rem_f32_rhs, f32, rem);
// impl_rhs_arithmetic!(rem_f64_rhs, f64, rem);


macro_rules! impl_eq_num {
    ($name:ident, $type:ty) => {
        //#[pymethods]
        impl PySeries {
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.equal(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
            }
        }
    };
}

// impl_eq_num!(eq_u8, u8);
// impl_eq_num!(eq_u16, u16);
// impl_eq_num!(eq_u32, u32);
// impl_eq_num!(eq_u64, u64);
// impl_eq_num!(eq_i8, i8);
// impl_eq_num!(eq_i16, i16);
// impl_eq_num!(eq_i32, i32);
// impl_eq_num!(eq_i64, i64);
// impl_eq_num!(eq_f32, f32);
// impl_eq_num!(eq_f64, f64);
// impl_eq_num!(eq_str, &str);

macro_rules! impl_neq_num {
    ($name:ident, $type:ty) => {
        #[allow(clippy::nonstandard_macro_braces)]
        //#[pymethods]
        impl PySeries {
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.not_equal(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
            }
        }
    };
}

// impl_neq_num!(neq_u8, u8);
// impl_neq_num!(neq_u16, u16);
// impl_neq_num!(neq_u32, u32);
// impl_neq_num!(neq_u64, u64);
// impl_neq_num!(neq_i8, i8);
// impl_neq_num!(neq_i16, i16);
// impl_neq_num!(neq_i32, i32);
// impl_neq_num!(neq_i64, i64);
// impl_neq_num!(neq_f32, f32);
// impl_neq_num!(neq_f64, f64);
// impl_neq_num!(neq_str, &str);

macro_rules! impl_gt_num {
    ($name:ident, $type:ty) => {
        //#[pymethods]
        impl PySeries {
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.gt(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
            }
        }
    };
}

impl_gt_num!(gt_u8, u8);
impl_gt_num!(gt_u16, u16);
impl_gt_num!(gt_u32, u32);
impl_gt_num!(gt_u64, u64);
impl_gt_num!(gt_i8, i8);
impl_gt_num!(gt_i16, i16);
impl_gt_num!(gt_i32, i32);
impl_gt_num!(gt_i64, i64);
impl_gt_num!(gt_f32, f32);
impl_gt_num!(gt_f64, f64);
impl_gt_num!(gt_str, &str);

macro_rules! impl_gt_eq_num {
    ($name:ident, $type:ty) => {
        //#[pymethods]
        impl PySeries {
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.gt_eq(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
            }
        }
    };
}

impl_gt_eq_num!(gt_eq_u8, u8);
impl_gt_eq_num!(gt_eq_u16, u16);
impl_gt_eq_num!(gt_eq_u32, u32);
impl_gt_eq_num!(gt_eq_u64, u64);
impl_gt_eq_num!(gt_eq_i8, i8);
impl_gt_eq_num!(gt_eq_i16, i16);
impl_gt_eq_num!(gt_eq_i32, i32);
impl_gt_eq_num!(gt_eq_i64, i64);
impl_gt_eq_num!(gt_eq_f32, f32);
impl_gt_eq_num!(gt_eq_f64, f64);
impl_gt_eq_num!(gt_eq_str, &str);

macro_rules! impl_lt_num {
    ($name:ident, $type:ty) => {
        #[allow(clippy::nonstandard_macro_braces)]
        //#[pymethods]
        impl PySeries {
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.lt(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
            }
        }
    };
}

impl_lt_num!(lt_u8, u8);
impl_lt_num!(lt_u16, u16);
impl_lt_num!(lt_u32, u32);
impl_lt_num!(lt_u64, u64);
impl_lt_num!(lt_i8, i8);
impl_lt_num!(lt_i16, i16);
impl_lt_num!(lt_i32, i32);
impl_lt_num!(lt_i64, i64);
impl_lt_num!(lt_f32, f32);
impl_lt_num!(lt_f64, f64);
impl_lt_num!(lt_str, &str);

macro_rules! impl_lt_eq_num {
    ($name:ident, $type:ty) => {
        //#[pymethods]
        impl PySeries {
            fn $name(&self, rhs: $type) -> PyResult<Self> {
                let s = self.series.lt_eq(rhs).map_err(PyPolarsErr::from)?;
                Ok(s.into_series().into())
            }
        }
    };
}

impl_lt_eq_num!(lt_eq_u8, u8);
impl_lt_eq_num!(lt_eq_u16, u16);
impl_lt_eq_num!(lt_eq_u32, u32);
impl_lt_eq_num!(lt_eq_u64, u64);
impl_lt_eq_num!(lt_eq_i8, i8);
impl_lt_eq_num!(lt_eq_i16, i16);
impl_lt_eq_num!(lt_eq_i32, i32);
impl_lt_eq_num!(lt_eq_i64, i64);
impl_lt_eq_num!(lt_eq_f32, f32);
impl_lt_eq_num!(lt_eq_f64, f64);
impl_lt_eq_num!(lt_eq_str, &str);


// Init with numpy arrays
macro_rules! init_method {
    ($name:ident, $type:ty) => {
        //#[pymethods]
        impl PySeries {
            //#[staticmethod]
            fn $name(py: Python, name: &str, array: &PyArray1<$type>, _strict: bool) -> PySeries {
                let array = array.readonly();
                let vals = array.as_slice().unwrap();
                py.allow_threads(|| PySeries {
                    series: Series::new(name, vals),
                })
            }
        }
    };
}

init_method!(new_i8, i8);
init_method!(new_i16, i16);
init_method!(new_i32, i32);
init_method!(new_i64, i64);
init_method!(new_bool, bool);
init_method!(new_u8, u8);
init_method!(new_u16, u16);
init_method!(new_u32, u32);
init_method!(new_u64, u64);


fn new_primitive<'a, T>(name: &str, obj: &'a PyAny, strict: bool) -> PyResult<PySeries>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
    T::Native: FromPyObject<'a>,
{
    let len = obj.len()?;
    let mut builder = PrimitiveChunkedBuilder::<T>::new(name, len);

    for res in obj.iter()? {
        let item = res?;

        if item.is_none() {
            builder.append_null()
        } else {
            match item.extract::<T::Native>() {
                Ok(val) => builder.append_value(val),
                Err(e) => {
                    if strict {
                        return Err(e);
                    }
                    builder.append_null()
                }
            }
        }
    }
    let ca = builder.finish();

    let s = ca.into_series();
    Ok(PySeries { series: s })
}

// Init with lists that can contain Nones
macro_rules! init_method_opt {
    ($name:ident, $type:ty, $native: ty) => {
        //#[pymethods]
        impl PySeries {
            //#[staticmethod]
            fn $name(name: &str, obj: &PyAny, strict: bool) -> PyResult<PySeries> {
                new_primitive::<$type>(name, obj, strict)
            }
        }
    };
}

init_method_opt!(new_opt_u8, UInt8Type, u8);
init_method_opt!(new_opt_u16, UInt16Type, u16);
init_method_opt!(new_opt_u32, UInt32Type, u32);
init_method_opt!(new_opt_u64, UInt64Type, u64);
init_method_opt!(new_opt_i8, Int8Type, i8);
init_method_opt!(new_opt_i16, Int16Type, i16);
init_method_opt!(new_opt_i32, Int32Type, i32);
init_method_opt!(new_opt_i64, Int64Type, i64);
init_method_opt!(new_opt_f32, Float32Type, f32);
init_method_opt!(new_opt_f64, Float64Type, f64);


#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySeries {
    pub series: Series,
}

impl From<Series> for PySeries {
    fn from(series: Series) -> Self {
        PySeries { series }
    }
}

impl PySeries {
    pub(crate) fn new(series: Series) -> Self {
        PySeries { series }
    }
}

pub(crate) trait ToSeries {
    fn to_series(self) -> Vec<Series>;
}

impl ToSeries for Vec<PySeries> {
    fn to_series(self) -> Vec<Series> {
        // Safety
        // repr is transparent
        unsafe { std::mem::transmute(self) }
    }
}

pub(crate) trait ToPySeries {
    fn to_pyseries(self) -> Vec<PySeries>;
}

impl ToPySeries for Vec<Series> {
    fn to_pyseries(self) -> Vec<PySeries> {
        // Safety
        // repr is transparent
        unsafe { std::mem::transmute(self) }
    }
}


fn get_ptr<T: PolarsNumericType>(ca: &ChunkedArray<T>) -> usize {
    let arr = ca.downcast_iter().next().unwrap();
    arr.values().as_ptr() as usize
}

macro_rules! impl_set_with_mask {
    ($name:ident, $native:ty, $cast:ident, $variant:ident) => {
        fn $name(
            series: &Series,
            filter: &PySeries,
            value: Option<$native>,
        ) -> PolarsResult<Series> {
            let mask = filter.series.bool()?;
            let ca = series.$cast()?;
            let new = ca.set(mask, value)?;
            Ok(new.into_series())
        }

        //#[pymethods]
        impl PySeries {
            fn $name(&self, filter: &PySeries, value: Option<$native>) -> PyResult<Self> {
                let series = $name(&self.series, filter, value).map_err(PyPolarsErr::from)?;
                Ok(Self::new(series))
            }
        }
    };
}

impl_set_with_mask!(set_with_mask_str, &str, utf8, Utf8);
impl_set_with_mask!(set_with_mask_f64, f64, f64, Float64);
impl_set_with_mask!(set_with_mask_f32, f32, f32, Float32);
impl_set_with_mask!(set_with_mask_u8, u8, u8, UInt8);
impl_set_with_mask!(set_with_mask_u16, u16, u16, UInt16);
impl_set_with_mask!(set_with_mask_u32, u32, u32, UInt32);
impl_set_with_mask!(set_with_mask_u64, u64, u64, UInt64);
impl_set_with_mask!(set_with_mask_i8, i8, i8, Int8);
impl_set_with_mask!(set_with_mask_i16, i16, i16, Int16);
impl_set_with_mask!(set_with_mask_i32, i32, i32, Int32);
impl_set_with_mask!(set_with_mask_i64, i64, i64, Int64);
impl_set_with_mask!(set_with_mask_bool, bool, bool, Boolean);

macro_rules! impl_get {
    ($name:ident, $series_variant:ident, $type:ty) => {
        //#[pymethods]
        impl PySeries {
            fn $name(&self, index: i64) -> Option<$type> {
                if let Ok(ca) = self.series.$series_variant() {
                    let index = if index < 0 {
                        (ca.len() as i64 + index) as usize
                    } else {
                        index as usize
                    };
                    ca.get(index)
                } else {
                    None
                }
            }
        }
    };
}

impl_get!(get_f32, f32, f32);
impl_get!(get_f64, f64, f64);
impl_get!(get_u8, u8, u8);
impl_get!(get_u16, u16, u16);
impl_get!(get_u32, u32, u32);
impl_get!(get_u64, u64, u64);
impl_get!(get_i8, i8, i8);
impl_get!(get_i16, i16, i16);
impl_get!(get_i32, i32, i32);
impl_get!(get_i64, i64, i64);
impl_get!(get_str, utf8, &str);
impl_get!(get_date, date, i32);
impl_get!(get_datetime, datetime, i64);
impl_get!(get_duration, duration, i64);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn transmute_to_series() {
        // NOTE: This is only possible because PySeries is #[repr(transparent)]
        // https://doc.rust-lang.org/reference/type-layout.html
        let ps = PySeries {
            series: [1i32, 2, 3].iter().collect(),
        };

        let s = unsafe { std::mem::transmute::<PySeries, Series>(ps.clone()) };

        assert_eq!(s.sum::<i32>(), Some(6));
        let collection = vec![ps];
        let s = collection.to_series();
        assert_eq!(
            s.iter().map(|s| s.sum::<i32>()).collect::<Vec<_>>(),
            vec![Some(6)]
        );
    }
}


/// Create an empty numpy array arrows 64 byte alignment
///
/// # Safety
/// All elements in the array are non initialized
///
/// The array is also writable from Python.
unsafe fn aligned_array<T: Element + NativeType>(
    py: Python<'_>,
    size: usize,
) -> (&PyArray1<T>, Vec<T>) {
    let mut buf = vec![T::default(); size];

    // modified from
    // numpy-0.10.0/src/array.rs:375

    let len = buf.len();
    let buffer_ptr = buf.as_mut_ptr();

    let mut dims = [len].into_dimension();
    let strides = [mem::size_of::<T>() as npy_intp];

    let ptr = PY_ARRAY_API.PyArray_NewFromDescr(
        py,
        PY_ARRAY_API.get_type_object(py, npyffi::NpyTypes::PyArray_Type),
        T::get_dtype(py).into_dtype_ptr(),
        dims.ndim_cint(),
        dims.as_dims_ptr(),
        strides.as_ptr() as *mut _, // strides
        buffer_ptr as _,            // data
        flags::NPY_ARRAY_OUT_ARRAY, // flag
        ptr::null_mut(),            //obj
    );
    (PyArray1::from_owned_ptr(py, ptr), buf)
}

macro_rules! impl_ufuncs {
    ($name:ident, $type:ident, $unsafe_from_ptr_method:ident) => {
        //#[pymethods]
        impl PySeries {
            // applies a ufunc by accepting a lambda out: ufunc(*args, out=out)
            // the out array is allocated in this method, send to Python and once the ufunc is applied
            // ownership is taken by Rust again to prevent memory leak.
            // if the ufunc fails, we first must take ownership back.
            fn $name(&self, lambda: &PyAny) -> PyResult<PySeries> {
                // numpy array object, and a *mut ptr
                Python::with_gil(|py| {
                    let size = self.len();
                    let (out_array, av) =
                        unsafe { aligned_array::<<$type as PolarsNumericType>::Native>(py, size) };

                    // inserting it in a tuple increase the reference count by 1.
                    let args = PyTuple::new(py, &[out_array]);

                    // whatever the result, we must take the leaked memory ownership back
                    let s = match lambda.call1(args) {
                        Ok(_) => {
                            // if this assert fails, the lambda has taken a reference to the object, so we must panic
                            // args and the lambda return have a reference, making a total of 3

                            let validity = self.series.chunks()[0].validity().cloned();
                            let ca = ChunkedArray::<$type>::new_from_owned_with_null_bitmap(
                                self.name(),
                                av,
                                validity,
                            );
                            PySeries::new(ca.into_series())
                        }
                        Err(e) => {
                            // return error information
                            return Err(e);
                        }
                    };

                    Ok(s)
                })
            }
        }
    };
}

impl_ufuncs!(apply_ufunc_f32, Float32Type, unsafe_from_ptr_f32);
impl_ufuncs!(apply_ufunc_f64, Float64Type, unsafe_from_ptr_f64);
impl_ufuncs!(apply_ufunc_u8, UInt8Type, unsafe_from_ptr_u8);
impl_ufuncs!(apply_ufunc_u16, UInt16Type, unsafe_from_ptr_u16);
impl_ufuncs!(apply_ufunc_u32, UInt32Type, unsafe_from_ptr_u32);
impl_ufuncs!(apply_ufunc_u64, UInt64Type, unsafe_from_ptr_u64);
impl_ufuncs!(apply_ufunc_i8, Int8Type, unsafe_from_ptr_i8);
impl_ufuncs!(apply_ufunc_i16, Int16Type, unsafe_from_ptr_i16);
impl_ufuncs!(apply_ufunc_i32, Int32Type, unsafe_from_ptr_i32);
impl_ufuncs!(apply_ufunc_i64, Int64Type, unsafe_from_ptr_i64);


fn set_at_idx(mut s: Series, idx: &Series, values: &Series) -> PolarsResult<Series> {
    let logical_dtype = s.dtype().clone();
    let idx = idx.cast(&IDX_DTYPE)?;
    let idx = idx.rechunk();
    let idx = idx.idx().unwrap();
    let idx = idx.downcast_iter().next().unwrap();

    if idx.null_count() > 0 {
        return Err(PolarsError::ComputeError(
            "index values should not be null".into(),
        ));
    }

    let idx = idx.values().as_slice();

    let values = values.to_physical_repr().cast(&s.dtype().to_physical())?;

    // do not shadow, otherwise s is not dropped immediately
    // and we want to have mutable access
    s = s.to_physical_repr().into_owned();
    let mutable_s = s._get_inner_mut();

    let s = match logical_dtype.to_physical() {
        DataType::Int8 => {
            let ca: &mut ChunkedArray<Int8Type> = mutable_s.as_mut();
            let values = values.i8()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::Int16 => {
            let ca: &mut ChunkedArray<Int16Type> = mutable_s.as_mut();
            let values = values.i16()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::Int32 => {
            let ca: &mut ChunkedArray<Int32Type> = mutable_s.as_mut();
            let values = values.i32()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::Int64 => {
            let ca: &mut ChunkedArray<Int64Type> = mutable_s.as_mut();
            let values = values.i64()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::UInt8 => {
            let ca: &mut ChunkedArray<UInt8Type> = mutable_s.as_mut();
            let values = values.u8()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::UInt16 => {
            let ca: &mut ChunkedArray<UInt16Type> = mutable_s.as_mut();
            let values = values.u16()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::UInt32 => {
            let ca: &mut ChunkedArray<UInt32Type> = mutable_s.as_mut();
            let values = values.u32()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::UInt64 => {
            let ca: &mut ChunkedArray<UInt64Type> = mutable_s.as_mut();
            let values = values.u64()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::Float32 => {
            let ca: &mut ChunkedArray<Float32Type> = mutable_s.as_mut();
            let values = values.f32()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::Float64 => {
            let ca: &mut ChunkedArray<Float64Type> = mutable_s.as_mut();
            let values = values.f64()?;
            std::mem::take(ca).set_at_idx2(idx, values.into_iter())
        }
        DataType::Boolean => {
            let ca = s.bool()?;
            let values = values.bool()?;
            ca.set_at_idx2(idx, values)
        }
        DataType::Utf8 => {
            let ca = s.utf8()?;
            let values = values.utf8()?;
            ca.set_at_idx2(idx, values)
        }
        _ => panic!("not yet implemented for dtype: {logical_dtype}"),
    };

    s.and_then(|s| s.cast(&logical_dtype))
}

#[pymethods]
impl PySeries {
    fn arg_max(&self) -> Option<usize> {
        self.series.arg_max()
    }

    fn arg_min(&self) -> Option<usize> {
        self.series.arg_min()
    }

    fn max(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .max_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn mean(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.mean()
            }
            _ => self.series.mean(),
        }
    }

    fn median(&self) -> Option<f64> {
        match self.series.dtype() {
            DataType::Boolean => {
                let s = self.series.cast(&DataType::UInt8).unwrap();
                s.median()
            }
            _ => self.series.median(),
        }
    }

    fn min(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .min_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn quantile(&self, quantile: f64, interpolation: Wrap<QuantileInterpolOptions>) -> PyObject {
        Python::with_gil(|py| {
            Wrap(
                self.series
                    .quantile_as_series(quantile, interpolation.0)
                    .expect("invalid quantile")
                    .get(0)
                    .unwrap_or(AnyValue::Null),
            )
            .into_py(py)
        })
    }

    fn sum(&self, py: Python) -> PyResult<PyObject> {
        Ok(Wrap(
            self.series
                .sum_as_series()
                .get(0)
                .map_err(PyPolarsErr::from)?,
        )
        .into_py(py))
    }

    fn add(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .try_add(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
    fn sub(&self, other: &PySeries) -> Self {
        (&self.series - &other.series).into()
    }
    fn div(&self, other: &PySeries) -> Self {
        (&self.series / &other.series).into()
    }
    fn mul(&self, other: &PySeries) -> Self {
        (&self.series * &other.series).into()
    }
    fn rem(&self, other: &PySeries) -> Self {
        (&self.series % &other.series).into()
    }

    fn eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.equal(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn neq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self
            .series
            .not_equal(&rhs.series)
            .map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn gt(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.gt(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn gt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.gt_eq(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn lt(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.lt(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    fn lt_eq(&self, rhs: &PySeries) -> PyResult<Self> {
        let s = self.series.lt_eq(&rhs.series).map_err(PyPolarsErr::from)?;
        Ok(s.into_series().into())
    }

    #[staticmethod]
    fn new_f32(py: Python, name: &str, array: &PyArray1<f32>, nan_is_null: bool) -> PySeries {
        let array = array.readonly();
        let vals = array.as_slice().unwrap();
        py.allow_threads(|| {
            if nan_is_null {
                let mut ca: Float32Chunked = vals
                    .iter()
                    .map(|&val| if f32::is_nan(val) { None } else { Some(val) })
                    .collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            } else {
                Series::new(name, vals).into()
            }
        })
    }

    #[staticmethod]
    fn new_f64(py: Python, name: &str, array: &PyArray1<f64>, nan_is_null: bool) -> PySeries {
        let array = array.readonly();
        let vals = array.as_slice().unwrap();
        py.allow_threads(|| {
            if nan_is_null {
                let mut ca: Float64Chunked = vals
                    .iter()
                    .map(|&val| if f64::is_nan(val) { None } else { Some(val) })
                    .collect_trusted();
                ca.rename(name);
                ca.into_series().into()
            } else {
                Series::new(name, vals).into()
            }
        })
    }

    #[staticmethod]
    fn new_opt_bool(name: &str, obj: &PyAny, strict: bool) -> PyResult<PySeries> {
        let len = obj.len()?;
        let mut builder = BooleanChunkedBuilder::new(name, len);

        for res in obj.iter()? {
            let item = res?;
            if item.is_none() {
                builder.append_null()
            } else {
                match item.extract::<bool>() {
                    Ok(val) => builder.append_value(val),
                    Err(e) => {
                        if strict {
                            return Err(e);
                        }
                        builder.append_null()
                    }
                }
            }
        }
        let ca = builder.finish();

        let s = ca.into_series();
        Ok(PySeries { series: s })
    }

    #[staticmethod]
    fn new_from_anyvalues(
        name: &str,
        val: Vec<Wrap<AnyValue<'_>>>,
        strict: bool,
    ) -> PyResult<PySeries> {
        let avs = slice_extract_wrapped(&val);
        // from anyvalues is fallible
        let s = Series::from_any_values(name, avs, strict).map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[staticmethod]
    fn new_str(name: &str, val: Wrap<Utf8Chunked>, _strict: bool) -> Self {
        let mut s = val.0.into_series();
        s.rename(name);
        s.into()
    }

    #[staticmethod]
    fn new_binary(name: &str, val: Wrap<BinaryChunked>, _strict: bool) -> Self {
        let mut s = val.0.into_series();
        s.rename(name);
        s.into()
    }

    #[staticmethod]
    fn new_null(name: &str, val: &PyAny, _strict: bool) -> PyResult<Self> {
        let s = Series::new_null(name, val.len()?);
        Ok(s.into())
    }

    #[staticmethod]
    pub fn new_object(name: &str, val: Vec<ObjectValue>, _strict: bool) -> Self {
        #[cfg(feature = "object")]
        {
            // object builder must be registered. this is done on import
            let s = ObjectChunked::<ObjectValue>::new_from_vec(name, val).into_series();
            s.into()
        }
        #[cfg(not(feature = "object"))]
        {
            todo!()
        }
    }

    #[staticmethod]
    fn new_series_list(name: &str, val: Vec<PySeries>, _strict: bool) -> Self {
        let series_vec = val.to_series();
        Series::new(name, &series_vec).into()
    }

    #[staticmethod]
    #[pyo3(signature = (width, inner, name, val, _strict))]
    fn new_array(
        width: usize,
        inner: Option<Wrap<DataType>>,
        name: &str,
        val: Vec<Wrap<AnyValue>>,
        _strict: bool,
    ) -> PyResult<Self> {
        let val = vec_extract_wrapped(val);
        let out = Series::new(name, &val);
        match out.dtype() {
            DataType::List(list_inner) => {
                let out = out
                    .cast(&DataType::Array(
                        Box::new(inner.map(|dt| dt.0).unwrap_or(*list_inner.clone())),
                        width,
                    ))
                    .map_err(PyPolarsErr::from)?;
                Ok(out.into())
            }
            _ => Err(PyValueError::new_err("could not create Array from input")),
        }
    }

    #[staticmethod]
    fn new_decimal(name: &str, val: Vec<Wrap<AnyValue<'_>>>, strict: bool) -> PyResult<PySeries> {
        // TODO: do we have to respect 'strict' here? it's possible if we want to
        let avs = slice_extract_wrapped(&val);
        // create a fake dtype with a placeholder "none" scale, to be inferred later
        let dtype = DataType::Decimal(None, None);
        let s = Series::from_any_values_and_dtype(name, avs, &dtype, strict)
            .map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[staticmethod]
    fn from_arrow(name: &str, array: &PyAny) -> PyResult<Self> {
        let arr = array_to_rust(array)?;

        match arr.data_type() {
            ArrowDataType::LargeList(_) => {
                let array = arr.as_any().downcast_ref::<LargeListArray>().unwrap();

                let mut previous = 0;
                let mut fast_explode = true;
                for &o in array.offsets().as_slice()[1..].iter() {
                    if o == previous {
                        fast_explode = false;
                        break;
                    }
                    previous = o;
                }
                let mut out = unsafe { ListChunked::from_chunks(name, vec![arr]) };
                if fast_explode {
                    out.set_fast_explode()
                }
                Ok(out.into_series().into())
            }
            _ => {
                let series: Series =
                    std::convert::TryFrom::try_from((name, arr)).map_err(PyPolarsErr::from)?;
                Ok(series.into())
            }
        }
    }

    fn to_arrow(&mut self) -> PyResult<PyObject> {
        self.rechunk(true);
        Python::with_gil(|py| {
            let pyarrow = py.import("pyarrow")?;

            arrow_interop::to_py::to_py_array(self.series.to_arrow(0), py, pyarrow)
        })
    }

    /// For numeric types, this should only be called for Series with null types.
    /// Non-nullable types are handled with `view()`.
    /// This will cast to floats so that `None = np.nan`.
    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        let s = &self.series;
        match s.dtype() {
            dt if dt.is_numeric() => {
                if s.bit_repr_is_large() {
                    let s = s.cast(&DataType::Float64).unwrap();
                    let ca = s.f64().unwrap();
                    let np_arr = PyArray1::from_iter(
                        py,
                        ca.into_iter().map(|opt_v| opt_v.unwrap_or(f64::NAN)),
                    );
                    Ok(np_arr.into_py(py))
                } else {
                    let s = s.cast(&DataType::Float32).unwrap();
                    let ca = s.f32().unwrap();
                    let np_arr = PyArray1::from_iter(
                        py,
                        ca.into_iter().map(|opt_v| opt_v.unwrap_or(f32::NAN)),
                    );
                    Ok(np_arr.into_py(py))
                }
            }
            DataType::Utf8 => {
                let ca = s.utf8().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                Ok(np_arr.into_py(py))
            }
            DataType::Binary => {
                let ca = s.binary().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                Ok(np_arr.into_py(py))
            }
            DataType::Boolean => {
                let ca = s.bool().unwrap();
                let np_arr = PyArray1::from_iter(py, ca.into_iter().map(|s| s.into_py(py)));
                Ok(np_arr.into_py(py))
            }
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                let ca = s
                    .as_any()
                    .downcast_ref::<ObjectChunked<ObjectValue>>()
                    .unwrap();
                let np_arr =
                    PyArray1::from_iter(py, ca.into_iter().map(|opt_v| opt_v.to_object(py)));
                Ok(np_arr.into_py(py))
            }
            dt => {
                raise_err!(
                    format!("'to_numpy' not supported for dtype: {dt:?}"),
                    ComputeError
                );
            }
        }
    }

    pub fn to_list(&self) -> PyObject {
        Python::with_gil(|py| {
            let series = &self.series;

            fn to_list_recursive(py: Python, series: &Series) -> PyObject {
                let pylist = match series.dtype() {
                    DataType::Boolean => PyList::new(py, series.bool().unwrap()),
                    DataType::UInt8 => PyList::new(py, series.u8().unwrap()),
                    DataType::UInt16 => PyList::new(py, series.u16().unwrap()),
                    DataType::UInt32 => PyList::new(py, series.u32().unwrap()),
                    DataType::UInt64 => PyList::new(py, series.u64().unwrap()),
                    DataType::Int8 => PyList::new(py, series.i8().unwrap()),
                    DataType::Int16 => PyList::new(py, series.i16().unwrap()),
                    DataType::Int32 => PyList::new(py, series.i32().unwrap()),
                    DataType::Int64 => PyList::new(py, series.i64().unwrap()),
                    DataType::Float32 => PyList::new(py, series.f32().unwrap()),
                    DataType::Float64 => PyList::new(py, series.f64().unwrap()),
                    DataType::Categorical(_) => {
                        PyList::new(py, series.categorical().unwrap().iter_str())
                    }
                    #[cfg(feature = "object")]
                    DataType::Object(_) => {
                        let v = PyList::empty(py);
                        for i in 0..series.len() {
                            let obj: Option<&ObjectValue> =
                                series.get_object(i).map(|any| any.into());
                            let val = obj.to_object(py);

                            v.append(val).unwrap();
                        }
                        v
                    }
                    DataType::List(_) => {
                        let v = PyList::empty(py);
                        let ca = series.list().unwrap();
                        for opt_s in ca.amortized_iter() {
                            match opt_s {
                                None => {
                                    v.append(py.None()).unwrap();
                                }
                                Some(s) => {
                                    let pylst = to_list_recursive(py, s.as_ref());
                                    v.append(pylst).unwrap();
                                }
                            }
                        }
                        v
                    }
                    DataType::Array(_, _) => {
                        let v = PyList::empty(py);
                        let ca = series.array().unwrap();
                        for opt_s in ca.amortized_iter() {
                            match opt_s {
                                None => {
                                    v.append(py.None()).unwrap();
                                }
                                Some(s) => {
                                    let pylst = to_list_recursive(py, s.as_ref());
                                    v.append(pylst).unwrap();
                                }
                            }
                        }
                        v
                    }
                    DataType::Date => {
                        let ca = series.date().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Time => {
                        let ca = series.time().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Datetime(_, _) => {
                        let ca = series.datetime().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Decimal(_, _) => {
                        let ca = series.decimal().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Utf8 => {
                        let ca = series.utf8().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Struct(_) => {
                        let ca = series.struct_().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Duration(_) => {
                        let ca = series.duration().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Binary => {
                        let ca = series.binary().unwrap();
                        return Wrap(ca).to_object(py);
                    }
                    DataType::Null => {
                        let null: Option<u8> = None;
                        let n = series.len();
                        let iter = std::iter::repeat(null).take(n);
                        use std::iter::{Repeat, Take};
                        struct NullIter {
                            iter: Take<Repeat<Option<u8>>>,
                            n: usize,
                        }
                        impl Iterator for NullIter {
                            type Item = Option<u8>;

                            fn next(&mut self) -> Option<Self::Item> {
                                self.iter.next()
                            }
                            fn size_hint(&self) -> (usize, Option<usize>) {
                                (self.n, Some(self.n))
                            }
                        }
                        impl ExactSizeIterator for NullIter {}

                        PyList::new(py, NullIter { iter, n })
                    }
                    DataType::Unknown => {
                        panic!("to_list not implemented for unknown")
                    }
                };
                pylist.to_object(py)
            }

            let pylist = to_list_recursive(py, series);
            pylist.to_object(py)
        })
    }

    fn struct_unnest(&self) -> PyResult<PyDataFrame> {
        let ca = self.series.struct_().map_err(PyPolarsErr::from)?;
        let df: DataFrame = ca.clone().into();
        Ok(df.into())
    }

    fn struct_fields(&self) -> PyResult<Vec<&str>> {
        let ca = self.series.struct_().map_err(PyPolarsErr::from)?;
        Ok(ca.fields().iter().map(|s| s.name()).collect())
    }

    fn is_sorted_ascending_flag(&self) -> bool {
        matches!(self.series.is_sorted_flag(), IsSorted::Ascending)
    }

    fn is_sorted_descending_flag(&self) -> bool {
        matches!(self.series.is_sorted_flag(), IsSorted::Descending)
    }

    fn can_fast_explode_flag(&self) -> bool {
        match self.series.list() {
            Err(_) => false,
            Ok(list) => list._can_fast_explode(),
        }
    }

    fn estimated_size(&self) -> usize {
        self.series.estimated_size()
    }

    #[cfg(feature = "object")]
    fn get_object(&self, index: usize) -> PyObject {
        Python::with_gil(|py| {
            if matches!(self.series.dtype(), DataType::Object(_)) {
                let obj: Option<&ObjectValue> = self.series.get_object(index).map(|any| any.into());
                obj.to_object(py)
            } else {
                py.None()
            }
        })
    }

    fn get_fmt(&self, index: usize, str_lengths: usize) -> String {
        let val = format!("{}", self.series.get(index).unwrap());
        if let DataType::Utf8 | DataType::Categorical(_) = self.series.dtype() {
            let v_trunc = &val[..val
                .char_indices()
                .take(str_lengths)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0)];
            if val == v_trunc {
                val
            } else {
                format!("{v_trunc}…")
            }
        } else {
            val
        }
    }

    fn rechunk(&mut self, in_place: bool) -> Option<Self> {
        let series = self.series.rechunk();
        if in_place {
            self.series = series;
            None
        } else {
            Some(series.into())
        }
    }

    fn get_idx(&self, py: Python, idx: usize) -> PyResult<PyObject> {
        let av = self.series.get(idx).map_err(PyPolarsErr::from)?;
        if let AnyValue::List(s) = av {
            let pyseries = PySeries::new(s);
            let out = POLARS
                .getattr(py, "wrap_s")
                .unwrap()
                .call1(py, (pyseries,))
                .unwrap();

            Ok(out.into_py(py))
        } else {
            Ok(Wrap(self.series.get(idx).map_err(PyPolarsErr::from)?).into_py(py))
        }
    }

    fn bitand(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitand(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn bitor(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitor(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }
    fn bitxor(&self, other: &PySeries) -> PyResult<Self> {
        let out = self
            .series
            .bitxor(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn chunk_lengths(&self) -> Vec<usize> {
        self.series.chunk_lengths().collect()
    }

    fn name(&self) -> &str {
        self.series.name()
    }

    fn rename(&mut self, name: &str) {
        self.series.rename(name);
    }

    fn dtype(&self, py: Python) -> PyObject {
        Wrap(self.series.dtype().clone()).to_object(py)
    }

    fn inner_dtype(&self, py: Python) -> Option<PyObject> {
        self.series
            .dtype()
            .inner_dtype()
            .map(|dt| Wrap(dt.clone()).to_object(py))
    }

    fn set_sorted_flag(&self, descending: bool) -> Self {
        let mut out = self.series.clone();
        if descending {
            out.set_sorted_flag(IsSorted::Descending);
        } else {
            out.set_sorted_flag(IsSorted::Ascending)
        }
        out.into()
    }

    fn n_chunks(&self) -> usize {
        self.series.n_chunks()
    }

    fn append(&mut self, other: &PySeries) -> PyResult<()> {
        self.series
            .append(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    fn extend(&mut self, other: &PySeries) -> PyResult<()> {
        self.series
            .extend(&other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    fn new_from_index(&self, index: usize, length: usize) -> PyResult<Self> {
        if index >= self.series.len() {
            Err(PyValueError::new_err("index is out of bounds"))
        } else {
            Ok(self.series.new_from_index(index, length).into())
        }
    }

    fn filter(&self, filter: &PySeries) -> PyResult<Self> {
        let filter_series = &filter.series;
        if let Ok(ca) = filter_series.bool() {
            let series = self.series.filter(ca).map_err(PyPolarsErr::from)?;
            Ok(PySeries { series })
        } else {
            Err(PyRuntimeError::new_err("Expected a boolean mask"))
        }
    }

    fn sort(&mut self, descending: bool) -> Self {
        self.series.sort(descending).into()
    }

    fn value_counts(&self, sorted: bool) -> PyResult<PyDataFrame> {
        let df = self
            .series
            .value_counts(true, sorted)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.idx().map_err(PyPolarsErr::from)?;
        let take = self.series.take(idx).map_err(PyPolarsErr::from)?;
        Ok(take.into())
    }

    fn null_count(&self) -> PyResult<usize> {
        Ok(self.series.null_count())
    }

    fn has_validity(&self) -> bool {
        self.series.has_validity()
    }

    fn series_equal(&self, other: &PySeries, null_equal: bool, strict: bool) -> bool {
        if strict {
            self.series.eq(&other.series)
        } else if null_equal {
            self.series.series_equal_missing(&other.series)
        } else {
            self.series.series_equal(&other.series)
        }
    }

    fn _not(&self) -> PyResult<Self> {
        let bool = self.series.bool().map_err(PyPolarsErr::from)?;
        Ok((!bool).into_series().into())
    }

    fn as_str(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.series))
    }

    fn len(&self) -> usize {
        self.series.len()
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    fn as_single_ptr(&mut self) -> PyResult<usize> {
        let ptr = self.series.as_single_ptr().map_err(PyPolarsErr::from)?;
        Ok(ptr)
    }

    fn clone(&self) -> Self {
        self.series.clone().into()
    }

    #[pyo3(signature = (lambda, output_type, skip_nulls))]
    fn apply_lambda(
        &self,
        lambda: &PyAny,
        output_type: Option<Wrap<DataType>>,
        skip_nulls: bool,
    ) -> PyResult<PySeries> {
        let series = &self.series;

        if skip_nulls && (series.null_count() == series.len()) {
            if let Some(output_type) = output_type {
                return Ok(Series::full_null(series.name(), series.len(), &output_type.0).into());
            }
            let msg = "The output type of 'apply' function cannot determined.\n\
            The function was never called because 'skip_nulls=True' and all values are null.\n\
            Consider setting 'skip_nulls=False' or setting the 'return_dtype'.";
            raise_err!(msg, ComputeError)
        }

        let output_type = output_type.map(|dt| dt.0);

        macro_rules! dispatch_apply {
            ($self:expr, $method:ident, $($args:expr),*) => {
                match $self.dtype() {
                    #[cfg(feature = "object")]
                    DataType::Object(_) => {
                        let ca = $self.0.unpack::<ObjectType<ObjectValue>>().unwrap();
                        ca.$method($($args),*)
                    },
                    _ => {
                        apply_method_all_arrow_series2!(
                            $self,
                            $method,
                            $($args),*
                        )
                    }

                }
            }

        }

        Python::with_gil(|py| {
            if matches!(
                self.series.dtype(),
                DataType::Datetime(_, _)
                    | DataType::Date
                    | DataType::Duration(_)
                    | DataType::Categorical(_)
                    | DataType::Binary
                    | DataType::Array(_, _)
                    | DataType::Time
            ) || !skip_nulls
            {
                let mut avs = Vec::with_capacity(self.series.len());
                let iter = self.series.iter().map(|av| match (skip_nulls, av) {
                    (true, AnyValue::Null) => AnyValue::Null,
                    (_, av) => {
                        let input = Wrap(av);
                        call_lambda_and_extract::<_, Wrap<AnyValue>>(py, lambda, input)
                            .unwrap()
                            .0
                    }
                });
                avs.extend(iter);
                return Ok(Series::new(self.name(), &avs).into());
            }

            let out = match output_type {
                Some(DataType::Int8) => {
                    let ca: Int8Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::Int16) => {
                    let ca: Int16Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::Int32) => {
                    let ca: Int32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::Int64) => {
                    let ca: Int64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::UInt8) => {
                    let ca: UInt8Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::UInt16) => {
                    let ca: UInt16Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::UInt32) => {
                    let ca: UInt32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::UInt64) => {
                    let ca: UInt64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::Float32) => {
                    let ca: Float32Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::Float64) => {
                    let ca: Float64Chunked = dispatch_apply!(
                        series,
                        apply_lambda_with_primitive_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::Boolean) => {
                    let ca: BooleanChunked = dispatch_apply!(
                        series,
                        apply_lambda_with_bool_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                Some(DataType::Utf8) => {
                    let ca = dispatch_apply!(
                        series,
                        apply_lambda_with_utf8_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;

                    ca.into_series()
                }
                #[cfg(feature = "object")]
                Some(DataType::Object(_)) => {
                    let ca = dispatch_apply!(
                        series,
                        apply_lambda_with_object_out_type,
                        py,
                        lambda,
                        0,
                        None
                    )?;
                    ca.into_series()
                }
                None => return dispatch_apply!(series, apply_lambda_unknown, py, lambda),

                _ => return dispatch_apply!(series, apply_lambda_unknown, py, lambda),
            };

            Ok(out.into())
        })
    }

    fn zip_with(&self, mask: &PySeries, other: &PySeries) -> PyResult<Self> {
        let mask = mask.series.bool().map_err(PyPolarsErr::from)?;
        let s = self
            .series
            .zip_with(mask, &other.series)
            .map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    #[pyo3(signature = (separator, drop_first=false))]
    fn to_dummies(&self, separator: Option<&str>, drop_first: bool) -> PyResult<PyDataFrame> {
        let df = self
            .series
            .to_dummies(separator, drop_first)
            .map_err(PyPolarsErr::from)?;
        Ok(df.into())
    }

    fn get_list(&self, index: usize) -> Option<Self> {
        if let Ok(ca) = &self.series.list() {
            let s = ca.get(index);
            s.map(|s| s.into())
        } else {
            None
        }
    }

    fn peak_max(&self) -> Self {
        self.series.peak_max().into_series().into()
    }

    fn peak_min(&self) -> Self {
        self.series.peak_min().into_series().into()
    }

    fn n_unique(&self) -> PyResult<usize> {
        let n = self.series.n_unique().map_err(PyPolarsErr::from)?;
        Ok(n)
    }

    fn floor(&self) -> PyResult<Self> {
        let s = self.series.floor().map_err(PyPolarsErr::from)?;
        Ok(s.into())
    }

    fn shrink_to_fit(&mut self) {
        self.series.shrink_to_fit();
    }

    fn dot(&self, other: &PySeries) -> Option<f64> {
        self.series.dot(&other.series)
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        // Used in pickle/pickling
        let mut writer: Vec<u8> = vec![];
        ciborium::ser::into_writer(&self.series, &mut writer)
            .map_err(|e| PyPolarsErr::Other(format!("{}", e)))?;

        Ok(PyBytes::new(py, &writer).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        // Used in pickle/pickling
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.series = ciborium::de::from_reader(s.as_bytes())
                    .map_err(|e| PyPolarsErr::Other(format!("{}", e)))?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn skew(&self, bias: bool) -> PyResult<Option<f64>> {
        let out = self.series.skew(bias).map_err(PyPolarsErr::from)?;
        Ok(out)
    }

    fn kurtosis(&self, fisher: bool, bias: bool) -> PyResult<Option<f64>> {
        let out = self
            .series
            .kurtosis(fisher, bias)
            .map_err(PyPolarsErr::from)?;
        Ok(out)
    }

    fn cast(&self, dtype: Wrap<DataType>, strict: bool) -> PyResult<Self> {
        let dtype = dtype.0;
        let out = if strict {
            self.series.strict_cast(&dtype)
        } else {
            self.series.cast(&dtype)
        };
        let out = out.map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn time_unit(&self) -> Option<&str> {
        if let DataType::Datetime(time_unit, _) | DataType::Duration(time_unit) =
            self.series.dtype()
        {
            Some(match time_unit {
                TimeUnit::Nanoseconds => "ns",
                TimeUnit::Microseconds => "us",
                TimeUnit::Milliseconds => "ms",
            })
        } else {
            None
        }
    }

    fn get_chunks(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let wrap_s = py_modules::POLARS.getattr(py, "wrap_s").unwrap();
            flatten_series(&self.series)
                .into_iter()
                .map(|s| wrap_s.call1(py, (Self::new(s),)))
                .collect()
        })
    }

    fn is_sorted(&self, descending: bool) -> PyResult<bool> {
        let options = SortOptions {
            descending,
            nulls_last: descending,
            multithreaded: true,
            maintain_order: false,
        };
        Ok(self.series.is_sorted(options).map_err(PyPolarsErr::from)?)
    }

    fn clear(&self) -> Self {
        self.series.clear().into()
    }

    fn hist(&self, bins: Option<Self>, bin_count: Option<usize>) -> PyResult<PyDataFrame> {
        let bins = bins.map(|s| s.series);
        let out = hist(&self.series, bins.as_ref(), bin_count).map_err(PyPolarsErr::from)?;
        Ok(out.into())
    }

    fn get_ptr(&self) -> PyResult<usize> {
        let s = self.series.to_physical_repr();
        let arrays = s.chunks();
        if arrays.len() != 1 {
            let msg = "Only can take pointer, if the 'series' contains a single chunk";
            raise_err!(msg, ComputeError);
        }
        match s.dtype() {
            DataType::Boolean => {
                let ca = s.bool().unwrap();
                let arr = ca.downcast_iter().next().unwrap();
                // this one is quite useless as you need to know the offset
                // into the first byte.
                let (slice, start, _len) = arr.values().as_slice();
                if start == 0 {
                    Ok(slice.as_ptr() as usize)
                } else {
                    let msg = "Cannot take pointer boolean buffer as it is not perfectly aligned.";
                    raise_err!(msg, ComputeError);
                }
            }
            dt if dt.is_numeric() => Ok(with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                get_ptr(ca)
            })),
            _ => {
                let msg = "Cannot take pointer of nested type";
                raise_err!(msg, ComputeError);
            }
        }
    }

    fn set_at_idx(&mut self, idx: PySeries, values: PySeries) -> PyResult<()> {
        // we take the value because we want a ref count
        // of 1 so that we can have mutable access
        let s = std::mem::take(&mut self.series);
        match set_at_idx(s, &idx.series, &values.series) {
            Ok(out) => {
                self.series = out;
                Ok(())
            }
            Err(e) => Err(PyErr::from(PyPolarsErr::from(e))),
        }
    }
}
