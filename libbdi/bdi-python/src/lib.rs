use numpy::PyArray1;
use pyo3::prelude::*;

use bdi;

#[pyclass]
#[derive(Clone, Debug)]
pub struct CompressionBaseConfig {
    #[pyo3(get)]
    pub num_bases: usize,
    #[pyo3(get)]
    pub base_size: usize,
    #[pyo3(get)]
    pub delta_size: usize,
    #[pyo3(get)]
    pub name: String,
}

#[pymethods]
impl CompressionBaseConfig {
    #[new]
    fn new(num_bases: usize, base_size: usize, delta_size: usize, name: String) -> Self {
        CompressionBaseConfig {
            num_bases,
            base_size,
            delta_size,
            name,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct CompressionResult {
    #[pyo3(get)]
    pub bases: Vec<bdi::Integer>,
    #[pyo3(get)]
    pub compression_duration: u128,
    #[pyo3(get)]
    pub decompression_duration: u128,
    #[pyo3(get)]
    pub elements: Vec<(bdi::Integer, bdi::Integer)>,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub n_bases: usize,
    #[pyo3(get)]
    pub size: usize,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct CompressionConfig {
    #[pyo3(get)]
    pub base_configs: Vec<CompressionBaseConfig>,
    #[pyo3(get)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub scale_factor_base: usize,
    #[pyo3(get)]
    pub scale_factor_delta: usize,
    #[pyo3(get)]
    pub scale_factor_init: usize,
    #[pyo3(get)]
    pub use_randomized_base_selection: bool,
    #[pyo3(get)]
    pub use_stochastic_rounding: bool,
    #[pyo3(get)]
    pub use_zero_compression: bool,
}

#[pymethods]
impl CompressionConfig {
    #[new]
    fn new(
        base_configs: Vec<CompressionBaseConfig>,
        batch_size: usize,
        scale_factor_init: usize,
        scale_factor_base: usize,
        scale_factor_delta: usize,
        use_randomized_base_selection: bool,
        use_stochastic_rounding: bool,
        use_zero_compression: bool,
    ) -> Self {
        CompressionConfig {
            base_configs,
            batch_size,
            scale_factor_init,
            scale_factor_base,
            scale_factor_delta,
            use_randomized_base_selection,
            use_stochastic_rounding,
            use_zero_compression,
        }
    }
}

#[pymodule]
fn libbdi<'python>(_python: Python<'python>, python_module: &PyModule) -> PyResult<()> {
    python_module.add_class::<CompressionBaseConfig>()?;
    python_module.add_class::<CompressionConfig>()?;
    python_module.add_class::<CompressionResult>()?;

    // #[pyfn(python_module, "get_compression_stats")]
    // fn get_compression_stats<'python>(
    //     python: Python<'python>,
    //     buffer: &PyArray1<bdi::Integer>,
    // ) -> PyResult<(&'python PyArray1<bdi::Integer>, usize, Vec<&'static str>)> {
    //     let buffer = buffer.as_slice().unwrap();
    //     let (decompressed_values, compressed_size, schemes) = bdi::get_compression_stats(buffer);
    //     let decompressed_values = decompressed_values.into_pyarray(python);
    //     Ok((decompressed_values, compressed_size, schemes))
    // }

    #[pyfn(python_module, "replace_with_compressed_repr")]
    fn replace_with_compressed_repr<'python>(
        _python: Python<'python>,
        config: CompressionConfig,
        buffer: &PyArray1<bdi::Integer>,
        dest: usize,
        scale: f32,
        zero_point: usize,
    ) -> PyResult<(Vec<CompressionResult>, usize, Vec<String>, (u128, u128))> {
        let buffer = buffer.as_slice().unwrap();
        let compression = bdi::Compression {
            batch_size: config.batch_size,
            compression_configs: config
                .base_configs
                .into_iter()
                .map(|config| bdi::CompressionConfig {
                    num_bases: config.num_bases,
                    base_size: config.base_size,
                    delta_size: config.delta_size,
                    name: config.name,
                })
                .collect::<Vec<_>>(),
            immediates: vec![0],
            scale_factor_init: config.scale_factor_init,
            scale_factor_base: config.scale_factor_base,
            scale_factor_delta: config.scale_factor_delta,
            use_randomized_base_selection: config.use_randomized_base_selection,
            use_stochastic_rounding: config.use_stochastic_rounding,
            use_zero_compression: config.use_zero_compression,
        };
        let (
            data,
            decompressed_values,
            compressed_size,
            schemes,
            (compression_duration, decompression_duration),
        ) = compression.get_compression_stats(buffer, scale, zero_point);

        let data = data
            .into_iter()
            .map(|result| CompressionResult {
                bases: result.bases,
                compression_duration: result.compression_duration,
                decompression_duration: result.decompression_duration,
                elements: result
                    .elements
                    .into_iter()
                    .map(|element| match element {
                        bdi::BufferElement::Compressed { base, delta } => (base, delta),
                        _ => (0, 0),
                    })
                    .collect::<Vec<_>>(),
                name: String::from(result.name),
                n_bases: result.n_bases,
                size: result.size,
            })
            .collect::<Vec<_>>();

        unsafe {
            decompressed_values
                .as_ptr()
                .copy_to(dest as *mut f32, buffer.len())
        }
        Ok((
            data,
            compressed_size,
            schemes,
            (
                compression_duration.as_nanos(),
                decompression_duration.as_nanos(),
            ),
        ))
    }

    Ok(())
}
