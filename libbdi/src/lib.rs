#![feature(const_fn_union, const_mut_refs, test, try_blocks)]

extern crate test;

#[macro_use]
extern crate derive_builder;

use std::mem;
use std::time;

use rand::prelude::*;
use rayon::prelude::*;

pub type Integer = u32;

pub const BYTE_SIZE: usize = 8;
pub const INT_SIZE: usize = mem::size_of::<Integer>() * BYTE_SIZE;

#[derive(Clone, Debug)]
pub struct CompressionConfig {
    pub num_bases: usize,
    pub base_size: usize,
    pub delta_size: usize,
    pub name: String,
}

#[derive(Builder, Debug)]
#[builder(setter(into))]
pub struct Compression {
    #[builder(default = "64")]
    pub batch_size: usize,
    pub compression_configs: Vec<CompressionConfig>,
    #[builder(default = "vec![0]")]
    pub immediates: Vec<Integer>,
    #[builder(default = "0")]
    pub scale_factor_init: usize,
    #[builder(default = "0")]
    pub scale_factor_base: usize,
    #[builder(default = "0")]
    pub scale_factor_delta: usize,
    #[builder(default = "true")]
    pub use_stochastic_rounding: bool,
    #[builder(default = "true")]
    pub use_zero_compression: bool,
    #[builder(default = "true")]
    pub use_randomized_base_selection: bool,
}

#[derive(Debug)]
pub struct CompressionResult {
    pub zero_mask: Option<usize>,
    pub bases: Vec<Integer>,
    pub elements: Vec<BufferElement>,
    pub name: String,
    pub n_bases: usize,
    pub size: usize,
    pub compression_duration: u128,
    pub decompression_duration: u128,
}

#[derive(Clone, Copy, Debug)]
pub enum BufferElement {
    Compressed { base: Integer, delta: Integer },
    Uncompressed(Integer),
    Zero,
}

/// Gets the max and min values for a specific size.
const fn size_to_limit(size: usize) -> (Integer, Integer) {
    let shift_amount = INT_SIZE - size;
    (
        Integer::min_value() >> shift_amount,
        Integer::max_value() >> shift_amount,
    )
}

impl Compression {
    /// Processes a cache line, finds all potential values that are compressible for a given base, and marks those values as compressed.
    /// returns the number of elements compressed in this iteration
    fn run_base_delta(
        &self,
        elements: &mut Vec<BufferElement>,
        base: Integer,
        delta_size: usize,
        rng: &mut ThreadRng,
    ) -> usize {
        let (min, max) = size_to_limit(delta_size);
        elements
            .iter_mut()
            .map(|element| {
                match element {
                    BufferElement::Uncompressed(value) => {
                        let value = *value;
                        if Integer::min_value() == 0 && base > value {
                            return 0;
                        }
                        let delta = value - base;
                        let delta_shifted = delta >> self.scale_factor_delta;
                        // create the left shifted version with stochastic rounding
                        let delta_shifted = if self.use_stochastic_rounding
                            && min <= (delta_shifted + 1)
                            && max <= (delta_shifted + 1)
                        {
                            // if rand(0, 1) <= (((delta >> SCALE_FACTOR) + 1) << SCALE_FACTOR) - delta
                            let plus_one_shifted_recovered =
                                ((delta_shifted + 1) as u32) << self.scale_factor_delta;
                            let plus_one_shifted_diff =
                                (plus_one_shifted_recovered - delta as u32) as Integer;
                            let ratio = (plus_one_shifted_diff as f32)
                                / ((1 << self.scale_factor_delta) as f32);
                            let prob = rng.gen::<f32>();

                            if prob <= ratio {
                                delta_shifted + 1
                            } else {
                                delta_shifted
                            }
                        } else {
                            delta_shifted
                        };

                        if min <= delta_shifted && delta_shifted <= max {
                            *element = BufferElement::Compressed {
                                base,
                                delta: delta_shifted,
                            };
                            1
                        } else {
                            0
                        }
                    }
                    _ => 1,
                }
            })
            .sum::<usize>()
    }

    fn get_base(
        &self,
        elements: &Vec<BufferElement>,
        base_size: usize,
        num_left: usize,
        rng: &mut ThreadRng,
    ) -> Option<Integer> {
        let (min, max) = size_to_limit(base_size);
        let mut mapped = elements.iter().filter_map(|element| match element {
            BufferElement::Uncompressed(value) => {
                let value = *value >> self.scale_factor_base;
                if min <= value && value <= max {
                    Some(value)
                } else {
                    None
                }
            }
            _ => None,
        });
        if self.use_randomized_base_selection {
            mapped.enumerate().find_map(|(i, element)| {
                if rng.gen_range(0, num_left) <= i {
                    Some(element)
                } else {
                    None
                }
            })
        } else {
            mapped.next()
        }
    }

    fn calculate_zero_mask(&self, elements: &mut Vec<BufferElement>) -> usize {
        elements
            .iter_mut()
            .enumerate()
            .map(|(i, element)| match element {
                BufferElement::Uncompressed(value)
                    if *value == 0 || (*value >> self.scale_factor_delta) == 0 =>
                {
                    *element = BufferElement::Zero;
                    0
                }
                _ => (0b1 << i),
            })
            .fold(0, |x, y| x | y)
    }

    pub fn compress(
        &self,
        mut elements: Vec<BufferElement>,
        config: &CompressionConfig,
    ) -> Option<CompressionResult> {
        let start_time = time::Instant::now();
        let mut rng = thread_rng();

        let mut bases: Vec<Integer> = Vec::new();
        bases.reserve(config.num_bases);
        let num_elements = elements.len();
        let size: usize = {
            let zero_mask_size = if self.use_zero_compression {
                num_elements
            } else {
                0
            };
            zero_mask_size + (config.num_bases * config.base_size)
        };

        // compress zeros
        let zero_mask = if self.use_zero_compression {
            Some(self.calculate_zero_mask(&mut elements))
        } else {
            None
        };

        // process immediates
        let mut num_left = num_elements;
        for immediate in &self.immediates {
            num_left = num_elements
                - self.run_base_delta(&mut elements, *immediate, config.delta_size, &mut rng);
            if num_left == 0 {
                break;
            }
        }

        if num_left != 0 {
            // treat the zero mask as a base and run it on that.
            for _ in 0..config.num_bases {
                if let Some(base) = self.get_base(&elements, config.base_size, num_left, &mut rng) {
                    bases.push(base);
                    num_left = num_elements
                        - self.run_base_delta(&mut elements, base, config.delta_size, &mut rng);
                    if num_left == 0 {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        if num_left != 0 {
            None
        } else {
            let size = size
                + elements
                    .iter()
                    .map(|element| match element {
                        BufferElement::Uncompressed(_) => INT_SIZE,
                        BufferElement::Compressed { base: _, delta: _ } => config.delta_size,
                        BufferElement::Zero => 0,
                    })
                    .sum::<usize>();
            let n_bases = bases.len();
            Some(CompressionResult {
                bases,
                compression_duration: start_time.elapsed().as_nanos(),
                decompression_duration: 0,
                elements,
                name: config.name.clone(),
                n_bases,
                size,
                zero_mask,
            })
        }
    }

    /// Compresses (in parallel) the given cache line with all possible configurations. Returns the best compression.
    pub fn compress_with_best_config(&self, buffer: Vec<Integer>) -> CompressionResult {
        let elements = buffer
            .into_iter()
            .map(|element| BufferElement::Uncompressed(element))
            .collect::<Vec<_>>();
        let result = self
            .compression_configs
            .par_iter()
            .filter_map(|config| self.compress(elements.clone(), config))
            .reduce(
                || CompressionResult {
                    bases: Vec::new(),
                    compression_duration: u128::max_value(),
                    decompression_duration: u128::max_value(),
                    elements: Vec::new(),
                    name: String::from("invalid"),
                    n_bases: 0,
                    size: usize::max_value(),
                    zero_mask: None,
                },
                |x, y| if x.size > y.size { y } else { x },
            );
        if result.size == usize::max_value() {
            panic!("Could not compress chunk...")
        }
        result
    }

    /// Partitions the input buffer into chunks (cache lines) and processes each cache line in parallel.
    pub fn compress_block(&self, buffer: &[Integer]) -> Vec<CompressionResult> {
        buffer
            .into_par_iter()
            .chunks(self.batch_size)
            .map(|buffer| {
                self.compress_with_best_config(
                    buffer
                        .into_iter()
                        .map(|value| *value >> self.scale_factor_init)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    }

    /// Helper function for running BDI on a block of memory and returning some important stats on it.
    /// Stats returned: Compressed results, decompressed values (i.e., value -> compress -> decompress. used to assess data loss), ouptut size, compression schemes used, and total time taken to compress and decompress.
    pub fn get_compression_stats(
        &self,
        buffer: &[Integer],
        scale: f32,
        zero_point: usize,
    ) -> (
        Vec<CompressionResult>,
        Vec<f32>,
        usize,
        Vec<String>,
        (time::Duration, time::Duration),
    ) {
        let num_elements = buffer.len();
        let start = time::Instant::now();
        let mut results = self.compress_block(buffer);
        let compression_duration = start.elapsed();
        let output_size = results.iter().map(|result| result.size).sum::<usize>();
        let schemes = results
            .iter()
            .map(|result| result.name.clone())
            .collect::<Vec<_>>();
        let start = time::Instant::now();
        let (decompressed_values, decompression_durations): (Vec<_>, Vec<_>) = results
            .iter()
            .flat_map(|result| {
                result.elements.iter().map(|element| {
                    let start = time::Instant::now();
                    match element {
                        BufferElement::Compressed { base, delta } => {
                            let value = (base << self.scale_factor_base)
                                + (delta << self.scale_factor_delta);
                            let value = value << self.scale_factor_init;
                            let value = value as i32 + zero_point as i32;
                            ((value as f32) * scale, start.elapsed())
                        }
                        BufferElement::Uncompressed(_) => panic!("could not compress..."),
                        BufferElement::Zero => ((zero_point as f32) * scale, start.elapsed()),
                    }
                })
            })
            .unzip();
        let decompression_duration = start.elapsed();
        assert_eq!(num_elements, decompressed_values.len());

        for (result, duration) in results.iter_mut().zip(decompression_durations) {
            result.decompression_duration = duration.as_nanos()
        }
        (
            results,
            decompressed_values,
            output_size,
            schemes,
            (compression_duration, decompression_duration),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    use std::fs::File;
    use std::io::{self, BufRead, BufReader};

    #[bench]
    fn bench_compression(b: &mut Bencher) {
        fn read_input_file(path: String) -> io::Result<Vec<Integer>> {
            let f = File::open(path)?;
            Ok(BufReader::new(f)
                .lines()
                .filter_map(|line| match line {
                    Ok(line) => match line.parse::<Integer>() {
                        Ok(value) => Some(value),
                        _ => None,
                    },
                    _ => None,
                })
                .take(256)
                .collect::<Vec<_>>())
        }

        let data = read_input_file(String::from("/workspaces/bdi-rust/example/data.txt")).unwrap();
        let compression = CompressionBuilder::default()
            .scale_factor_base(24usize)
            .scale_factor_delta(24usize)
            .build()
            .unwrap();

        b.iter(|| {
            compression.compress_block(data.as_slice());
        });
    }

    fn compress_with_best_config_and_get_results(buffer: Vec<Integer>) -> Vec<Integer> {
        let compression = CompressionBuilder::default()
            .scale_factor_base(0usize)
            .scale_factor_delta(0usize)
            .use_stochastic_rounding(false)
            .use_zero_compression(false)
            .build()
            .unwrap();
        let mut result: CompressionResult = compression.compress_with_best_config(buffer);
        let mut elements = result
            .elements
            .into_iter()
            .map(|element| match element {
                BufferElement::Uncompressed(_) => panic!("uncompressed element detected!"),
                BufferElement::Compressed { base: _, delta } => delta,
                BufferElement::Zero => 0,
            })
            .collect::<Vec<_>>();

        result.bases.append(&mut elements);
        result.bases
    }

    fn assert_compression(input: Vec<Integer>, output: Vec<Integer>) {
        assert_eq!(compress_with_best_config_and_get_results(input), output)
    }

    #[test]
    fn it_compresses_correctly_with_the_presentation_exmaple() {
        assert_compression(
            vec![0xc04039c0, 0xc04039c8, 0xc04039d0, 0xc04039f8],
            vec![0xc04039c0, 0x00, 0x08, 0x10, 0x38],
        )
    }

    #[test]
    fn it_works_on_perl_bench_example() {
        assert_compression(
            vec![
                0xc04039c0, 0xc04039c8, 0xc04039d0, 0xc04039d8, 0xc04039e0, 0xc04039e8, 0xc04039f0,
                0xc04039f8,
            ],
            vec![0xc04039c0, 0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38],
        )
    }

    #[test]
    fn it_cant_compress_data_without_locality() {
        assert_compression(
            vec![0xc04039c0, 0xc04039c8, 0xc04039d0, 0xc04039f8],
            vec![0xc04039c0, 0x00, 0x08, 0x10, 0x38],
        )
    }
}
