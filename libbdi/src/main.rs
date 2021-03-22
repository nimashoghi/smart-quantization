use std::fs::File;
use std::io::{self, BufRead, BufReader};

use bdi::*;
use clap::Clap;

#[derive(Clap)]
#[clap(version = "1.0", author = "Nima Shoghi")]
struct Opts {
    #[clap(default_value = "./example/data.txt")]
    input_file_path: String,
}

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
        .collect::<Vec<_>>())
}

pub fn main() -> io::Result<()> {
    let opts: Opts = Opts::parse();
    let buffer = read_input_file(opts.input_file_path)?;
    // let configs = make_bdi_configs((MIN_BASES, MAX_BASES));
    let compression = bdi::CompressionBuilder::default()
        .scale_factor_base(24usize)
        .scale_factor_delta(24usize)
        .build()
        .unwrap();

    let outputs = compression.compress_block(buffer.as_slice());

    println!(
        "{:?}",
        outputs
            .into_iter()
            .map(|value| value.name)
            .collect::<Vec<_>>()
    );
    Ok(())
}
