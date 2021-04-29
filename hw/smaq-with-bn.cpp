#include <array>
#include <cmath>
#include <utility>

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

//#define RANGE_STD_DEV
//#define NO_CALC_STATS
#define SAMPLE_STATS

constexpr auto N = 2048;
constexpr auto N_SAMPLE = 16;
// formula = 1 / sqrt(2 * log(N))
constexpr auto C_N = 0.3467341730212743f;

constexpr int get_sample_size()
{
#ifdef SAMPLE_STATS
	return N_SAMPLE;
#else
	return N;
#endif
}

template<typename T>
struct axi_t
{
    T data;
    ap_int<1> last;
};

using array_t = std::array<float, N>;
using quantized_array_t = std::array<int, N>;

// constexpr auto N = 256;
// using array_t = std::vector<float>;

#ifndef RANGE_STD_DEV
static inline std::pair<float, float> get_stats(const axi_t<float> *array)
{
	constexpr auto sample_size = get_sample_size();

    auto sum = 0.f;
    auto sum_of_squares = 0.f;

    for (auto i = 0u; i < sample_size; ++i) {
#pragma HLS UNROLL factor=64 skip_exit_check
        sum += array[i].data;
        sum_of_squares += (array[i].data * array[i].data);
    }

    const auto mean = sum / sample_size;
    const auto square_mean = sum_of_squares / sample_size;

   return std::make_pair(mean, hls::rsqrt(square_mean - (mean * mean)));
}
#else
static inline std::pair<float, float> get_stats(const axi_t<float> *array)
{
    auto sum = 0.f;
    auto min = std::numeric_limits<float>::min();
    auto max = std::numeric_limits<float>::max();

    for (auto i = 0u; i < N; ++i) {
#pragma HLS UNROLL factor=64 skip_exit_check
        sum += array[i].data;
        min = std::min(min, array[i].data);
        max = std::max(max, array[i].data);
    }

    const auto range = max - min;
    const auto mean = sum / N;

    return std::make_pair(mean, 1.f / (C_N * range));
}
#endif

constexpr float get_scale(const std::size_t n_bits) {
    return float(1 << (n_bits - 1)) / 3.f;
}

constexpr float scale_1std = get_scale(6);
constexpr float scale_else = get_scale(8);

inline int quantize(const float data) {
    const auto is_1std = (data <= 1.f && data >= -1.f);
    const auto scale =
        is_1std ? scale_1std : scale_else;
    const auto quantized_float = data * scale;
    return int(quantized_float) << 1 | (is_1std ? 0b1 : 0b0);
}

constexpr float dequant_scale_1std = 1.f / get_scale(6);
constexpr float dequant_scale_else = 1.f / get_scale(8);

inline float dequantize(const int data) {
	const auto is_1std = (data & 0b1);
	const auto scale =
        is_1std ? dequant_scale_1std : dequant_scale_else;
    return float(data >> 1) * scale;
}

inline void compress(hls::stream<float> &in_stream, hls::stream<int> &out_stream, const float mean, const float recpr_std_dev, const float shift, const float scalar)
{
    for (auto i = 0u; i < N; ++i) {
#pragma HLS PIPELINE II=1
        const auto z_score = (in_stream.read() - mean) * recpr_std_dev;
        const auto normalized = (z_score * scalar) + shift;
        out_stream << quantize(normalized);
    }
}

inline void decompress(hls::stream<int> &in_stream, hls::stream<float> &out_stream, const float mean, const float std_dev, const float shift, const float scalar)
{
    for (auto i = 0u; i < N; ++i) {
#pragma HLS PIPELINE II=1
        const auto dequantized = dequantize(in_stream.read());
        const auto z_score = (dequantized - shift) / scalar;
        out_stream << (z_score * std_dev) + mean;
    }
}

template<typename T>
inline void read_input(axi_t<T> *src, hls::stream<T> &in_stream)
{
    for (auto i = 0u; i < N; ++i) {
#pragma HLS PIPELINE II=1
        in_stream << src[i].data;
    }
}

template<typename T>
inline void write_result(hls::stream<T> &out_stream, axi_t<T> *dst)
{
    for (auto i = 0u; i < N; ++i) {
#pragma HLS PIPELINE II=1
        dst[i].data = out_stream.read();
        dst[i].last = i == N - 1;
    }
}

inline void compress_inner(const float mean, const float recpr_std_dev, const float shift, const float scalar, axi_t<float> *src, axi_t<int> *dst)
{
	hls::stream<float> in_stream{"input_stream"};
#pragma HLS STREAM variable=in_stream depth=64
	hls::stream<int> out_stream{"output_stream"};
#pragma HLS STREAM variable=out_stream depth=64

#pragma HLS DATAFLOW
	read_input(src, in_stream);
	compress(in_stream, out_stream, mean, recpr_std_dev, shift, scalar);
	write_result(out_stream, dst);
}

extern "C"
{
#ifndef NO_CALC_STATS
	void compress_accel(const float shift, const float scalar, axi_t<float> *src, axi_t<int> *dst)
#else
   void compress_accel(const float mean, const float recpr_std_dev, const float shift, const float scalar, axi_t<float> *src, axi_t<int> *dst)
#endif
    {
#ifdef NO_CALC_STATS
#pragma HLS INTERFACE s_axilite port=mean
#pragma HLS INTERFACE s_axilite port=recpr_std_dev
#endif
#pragma HLS INTERFACE s_axilite port=shift
#pragma HLS INTERFACE s_axilite port=scalar
#pragma HLS INTERFACE axis register both port=src
#pragma HLS INTERFACE axis register both port=dst
#pragma HLS INTERFACE s_axilite port=return

#ifndef NO_CALC_STATS
		const auto stats = get_stats(src);
		const auto mean = stats.first;
		const auto recpr_std_dev = stats.second;
#endif

		compress_inner(mean, recpr_std_dev, shift, scalar, src, dst);
    }

    void decompress_accel(const float mean, const float recpr_std_dev, const float shift, const float scalar, axi_t<int> *src, axi_t<float> *dst)
    {
#pragma HLS INTERFACE s_axilite port=mean
#pragma HLS INTERFACE s_axilite port=recpr_std_dev
#pragma HLS INTERFACE s_axilite port=shift
#pragma HLS INTERFACE s_axilite port=scalar
#pragma HLS INTERFACE axis register both port=src
#pragma HLS INTERFACE axis register both port=dst
#pragma HLS INTERFACE s_axilite port=return


    	hls::stream<int> in_stream{"input_stream"};
#pragma HLS STREAM variable=in_stream depth=64
    	hls::stream<float> out_stream{"output_stream"};
#pragma HLS STREAM variable=out_stream depth=64

#pragma HLS DATAFLOW
		read_input(src, in_stream);
		decompress(in_stream, out_stream, mean, recpr_std_dev, shift, scalar);
		write_result(out_stream, dst);
    }
}
