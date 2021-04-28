#include <array>
#include <cmath>
#include <utility>

#include <ap_int.h>
#include <hls_math.h>

constexpr auto N = 1024;

struct axi_interface_type
{
    ap_uint<32> data;
    ap_int<1> last;
};

using array_t = std::array<float, N>;
using quantized_array_t = std::array<int, N>;

// constexpr auto N = 256;
// using array_t = std::vector<float>;

static inline std::pair<float, float> get_stats(const array_t &array)
{
    auto sum = 0.f;
    auto sum_of_squares = 0.f;

    for (const auto value : array) {
#pragma HLS PIPELINE
        sum += value;
        sum_of_squares += value * value;
    }

    const auto mean = sum / N;
    const auto square_mean = sum_of_squares / N;

    return std::make_pair(mean, hls::rsqrt(square_mean - (mean * mean)));
}

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
constexpr float dequant_scale_else = 1.f/ get_scale(8);

inline float dequantize(const int data) {
	const auto is_1std = (data & 0b1);
	const auto scale =
        is_1std ? dequant_scale_1std : dequant_scale_else;
    return float(data >> 1) * scale;
}

inline quantized_array_t compress(array_t array, const float shift, const float scalar)
{
    quantized_array_t return_array = {0};
// #pragma HLS ARRAY_PARTITION variable=return_array complete dim=1

    const auto stats = get_stats(array);

    const auto sum = 0;
    for (auto i = 0u; i < N; ++i) {
#pragma PIPELINE
        const auto z_score = (array[i] - stats.first) / stats.second;
        const auto normalized = (z_score * scalar) + shift;
        return_array[i] = quantize(normalized);
    }

    return return_array;
}

inline array_t decompress(quantized_array_t array, const float mean, const float std_dev, const float shift, const float scalar)
{
	array_t return_array = {0};
// #pragma HLS ARRAY_PARTITION variable=return_array complete dim=1

    const auto sum = 0;
    for (auto i = 0u; i < N; ++i) {
#pragma PIPELINE
        const auto dequantized = dequantize(array[i]);
        const auto z_score = (dequantized - shift) / scalar;
        return_array[i] = (z_score * std_dev) + mean;
    }

    return return_array;
}

inline void read_input(axi_interface_type *src, hls::stream<float> &in_stream)
{
    union {
        int ival;
        float oval;
    } converter;

    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        converter.ival = src[i];
        in_stream << converter.oval;
    }
}

extern "C"
{
    void compress_accel(const float shift, const float scalar, axi_interface_type *src, axi_interface_type *dst)
    {
#pragma HLS INTERFACE s_axilite port=shift
#pragma HLS INTERFACE s_axilite port=scalar
#pragma HLS INTERFACE axis register both port=src
#pragma HLS INTERFACE axis register both port=dst
#pragma HLS INTERFACE s_axilite port=return

#pragma DATAFLOW

    	union {
    	    int ival;
    	    float oval;
    	} converter;

        array_t array = {0};
// #pragma HLS ARRAY_PARTITION variable=array complete dim=1

        for (auto i = 0u; i < N; ++i) {
#pragma PIPELINE
        	converter.ival = src[i].data;
            array[i] = converter.oval;
        }

        const auto return_array = compress(array, shift, scalar);
        for (auto i = 0u; i < N; ++i) {
#pragma PIPELINE
            dst[i].data = return_array[i];
            dst[i].last = i == (N - 1);
        }
    }

    void decompress_accel(const float mean, const float std_dev, const float shift, const float scalar, axi_interface_type *src, axi_interface_type *dst)
    {
#pragma HLS INTERFACE s_axilite port=mean
#pragma HLS INTERFACE s_axilite port=std_dev
#pragma HLS INTERFACE s_axilite port=shift
#pragma HLS INTERFACE s_axilite port=scalar
#pragma HLS INTERFACE axis register both port=src
#pragma HLS INTERFACE axis register both port=dst
#pragma HLS INTERFACE s_axilite port=return

    	union {
    	    int ival;
    	    float oval;
    	} converter;

        quantized_array_t array = {0};

        for (auto i = 0u; i < N; ++i) {
#pragma PIPELINE
            array[i] = src[i].data;
        }

        const auto return_array = decompress(array, mean, std_dev, shift, scalar);
        for (auto i = 0u; i < N; ++i) {
#pragma PIPELINE
        	converter.oval = return_array[i];
            dst[i].data = converter.ival;
            dst[i].last = i == (N - 1);
        }
    }
}
