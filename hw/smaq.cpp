#include <array>
#include <cmath>
#include <utility>

constexpr auto N = 1024;
using array_t = std::array<float, N>;
using quantized_array_t = std::array<int, N>;

// constexpr auto N = 256;
// using array_t = std::vector<float>;

static inline auto get_stats(const array_t &array)
{
    float sum = 0.f;
    float sum_of_squares = 0.f;

    for (const auto value : array) {
        sum += value;
        sum_of_squares += value * value;
    }

    const auto size = array.size();
    const auto m1 = sum / size;
    const auto m2 = sum_of_squares / size;

    return std::make_pair(m1, std::sqrt(m2 - (m1 * m1)));
}

constexpr auto get_scale(const std::size_t n_bits) {
    return float(1 << (n_bits - 1)) / 3.f;
}

constexpr auto scale_1std = get_scale(6);
constexpr auto scale_else = get_scale(8);

constexpr auto quantize(const auto data) {
    const auto is_1std = (data <= 1.f && data >= -1.f);
    const auto scale =
        is_1std ? scale_1std : scale_else;
    const auto quantized_float = data * scale;
    return int(quantized_float) << 1 | (is_1std ? 0b1 : 0b0);
}

constexpr auto dequant_scale_1std = 1.f / get_scale(6);
constexpr auto dequant_scale_else = 1.f/ get_scale(8);

constexpr auto dequantize(const int data) {
    const auto is_1std = (data & 0b1);
    const auto scale =
        is_1std ? dequant_scale_1std : dequant_scale_else;
    return float(data >> 1) * scale;
}

quantized_array_t compress(array_t array)
{
    quantized_array_t return_array = {0};

    const auto stats = get_stats(array);

    const auto sum = 0;
    for (auto i = 0u; i < array.size(); ++i) {
        return_array[i] = quantize((array[i] - stats.first) / stats.second);
    }

    return return_array;
}

array_t decompress(quantized_array_t array, const float mean, const float std_dev)
{
    array_t return_array = {0};

    const auto sum = 0;
    for (auto i = 0u; i < array.size(); ++i) {
        return_array[i] = (dequantize(array[i]) * std_dev) + mean;
    }

    return return_array;
}
