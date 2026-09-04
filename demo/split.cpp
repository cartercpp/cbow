//
// Created by cartercpp on 9/3/26.
//
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <ranges>
#include <cstddef>

std::vector<std::string> Split(std::string_view text, std::string_view delimiters)
{
    std::vector<std::string> output;

    std::size_t start = 0;
    while (start < text.size())
    {
        const std::size_t end = text.find_first_of(delimiters, start);

        if (end == std::string_view::npos)
        {
            output.emplace_back(text.substr(start));
            break;
        }

        if (end > start)
            output.emplace_back(text.substr(start, end - start));

        start = end + 1;
    }

    return output;
}