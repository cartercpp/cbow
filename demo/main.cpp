#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "matrix.h"

std::vector<std::string> Split(std::string_view, std::string_view);

extern matrix<double> embeddingMatrix;
extern matrix<double> weightMatrix;

namespace ansi
{
    constexpr std::string_view reset   = "\033[0m";
    constexpr std::string_view bold    = "\033[1m";
    constexpr std::string_view dim     = "\033[2m";
    constexpr std::string_view cyan    = "\033[96m";
    constexpr std::string_view magenta = "\033[95m";
    constexpr std::string_view yellow  = "\033[93m";
    constexpr std::string_view green   = "\033[92m";
    constexpr std::string_view red     = "\033[91m";
    constexpr std::string_view gray    = "\033[90m";
}

constexpr std::string_view delimiters = " \t\n.,!?;:'\"()[]{}<>+-=*/\\|&%^~`_";
constexpr std::size_t barWidth = 16;

std::string Lower(std::string word)
{
    std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c)
    {
        return static_cast<char>(std::tolower(c));
    });
    return word;
}

bool IsGoodWindow(const std::array<std::string, 5>& words)
{
    // Skip mostly-nonsemantic ad-libs so the samples read better on screen.
    static const std::set<std::string> adlibs{
        "ooh", "uh", "ayy", "rrr", "rr", "haha", "hey", "slide"
    };

    // Split() breaks contractions at apostrophes ("I'm" -> "I", "m").
    // Avoid windows containing those fragments because they look awkward alone.
    static const std::set<std::string> contractionFragments{
        "m", "re", "ve", "ll", "d", "t", "s"
    };

    std::set<std::string> distinct;

    for (const auto& word : words)
    {
        const std::string lower = Lower(word);

        if (adlibs.contains(lower) || contractionFragments.contains(lower))
            return false;

        distinct.insert(lower);
    }

    // A tiny center word makes for a less interesting TikTok prediction.
    if (words[2].size() < 3)
        return false;

    // Avoid samples that are almost entirely the same repeated word.
    return distinct.size() >= 4;
}

std::vector<std::array<std::string, 5>> BuildCandidateWindows(const std::string& text)
{
    std::vector<std::array<std::string, 5>> candidates;

    std::size_t lineStart = 0;
    while (lineStart <= text.size())
    {
        const std::size_t lineEnd = text.find('\n', lineStart);
        const std::string_view line{
            text.data() + lineStart,
            (lineEnd == std::string::npos ? text.size() : lineEnd) - lineStart
        };

        // Don't sample metadata such as [Chorus] or [Verse].
        const auto firstVisible = line.find_first_not_of(" \t\r");
        if (firstVisible != std::string_view::npos && line[firstVisible] != '[')
        {
            const auto words = Split(line, delimiters);

            for (std::size_t i = 0; i + 5 <= words.size(); ++i)
            {
                std::array<std::string, 5> window{
                    words[i], words[i + 1], words[i + 2], words[i + 3], words[i + 4]
                };

                if (IsGoodWindow(window))
                    candidates.push_back(std::move(window));
            }
        }

        if (lineEnd == std::string::npos)
            break;

        lineStart = lineEnd + 1;
    }

    return candidates;
}

void PrintBar(double probability, std::string_view color)
{
    const auto filled = std::min(
        barWidth,
        static_cast<std::size_t>(std::round(probability * barWidth))
    );

    std::cout << color;
    for (std::size_t i = 0; i < filled; ++i)
        std::cout << "━";

    std::cout << ansi::gray;
    for (std::size_t i = filled; i < barWidth; ++i)
        std::cout << "─";

    std::cout << ansi::reset;
}

int main()
{
    std::ifstream file{"/home/cartercpp/Documents/C++/CBOW/yeat.txt"};
    const std::string text(std::istreambuf_iterator<char>{file},
                           std::istreambuf_iterator<char>{});

    const std::vector<std::string> splitText{Split(text, delimiters)};
    const std::set<std::string> uniqueWords(splitText.begin(), splitText.end());

    std::map<std::string, std::size_t> wordToID;
    std::vector<std::string> idToWord;
    idToWord.reserve(uniqueWords.size());

    std::size_t id = 0;
    for (const auto& word : uniqueWords)
    {
        wordToID[word] = id++;
        idToWord.push_back(word);
    }

    constexpr std::size_t windowSize = 2;
    constexpr std::size_t topK = 5;

    if (embeddingMatrix.rows() != uniqueWords.size() ||
        weightMatrix.columns() != uniqueWords.size())
    {
        std::cerr << "Saved matrices do not match this vocabulary.\n";
        return 1;
    }

    const auto candidates = BuildCandidateWindows(text);
    if (candidates.empty())
    {
        std::cerr << "No suitable 5-word samples found.\n";
        return 1;
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<std::size_t> sampleDist(0, candidates.size() - 1);

    std::size_t sampleNumber = 1;

    while (true)
    {
        const auto& words = candidates[sampleDist(rng)];
        const std::string& actual = words[windowSize];

        math_vector<double> contextEmbedding(embeddingMatrix.columns(), 0.0);

        for (std::size_t i = 0; i < words.size(); ++i)
        {
            if (i == windowSize)
                continue;

            contextEmbedding += embeddingMatrix[wordToID.at(words[i])];
        }

        contextEmbedding /= static_cast<double>(windowSize * 2);

        auto probabilities = contextEmbedding * weightMatrix;

        double maxLogit = std::numeric_limits<double>::lowest();
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            maxLogit = std::max(maxLogit, probabilities[i]);

        double softmaxSum = 0.0;
        for (std::size_t i = 0; i < probabilities.size(); ++i)
        {
            probabilities[i] = std::exp(probabilities[i] - maxLogit);
            softmaxSum += probabilities[i];
        }

        for (std::size_t i = 0; i < probabilities.size(); ++i)
            probabilities[i] /= softmaxSum;

        std::vector<std::pair<double, std::size_t>> ranked;
        ranked.reserve(probabilities.size());

        for (std::size_t i = 0; i < probabilities.size(); ++i)
            ranked.emplace_back(probabilities[i], i);

        std::partial_sort(
            ranked.begin(),
            ranked.begin() + std::min(topK, ranked.size()),
            ranked.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );

        const bool correct = idToWord[ranked[0].second] == actual;
        const std::string_view topColor = correct ? ansi::green : ansi::red;

        // Clear terminal and move cursor to top-left.
        std::cout << "\033[2J\033[H";

        std::cout << ansi::dim << "╭──────────────────────────────────────────╮\n" << ansi::reset;
        std::cout << "  " << ansi::bold << ansi::magenta << "CBOW" << ansi::reset
                  << ansi::gray << "  //  WORD PREDICTOR" << ansi::reset
                  << ansi::dim << "                 #" << sampleNumber << "\n" << ansi::reset;
        std::cout << ansi::dim << "╰──────────────────────────────────────────╯\n\n" << ansi::reset;

        std::cout << ansi::gray << "CONTEXT\n" << ansi::reset;
        std::cout << "  " << words[0] << ' '
                  << words[1] << ' '
                  << ansi::yellow << ansi::bold << "[ MASK ]" << ansi::reset << ' '
                  << words[3] << ' '
                  << words[4] << "\n\n";

        std::cout << ansi::gray << "TOP 5\n" << ansi::reset;
        for (std::size_t i = 0; i < std::min(topK, ranked.size()); ++i)
        {
            const auto [probability, wordID] = ranked[i];
            const std::string_view rowColor = (i == 0) ? topColor : ansi::cyan;

            std::cout << "  " << ansi::gray << (i + 1) << ansi::reset << "  "
                      << rowColor << ansi::bold
                      << std::left << std::setw(11) << idToWord[wordID]
                      << ansi::reset << std::right << ' ';

            PrintBar(probability, rowColor);

            std::cout << "  " << rowColor << std::setw(5) << std::fixed
                      << std::setprecision(1) << probability * 100.0 << "%"
                      << ansi::reset << "\n";
        }

        std::cout << "\n" << ansi::dim
                  << "  next sample in 2s  •  Ctrl+C to stop"
                  << ansi::reset << std::flush;

        ++sampleNumber;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}
