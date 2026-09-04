#include <print>
#include <fstream>
#include <string>
#include <string_view>
#include <set>
#include <vector>
#include <map>
#include <random>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "matrix.h"
#include "math_vector.h"

std::vector<std::string> Split(std::string_view, std::string_view);

int main()
{
    std::ifstream file{"/home/cartercpp/Documents/C++/CBOW/yeat.txt"};
    const std::string text(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});

    const std::vector<std::string> splitText{Split(text, " \t\n.,!?;:'\"()[]{}<>+-=*/\\|&%^~`_")};
    const std::set<std::string> uniqueWords(splitText.begin(), splitText.end());
    const std::size_t vocabularySize = uniqueWords.size();

    std::map<std::string, std::size_t> wordToID;
    for (std::size_t i = 0; i < vocabularySize; ++i)
    {
        const std::string& word{*std::next(uniqueWords.begin(), i)};
        wordToID[word] = i;
    }

    constexpr std::size_t embeddingSize = 50,
                          windowSize = 2;

    matrix<double> embeddingMatrix(vocabularySize, embeddingSize, 0),
                   weightMatrix(embeddingSize, vocabularySize, 0);
    constexpr double learningRate = 0.01;

    std::random_device rd;
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (std::size_t i = 0; i < vocabularySize; ++i)
        for (std::size_t i2 = 0; i2 < embeddingSize; ++i2)
        {
            embeddingMatrix[i][i2] = dist(rd);
            weightMatrix[i2][i] = dist(rd);
        }

    for (std::size_t epoch = 0; epoch < 100; ++epoch)
    {
        double averageLoss = 0;
        std::size_t examples = 0;

        for (std::size_t start = 0; start + windowSize * 2 + 1 < splitText.size(); ++start)
        {
            const std::vector<std::string>
                sentence(splitText.begin() + start, splitText.begin() + start + windowSize * 2 + 1);

            const std::string& targetWord{sentence[sentence.size() / 2]};
            const std::size_t targetWordID = wordToID[targetWord];

            math_vector<double> target(vocabularySize, 0);
            target[targetWordID] = 1;

            std::vector<math_vector<double>> wordEmbeddings;
            wordEmbeddings.reserve(windowSize * 2);

            for (std::size_t i = 0; i < windowSize; ++i)
            {
                const std::string& word{sentence[i]};
                const std::size_t wordID = wordToID[word];
                wordEmbeddings.emplace_back(embeddingMatrix[wordID]);
            }

            for (std::size_t i = 0; i < windowSize; ++i)
            {
                const std::string& word{sentence[sentence.size() - i - 1]};
                const std::size_t wordID = wordToID[word];
                wordEmbeddings.emplace_back(embeddingMatrix[wordID]);
            }

            math_vector<double> contextEmbeddings(embeddingSize, 0);
            for (const auto& wordEmbedding : wordEmbeddings)
                contextEmbeddings += wordEmbedding / static_cast<double>(windowSize * 2);

            auto probabilities = contextEmbeddings * weightMatrix;
            double max = std::numeric_limits<double>::lowest();

            for (std::size_t i = 0; i < probabilities.size(); ++i)
                if (probabilities[i] > max)
                    max = probabilities[i];

            double softmaxSum = 0;

            for (std::size_t i = 0; i < probabilities.size(); ++i)
            {
                probabilities[i] -= max;
                probabilities[i] = std::exp(probabilities[i]);
                softmaxSum += probabilities[i];
            }

            for (std::size_t i = 0; i < probabilities.size(); ++i)
                probabilities[i] /= softmaxSum;

            const double loss = -std::log(probabilities[targetWordID]);
            averageLoss += loss;
            ++examples;

            const auto outputGradient = probabilities - target;
            const auto weightGradient = outer_product(contextEmbeddings, outputGradient);
            const auto embeddingGradient = outputGradient * weightMatrix.transpose();

            weightMatrix -= learningRate * weightGradient;

            for (std::size_t i = 0; i < windowSize; ++i)
            {
                const std::string& word{sentence[i]};
                const std::size_t wordID = wordToID[word];
                embeddingMatrix[wordID] -= 1.0 / (2 * windowSize) * embeddingGradient;
            }

            for (std::size_t i = 0; i < windowSize; ++i)
            {
                const std::string& word{sentence[sentence.size() - i - 1]};
                const std::size_t wordID = wordToID[word];
                embeddingMatrix[wordID] -= 1.0 / (2 * windowSize) * embeddingGradient;
            }
        }

        averageLoss /= examples;
        std::println("Epoch {} average loss: {:.2f}", epoch, averageLoss);
    }

    std::ofstream weightsFile{"/home/cartercpp/Documents/C++/CBOW/weights.txt"};
    std::ofstream embeddingsFile{"/home/cartercpp/Documents/C++/CBOW/embeddings.txt"};

    weightsFile << std::format("{}", weightMatrix);
    embeddingsFile << std::format("{}", embeddingMatrix);
}
