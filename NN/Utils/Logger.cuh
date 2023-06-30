//
// Created by sid on 30/6/23.
//

#ifndef NNCPP_LOGGER_CUH
#define NNCPP_LOGGER_CUH
#include <utility>

#include "../../pch.cuh"

namespace NN {
    class Logger {
    public:
        explicit Logger(bool debug, std::string prefix = "") : debug_enabled(debug),
                                                               prefix(std::move(prefix) + " : \t") {}

        // Handle << operator for all types of input
        template<typename T>
        Logger &operator<<(const T &msg) {
            if (debug_enabled) {
                std::cout << prefix << msg;
            }
            return *this;
        }

        // Specialization for std::vector
        template<typename T>
        Logger &operator<<(const std::vector<T> &vec) {
            if (debug_enabled) {
                for (const auto &element: vec) {
                    std::cout << prefix << element << ' ';
                }
                std::cout << std::endl;
            }
            return *this;
        }

        // Handle std::endl and other iomanip
        Logger &operator<<(std::ostream &(*func)(std::ostream &)) {
            if (debug_enabled) {
                func(std::cout);
            }
            return *this;
        }

        ~Logger() = default;

    private:
        bool debug_enabled;
        std::string prefix;
    };

}
#endif //NNCPP_LOGGER_CUH
