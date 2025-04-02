#ifndef UTILS_H
#define UTILS_H

#include <regex>
#include <queue>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <filesystem>
#include <set>
#include <map>
#include <iostream>
#include <functional>
#include <sstream>
#include <numeric>

#include <Logger.hpp>

#include "matrix.hpp"


/* ====================================================================================================
 * Typedefs
 */
template <typename Value>
using Table = std::vector<std::vector<Value>>;

template <typename Key, typename Value>
using NamedTable = std::unordered_map<Key, std::unordered_map<Key, Value>>;

/* ====================================================================================================
 * Structs
 */
struct Point {
	int x;
	int y;
};

template<>
struct std::hash<Point> {
	size_t operator()(const Point& point) const {
		return static_cast<size_t>(point.x) ^ (static_cast<size_t>(point.y) << 32);
	}
};

/* ====================================================================================================
 * Whitespace Correction
 */

/**
 * fills the left side of a string to the given length using the filler character
 * @param s string to be filled
 * @param len length to be filled to
 * @param filler char used to fill the string - default space
 * @return filled string
 */
std::string pad_left(const std::string& s, int len, char filler = ' ') {
	return std::string(std::max(len - s.length(), 0ull), filler) + s;
}

/**
 * fills the right side of a string to the given length using the filler character
 * @param s string to be filled
 * @param len length to be filled to
 * @param filler char used to fill the string - default space
 * @return filled string
 */
std::string pad_right(const std::string& s, int len, char filler = ' ') {
	return s + std::string(std::max(len - s.length(), 0ull), filler);
}

std::string pad_center(const std::string& s, int len, char filler = ' ') {
	auto len_half = s.length() / 2;
	return pad_right(pad_left(s, len_half, filler), len - len_half, filler);
}

std::string repeat(char c, int n) {
	return std::string(n, c);
}

/**
 * trims whitespace (recognized using
 * std::isspace()) from both ends of the string
 * @param s string to be trimmed
 * @return trimmed string
 */
std::string trim(std::string s) {
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) {
		return !std::isspace(c);
	}));

	s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) {
		return !std::isspace(c);
	}).base(), s.end());

	return s;
}

std::string replace_all(const std::string& str, const std::string& pattern, const std::string& replace) {
	size_t n = 0;
	std::string str_cpy = str;

	while ((n = str_cpy.find(pattern, n)) != std::string::npos){
		str_cpy.replace(n, pattern.size(), replace);
		n += replace.size();
	}

	return str_cpy;
}

std::string replace_regex(const std::string& str, const std::regex& pattern, const char* replace) {
	return std::regex_replace(str, pattern, replace, std::regex_constants::match_flag_type::format_first_only);
}

std::string replace_regex_all(const std::string& str, const std::regex& pattern, const char* replace) {
	return std::regex_replace(str, pattern, replace);
}

size_t find_nth(const std::string& str, const std::string& pattern, size_t n) {
	int count = 0;
	size_t last_idx = str.find(pattern, n);
	while (last_idx != std::string::npos) {
		if (count == n) {
			return last_idx;
		}
		last_idx = str.find(pattern, last_idx+1);
		++count;
	}
	return std::string::npos;
}

size_t find_nth(const std::string& str, const std::regex& pattern, size_t n) {
	std::sregex_iterator iter(str.begin(), str.end(), pattern);
	std::sregex_iterator end;
	for (int i = 0; i < n && iter != end; ++iter, ++i) {}
	if (iter == end) {
		return std::string::npos;
	}
	return iter->position(0);
}

std::optional<std::string> replace_nth(const std::string& str, const std::string& pattern, std::string replace, int n) {
	auto idx = find_nth(str, pattern, n);
	if (idx == std::string::npos) {
		return {};
	}

	return str.substr(0, idx) + replace + str.substr(idx + pattern.size());
}

/* ====================================================================================================
 * Reading Data
 */
/**
 * Reads file
 * @param filename
 * @return content
 * @throws 0xDEAD If file could not be found or opened
 */
std::string read_file(const std::string& filename) {
	auto dir_name = std::filesystem::current_path().parent_path().filename().string();
	dir_name = dir_name.substr(0, dir_name.size() - 2);
	auto cwd = std::filesystem::current_path() / "../../../" / "src" / dir_name;
	cwd = std::filesystem::canonical(cwd);
	auto file = std::ifstream(cwd / filename, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
		Logger::critical("Failed to open file '{}'. CWD: {}", filename, cwd.string());
    }

	std::stringstream contents;
	contents << file.rdbuf();

	return replace_all(contents.str(), "\r\n", "\n");
}

std::vector<std::string> split_lines(const std::string& s) {
	std::vector<std::string> result;
	std::stringstream ss(s);
	std::string line;
	while (std::getline(ss, line)) {
		line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
		result.emplace_back(line);
	}
	return result;
}

/* ====================================================================================================
 * Printing Data
 */
/**
 * Prints a 2d unordered_map
 * @param mat data to print
 * @param keyFormatter transformation function for key-values
 * @param valueFormatter transformation function for data-values
 */
template<typename Key, typename Value>
void printAdjacencyMatrix(const NamedTable<Key, Value>& mat,
	std::function<std::string(Key)> keyFormatter, std::function<std::string(Value)> valueFormatter) {

	std::set<Key> keys;
	int max_key_len = 0;
	for (const auto& [key, row] : mat) {
		keys.emplace(key);
		max_key_len = std::max(max_key_len, (int)(keyFormatter(key).size()));
		for (const auto& [key2, _] : row) {
			keys.emplace(key2);
			max_key_len = std::max(max_key_len, (int)(keyFormatter(key2).size()));
		}
	}

	int keyCounter = 0;
	std::unordered_map<Key, int> keyIdx;
	for (const auto& key : keys) {
		keyIdx.emplace(key, keyCounter);
		keyCounter++;
	}

	std::vector<std::vector<std::string>> values;
	for (int i = 0; i < keyCounter; i++) {
		values.emplace_back(keyCounter, pad_left("-", max_key_len) + "  ");
	}

	for (const auto& [key, row] : mat) {
		const auto x = keyIdx.find(key);
		for (const auto& [key2, value] : row) {
			const auto y = keyIdx.find(key2);
			values[x->second][y->second] = pad_left(valueFormatter(value), max_key_len) + "  ";
		}
	}

	std::cout << pad_right("", max_key_len);
	for (const auto& key : keys) {
		std::cout << "  " << pad_right(std::string(key), max_key_len);
	}
	std::cout << std::endl;

	for (const auto& [key, row] : mat) {
		std::cout << pad_right(std::string(key), max_key_len) << "  ";
		const auto idx = keyIdx.find(key);
		for (const auto& value : values[idx->second]) {
			std::cout << value;
		}
		std::cout << std::endl;
	}
}

/* ====================================================================================================
 * Splitting
 */

template<typename T>
T string_to_generic(std::string s) {
	Logger::critical("No implementation for generating this generic");
	return T();
}

template<>
inline std::string string_to_generic<std::string>(std::string s) {
	return s;
}

template<>
inline int string_to_generic<int>(std::string s) {
	return std::stoi(s);
}

template<>
inline long string_to_generic<long>(std::string s) {
	return std::stol(s);
}

template<>
inline double string_to_generic<double>(std::string s) {
	return std::stod(s);
}

template<>
inline char string_to_generic<char>(std::string s) {
	return s[0];
}
template<>
inline long long string_to_generic<long long>(std::string s) {
	return std::stoll(s);
}

template<>
inline unsigned long long string_to_generic<unsigned long long>(std::string s) {
	return std::stoull(s);
}


/**
 *	Splits a given string at the given delimiter and trims the parts
 *	@param s string to be split
 *	@param delim delimiter (can be longer than 1 char)
 */
std::vector<std::string> split(const std::string& s, const std::string& delim) {
	if (delim.empty()) {
		Logger::critical("`split` received an empty delimiter");
		return {};
	}
	if (trim(s).empty()) {
		return {};
	}

	size_t lastDelim = 0;
	auto curDelim = s.find(delim, lastDelim);
	std::vector<std::string> parts{};

	while (curDelim != std::string::npos) {
		std::string part = s.substr(lastDelim, curDelim - lastDelim);
		parts.emplace_back(trim(part));
		lastDelim = curDelim + delim.size();
		curDelim = s.find(delim, lastDelim);
	}

	auto lastPart = s.substr(lastDelim);
	parts.emplace_back(trim(lastPart));

	return parts;
}

/**
 *	Splits a given string at the given delimiter and trims the parts before converting them using the given function.
 *	@param s string to be split
 *	@param delim delimiter (can be longer than 1 char)
 *	@param fn converts parts after splitting using this function
 */
template<typename T>
std::vector<T> split(const std::string& s, const std::string& delim, std::function<T(std::string)> fn) {
	auto splitted = split(s, delim);
	auto result = std::vector<T>();
	result.reserve(splitted.size());
	for (auto str : splitted) {
		result.emplace_back(fn(str));
	}
	return result;
}

template<typename T>
std::vector<T> split(const std::string& s, const std::string& delim) {
	auto splitted = split(s, delim);
	auto result = std::vector<T>();
	result.reserve(splitted.size());
	for (auto str : splitted) {
		result.emplace_back(string_to_generic<T>(str));
	}
	return result;
}

std::vector<int> split_int(const std::string& s, const std::string& delim) {
	return split<int>(s, delim);
}

/**
 *	Splits a given string at the first occurrence of the given delimiter and returns both parts as pair
 *	@param s string to be split
 *	@param delim delimiter (can be longer than 1 char)
 *	@param fn converts parts after splitting using this function
 */
std::pair<std::string, std::string> split_once(const std::string& s, const std::string& delim) {
	auto idx = s.find(delim);
	if (idx == std::string::npos) {
		Logger::critical("Failed split_once");
	}

	return {s.substr(0, idx), s.substr(idx + delim.size())};
}

template<typename T, typename U>
std::pair<T, U> split_once(const std::string& s, const std::string& delim) {
	auto idx = s.find(delim);
	if (idx == std::string::npos) {
		Logger::critical("Failed split_once");
	}

	return {
		string_to_generic<T>(s.substr(0, idx)),
		string_to_generic<U>(s.substr(idx + delim.size()))
	};
}

template<typename T>
std::pair<T, T> split_once(const std::string& s, const std::string& delim, std::function<T(std::string)> fn) {
	auto idx = s.find(delim);
	if (idx == std::string::npos) {
		Logger::critical("Failed split_once");
	}

	return {
		fn(s.substr(0, idx)),
		fn(s.substr(idx + delim.size()))
	};
}

template<typename T, typename... Args> requires (std::is_same_v<T, Args...>)
T max(T arg, Args... args) {
	return max(arg, max(args));
}

template<typename T, typename U> requires (std::is_same_v<T, U>)
T max(T t, U u) {
	return t > u ? t : u;
}

template<typename T, typename... Args> requires (std::is_same_v<T, Args...>)
T min(T arg, Args... args) {
	return min(arg, min(args));
}

template<typename T, typename U> requires (std::is_same_v<T, U>)
T min(T t, U u) {
	return t < u ? t : u;
}

template<typename... Args, std::size_t... Indices>
std::tuple<Args...> make_tuple_from_match(const std::smatch& match, std::index_sequence<Indices...>) {
	return std::make_tuple<Args...>(string_to_generic<Args>(match[Indices + 1].str())...);
}

/**
 * Extracts data from a given string using a regex and converts the matches to the correct type using the generics provided
 * @tparam Args Types of the matches. A template specialization of string_to_generic must be provided
 * @param pattern regex pattern to match
 * @param s	string to be matched
 * @return tuple of the converted matches
 */
template<typename... Args>
std::tuple<Args...> extract_data(std::string s, const std::regex& pattern) {
	std::smatch match;
	if (!std::regex_match(s, match, pattern)) {
		Logger::critical("Failed to match regex for '{}'", s);
	}

	return make_tuple_from_match<Args...>(match, std::index_sequence_for<Args...>{});
}

template<typename... Args>
std::vector<std::tuple<Args...>> extract_data_all(std::string s, const std::regex& pattern) {
	std::sregex_iterator iter(s.begin(), s.end(), pattern);
	std::sregex_iterator end;
	std::vector<std::tuple<Args...>> result{};
	while (iter != end) {
		std::smatch match = *iter;
		result.push_back(make_tuple_from_match<Args...>(match, std::index_sequence_for<Args...>{}));
		++iter;
	}

	return result;
}

template<typename... Args>
std::optional<std::tuple<Args...>> extract_data_opt(std::string s, const std::regex& pattern) {
	std::smatch match;
	if (!std::regex_match(s, match, pattern)) {
		return std::nullopt;
	}

	return make_tuple_from_match<Args...>(match, std::index_sequence_for<Args...>{});
}

std::vector<std::string> split_regex(const std::string& s, std::regex& pattern) {
	std::sregex_token_iterator iter(s.begin(), s.end(), pattern, -1);
	std::sregex_token_iterator end;
	return {iter, end};
}

template<typename T>
std::vector<std::string> split_regex(const std::string& s, std::regex& pattern) {
	std::sregex_token_iterator iter(s.begin(), s.end(), pattern, -1);
	std::sregex_token_iterator end;
	auto result = std::vector<T>();
	for (; iter != end; ++iter) {
		result.emplace_back(string_to_generic<T>(*iter));
	}
	return result;
}

template<typename T>
std::vector<std::string> split_regex(const std::string& s, std::regex& pattern, std::function<T(std::string)> fn) {
	std::sregex_token_iterator iter(s.begin(), s.end(), pattern, -1);
	std::sregex_token_iterator end;
	auto result = std::vector<T>();
	for (; iter != end; ++iter) {
		result.emplace_back(fn(*iter));
	}
	return result;
}

std::pair<std::string, std::string> split_once_regex(const std::string& s, const std::regex& pattern) {
	std::smatch match;
	if (!std::regex_search(s, match, pattern)) {
		Logger::critical("Failed to find regex in '{}'", s);
	}

	auto match_start = match.position(0);
	auto match_end = match_start + match.length(0);

	return {
		s.substr(0, match_start),
		s.substr(match_end)
	};
}

template<typename T, typename U>
std::pair<T, U> split_once_regex(const std::string& s, const std::regex& pattern) {
	std::smatch match;
	if (!std::regex_search(s, match, pattern)) {
		Logger::critical("Failed to find regex in '{}'", s);
	}

	auto match_start = match.position(0);
	auto match_end = match_start + match.length(0);

	return {
		string_to_generic<T>(s.substr(0, match_start)),
		string_to_generic<U>(s.substr(match_end))
	};
}

template<typename T>
std::pair<T, T> split_once_regex(const std::string& s, const std::regex& pattern, std::function<T(std::string)> fn) {
	std::smatch match;
	if (!std::regex_search(s, match, pattern)) {
		Logger::critical("Failed to find regex in '{}'", s);
	}

	auto match_start = match.position(0);
	auto match_end = match_start + match.length(0);

	return {
		fn(s.substr(0, match_start)),
		fn(s.substr(match_end))
	};
}

std::vector<std::string> find_all_regex(const std::string& s, std::regex& pattern) {
	std::sregex_iterator iter(s.begin(), s.end(), pattern);
	std::sregex_iterator end;
	std::vector<std::string> result{};
	while (iter != end) {
		std::smatch match = *iter;
		result.push_back(match[1]);
		++iter;
	}
	return result;
}

bool isDigit(char c) {
	return '0' <= c && c <= '9';
}

bool isLowercase(char c) {
	return 'a' <= c && c <= 'z';
}

bool isUppercase(char c) {
	return 'A' <= c && c <= 'Z';
}

bool isHex(char c) {
	return ('0' <= c && c <= '9') || ('a' <= c && c <= 'f');
}

std::string format_time(std::chrono::duration<std::chrono::nanoseconds::rep, std::chrono::nanoseconds::period> duration) {
	std::string result = "";
	const int64_t lengths[] = {
		// nano
		1000, //micro
		1000, //milli
		1000, //sec
		60, // min
		60, // hours
		24, // days
		365 // years
	};

	const char* names[] = {" ns", " \xE6s ", " ms ", " s ", " min ", " h ", " d "};

	auto rest = duration.count();
	int i = 0;
	while (rest != 0 && i < 7) {
		result.insert(0, std::to_string(rest % lengths[i]) + names[i]);
		rest /= lengths[i];
		i++;
	}
	if (rest != 0) {
		result = std::to_string(rest) + " a " + result;
	}
	return result;
}

template<typename Result, typename... Args>
struct Test {
	std::string input;
	Result expected;
	bool file;
	std::tuple<Args...> args;
};

template<typename... Args>
struct Input {
	std::string input;
	bool file;
	std::tuple<Args...> args;
};

template <typename Result, typename... Args>
class Runner {
private:
	typedef std::function<Result(std::string, Args...)> SolverFn;
	typedef std::function<std::string(Result)> ResultTransformFn;

	SolverFn solve_fn;
	ResultTransformFn result_transform_fn = nullptr;

	std::vector<Test<Result, Args...>> tests;
	std::vector<Input<Args...>> inputs;
	std::vector<Result> results;

	unsigned tests_failed;
	unsigned tests_succeeded;

public:
	Runner(SolverFn solve_fn, const int year, const int day) : solve_fn(solve_fn), tests_failed(0), tests_succeeded(0) {
		Logger::init();
		Logger::info("==================================================");
		Logger::info("=========== Advent of Code {} Day {} ===========", year, pad_left(std::to_string(day), 2, '0'));
		Logger::info("==================================================");
	}

	void set_result_transformation(std::function<std::string(Result)> result_transform_fn) {
		this->result_transform_fn = result_transform_fn;
	}

	void add_test_string(const std::string& input, Result expected, Args... args) {
		tests.push_back(Test<Result, Args...>(input, expected, false, {args...}));
	}

	void add_test_file(const std::string& filename, Result expected, Args... args) {
		tests.push_back(Test<Result, Args...>(filename, expected, true, {args...}));
	}

	void add_input_string(const std::string& input, Args... args) {
		inputs.push_back(Input<Args...>(input, false, {args...}));
	}

	void add_input_file(const std::string& filename, Args... args) {
		inputs.push_back(Input<Args...>(filename, true, {args...}));
	}

	bool run_test(const Test<Result, Args...>& test) {
		auto input = test.input;
		if (test.file) {
			input = read_file(input);
		}
		Result result = std::apply(
			[=](auto&&... args) -> Result {
				return solve_fn(input, args...);
			},
			test.args
		);

		if (result == test.expected) {
			tests_succeeded++;
			return true;
		}

		std::stringstream ss;
		ss << "Failed Test '" << test.input << "': Expected " << test.expected << " but got " << result;

		Logger::error("{}", ss.str());
		tests_failed++;
		return false;
	}

	bool run_tests() {
		if (tests.empty()) return true;

		Logger::info("==================================================");
		Logger::info("Running {} Tests", tests.size());
		Logger::info("==================================================");

		tests_failed = 0;
		tests_succeeded = 0;

		for (auto& test : tests) {
			run_test(test);
		}

		Logger::info("==================================================");
		Logger::info("Tests Finished");
		Logger::info("--------------------------------------------------");
		if (tests_succeeded != 0) {
			Logger::info("Succeded: {}", tests_succeeded);
		}

		if (tests_failed != 0) {
			Logger::error("Failed: {}", tests_failed);
		}
		Logger::info("==================================================");

		return tests_failed == 0;
	}

	Result run_input(const Input<Args...>& input) {
		auto input_str = input.input;
		if (input.file) {
			input_str = read_file(input_str);
		}
		auto start_time = std::chrono::high_resolution_clock::now();
		Result result = std::apply(
			[=](auto&&... args) -> Result {
				return solve_fn(input_str, args...);
			},
			input.args
		);
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration =
			std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

		std::stringstream ss;
		ss << "Input Finished '" << input.input << "': ";
		if (result_transform_fn == nullptr) {
			ss << result;
		} else {
			ss << result_transform_fn(result);
		}
		ss << " (" << format_time(duration) << ")";
		Logger::info("{}", ss.str());

		return result;
	}

	std::vector<Result> run_inputs() {
		results.clear();
		for (auto& input : inputs) {
			results.emplace_back(run_input(input));
		}
		return results;
	}

	std::vector<Result> run() {
		if (!run_tests()) return {};
		run_inputs();

		Logger::info("");
		return results;
	}
};

std::vector<size_t> find_all_idx(const std::string& s, const std::string& pattern) {
	std::vector<size_t> idxs{};
	size_t last_idx = s.find(pattern);
	while (last_idx != std::string::npos) {
		idxs.push_back(last_idx);
		last_idx = s.find(pattern, last_idx+1);
	}
	return idxs;
}

template<typename T>
std::ostream& operator<< (std::ostream& os, const std::vector<T>& list) {
	os << "vector{ ";
	for (const T& n : list) {
		os << n << ", ";
	}
	os << "}";
	return os;
}

template<typename T>
std::ostream& operator<< (std::ostream& os, const std::unordered_set<T>& set) {
	os << "set( ";
	for (const T& n : set) {
		os << n << ", ";
	}
	os << ")";
	return os;
}

template<typename T>
T sum(std::vector<T> list) {
	auto sum = static_cast<T>(0);
	for (auto n : list) {
		sum += n;
	}
	return sum;
}

template <typename T>
std::vector<std::vector<T>> create_mat(unsigned size, T fill = static_cast<T>(0)) {
	return std::vector<std::vector<T>>(size, std::vector<T>(size, fill));
}

template <typename T>
T max(std::vector<T> list) {
	T max_val = list[0];
	for (const auto& n : list) {
		max_val = max(max_val, n);
	}
	return max_val;
}

template <typename T>
T min(std::vector<T> list) {
	T min_val = list[0];
	for (const auto& n : list) {
		min_val = min(min_val, n);
	}
	return min_val;
}

enum class Dir {
	LEFT,
	RIGHT,
	UP,
	DOWN
};

Vec2i dir_vec(Dir dir) {
	switch (dir) {
    	case Dir::LEFT: return {-1, 0};
    	case Dir::RIGHT: return {1, 0};
    	case Dir::UP: return {0, -1};
    	case Dir::DOWN: return {0, 1};
	}
	return {0, 0};
}

std::vector<Vec2i> all_dirs() {
	return {
		{1,0},
		{0,1},
		{-1,0},
		{0,-1}
	};
}

std::vector<Vec2i> all_dirs_diag() {
	return {
		{1,0},
		{1,1},
		{0,1},
		{-1,1},
		{-1,0},
		{-1,-1},
		{0,-1},
		{1,-1}
	};
}

std::string str(char c) {
	return {c};
}

std::string str(int i) {
	return std::to_string(i);
}

std::string str(double i) {
	return std::to_string(i);
}

std::string str(long i) {
	return std::to_string(i);
}

std::string str(uint64_t i) {
	return std::to_string(i);
}

std::vector<size_t> find_all_idx(const std::string& s, char pattern) {
	return find_all_idx(s, str(pattern));
}

size_t find_nth(const std::string& s, char pattern, size_t n) {
	return find_nth(s, str(pattern), n);
}

std::string replace_all(const std::string& s, char pattern, const std::string& replace) {
	return replace_all(s, str(pattern), replace);
}

template<typename T>
std::vector<T> max_n(std::vector<T> list, size_t n) {
	std::sort(list.begin(), list.end(), std::greater<T>());
	if (list.size() < n) {
		return list;
	}
	return std::vector<T>(list.begin(), list.begin() + n);
}

template<typename T>
std::vector<T> min_n(std::vector<T> list, size_t n) {
	std::sort(list.begin(), list.end(), std::less<T>());
	if (list.size() < n) {
		return list;
	}
	return std::vector<T>(list.begin(), list.begin() + n);
}

template<typename T>
std::set<T> set_intersection(std::set<T> a, std::set<T> b) {
	std::set<T> res{};
	std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::inserter(res, res.begin()));
	return res;
}

template<typename T>
std::set<T> set_sym_diff(std::set<T> a, std::set<T> b) {
	std::set<T> res{};
	std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(res, res.begin()));
	return res;
}

template<typename T>
std::set<T> set_diff(std::set<T> a, std::set<T> b) {
	std::set<T> res{};
	std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(res, res.begin()));
	return res;
}

template<typename T, typename U>
std::set<T> map_key_set(const std::map<T, U>& map) {
	std::set<T> res{};
	for (auto it = map.begin(); it != map.end(); ++it) {
		res.insert(it->first);
	}
	return res;
}

template<typename T, typename U>
std::vector<T> map_key_list(const std::map<T, U>& map) {
	std::vector<T> res{};
	for (auto it = map.begin(); it != map.end(); ++it) {
		res.push_back(it->first);
	}
	return res;
}

template<typename T, typename U>
std::set<T> map_key_set(const std::unordered_map<T, U>& map) {
	std::set<T> res{};
	for (auto it = map.begin(); it != map.end(); ++it) {
		res.insert(it->first);
	}
	return res;
}

template<typename T, typename U>
std::vector<T> map_key_list(const std::unordered_map<T, U>& map) {
	std::vector<T> res{};
	for (auto it = map.begin(); it != map.end(); ++it) {
		res.push_back(it->first);
	}
	return res;
}

template<typename T, typename U>
std::unordered_map<T, U> invert_map(const std::unordered_map<U, T>& map) {
	std::unordered_map<T, U> res{};
	for (const auto& [key, value] : map) {
		res[value] = key;
	}
	return res;
}

template<typename T, typename U>
std::unordered_map<T, std::vector<U>> invert_map_vec(const std::unordered_map<U, std::vector<T>>& map) {
	std::unordered_map<T, std::vector<U>> res{};
	for (const auto& [key, values] : map) {
		for (const auto& value : values) {
			auto it = res.find(value);
			if (it == res.end()) {
				res.emplace(value, std::vector<U>{key});
			} else {
				it->second.push_back(key);
			}
		}
	}
	return res;
}

template<typename T, typename U>
std::map<T, U> invert_map(const std::map<U, T>& map) {
	std::map<T, U> res{};
	for (const auto& [key, value] : map) {
		res[value] = key;
	}
	return res;
}

template<typename T, typename U>
std::map<T, std::vector<U>> invert_map_vec(const std::map<U, std::vector<T>>& map) {
	std::map<T, std::vector<U>> res{};
	for (const auto& [key, values] : map) {
		for (const auto& value : values) {
			auto it = res.find(value);
			if (it == res.end()) {
				res.emplace(value, std::vector<U>{key});
			} else {
				it->second.push_back(key);
			}
		}
	}
	return res;
}

std::vector<int> diffs(const std::vector<int>& vec) {
	std::vector<int> res{};
	for (int i = 0; i < vec.size()-1; ++i) {
		res.push_back(vec[i+1] - vec[i]);
	}
	return res;
}

std::vector<int> diffs(const std::vector<int>& vec1, const std::vector<int>& vec2) {
	std::vector<int> res{};
	for (int i = 0; i < min(vec1.size(), vec2.size()); ++i) {
		res.push_back(vec2[i] - vec1[i]);
	}
	return res;
}

std::vector<std::string> rotate90c(const std::vector<std::string>& matrix) {
	int n = matrix.size();
	auto res = std::vector<std::string>{};

	for (int i = 0; i < n; ++i) {
		res.emplace_back("");
		for (int j = 0; j < n; ++j) {
			res[i] += matrix[n-j-1][i];
		}
	}

	return res;
}

template<typename T>
std::vector<std::vector<T>> rotate90c(const std::vector<std::vector<T>>& matrix) {
	int n = matrix.size();
	auto res = std::vector<std::vector<T>>{};
	for (int i = 0; i < n; ++i) {
		res.emplace_back(std::vector<T>{});
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			res[i].emplace_back(matrix[n-j-1][i]);
		}
	}

	return res;
}

template<typename T>
std::vector<std::vector<T>> rotate90cc(const std::vector<std::vector<T>>& matrix) {
	int n = matrix.size();
	auto res = std::vector<std::vector<T>>{};
	for (int i = 0; i < n; ++i) {
		res.emplace_back(std::vector<T>{});
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			res[i].emplace_back(matrix[j][n-i-1]);
		}
	}

	return res;
}

bool inbounds(int x, int y, int w, int h) {
	return x >= 0 && y >= 0 && x < w && y < h;
}

bool inbounds(int x, int y, int w, int h, int bw, int bh) {
	return x >= 0 && y >= 0 && x < w-bw+1 && y < h-bh+1;
}

Vec2i arrow_dir(char c) {
	switch (c) {
		case '^': return {0, -1};
		case 'v': return {0, 1};
		case '>': return {1, 0};
		case '<': return {-1, 0};
		default: Logger::critical("Unrecognized arrow direction '{}'", c);
	}
}

template<typename T, typename U>
struct std::hash<std::pair<T, U>> {
	size_t operator()(const std::pair<T, U>& pair) const noexcept {
		return std::hash<T>()(pair.first) ^ std::hash<U>()(pair.second);
	}
};

template<typename... Ts>
struct std::hash<std::tuple<Ts...>> {
	size_t operator()(const std::tuple<Ts...>& tuple) const noexcept {
		return std::apply([](const auto& ... xs){ return (std::hash<Ts>()(xs) ^ ...); }, tuple);
	}
};

template<typename T, size_t Size>
struct std::hash<std::array<T, Size>> {
	size_t operator()(const std::array<T, Size>& arr) const noexcept {
		size_t result = 0;
		for (size_t i = 0; i < Size; ++i) {
			result ^= std::hash<T>()(arr[i]);
		}
		return result;
	}
};

template<typename T>
struct std::hash<std::vector<T>> {
	size_t operator()(const std::vector<T>& arr) const noexcept {
		size_t result = 0;
		for (const auto& e : arr) {
			result ^= std::hash<T>()(e);
		}
		return result;
	}
};

template<typename T>
struct std::hash<std::unordered_set<T>> {
	size_t operator()(const std::unordered_set<T>& arr) const noexcept {
		size_t result = 0;
		for (const auto& e : arr) {
			result ^= std::hash<T>()(e);
		}
		return result;
	}
};

inline int num_len(long long n) {
	return static_cast<int>(std::log10(n)) + 1;
}

template<typename T>
T mod_math(T a, T b) {
	return (a % b + b) % b;
}

template<typename T>
constexpr std::optional<int> leading_zeros(T n) {
	if (n == 0) return std::nullopt;
	int i = sizeof(T) * 8 -1;
	while (((1 << i) & n) == 0) --i;
	return i;
}

#endif //UTILS_H
