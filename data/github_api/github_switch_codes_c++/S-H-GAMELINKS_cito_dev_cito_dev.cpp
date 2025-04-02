#include <iostream>
#include <string>
#include "cpp-linenoise/linenoise.hpp"
#include "cpp-peglib/peglib.h"
//#include "cparse/shunting-yard.h"

int main() {

    std::cout << "Cito dev version 0.0.0.2" << std::endl;

    std::string cmd;

    const auto path = "history.txt";

    linenoise::SetMultiLine(true);

    linenoise::SetHistoryMaxLen(20);

    linenoise::LoadHistory(path);

    auto reduce = [](const peg::SemanticValues& sv) -> long {
        auto result = sv[0].get<long>();
        for (auto i = 1u; i < sv.size(); i += 2) {
            auto num = sv[i + 1].get<long>();
            auto ope = sv[i].get<char>();
            switch (ope) {
                case '+': result += num; break;
                case '-': result -= num; break;
                case '*': result *= num; break;
                case '/': result /= num; break;
            }
        }
        return result;
    };

    peg::parser parser(R"(
        EXPRESSION       <-  _ TERM (TERM_OPERATOR TERM)*
        TERM             <-  FACTOR (FACTOR_OPERATOR FACTOR)*
        FACTOR           <-  NUMBER / '(' _ EXPRESSION ')' _
        TERM_OPERATOR    <-  < [-+] > _
        FACTOR_OPERATOR  <-  < [/*] > _
        NUMBER           <- '-'? [0-9]+ ('.' [0-9]+)?
        ~_               <-  [ \t\r\n]*
    )");

    parser["EXPRESSION"]      = reduce;
    parser["TERM"]            = reduce;
    parser["TERM_OPERATOR"]   = [](const peg::SemanticValues& sv) { return static_cast<char>(*sv.c_str()); };
    parser["FACTOR_OPERATOR"] = [](const peg::SemanticValues& sv) { return static_cast<char>(*sv.c_str()); };
    parser["NUMBER"]          = [](const peg::SemanticValues& sv) { return atol(sv.c_str()); };

    long val = 0;
    bool quit = false;
    
    while(true) {
        quit = linenoise::Readline("cito > ", cmd);

        if (quit)
            break;

        if (parser.parse(cmd.c_str(), val)) {
            std::cout << val << std::endl;
            linenoise::AddHistory(cmd.c_str());
        }
    }

    linenoise::SaveHistory(path);

    return 0;
}