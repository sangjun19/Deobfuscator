// Repository: pvmoore/x86_64_disassembler
// File: src/disassembler/parse/parse_3dnow.d

module disassembler.parse.parse_3dnow;

import disassembler.all;

void parse3dnow(Parser p, uint loNibble) {
    switch(loNibble) {
        case 0xD:
        case 0xE:
        case 0xF:
            break;
        default: break;
    }
    todo();
}

private:
