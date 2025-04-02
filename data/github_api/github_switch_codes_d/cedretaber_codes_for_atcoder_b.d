// Repository: cedretaber/codes_for_atcoder
// File: arc/120/b.d

import std.stdio, std.algorithm, std.conv, std.array, std.string, std.math, std.typecons, std.numeric, std.container, std.range;

void get(Args...)(ref Args args)
{
    import std.traits, std.meta, std.typecons;

    static if (Args.length == 1) {
        alias Arg = Args[0];
        
        static if (isArray!Arg) {
          static if (isSomeChar!(ElementType!Arg)) {
            args[0] = readln.chomp.to!Arg;
          } else {
            args[0] = readln.split.to!Arg;
          }
        } else static if (isTuple!Arg) {
            auto input = readln.split;
            static foreach (i; 0..Fields!Arg.length) {
                args[0][i] = input[i].to!(Fields!Arg[i]);
            }
        } else {
            args[0] = readln.chomp.to!Arg;
        }
    } else {
        auto input = readln.split;
        assert(input.length == Args.length);

        static foreach (i; 0..Args.length) {
            args[i] = input[i].to!(Args[i]);
        }
    }
}

void get_lines(Args...)(size_t N, ref Args args)
{
    import std.traits, std.range;

    static foreach (i; 0..Args.length) {
        static assert(isArray!(Args[i]));
        args[i].length = N;
    }

    foreach (i; 0..N) {
        static if (Args.length == 1) {
            get(args[0][i]);
        } else {
            auto input = readln.split;
            static foreach (j; 0..Args.length) {
                args[j][i] = input[j].to!(ElementType!(Args[j]));
            }
        }
    }
}

enum P = 998244353L;

void main()
{
    int H, W; get(H, W);
    char[][] BB; get_lines(H, BB);

    int i, j;
    long res = 1;
    for (;;) {
        bool r, b;
        int ii = i, jj = j;
        while (0 <= ii && jj < W) {
            switch (BB[ii][jj]) {
                case 'R': r = true; break;
                case 'B': b = true; break;
                default: 
            }
            --ii;
            ++jj;
        }
        if (r && b) {
            return writeln(0);
        } else if (!r && !b) {
            (res *= 2) %= P;
        }
        if (i + 1 < H) {
            ++i;
        } else {
            ++j;
        }
        if (j == W) break;
    }
    writeln(res);
}
