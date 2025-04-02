// Repository: cedretaber/d_emotion
// File: emotion.d

import mecab;
import std.algorithm;
import std.array;
import std.string;
import std.math : log;
import std.conv : to;
import std.stdio;
import std.format;
import std.file;
import std.range;

enum Parts { NOUN, VERB, ADJECTIVE }

struct Dic
{
    Word[string] words;

    alias words this;

    static struct Word
    {
        string word;
        string yomi;
        Parts part;
        double point;
    }
}

struct Word
{
    string word;
    string normal;
    Parts part;
}

Parts dictPart(string word)
{
    Parts part;
    switch(word)
    {
        case "名詞":
            part = Parts.NOUN;
            break;
        case "動詞":
            part = Parts.VERB;
            break;
        case "形容詞":
            part = Parts.ADJECTIVE;
            break;
        default:
            break;
  }
  return part;
}

Dic mkDic(string dicFileName)
{
    auto dicFile = File(dicFileName, "r");
    Dic dic;

    foreach(line; dicFile.byLine)
    {
        auto es = line.strip.split(":").map!(to!string).array;

        dic[es[0]] = Dic.Word(es[0], es[1], es[2].dictPart, es[3].to!double);
    }

    return dic;
}

Word[] textToWord(string text, mecab_t * mecab)
{
    Word[] words;
    auto node = mecab_sparse_tonode(mecab, text.toStringz);

    for(; node; node = node.next)
    {
        if(node.stat != MECAB_NOR_NODE && node.stat != MECAB_UNK_NODE) continue;
        switch(node.posid)
        {
            case 10: .. case 12: case 31: .. case 33: case 36: .. case 47:
                auto data = node.feature.to!string.split(",").array;
                words ~= Word(node.surface.to!string.strip, data[6].to!string, data[0].dictPart);
                break;
            default:
                break;
        }
    }

    return words;
}

void main(string[] args)
{
    if(args.length < 2) return;

    auto mecab = mecab_new2("");
    scope(exit) mecab_destroy(mecab);

    auto dic = mkDic("./pn_ja.dic");

    auto words = textToWord(args[1], mecab);

    double sumPoints = 0;
    foreach(word; words)
    {
        double point = 0;
        auto wd = (word.normal in dic);
        if(wd !is null)
            point = wd.point;

        sumPoints += point;
    }

    writeln(sumPoints);
}
