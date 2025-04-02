// Repository: VerbalExpressions/DVerbalExpressions
// File: verex.d

/**
 * Verbal Expressions v0.1 ported to D from JavaScript version.
 * 
 * https://github.com/VerbalExpressions
 * 
 * @author Diego Lago <diego.lago.gonzalez@gmail.com>
 * @version 0.1
 * @date 2013-08-09
 * 
 * The MIT License (MIT)
 * 
 * Copyright (c) 2013 whackashoe
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 */

module verex;

import std.regex;
import std.string;

debug {
	import std.stdio;
}

class VerEx {
	
	private:
	
		string		prefixes;
		string		source;
		string		suffixes;
		string		pattern;
		
		enum Flags {
			Global			= 1,
			Multiline		= 2,
			CaseInsensitive	= 4
		}
		
		string createFlags(string append = "") {
			return (modifiers & Flags.CaseInsensitive ? "i" : "") ~ append;
		}
		
		string reduceLines(string value) {
			ptrdiff_t index = indexOf(value, '\n');
			return (index == -1 ? value : value[0..index]);
		}
		
		string sanitize(string value) {
			// Is the use of this method really needed using D regular expressions?
			return std.regex.replace(value, regex("[^\\w]", "g"), `\$&`);
		}
		
	public:
	
		uint modifiers;
		
		this() {}
		
		~this() {}
		
		ref VerEx add(string value) {
			source ~= value;
			pattern = prefixes ~ source ~ suffixes;
			debug(verbose) writefln("prefixes=%s — source=%s — suffixes=%s — pattern=%s", prefixes, source, suffixes, pattern);
			debug writefln("pattern=%s", pattern);
			return this;
		}
		
		ref VerEx startOfLine(bool enable = true) {
			prefixes = enable ? "^" : "";
			return add("");
		}
		
		ref VerEx endOfLine(bool enable = true) {
			suffixes = enable ? "$" : "";
			return add("");
		}
		
		ref VerEx then(string value) {
			return add("(?:" ~ value ~ ")");
		}
		
		ref VerEx find(string value) {
			return then(value);
		}
		
		ref VerEx maybe(string value) {
			return add("(?:" ~ value ~ ")?");
		}
		
		ref VerEx anything() {
			return add("(?:.*)");
		}
		
		ref VerEx anythingBut(string value) {
			return add("(?:[^" ~ value ~ "]*)");
		}
		
		ref VerEx something() {
			return add("(?:.+)");
		}
		
		ref VerEx somethingBut(string value) {
			return add("(?:[^" ~ value ~ "]+)");
		}
		
		string replace(string source, string value) {
			return std.regex.replace(source, std.regex.regex(pattern, createFlags()), value);
		}
		
		ref VerEx lineBreak() {
			return add("(?:(?:\\n)|(?:\\r\\n))");
		}
		
		ref VerEx br() {
			return lineBreak();
		}
		
		ref VerEx tab() {
			return add("\\t");
		}
		
		ref VerEx word() {
			return add("\\w+");
		}
		
		ref VerEx anyOf(string value) {
			return add("[" ~ value ~ "]");
		}
		
		ref VerEx any(string value) {
			return anyOf(value);
		}
		
		ref VerEx range(string[] args) {
			
			if(args.length <= 0 || args.length % 2 != 0) {
				// We do not raise an error if there are no arguments or there are no paired arguments.
				// We simply return this. Is this correct?
				return this;
			}
			
			string value = "[";
			
			for(int from = 0, to = from + 1; from < args.length; from += 2, to += 2) {
				value ~= args[from] ~ "-" ~ args[to];
			}
			
			value ~= "]";
			
			add(value);
			
			return this;
		}
		
		ref VerEx addModifier(char modifier) {
			switch (modifier) {
				case 'i':
					modifiers |= Flags.CaseInsensitive;
					break;
				case 'm':
					modifiers |= Flags.Multiline;
					break;
				case 'g':
					modifiers |= Flags.Global;
					break;
				default:
					break;
			}

			return this;
		}
		
		ref VerEx removeModifier(char modifier) {
			switch (modifier) {
				case 'i':
					modifiers ^= Flags.CaseInsensitive;
					break;
				case 'm':
					modifiers ^= Flags.Multiline;
					break;
				case 'g':
					modifiers ^= Flags.Global;
					break;
				default:
					break;
			}

			return this;
		}
	
		ref VerEx withAnyCase(bool enable = true) {
			enable ? addModifier('i') : removeModifier('i');
			return this;
		}
		
		ref VerEx searchOneLine(bool enable = true) {
			enable ? addModifier('m') : removeModifier('m');
			return this;
		}
		
		ref VerEx searchGlobal(bool enable = true) {
			enable ? addModifier('g') : removeModifier('g');
			return this;
		}
		
		ref VerEx multiple(string value) {
			if(value.length > 0 && value[0] != '*' && value[0] != '+') {
				add("+");
			}
			return add(value);
		}
		
		ref VerEx or(string value) {
			if(indexOf(prefixes, '(') == -1) {
				prefixes ~= "(";
			}
			if(indexOf(suffixes, ')') == -1) {
				suffixes = suffixes ~ ")";
			}
			add(")|(");
			return then(value);
		}
		
		ref VerEx beginCapture() {
			suffixes ~= ")";
			add("(");
			return this;
		}
		
		ref VerEx endCapture() {
			suffixes = suffixes[0..$-1];
			add(")");
			return this;
		}
		
		bool test(string value) {
			string toTest = modifiers & Flags.Multiline ? value : reduceLines(value);
			return cast(bool)std.regex.match(toTest, std.regex.regex(pattern, createFlags(modifiers & Flags.Global ? "g" : "")));
		}
		
		override string toString() {
			return pattern;
		}
}

unittest {
	
	auto e = (new VerEx()).searchOneLine()
			.startOfLine()
			.then("http")
			.maybe("s")
			.then("://")
			.maybe("www.")
			.anythingBut(" ")
			.endOfLine();
			
	assert(e.toString() == "^(?:http)(?:s)?(?:://)(?:www.)?(?:[^ ]*)$");
			
	assert(e.test("http://www.google.es"));
	assert(e.test("https://gmail.com"));
	assert(!e.test("invalid.url.com"));
	
	assert((new VerEx()).find("red").replace("We have a red house", "blue") == "We have a blue house");
	
}
