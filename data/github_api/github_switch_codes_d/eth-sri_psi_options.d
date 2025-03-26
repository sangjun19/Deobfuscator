// Repository: eth-sri/psi
// File: options.d

// Written in the D programming language
// License: http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0

enum Format{
	default_,
	gnuplot,
	matlab,
	maple,
	mathematica,
	python,
	sympy,
	lisp,
}

enum ErrorFormat{
	default_,
	json,
}

enum InferenceMethod{
	symbolic,
	dp,
	simulate,
}

enum OutputForm{
	default_,
	raw,
	rawError,
}

enum IntegrationLevel{
	full,
	deltas,
	none,
}

struct Options{
	InferenceMethod backend;
	bool plot=false;
	string plotRange="[-10:10]";
	string plotFile="";
	bool cdf=false;
	bool kill=false;
	auto formatting=Format.default_;
	auto errorFormat=ErrorFormat.default_;
	bool casBench=false;
	bool noBoundsCheck=false;
	bool noCheck=false;
	bool noNormalize=false;
	bool trace=false;
	bool expectation=false;
	IntegrationLevel integrationLevel=IntegrationLevel.full;
	OutputForm outputForm;
	string[] importPath;
	string[] summarize;
	ulong numSimulations=1;
	bool dexpr=false;
}
Options opt; // TODO: get rid of global?

struct Expected{
	bool exists;
	bool todo;
	string ex;

	void parse(T,L)(string dexp,T err,L loc){
		import sym.dparse;
		bool todo=false;
		import std.string: strip, startsWith;
		auto ex=dexp.strip;
		if(ex.startsWith("TODO:")){
			todo=true;
			ex=ex["TODO:".length..$].strip;
		}
		if(!exists){
			this=Expected(true,todo,ex);
		}else if(this != Expected(true,todo,ex))
			err.error("can only have one 'expected' annotation, in 'main'.",loc);
	}
}
Expected expected;
string casExt(Format formatting=opt.formatting){
	final switch(formatting) with(Format){
		case default_: return "dexpr";
		case gnuplot: return "gnu";
		case matlab: return "m";
		case mathematica: return "m";
		case maple: return "mpl";
		case python: return "py";
		case sympy: return "py";
		case lisp: return "lisp";
	}
}
