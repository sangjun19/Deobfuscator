// Repository: UIMSolutions/uim
// File: lowlevel/consoles/uim/consoles/classes/consoles/optionparser.d

/****************************************************************************************************************
* Copyright: © 2018-2025 Ozan Nurettin Süel (aka UIManufaktur)                                                  *
* License: Subject to the terms of the Apache 2.0 license, as written in the included LICENSE.txt file.         *
* Authors: Ozan Nurettin Süel (aka UIManufaktur)                                                                *
*****************************************************************************************************************/
module uim.consoles.classes.consoles.optionparser;

import uim.consoles;
@safe:

version (test_uim_consoles) {
    unittest {
        writeln("-----  ", __MODULE__, "\t  -----");
    }
}

/**
 * Handles parsing the ARGV in the command line and provides support
 * for GetOpt compatible option definition. Provides a builder pattern implementation
 * for creating shell option parsers.
 *
 * ### Options
 *
 * Named arguments come in two forms, long and short. Long arguments are preceded
 * by two - and give a more verbose option name. i.e. `--version`. Short arguments are
 * preceded by one - and are only one character long. They usually match with a long option,
 * and provide a more terse alternative.
 *
 * ### Using Options
 *
 * Options can be defined with both long and short forms. By using ` aParser.addOption()`
 * you can define new options. The name of the option is used as its long form, and you
 * can supply an additional short form, with the `short` option. Short options should
 * only be one letter long. Using more than one letter for a short option will raise an exception.
 *
 * Calling options can be done using syntax similar to most *nix command line tools. Long options
 * cane either include an `=` or leave it out.
 *
 * `uim _command --connection default --name=something`
 *
 * Short options can be defined singly or in groups.
 *
 * `uim _command -cn`
 *
 * Short options can be combined into groups as seen above. Each letter in a group
 * will be treated as a separate option. The previous example is equivalent to:
 *
 * `uim _command -c -n`
 *
 * Short options can also accept values:
 *
 * `uim _command -c default`
 *
 * ### Positional arguments
 *
 * If no positional arguments are defined, all of them will be parsed. If you define positional
 * arguments any arguments greater than those defined will cause exceptions. Additionally you can
 * declare arguments as optional, by setting the required param to false.
 *
 * ```
 * aParser.addArgument("model", ["required": false]);
 * ```
 *
 * ### Providing Help text
 *
 * By providing help text for your positional arguments and named arguments, the DConsoleOptionParser buildOptionParser
 * can generate a help display for you. You can view the help for shells by using the `--help` or `-h` switch.
 */
class DConsoleOptionParser : UIMObject, IConsoleOptionParser {
  this() {
    super();
  }

  this(Json[string] initData) {
    super(initData);
  }

  // MapHelper of short ~ long options, generated when using addOption()
  protected STRINGAA _shortOptions;

  // #region description
  // Sets the description text for shell/task.
  protected string _description;
  string description() {
    return _description;
  }

  IConsoleOptionParser description(string[] descriptions...) {
    description(descriptions.dup);
    return this;
  }

  IConsoleOptionParser description(string[] descriptions) {
    _description = descriptions.join("\n");
    return this;
  }
  // #endregion description

    // Option definitions.
  protected DInputOptionConsole[string] _options;

  //  Positional argument definitions.
  protected DInputArgument[] _arguments;

  // Array of args (arguments).
  protected Json[string] _token;

  // #region rootName
  // Root alias used in help output
  protected string _rootName = "uim";
  // Set the root name used in the HelpFormatter
  @property rootName(string aName) {
    _rootName = name;
  }
  // #endregion rootName

  // #region epilog
  /**
        * Sets an epilog to the parser. The epilog is added to the end of
        * the options and arguments listing when help is generated. */
  mixin(TProperty!("string", "epilog"));

  @property void epilog(string[] texts) {
    epilog(texts.join("\n"));
  }
  // #endregion epilog

  // #region command
  // Command name.
  protected string _command = "";

  // Sets the command name for shell/task.
  void setCommand(string commandName) {
    _command = commandName.underscore;
  }

  // Gets the command name for shell/task.
  string getCommand() {
    return _command;
  }
  // #endregion command

  this(string commandName = "", bool isVerboseAndQuiet = true) {
    super();
    /* setCommand(command);

        addOption("help", MapHelper.create!(string, Json)
                .set("short", "h")
                .set("help", "Display this help.")
                .set("boolean", true));

        if (isVerboseAndQuiet) {
            addOption("verbose", MapHelper.create!(string, Json)
                    .set("short", "v")
                    .set("help", "Enable verbose output.")
                    .set("boolean", true));

            addOption("quiet", MapHelper.create!(string, Json)
                    .set("short", "q")
                    .set("help", "Enable quiet output.")
                    .set("boolean", true));
        } */
  }

  // Static factory method for creating new DOptionParsers so you can chain methods off of them.
  /* static auto create(string commandName, bool useDefaultOptions = true) {
        /* return new static(commandName, useDefaultOptions); * /
        return this;
    } */

  /**
     * Build a parser from an array. Uses an array like
     *
     * ```
     * spec = [
     *    "description": "text",
     *    "epilog": "text",
     *    "arguments": [
     *        // list of arguments compatible with addArguments.
     *    ],
     *    "options": [
     *        // list of options compatible with addOptions
     *    ]
     * ];
     * ```
     * Params:
     * Json[string] sepcData The sepcData to build the OptionParser with.
     */
  static auto buildFromArray(Json[string] sepcData, bool isVerboseAndQuiet = true) {
    /* auto aParser = new static(sepcData["command"], isVerboseAndQuiet);
        if (!sepcData.isEmpty("arguments")) {
            aParser.addArguments(sepcData["arguments"]);
        }
        if (!sepcData.isEmpty("options")) {
            aParser.addOptions(sepcData["options"]);
        }
        if (!sepcData.isEmpty("description") {
            aParser.description(sepcData["description"]);
        }
        if (!sepcData.isEmpty("epilog")) {
            aParser.epilog(sepcData["epilog"]);
        }
        return aParser; */
    return null;
  }

  // Returns an array representation of this parser.
  Json[string] toArray() {
    return [
      /* "command": Json(_command), */
      /* "arguments": Json(_arguments), */
      /* "options": Json(_options), */
      "description": Json(_description),
      "epilog": Json(_epilog),
    ];
  }

  // Get or set the command name for shell/task.
  IConsoleOptionParser merge(IConsoleOptionParser buildOptionParser) {
    /*  merge(spec.toJString()); */
    return this; 
  }

  IConsoleOptionParser merge(Json[string] options) {
    /* if (!options.isEmpty("arguments")) {
      addArguments(options.get("arguments"));
    }
    if (!options.isEmpty("options")) {
      addOptions(options.get("options"));
    }
    if (!options.isEmpty("description")) {
      description(options.get("description"));
    }
    if (!options.isEmpty("epilog")) {
      epilog(options.get("epilog"));
    } */

    return this; 
  }

  /**
     * Add an option to the option parser. Options allow you to define optional or required
     * parameters for your console application. Options are defined by the parameters they use.
     *
     * ### Options
     *
     * - `short` - The single letter variant for this option, leave undefined for none.
     * - `help` - Help text for this option. Used when generating help for the option.
     * - `default` - The default value for this option. Defaults are added into the parsed params when the
     *  attached option is not provided or has no value. Using default and boolean together will not work.
     *  are added into the parsed parameters when the option is undefined. Defaults to null.
     * - `boolean` - The option uses no value, it`s just a boolean switch. Defaults to false.
     *  If an option is defined as boolean, it will always be added to the parsed params. If no present
     *  it will be false, if present it will be true.
     * - `multiple` - The option can be provided multiple times. The parsed option
     * will be an array of values when this option is enabled.
     * - `choices` A list of valid choices for this option. If left empty all values are valid..
     * An exception will be raised when parse() encounters an invalid value.
     * Params:
     * \UIM\Console\InputOptionConsole|string aName The long name you want to the value to be parsed out
     * as when options are parsed. Will also accept an instance of InputOptionConsole.
     * options An array of parameters that define the behavior of the option
     */
  void addOption(string optionName, Json[string] behaviorOptions = null) {
    behaviorOptions = behaviorOptions
      .merge("short", "")
      .merge("help", "")
      .merge("boolean", false)
      .merge("multiple", false)
      .merge("required", false)
      .merge("choices", Json.emptyArray)
      .merge("default", Json(null))
      .merge("prompt", Json(null));

    /* auto inputOption = new DInputOptionConsole(
            name,
            behaviorOptions.getMap("short", "help", "boolean", "default", "choices", "multiple", "required", "prompt")
        ); */
  }

  // TODO 
  /* addOption(inputOption, behaviorOptions) {
    } */

  void addOption(DInputOptionConsole inputOption, Json[string] behaviorOptions = null) {
    string optionName = inputOption.name();

    /* _options.set(optionName, inputOption);
        asort(_options);
        if (inputOption.short()) {
            _shortoptions.get(inputOption.short()] = optionName;
            asort(_shortOptions);
        } */
  }

  // Remove an option from the option parser.
  void removeOption(string name) {
    _options.removeKey(name);
  }

  /**
     * Add a positional argument to the option parser.
     *
     * ### Params
     *
     * - `help` The help text to display for this argument.
     * - `required` Whether this parameter is required.
     * - `index` The index for the arg, if left undefined the argument will be put
     * onto the end of the arguments. If you define the same index twice the first
     * option will be overwritten.
     * - `choices` A list of valid choices for this argument. If left empty all values are valid..
     * An exception will be raised when parse() encounters an invalid value.
     * Params:
     * \UIM\Console\InputConsoleArgument|string aName The name of the argument.
     * Will also accept an instance of InputConsoleArgument.
     */
  IConsoleOptionParser addArgument(DInputArgument argument, Json[string] options = new Json[string]) {
    // TODO
    return this;
  }

  IConsoleOptionParser addArgument(string argName, Json[string] options = new Json[string]) {
    Json[string] defaultOptions;
      defaultOptions.set("name", argName);
      defaultOptions.set("help", "");
      defaultOptions.set("index", _arguments.length);
      defaultOptions.set("required", false);
      defaultOptions.set("choices", Json.emptyArray);

    auto newParams = options.merge(defaultOptions);
    auto anIndex = newParams.shift("index");
    auto inputArgument = new DInputArgument(newParams);

    /* _arguments.each!((arg) {
      if (arg.isEqualTo(inputArgument)) {
        return;
      }
      if (options.hasKey("required") && !arg.isRequired()) {
        throw new DLogicException("A required argument cannot follow an optional one");
      }
    }); */

    // TODO _arguments.set(anIndex, arg);
    // TODO ksort(_arguments);
    return this;
  }

  /**
     * Add multiple arguments at once. Take an array of argument definitions.
     * The keys are used as the argument names, and the values as params for the argument.
     * Params:
     * array<string, Json[string]|\UIM\Console\InputConsoleArgument> someArguments Array of arguments to add.
     */
  void addArguments(Json[string] someArguments) {
    foreach (name, params; someArguments) {
      /* if (cast(DInputArgument) params) {
                name = params;
                params = null;
            }
            this.addArgument(name, params); */
    }
  }

  /**
     * Add multiple options at once. Takes an array of option definitions.
     * The keys are used as option names, and the values as params for the option.
     */
  void addOptions(Json[string] optionsToAdd = null) {
    foreach (name, params; optionsToAdd) {
      /*      if (cast(DInputOptionConsole) params) {
                name = params;
                params = null;
            }
            this.addOption(name, params); */
    }
  }

  // Gets the arguments defined in the parser.
  DInputArgument[] arguments() {
    return _arguments;
  }

  // Get the list of argument names.
  string[] argumentNames() {
    /* auto results = _arguments.map(arg => arg.name()).array;
        return results; */
    return null;
  }

  // Get the defined options in the parser.
  DInputOptionConsole[string] options() {
    /* return _options; */
    return null;
  }

  // Parse the arguments array into a set of params and args.
  Json[string] parse(Json[string] arguments, DConsole console = null) {
    // _tokens = arguments;
    /* auto params = someArguments = null;

        bool afterDoubleDash = false;
        while ((token = _tokens.shift()) !is null) {
            token = to!string(token);
            if (token == "--") {
                afterDoubleDash = true;
                continue;
            }
            if (afterDoubleDash) {
                // only positional arguments after --
                someArguments = _parseArg(token, someArguments);
                continue;
            }
            if (token.startsWith("--")) {
                params = _parseLongOption(token, params);
            } else if (token.startsWith("-")) {
                params = _parseShortOption(token, params);
            } else {
                someArguments = _parseArg(token, someArguments);
            }
        }
        if (params.hasKey("help")) {
            return [params, someArguments];
        }
        foreach (index, arg; arguments) {
            if (arg.isRequired() && !someArguments.has(index)) {
                throw new DConsoleException(
                    "Missing required argument. The `%s` argument is required.".format(arg.name())
                );
            }
        }
        _options.each!((option) {
            auto name = option.name();
            auto isBoolean = option.isBoolean();
            auto defaultValue = option.defaultValue();

            auto useDefault = !params.hasKey(name);
            if (defaultValue !is null && useDefault && !isBoolean) {
                params.set(name, defaultValue);
            }
            if (isBoolean && useDefault) {
                params.set(name, false);
            }
            auto prompt = option.prompt();
            if (!params.hasKey(name) && prompt) {
                if (!console) {
                    throw new DConsoleException(
                        "Cannot use interactive option prompts without a Console instance. " ~
                        "Please provide a ` console` parameter to `parse()`."
                    );
                }
                auto choices = option.choices();

                auto aValue = choices
                    ? console.askChoice(prompt, choices) : console.ask(prompt);

                params[name] = aValue;
            }
            if (option.isRequired() && !params.hasKey(name)) {
                throw new DConsoleException(
                    "Missing required option. The `%s` option is required and has no default value."
                    .format(name)
                );
            }
        });
        return [params, someArguments]; */
    return null;
  }

  /**
     * Gets formatted help for this parser object.
     *
     * Generates help text based on the description, options, arguments and epilog
     * in the parser.
     */
  string help(string outputFormat = "text", int formatWidth = 72) {
    /* auto formatter = new DHelpFormatter(this);
        formatter.aliasName(_rootName);

        if (outputFormat == "text") {
            return formatter.text(formatWidth);
        }
        if (outputFormat == "xml") {
            return to!string(formatter.xml());
        }
        throw new DConsoleException("Invalid format. Output format can be text or xml."); */
    return null;
  }

  /**
     * Parse the value for a long option out of _tokens. Will handle
     * options with an `=` in them.
     */
  protected Json[string] _parseLongOption(string optionToParse, Json[string] paramsData) {
    string name = subString(optionToParse, 2);
    /* if (name.contains("=")) {
            [name, aValue] = split("=", name, 2);
            _tokens.unshift(aValue);
        } */
    return _parseOption(name, paramsData);
  }

  /**
     * Parse the value for a short option out of _tokens
     * If the option is a combination of multiple shortcuts like -otf
     * they will be shifted onto the token stack and parsed individually.
     */
  protected Json[string] _parseShortOption(string optionToParse, Json[string] paramsToAppen) {
    /* string aKey = subString(optionToParse, 1);
        if (aKey.length > 1) {
            string[] flags = aKey.split;
            aKey = flags[0];
            for (index = 1, len = count(flags); index < len; index++) {
                _tokens.unshift("-" ~ flags[index]);
            }
        }
        if (!_shortOptions.hasKey(aKey)) {
            auto options = _shortOptions.byKeyValue
                .map!(shortLong => shortLong.key ~ " (short for `--" ~ shortLong.value ~ "`)");

            throw new DMissingOptionException(
                "Unknown short option `%s`.".format(aKey),
                aKey, options
            );
        }

        auto name = _shortOptions.getString(aKey);
        return _parseOption(name, paramsToAppen); */
    return null;
  }

  /**
     * Parse an option by its name index.
     * Params:
     * params The params to append the parsed value into
     * returns Params with option added in.
     */
  protected Json[string] _parseOption(string nameToParse, Json[string] params) {
    /* if (!_options.hasKey(nameToParse)) {
            throw new DMissingOptionException(
                "Unknown option `%s`.".format(nameToParse), nameToParse, _options.keys
            );
        }
        auto option = _options.get(nameToParse);
        auto isBoolean = option.isBoolean();
        auto nextValue = _nextToken();
        auto emptyNextValue = (isEmpty(nextValue) && nextValue != "0");
        if (!isBoolean && !emptyNextValue && !_optionhasKey(nextValue)) {
            _tokens.shift();
            aValue = nextValue;
        } else if (isBoolean) {
            aValue = true;
        } else {
            aValue = to!string(option.defaultValue());
        }
        option.validChoice(aValue);

        if (option.acceptsMultiple()) {
            params.append(nameToParse, aValue);
        } else {
            params.set(nameToParse, aValue);
        }
        return params; */
    return null;
  }

  // Check to see if name has an option (short/long) defined for it.
  protected bool _optionhasKey(string optionName) {
    /*         if (optionName.startsWith("--")) {
            return _options.hasKey(subString(optionName, 2));
        }
        if (optionName[0] == "-" && optionName[1] != "-") {
            return _shortOptions.hasKey(optionName[1]);
        }
 */
    return false;
  }

  /**
     * Parse an argument, and ensure that the argument doesn`t exceed the number of arguments
     * and that the argument is a valid choice.
     */
  protected string[] _parseArg(string argumentToAppend, Json[string] someArguments) {
    /* if (_arguments.isEmpty) {
            someArguments ~= argumentToAppend;
            return someArguments;
        }

        auto next = count(someArguments);
        if (!_arguments.hasKey(next)) {
            auto expected = count(_arguments);
            throw new DConsoleException(
                "Received too many arguments. Got `%s` but only `%s` arguments are defined."
                    .format(next, expected)
            );
        }
        _arguments[next].validChoice(argument);
        someArguments ~= argument;

        return someArguments; */
    return null;
  }

  // Find the next token in the arguments set.
  protected string _nextToken() {
    /* return _tokens[0] ? _tokens[0] : ""; */
    return null;
  }
}
