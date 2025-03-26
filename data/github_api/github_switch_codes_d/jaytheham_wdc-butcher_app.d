// Repository: jaytheham/wdc-butcher
// File: app.d

import std.math,	std.conv,	std.stdio,	std.file,
	std.string,		std.random,	std.format,	std.zlib,
	std.typecons,
	core.thread,
	gfm.logger,		gfm.sdl2,	gfm.math,	gfm.opengl,
	wdc.car,		wdc.carFromObj,			wdc.carToObj,
	wdc.track,		wdc.drawable,			wdc.binary,
	camera,	timekeeper;

private
{
	enum int WINDOW_WIDTH = 1280;
	enum int WINDOW_HEIGHT = 720;
	enum string RELEASE_VERSION = "1.0.0 Feb 21 2017";

	Binary binaryFile;
	SDL2Window window;
	bool windowVisible = false;

	int mode;

	OpenGL gl;

	Drawable selectedObject;

	string outputDir;

	struct UserCommand
	{
		string shortCommand;
		string longCommand;
		string description;
		string usage;
		void function(string[] args) run;
	}
	UserCommand[] commands;
}

void main(string[] args)
{
	writeln("World Driver Championship Butcher for N64");
	writeln("Created by jaytheham@gmail.com");
	writeln("--------------------------------\n");

	binaryFile = getWDCBinary(args);

	setupPathAndFolder();

	auto conLogger = new ConsoleLogger();
	SDL2 sdl2 = new SDL2(conLogger, SharedLibVersion(2, 0, 0));
	gl = new OpenGL(conLogger);

	window = createSDLWindow(sdl2);
	window.setTitle("World Driver Championship Viewer");
	window.hide();

	setOpenGLState();

	Camera basicCamera = new Camera(gfm.math.radians(45f), cast(float)WINDOW_WIDTH / WINDOW_HEIGHT);
	setupCommands();

	while(!sdl2.wasQuitRequested())
	{
		if (windowVisible)
		{
			TimeKeeper.startNewFrame();
			sdl2.processEvents();
			if (sdl2.keyboard.isPressed(SDLK_ESCAPE))
			{
				setWindowVisible(false);
				continue;
			}

			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			basicCamera.update(sdl2, TimeKeeper.getDeltaTime());

			handleInput(sdl2);

			selectedObject.draw(basicCamera, getKeys(sdl2));

			window.swapBuffers();
			Thread.sleep(TimeKeeper.uSecsUntilNextFrame().usecs);
		}
		else
		{
			handleCommands();
		}
	}
}

private void setupPathAndFolder()
{
	import std.path;
	string exeFolder = dirName(thisExePath());
	chdir(exeFolder);
	outputDir = exeFolder ~ dirSeparator ~ "output" ~ dirSeparator;
	if (!exists(outputDir))
	{
		mkdir(outputDir);
	}
}

private char[] getKeys(SDL2 sdl2)
{
	char[] keys;
	if (sdl2.keyboard.testAndRelease(SDLK_1))
	{
		keys ~= '1';
	}
	if (sdl2.keyboard.testAndRelease(SDLK_2))
	{
		keys ~= '2';
	}
	if (sdl2.keyboard.testAndRelease(SDLK_3))
	{
		keys ~= '3';
	}
	if (sdl2.keyboard.testAndRelease(SDLK_4))
	{
		keys ~= '4';
	}
	return keys;
}

private Binary getWDCBinary(string[] args)
{
	string binaryPath;
	if (args.length == 1)
	{
		writeln("Drag and drop a World Driver Championship ROM on the exe to load it.");
		writeln("Otherwise you can enter the path to a ROM and press Enter now:");
		binaryPath = chomp(readln());
		if (binaryPath[0] == '"' || binaryPath[0] == '\'')
		{
			binaryPath = binaryPath[1..$ - 1];
		}
	}
	else
	{
		binaryPath = args[1];
	}
	return new Binary(binaryPath);
}

private auto createSDLWindow(SDL2 sdl2)
{
	// You have to initialize each SDL subsystem you want by hand
	sdl2.subSystemInit(SDL_INIT_VIDEO);
	sdl2.subSystemInit(SDL_INIT_EVENTS);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	// create an OpenGL-enabled SDL window
	return new SDL2Window(sdl2, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
								WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_OPENGL);
}

private void handleCommands()
{
	std.stdio.write("\nWaiting for input: ");
	string[] args = splitCommands(chomp(readln()));
	if (args.length > 0)
	{
		writeln();
		foreach(cmd; commands)
		{
			if (cmd.shortCommand == args[0] || cmd.longCommand == args[0])
			{
				try
				{
					cmd.run(args);
				}
				catch (Exception e)
				{
					//writeln(e.info);
					writeln(e.msg, "\n", e.file, " ", e.line, "\n\n", cmd.usage);
				}
				return;
			}
		}
		writeHelp(null);
	}
}

private string[] splitCommands(string input)
{
	// assume no nested quotes
	string[] output;
	int start;
	bool inQuote = false;
	foreach (i, c; input)
	{
		if (c == '"') {

			if (inQuote)
			{
				output ~= input[start..i];
				inQuote = false;
				start = -1;
			}
			else
			{
				start = i + 1;
				inQuote = true;
			}
		}
		else if (c == ' ')
		{
			if (!inQuote && start != -1)
			{
				output ~= input[start..i];
				start = -1;
			}
		}
		else
		{
			if (start == -1)
			{
				start = i;
			}
		}
	}
	if (start != -1)
	{
		output ~= input[start..$];
	}
	return output;
}

private void handleInput(SDL2 sdl2)
{
	if (sdl2.keyboard.testAndRelease(SDLK_p))
	{
		mode = mode == GL_FILL ? GL_LINE : GL_FILL;
		glPolygonMode(GL_FRONT_AND_BACK, mode);
		if (mode == GL_LINE)
		{
			glDisable(GL_CULL_FACE);
		}
		else
		{
			glEnable(GL_CULL_FACE);
		}
	}
}

private void setWindowVisible(bool isVisible)
{
	if (isVisible)
	{
		SDL_SetRelativeMouseMode(SDL_TRUE);
		window.show();
		windowVisible = true;
		TimeKeeper.start(60);
	}
	else
	{
		window.hide();
		SDL_SetRelativeMouseMode(SDL_FALSE);
		windowVisible = false;
	}
}

private void setOpenGLState()
{
	// reload OpenGL now that a context exists
	gl.reload();
	mode = GL_LINE;
	glPointSize(3.0);
	glClearColor(0.1, 0.2, 0.4, 1);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// redirect OpenGL output to our Logger
	gl.redirectDebugOutput();
}

private void listCars(string[] args)
{
	writeln("\nIndex\tCar Name");
	writeln("-----\t--------\n");
	foreach(index, carName; binaryFile.getCarList()){
		writefln("%d\t%s", index, carName);
	}
}

private void displayCar(int index)
{
	selectedObject = binaryFile.getCar(index);
	selectedObject.setupDrawing(gl);
	setWindowVisible(true);
	writefln("\nDisplaying car #%d", index);
	writeln("Press Escape to return to command window");
	writeln("W A S D and mouse to move camera.\n1 & 2 to cycle through visible parts.\n3 & 4 to change current palette.\np to switch wireframe mode.");
}

private void extractCarObj(string[] args)
{
	int carIndex = parse!int(args[1]);
	string destinationFolder = outputDir ~ format("car%.2d/", carIndex);
	CarToObj.convert(binaryFile.getCar(carIndex), destinationFolder);
	writefln("Car %d extracted to .obj file in %s", carIndex, destinationFolder);
}

private void extractCarBinary(string[] args)
{
	int carIndex = parse!int(args[1]);
	binaryFile.dumpCarData(carIndex, outputDir);
}

private void importCarObj(string[] args)
{
	if (args.length < 3)
	{
		throw new Exception("Missing argument(s)");
	}
	selectedObject = CarFromObj.convert(args[1]);
	binaryFile.insertCar(cast(Car)selectedObject, parse!int(args[2]));
}

private void listTracks(string[] args)
{
	writeln("\nIndex\tTrack Name");
	writeln("-----\t--------\n");
	foreach(index, trackName; binaryFile.getTrackList()){
		writefln("%d\t%s", index, trackName);
	}
}

private void displayTrack(int index, int variation)
{
	selectedObject = binaryFile.getTrack(index, variation);

	selectedObject.setupDrawing(gl);
	setWindowVisible(true);
	writefln("\nDisplaying track #%d variation %d", index, variation);
	writeln("Press Escape to return to command window");
}

private void extractTrackBinary(string[] args)
{
	int trackIndex = parse!int(args[1]);
	binaryFile.dumpTrackData(trackIndex, outputDir);
}

private void extractZlibBlock(string[] args)
{
	bool hexPrefix = indexOf(args[1], "0x") == 0;
	string offset = chompPrefix(args[1], "0x");
	int dataOffset = hexPrefix ? parse!int(offset, 16) : parse!int(offset);

	string workingDir = getcwd();
	chdir(outputDir);
	ubyte[] output = binaryFile.decompressZlibBlock(dataOffset);
	std.file.write(format("Data_%.8x", dataOffset), output);
	writefln("Data from %d extracted to %s", dataOffset, outputDir);
	chdir(workingDir);
}

private void writeHelp(string[] args)
{
	writeln("\nAvailable commands:");
	foreach(cmd; commands)
	{
		writefln("\t%s\t%s\t\t%s", cmd.shortCommand, cmd.longCommand, cmd.description);
	}
	writeln();
}

private void setupCommands()
{
	commands ~= UserCommand("lc", "--list-cars", "List car names and indices", "lc", &listCars);
	commands ~= UserCommand("lt", "--list-tracks", "List track names and indices", "lt", &listTracks);
	commands ~= UserCommand("dc", "--display-car", "Display car {index}", "dc {index}",
		(string[] args) {
			if (args.length != 2)
			{
				throw new Exception("Wrong arg number");
			}
			displayCar(parse!int(args[1]));
		});
	commands ~= UserCommand("dt", "--display-track", "Display track {index} {variation}", "dt {index} {variation}",
		(string[] args) {
			if (args.length != 3)
			{
				throw new Exception("Wrong arg number");
			}
			displayTrack(parse!int(args[1]), parse!int(args[2]));
		});
	commands ~= UserCommand("e", "--extract", "Extract and inflate zlib data from ROM {offset}", "", &extractZlibBlock);
	commands ~= UserCommand("ecb", "--extract-car-binary", "Extract car {index} binary data", "", &extractCarBinary);
	commands ~= UserCommand("eco", "--extract-car-obj", "Extract car {index} converted to Wavefront Obj format", "", &extractCarObj);
	commands ~= UserCommand("ico", "--import-car-obj", "Import car from Wavefront Obj file", "ico {path/to/file.obj} {Car number to replace}", &importCarObj);
	commands ~= UserCommand("etb", "--extract-track", "Extract track {index} {variation} binary data", "", &extractTrackBinary);
	commands ~= UserCommand("h", "--help", "Display all available commands", "", &writeHelp);
	commands ~= UserCommand("v", "--version", "Version information", "", (string[] args) { writeln(RELEASE_VERSION); });
}