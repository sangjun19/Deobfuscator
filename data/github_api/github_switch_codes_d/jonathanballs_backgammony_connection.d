// Repository: jonathanballs/backgammony
// File: source/networking/fibs/connection.d

module networking.fibs.connection;

import formats.fibs;
import core.time;
import std.algorithm : startsWith;
import std.array;
import std.format;
import std.regex;
import std.socket;
import std.stdio;
import std.variant;
import networking.connection;
import networking.fibs.clipmessages;
public import networking.connection : TimeoutException;
import gameplay.gamestate;

/**
 * Status of connection with FIBS server
 */
enum FIBSConnectionStatus {
    Disconnected, Connecting, Connected, Failed, Crashed
}

/**
 * Handles connection with FIBS server as well as formatting requests and parsing
 * responses.
 */
class FIBSConnection : Connection {
    /// Meta status info to be displayed to the user
    FIBSConnectionStatus status;
    string statusMessage;

    /**
     * Create a new connection to a FIBS server and attempt to login
     */
    this(Address serverAddress, string username, string password) {
        super(serverAddress);

        this.status = FIBSConnectionStatus.Connecting;

        // Wait for login prompt...
        while (true) {
            try {
                this.readline(25.msecs);
            } catch (TimeoutException e) {
                if (this.recBuffer.startsWith("login:")) {
                    this.recBuffer = "";
                    break;
                }
            }
        }

        this.writeline("login backgammony-1.0.0 1008 " ~ username ~ " " ~ password);

        // Throw expception if receive another login prompt otherwise return
        // active connection ready to exchange messages messages. This has funny
        // behaviour. Usually a newline character will _not_ be emitted after
        // the "login:" prompt but sometimes and a random CLIP message will be
        // printed after and so a newline is found.
        try {
            this.readline(500.msecs); // Server will send a new line first
            auto l = this.readline(500.msecs);

            if (l.startsWith("login:")) {
                throw new Exception("Authentication Failure");
            }

            this.recBuffer = l ~ "\r\n" ~ recBuffer;
        } catch (TimeoutException e) {
            this.status = FIBSConnectionStatus.Failed;
            this.statusMessage = "Authentication Failure";
        }

        this.writeline("set boardstyle 3");

        writeln("Authenticated successfully to FIBS server ", serverAddress);
    }

    /*
     * Overrided writeline to send carriage return as well
     */
    override void writeline(string s = "") {
        if (this._debug) {
            writeln("NETSND: ", s);
        }
        this.conn.send(s ~ "\r\n");
    }

    /**
     * Read and return a CLIP message
     */
    Variant readMessage(Duration timeout = Duration.zero) {
        import std.datetime.stopwatch;
        auto timer = new StopWatch(AutoStart.yes);

        string[] lines;

        // Skip empty lines and useless 6
        do {
            lines ~= this.readline(timeout);
            if (lines[0] == "" || lines[0] == "6") {
                lines = [];
                continue;
            }

            // Is this a multi line output?
            string clipIdentifier = lines[0].split()[0];
            if (clipIdentifier == "3" || clipIdentifier == "7") {
                string lastLine = lines[$-1];
                if (lastLine.length) {
                    string lastClipIdentifier = lastLine.split()[0];
                    if (clipIdentifier == "3" && lastClipIdentifier == "4") {
                        break;
                    }
                    if (clipIdentifier == "7" && lastClipIdentifier == "8") {
                        break;
                    }
                }
            } else {
                break;
            }

        } while (timeout == Duration.zero || timer.peek < timeout);

        Variant v;
        switch (lines[0].split()[0]) {
            case "1":
                assert(lines.length == 1);
                v = CLIPWelcome(lines[0]);
                this.status = FIBSConnectionStatus.Connected;
                break;
            case "2":
                assert(lines.length == 1);
                v = CLIPOwnInfo(lines[0]); break;
            case "3":
                assert(lines.length >= 2);
                v = CLIPMOTD(lines); break;
            case "5":
                v = CLIPWho(lines[0]); break;
            case "7":
                v = CLIPLogin(lines[0]); break;
            case "8":
                v = CLIPLogout(lines[0]); break;
            case "9":
                v = CLIPMessage(lines[0]); break;
            case "10":
                v = CLIPMessageDelivered(lines[0]); break;
            case "11":
                v = CLIPMessageSaved(lines[0]); break;
            case "12":
                v = CLIPSays(lines[0]); break;
            case "13":
                v = CLIPShouts(lines[0]); break;
            case "14":
                v = CLIPWhispers(lines[0]); break;
            case "15":
                v = CLIPKibitz(lines[0]); break;
            case "16":
                v = CLIPYouSay(lines[0]); break;
            case "17":
                v = CLIPYouShout(lines[0]); break;
            case "18":
                v = CLIPYouWhisper(lines[0]); break;
            case "19":
                v = CLIPYouKibitz(lines[0]); break;
            default:
                // It's not a CLIP message, we'll have to try some REGEX
                if (lines[0].match(regex("^board:.*"))) {
                    v = CLIPMatchState(lines[0].parseFibsMatch(), lines[0]);
                    break;
                }

                string movementRegex = "([0-9]+|bar)-([0-9]|off)+";
                auto turnRegex = regex(format!"^.* moves? (%s )+"(movementRegex));
                if (lines[0].match(turnRegex)) {
                    PipMovement[] moves;
                    foreach (m; lines[0].split()[2..$-1]) {
                        moves ~= parseFibsMovement(m);
                    }
                    v = CLIPMatchMovement(lines[0].split()[0], moves);
                    break;
                }

                if (lines[0].match("^[a-zA-Z_<>]+ can't move")) {
                    v = CLIPMatchMovement(lines[0].split()[0], []);
                    break;
                }

                if (lines[0].match(regex("^[a-zA-Z_<>]+ rolls? [1-6] and [1-6]."))) {
                    auto sSplit = lines[0].split();
                    import std.conv;
                    v = CLIPMatchRoll(sSplit[0], sSplit[2].to!uint, sSplit[4][0..$-1].to!uint);
                    break;
                }

                v = "====> " ~ lines[0];
                break;
        }

        if (v.type != typeid(CLIPWho))
            writeln(v);

        return v;
    }
}
