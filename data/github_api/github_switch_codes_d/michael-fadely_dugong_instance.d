// Repository: michael-fadely/dugong
// File: source/http/instance.d

module http.instance;

public import std.socket;

import core.time;

import std.algorithm;
import std.array;
import std.ascii : isWhite;
import std.concurrency;
import std.conv;
import std.exception;
import std.functional : not;
import std.range;
import std.string;
import std.uni : sicmp;
import std.utf : byCodeUnit;

import http.common;
import http.enums;
import http.response;
import http.socket;

/// Interface for HTTP instances.
interface IHttpInstance
{
	/// Returns the connected state of this instance.
	@safe @property bool connected() const;
	/// Disconnects this instance.
	nothrow void disconnect();
	/// Main parsing routine.
	void run();
	/// Main receiving function.
	/// Returns: $(D true) if data has been received.
	bool receive();
	/// Sends the data stored in this instance to the given socket.
	void send(scope HttpSocket s);
	/// Sends the data in this instance to its connected socket.
	void send();
	/// Clears the data in this instance.
	nothrow void clear();
}

/// Base class for $(D HttpRequest) and $(D HttpResponse).
abstract class HttpInstance : IHttpInstance
{
private:
	bool persistent;
	bool hasBody_;
	bool chunked;
	bool isMultiPart_;
	string multiPartBoundary_;

protected:
	HttpSocket socket;
	HttpVersion version_;
	string[string] headers;
	ubyte[] body_;

public:
	this(HttpSocket socket, bool hasBody = true, int keepAlive = 5, lazy Duration timeout = 15.seconds)
	{
		enforce(socket !is null, "socket must not be null!");
		enforce(socket.isAlive, "socket must be connected!");

		this.socket = socket;
		this.hasBody_ = hasBody;
		this.socket.blocking = false;
		this.socket.setTimeouts(keepAlive, timeout);
	}

	nothrow void disconnect()
	{
		clear();

		if (socket !is null)
		{
			socket.disconnect();
		}

		socket = null;
	}

	nothrow void clear()
	{
		if (socket !is null)
		{
			socket.clear();
		}

		persistent = false;
		version_   = HttpVersion.v1_1;
		headers    = null;
		body_      = null;
		chunked    = false;
	}

	void send(scope HttpSocket s)
	{
		// This toString() is implemented by the derived class.
		s.sendAsync(toString());

		if (!body_.empty)
		{
			s.sendAsync(body_);
			return;
		}

		// TODO: connection abort

		if (isChunked)
		{
			foreach (chunk; byChunk())
			{
				if (s.sendAsync(chunk) < 1)
				{
					break;
				}
			}

			return;
		}

		if (isMultiPart)
		{
			sendMultiPart(s);
		}
		else
		{
			if (!hasBody)
			{
				return;
			}

			foreach (block; byBlock())
			{
				if (s.sendAsync(block) < 1)
				{
					break;
				}
			}
		}
	}

	final void sendMultiPart(scope HttpSocket s)
	{
		immutable start = "--" ~ multiPartBoundary;
		immutable end = start ~ "--";

		while (true)
		{
			char[] _boundary;

			if (socket.readln(_boundary) < 1)
			{
				break;
			}

			enforce(_boundary.startsWith(start), "Malformed multipart line: boundary not found");
			s.writeln(_boundary);

			if (_boundary == end)
			{
				break;
			}

			string[string] _headers;

			foreach (line; socket.byLine())
			{
				s.writeln(line);

				if (line.empty)
				{
					break;
				}

				/*
				auto key = line.munch("^:").idup;
				line.munch(": ");
				_headers[key] = line.idup;
				*/

				auto split_ = line.findSplit(":");
				auto key = split_[0].idup;
				_headers[key] = stripLeft(split_[2]).idup;
			}

			foreach (ubyte[] buffer; socket.byBlockUntil(start, true))
			{
				s.sendAsync(buffer);
			}
		}
	}

	void send()
	{
		send(socket);
	}

final:
	@safe @property
	{
		/// Indicates whether or not this instance uses connection persistence.
		nothrow bool isPersistent() const { return persistent; }
		/// Indicates whether or not this instance expects to have a body.
		nothrow bool hasBody() const { return chunked || hasBody_; }
		/// Indicates whether or not the Transfer-Encoding header is present in this instance.
		nothrow bool isChunked() const { return chunked; }
		/// Indicates whether or not this instance contains multi-part data.
		nothrow bool isMultiPart() const { return isMultiPart_; }
		/// Gets the multi-part boundary for this instance.
		nothrow auto multiPartBoundary() const { return multiPartBoundary_; }
		/// Indicates whether or not ths instance is connected.
		bool connected() const { return socket !is null && socket.isAlive; }
	}

	@nogc nothrow string getHeader(in string key, string* realKey = null)
	{
		import std.uni : sicmp;

		auto ptr = key in headers;
		if (ptr !is null)
		{
			return *ptr;
		}

		auto search = headers.byPair.find!(x => !sicmp(key, x[0]));

		if (!search.empty)
		{
			auto result = takeOne(search);

			if (realKey !is null)
			{
				*realKey = result.front[0];
			}

			return result.front[1];
		}

		return null;
	}

	/// Reads available headers from the socket and populates
	/// $(D headers), performs error handling, maybe more idk.
	void parseHeaders()
	{
		foreach (char[] header; socket.byLine())
		{
			if (header.empty)
			{
				break;
			}

			//auto key = header.munch("^:").idup;
			
			auto key = to!string(header.byCodeUnit.until(':'));

			//header.munch(": ");

			header = header
			         .byCodeUnit
			         .find(':')
			         .dropOne
			         .find!(x => !isWhite(x) && x != ':')
			         .source;

			// If more than one Content-Length header is
			// specified, take the smaller of the two.
			if (!sicmp(key, "Content-Length"))
			{
				try
				{
					immutable length = getHeader(key);
					if (!length.empty)
					{
						const existing = to!size_t(length);
						const received = to!size_t(header);

						if (existing < received)
						{
							continue;
						}
					}
				}
				catch (Exception)
				{
					// ignored
				}
			}

			headers[key] = header.idup;
		}

		string key;
		immutable contentLength = getHeader("Content-Length", &key);
		chunked = !getHeader("Transfer-Encoding").empty;

		if (!contentLength.empty && chunked)
		{
			enforce(headers.remove(key));
		}
		else if (hasBody_)
		{
			hasBody_ = !contentLength.empty;
		}

		auto connection = getHeader("Connection");
		if (connection.empty)
		{
			connection = getHeader("Proxy-Connection");
		}

		// Duplicating because we will be modifying this string.
		char[] contentType = getHeader("Content-Type").dup;

		if (!contentType.empty && contentType.toLower().canFind("multipart"))
		{

			import std.ascii : isWhite;
			alias notWhite = (x) => !isWhite(x);

			/*
			contentType.munch(" ");
			contentType.munch("^ ");
			contentType.munch(" ");
			*/

			contentType = contentType
			              .byCodeUnit
			              .find!(not!isWhite)
			              .find!(isWhite)
			              .find!(not!isWhite)
			              .source;

			/*
			immutable boundaryParam = contentType.munch("^=");
			contentType.munch(" =");
			*/

			const boundaryParam = contentType.byCodeUnit.until('=').array;

			contentType = contentType
			              .byCodeUnit
			              .find('=')
			              .dropOne
			              .find!(x => !isWhite(x) || x == '=')
			              .source;

			if (!boundaryParam.sicmp("boundary"))
			{
				//multiPartBoundary_ = contentType.munch("^ ;");
				// can I do this without a copy?
				contentType = contentType.byCodeUnit.until!(x => isWhite(x) || x == ';').array;
				multiPartBoundary_ = to!string(contentType);
				isMultiPart_ = true;
			}
		}

		switch (version_) with (HttpVersion)
		{
			case v1_0:
				persistent = !sicmp(connection, "keep-alive");
				break;

			case v1_1:
				persistent = !!sicmp(connection, "close");
				break;

			default:
				if (socket.isAlive)
				{
					socket.sendResponse(HttpStatus.httpVersionNotSupported);
				}

				disconnect();
				return;
		}
	}

	/// Converts the key-value pairs in $(D headers) to a string.
	string toHeaderString() const
	{
		if (!headers.length)
		{
			return null;
		}

		return headers.byKeyValue.map!(x => x.key ~ ": " ~ x.value).join("\r\n");
	}

	private void byChunkMethod()
	{
		// readChunk yields the buffer whenever possible
		if (!socket.readChunk(body_))
		{
			disconnect();
		}
	}

	/// Read data from this instance by chunk (Transfer-Encoding).
	auto byChunk()
	{
		enforce(isChunked, __PRETTY_FUNCTION__ ~ " called on instance with no chunked data.");
		return new Generator!(ubyte[])(&byChunkMethod);
	}

	/// Read data from this instance by block (requires Content-Length).
	auto byBlock()
	{
		immutable header = getHeader("Content-Length");
		enforce(!header.empty, __PRETTY_FUNCTION__ ~ " called on instance with no Content-Length header.");

		const length = to!size_t(header);
		return socket.byBlock(length);
	}
}
