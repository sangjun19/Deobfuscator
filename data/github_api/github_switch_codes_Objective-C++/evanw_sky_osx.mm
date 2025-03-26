// Repository: evanw/sky
// File: osx/osx.mm

#define SKEW_GC_MARK_AND_SWEEP
#define SKEW_GC_PARALLEL
#import <skew.h>

////////////////////////////////////////////////////////////////////////////////

struct FixedArray : virtual Skew::Object {
  FixedArray(int byteCount) {
    assert(byteCount >= 0);
    _data = new float[byteCount + 3 & ~3];
    _byteCount = byteCount;
  }

  ~FixedArray() {
    delete _data;
  }

  int byteCount() {
    return _byteCount;
  }

  int getByte(int byteIndex) {
    assert(0 <= byteIndex && byteIndex + 1 <= _byteCount);
    return bytesForCPP()[byteIndex];
  }

  void setByte(int byteIndex, int value) {
    assert(0 <= byteIndex && byteIndex + 1 <= _byteCount);
    bytesForCPP()[byteIndex] = value;
  }

  double getFloat(int byteIndex) {
    assert(0 <= byteIndex && byteIndex + 4 <= _byteCount && byteIndex % 4 == 0);
    return _data[byteIndex / 4];
  }

  void setFloat(int byteIndex, double value) {
    assert(0 <= byteIndex && byteIndex + 4 <= _byteCount && byteIndex % 4 == 0);
    _data[byteIndex / 4] = value;
  }

  FixedArray *getRange(int byteIndex, int byteCount) {
    return new FixedArray(this, byteIndex, byteCount);
  }

  void setRange(int byteIndex, FixedArray *array) {
    assert(byteIndex >= 0 && byteIndex + array->_byteCount <= _byteCount);
    assert(byteIndex % 4 == 0);
    memcpy(_data + byteIndex / 4, array->_data, array->_byteCount);
  }

  uint8_t *bytesForCPP() {
    return reinterpret_cast<uint8_t *>(_data);
  }

  #ifdef SKEW_GC_MARK_AND_SWEEP
    virtual void __gc_mark() override {
    }
  #endif

private:
  FixedArray(FixedArray *array, int byteIndex, int byteCount) {
    assert(byteIndex >= 0 && byteCount >= 0 && byteIndex + byteCount <= array->_byteCount);
    assert(byteCount % 4 == 0);
    _data = new float[byteCount / 4];
    _byteCount = byteCount;
    memcpy(_data, array->_data + byteIndex / 4, byteCount);
  }

  float *_data = nullptr;
  int _byteCount = 0;
};

namespace Log {
  void info(const Skew::string &text) {
    #ifndef NDEBUG
      puts(text.c_str());
    #endif
  }

  void warning(const Skew::string &text) {
    #ifndef NDEBUG
      puts(text.c_str());
    #endif
  }

  void error(const Skew::string &text) {
    #ifndef NDEBUG
      puts(text.c_str());
    #endif
  }
}

////////////////////////////////////////////////////////////////////////////////

#import "compiled.cpp"
#import <skew.cpp>
#import <codecvt>
#import <locale>
#import <sys/time.h>

#define Rect RectWhyIsThisAGlobalGoshDarnIt
#import <Cocoa/Cocoa.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>
#undef Rect

@class AppView;

////////////////////////////////////////////////////////////////////////////////

namespace OpenGL {
  struct Context;

  struct Texture : Graphics::Texture {
    Texture(Graphics::Context *context, Graphics::TextureFormat *format, int width, int height, FixedArray *pixels)
        : _context(context), _format(format), _width(0), _height(0) {
      glGenTextures(1, &_texture);
      glBindTexture(GL_TEXTURE_2D, _texture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, format->magFilter == Graphics::PixelFilter::NEAREST ? GL_NEAREST : GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, format->minFilter == Graphics::PixelFilter::NEAREST ? GL_NEAREST : GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, format->wrap == Graphics::PixelWrap::REPEAT ? GL_REPEAT : GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, format->wrap == Graphics::PixelWrap::REPEAT ? GL_REPEAT : GL_CLAMP_TO_EDGE);
      resize(width, height, pixels);
    }

    ~Texture() {
      glDeleteTextures(1, &_texture);
    }

    unsigned int texture() {
      return _texture;
    }

    virtual Graphics::Context *context() override {
      return _context;
    }

    virtual Graphics::TextureFormat *format() override {
      return _format;
    }

    virtual int width() override {
      return _width;
    }

    virtual int height() override {
      return _height;
    }

    virtual void resize(int width, int height, FixedArray *pixels) override {
      assert(width > 0);
      assert(height > 0);
      assert(pixels == nullptr || pixels->byteCount() == width * height * 4);

      if (width != _width || height != _height) {
        _width = width;
        _height = height;

        glBindTexture(GL_TEXTURE_2D, _texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels != nullptr ? pixels->bytesForCPP() : nullptr);
      }
    }

    virtual void upload(FixedArray *sourcePixels, int targetX, int targetY, int sourceWidth, int sourceHeight) override {
      assert(sourceWidth >= 0);
      assert(sourceHeight >= 0);
      assert(sourcePixels->byteCount() == sourceWidth * sourceHeight * 4);
      assert(targetX >= 0 && targetX + sourceWidth <= _width);
      assert(targetY >= 0 && targetY + sourceHeight <= _height);

      glBindTexture(GL_TEXTURE_2D, _texture);
      glTexSubImage2D(GL_TEXTURE_2D, 0, targetX, targetY, sourceWidth, sourceHeight, GL_RGBA, GL_UNSIGNED_BYTE, sourcePixels->bytesForCPP());
    }

    #ifdef SKEW_GC_MARK_AND_SWEEP
      virtual void __gc_mark() override {
        Graphics::Texture::__gc_mark();

        Skew::GC::mark(_context);
        Skew::GC::mark(_format);
      }
    #endif

  private:
    unsigned int _texture = 0;
    Graphics::Context *_context = nullptr;
    Graphics::TextureFormat *_format = nullptr;
    int _width = 0;
    int _height = 0;
  };

  ////////////////////////////////////////////////////////////////////////////////

  struct Material : Graphics::Material {
    Material(Graphics::Context *context, Graphics::VertexFormat *format, const char *vertexSource, const char *fragmentSource) : _context(context), _format(format) {
      _program = glCreateProgram();
      _vertexShader = _compileShader(GL_VERTEX_SHADER, vertexSource);
      _fragmentShader = _compileShader(GL_FRAGMENT_SHADER, fragmentSource);

      auto attributes = format->attributes();
      for (int i = 0; i < attributes->count(); i++) {
        glBindAttribLocation(_program, i, (*attributes)[i]->name.c_str());
      }

      glLinkProgram(_program);

      int status = 0;
      glGetProgramiv(_program, GL_LINK_STATUS, &status);

      if (!status) {
        char buffer[4096] = {'\0'};
        int length = 0;
        glGetProgramInfoLog(_program, sizeof(buffer), &length, buffer);
        puts(buffer);
        exit(1);
      }
    }

    ~Material() {
      glDeleteProgram(_program);
      glDeleteShader(_vertexShader);
      glDeleteShader(_fragmentShader);
    }

    void prepare() {
      glUseProgram(_program);
      for (const auto &it : _samplers) {
        auto texture = static_cast<Texture *>(it.second);
        glActiveTexture(GL_TEXTURE0 + it.first);
        glBindTexture(GL_TEXTURE_2D, texture != nullptr ? texture->texture() : 0);
      }
    }

    virtual Graphics::Context *context() override {
      return _context;
    }

    virtual Graphics::VertexFormat *format() override {
      return _format;
    }

    virtual void setUniformFloat(Skew::string name, double x) override {
      glUseProgram(_program);
      glUniform1f(_location(name), x);
    }

    virtual void setUniformInt(Skew::string name, int x) override {
      glUseProgram(_program);
      glUniform1i(_location(name), x);
    }

    virtual void setUniformVec2(Skew::string name, double x, double y) override {
      glUseProgram(_program);
      glUniform2f(_location(name), x, y);
    }

    virtual void setUniformVec3(Skew::string name, double x, double y, double z) override {
      glUseProgram(_program);
      glUniform3f(_location(name), x, y, z);
    }

    virtual void setUniformVec4(Skew::string name, double x, double y, double z, double w) override {
      glUseProgram(_program);
      glUniform4f(_location(name), x, y, z, w);
    }

    virtual void setUniformSampler(Skew::string name, Graphics::Texture *texture, int index) override {
      glUseProgram(_program);
      glUniform1i(_location(name), index);
      _samplers[index] = texture;
    }

    #ifdef SKEW_GC_MARK_AND_SWEEP
      virtual void __gc_mark() override {
        Graphics::Material::__gc_mark();

        Skew::GC::mark(_context);
        Skew::GC::mark(_format);

        for (const auto &it : _samplers) {
          Skew::GC::mark(it.second);
        }
      }
    #endif

  private:
    int _location(const Skew::string &name) {
      auto it = _locations.find(name.std_str());
      if (it == _locations.end()) {
        it = _locations.insert(std::make_pair(name.std_str(), glGetUniformLocation(_program, name.c_str()))).first;
      }
      return it->second;
    }

    unsigned int _compileShader(int type, const char *source) {
      auto shader = glCreateShader(type);
      glShaderSource(shader, 1, &source, nullptr);
      glCompileShader(shader);

      int status = 0;
      glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

      if (!status) {
        char buffer[4096] = {'\0'};
        int length = 0;
        glGetShaderInfoLog(shader, sizeof(buffer), &length, buffer);
        puts(buffer);
        exit(1);
      }

      glAttachShader(_program, shader);
      return shader;
    }

    unsigned int _program = 0;
    unsigned int _vertexShader = 0;
    unsigned int _fragmentShader = 0;
    Graphics::Context *_context = nullptr;
    Graphics::VertexFormat *_format = nullptr;
    std::unordered_map<std::string, int> _locations;
    std::unordered_map<int, Graphics::Texture *> _samplers;
  };

  ////////////////////////////////////////////////////////////////////////////////

  struct RenderTarget : Graphics::RenderTarget {
    RenderTarget(Graphics::Context *context, Graphics::Texture *texture) : _context(context), _texture(texture) {
      glGenFramebuffers(1, &_framebuffer);
    }

    ~RenderTarget() {
      glDeleteFramebuffers(1, &_framebuffer);
    }

    unsigned int framebuffer() {
      return _framebuffer;
    }

    virtual Graphics::Context *context() override {
      return _context;
    }

    virtual Graphics::Texture *texture() override {
      return _texture;
    }

    #ifdef SKEW_GC_MARK_AND_SWEEP
      virtual void __gc_mark() override {
        Graphics::RenderTarget::__gc_mark();

        Skew::GC::mark(_context);
        Skew::GC::mark(_texture);
      }
    #endif

  private:
    Graphics::Context *_context = nullptr;
    Graphics::Texture *_texture = nullptr;
    unsigned int _framebuffer = 0;
  };

  ////////////////////////////////////////////////////////////////////////////////

  struct Context : Graphics::Context {
    struct Viewport {
      int x = 0;
      int y = 0;
      int width = 0;
      int height = 0;
    };

    ~Context() {
      glDeleteBuffers(1, &_vertexBuffer);
    }

    virtual int width() override {
      return _width;
    }

    virtual int height() override {
      return _height;
    }

    virtual void addContextResetHandler(Skew::FnVoid0 *callback) override {
    }

    virtual void removeContextResetHandler(Skew::FnVoid0 *callback) override {
    }

    virtual void setViewport(int x, int y, int width, int height) override {
      _currentViewport.x = x;
      _currentViewport.y = y;
      _currentViewport.width = width;
      _currentViewport.height = height;
    }

    virtual void clear(int color) override {
      _updateRenderTargetAndViewport();
      _updateBlendState();

      if (color != _currentClearColor) {
        glClearColor(
          Graphics::RGBA::red(color) / 255.0,
          Graphics::RGBA::green(color) / 255.0,
          Graphics::RGBA::blue(color) / 255.0,
          Graphics::RGBA::alpha(color) / 255.0);
        _currentClearColor = color;
      }

      glClear(GL_COLOR_BUFFER_BIT);
    }

    virtual Graphics::Material *createMaterial(Graphics::VertexFormat *format, Skew::string vertexSource, Skew::string fragmentSource) override {
      std::string precision("precision highp float;");
      auto vertex = vertexSource.std_str();
      auto fragment = fragmentSource.std_str();
      auto v = vertex.find(precision);
      auto f = fragment.find(precision);
      if (v != std::string::npos) vertex = vertex.substr(v + precision.size());
      if (f != std::string::npos) fragment = fragment.substr(f + precision.size());
      return new Material(this, format, vertex.c_str(), fragment.c_str());
    }

    virtual Graphics::Texture *createTexture(Graphics::TextureFormat *format, int width, int height, FixedArray *pixels) override {
      return new Texture(this, format, width, height, pixels);
    }

    virtual Graphics::RenderTarget *createRenderTarget(Graphics::Texture *texture) override {
      return new RenderTarget(this, texture);
    }

    virtual void draw(Graphics::Primitive primitive, Graphics::Material *material, FixedArray *vertices) override {
      if (vertices == nullptr || vertices->byteCount() == 0) {
        return;
      }

      assert(vertices->byteCount() % material->format()->stride() == 0);

      // Update the texture set before preparing the material so uniform samplers can check for that they use different textures
      _updateRenderTargetAndViewport();
      static_cast<Material *>(material)->prepare();

      // Update the vertex buffer before updating the format so attributes can bind correctly
      if (_vertexBuffer == 0) {
        glGenBuffers(1, &_vertexBuffer);
      }
      glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
      glBufferData(GL_ARRAY_BUFFER, vertices->byteCount(), vertices->bytesForCPP(), GL_DYNAMIC_DRAW);
      _updateFormat(material->format());

      // Draw now that everything is ready
      _updateBlendState();
      glDrawArrays(primitive == Graphics::Primitive::TRIANGLES ? GL_TRIANGLES : GL_TRIANGLE_STRIP,
        0, vertices->byteCount() / material->format()->stride());
    }

    virtual void resize(int width, int height) override {
      assert(width >= 0);
      assert(height >= 0);
      _width = width;
      _height = height;
      setViewport(0, 0, width, height);
    }

    virtual void setRenderTarget(Graphics::RenderTarget *renderTarget) override {
      _currentRenderTarget = renderTarget;

      // Automatically update the viewport after changing render targets
      setViewport(0, 0,
        renderTarget != nullptr ? renderTarget->texture()->width() : _width,
        renderTarget != nullptr ? renderTarget->texture()->height() : _height);
    }

    virtual void setBlendState(Graphics::BlendOperation source, Graphics::BlendOperation target) override {
      _blendOperations = (int)source | (int)target << 4;
    }

    #ifdef SKEW_GC_MARK_AND_SWEEP
      virtual void __gc_mark() override {
        Graphics::Context::__gc_mark();

        Skew::GC::mark(_currentRenderTarget);
        Skew::GC::mark(_oldRenderTarget);
      }
    #endif

  private:
    void _updateRenderTargetAndViewport() {
      auto renderTarget = _currentRenderTarget;

      if (_oldRenderTarget != renderTarget) {
        glBindFramebuffer(GL_FRAMEBUFFER, renderTarget != nullptr ? static_cast<RenderTarget *>(renderTarget)->framebuffer() : 0);
        _oldRenderTarget = renderTarget;
      }

      if (_currentViewport.x != _oldViewport.x ||
          _currentViewport.y != _oldViewport.y ||
          _currentViewport.width != _oldViewport.width ||
          _currentViewport.height != _oldViewport.height) {
        glViewport(
          _currentViewport.x,
          (renderTarget != nullptr ? renderTarget->texture()->height() : _height) - _currentViewport.y - _currentViewport.height,
          _currentViewport.width,
          _currentViewport.height);
        _oldViewport = _currentViewport;
      }
    }

    void _updateBlendState() {
      if (_oldBlendOperations != _blendOperations) {
        int operations = _blendOperations;
        int oldOperations = _oldBlendOperations;
        int source = operations & 0xF;
        int target = operations >> 4;

        assert(_blendOperationMap.count(source));
        assert(_blendOperationMap.count(target));

        // Special-case the blend mode that just writes over the target buffer
        if (operations == COPY_BLEND_OPERATIONS) {
          glDisable(GL_BLEND);
        } else {
          if (oldOperations == COPY_BLEND_OPERATIONS) {
            glEnable(GL_BLEND);
          }

          // Otherwise, use actual blending
          glBlendFunc(_blendOperationMap[source], _blendOperationMap[target]);
        }

        _oldBlendOperations = operations;
      }
    }

    void _updateFormat(Graphics::VertexFormat *format) {
      // Update the attributes
      auto attributes = format->attributes();
      int count = attributes->count();
      for (int i = 0; i < count; i++) {
        auto attribute = (*attributes)[i];
        bool isByte = attribute->type == Graphics::AttributeType::BYTE;
        glVertexAttribPointer(i, attribute->count, isByte ? GL_UNSIGNED_BYTE : GL_FLOAT, isByte, format->stride(), reinterpret_cast<void *>(attribute->byteOffset));
      }

      // Update the attribute count
      while (_attributeCount < count) {
        glEnableVertexAttribArray(_attributeCount);
        _attributeCount++;
      }
      while (_attributeCount > count) {
        _attributeCount--;
        glDisableVertexAttribArray(_attributeCount);
      }
      _attributeCount = count;
    }

    enum {
      COPY_BLEND_OPERATIONS = (int)Graphics::BlendOperation::ONE | (int)Graphics::BlendOperation::ZERO << 4,
    };

    int _width = 0;
    int _height = 0;
    Graphics::RenderTarget *_currentRenderTarget = nullptr;
    Graphics::RenderTarget *_oldRenderTarget = nullptr;
    int _oldBlendOperations = COPY_BLEND_OPERATIONS;
    int _blendOperations = COPY_BLEND_OPERATIONS;
    int _currentClearColor = 0;
    int _attributeCount = 0;
    Viewport _oldViewport;
    Viewport _currentViewport;
    unsigned int _vertexBuffer = 0;

    static std::unordered_map<int, int> _blendOperationMap;
  };

  ////////////////////////////////////////////////////////////////////////////////

  std::unordered_map<int, int> Context::_blendOperationMap = {
    { (int)Graphics::BlendOperation::ZERO, GL_ZERO },
    { (int)Graphics::BlendOperation::ONE, GL_ONE },

    { (int)Graphics::BlendOperation::SOURCE_COLOR, GL_SRC_COLOR },
    { (int)Graphics::BlendOperation::TARGET_COLOR, GL_DST_COLOR },
    { (int)Graphics::BlendOperation::INVERSE_SOURCE_COLOR, GL_ONE_MINUS_SRC_COLOR },
    { (int)Graphics::BlendOperation::INVERSE_TARGET_COLOR, GL_ONE_MINUS_DST_COLOR },

    { (int)Graphics::BlendOperation::SOURCE_ALPHA, GL_SRC_ALPHA },
    { (int)Graphics::BlendOperation::TARGET_ALPHA, GL_DST_ALPHA },
    { (int)Graphics::BlendOperation::INVERSE_SOURCE_ALPHA, GL_ONE_MINUS_SRC_ALPHA },
    { (int)Graphics::BlendOperation::INVERSE_TARGET_ALPHA, GL_ONE_MINUS_DST_ALPHA },

    { (int)Graphics::BlendOperation::CONSTANT, GL_CONSTANT_COLOR },
    { (int)Graphics::BlendOperation::INVERSE_CONSTANT, GL_ONE_MINUS_CONSTANT_COLOR },
  };
}

////////////////////////////////////////////////////////////////////////////////

namespace OSX {
  template <typename T, void (*F)(T)>
  struct CDeleter {
    void operator () (T ref) {
      if (ref) {
        F(ref);
      }
    }
  };

  template <typename T, typename X, void (*F)(X)>
  using CPtr = std::unique_ptr<typename std::remove_pointer<T>::type, CDeleter<X, F>>;

  template <typename T>
  using CFPtr = CPtr<T, CFTypeRef, CFRelease>;

  using CGColorSpacePtr = CPtr<CGColorSpaceRef, CGColorSpaceRef, CGColorSpaceRelease>;
  using CGContextPtr = CPtr<CGContextRef, CGContextRef, CGContextRelease>;

  ////////////////////////////////////////////////////////////////////////////////

  struct FontInstance : UI::FontInstance {
    FontInstance(UI::Font font, Skew::List<Skew::string> *fontNames, double size, double lineHeight, int flags, double pixelScale)
      : _font(font), _size(size), _lineHeight(lineHeight), _flags(flags), _fontNames(fontNames) {
      changePixelScale(pixelScale);
    }

    virtual UI::Font font() override {
      return _font;
    }

    virtual double size() override {
      return _size;
    }

    virtual double lineHeight() override {
      return _lineHeight;
    }

    virtual int flags() override {
      return _flags;
    }

    void changePixelScale(double pixelScale) {
      if (_pixelScale == pixelScale) {
        return;
      }

      _pixelScale = pixelScale;
      auto fontSize = _size * pixelScale;

      _fonts.clear();

      // Find the first user-provided font name
      for (const auto &name : *_fontNames) {
        if (_addFont(name.c_str(), fontSize)) {
          Log::info("selected font '" + name.std_str() + "'");
          break;
        }

        // Try the next one
        Log::warning("failed to font font '" + name.std_str() + "'");
      }

      // Use the default font as a fallback
      if (_fonts.empty()) {
        _fonts.emplace_back(CTFontCreateUIFontForLanguage(_font == UI::Font::CODE_FONT ? kCTFontUIFontUserFixedPitch : kCTFontUIFontSystem, fontSize, nullptr));
      }

      // Get the font fallback list
      CFPtr<CFArrayRef> appleLanguages((__bridge_retained CFArrayRef)[[NSUserDefaults standardUserDefaults] stringArrayForKey:@"AppleLanguages"]);
      CFPtr<CFArrayRef> defaultCascade(CTFontCopyDefaultCascadeListForLanguages(_fonts.front().get(), appleLanguages.get()));

      // Create a font for each one because Core Text doesn't have a way of querying a whole cascade for a glyph
      for (int i = 0, length = (int)CFArrayGetCount(defaultCascade.get()); i < length; i++) {
        auto descriptor = (CTFontDescriptorRef)CFArrayGetValueAtIndex(defaultCascade.get(), i);
        _fonts.emplace_back(CTFontCreateWithFontDescriptor(descriptor, fontSize, nullptr));
      }

      // These aren't in the default font fallback list for some reason but
      // they are present in Chrome. I'm using the same code as Chrome far as
      // I can tell so I have no idea why the result is different. Oh well.
      _addFont("Arial Unicode MS", fontSize); // Test character: "☲"
      _addFont("Apple Symbols", fontSize); // Test character: "𝌆"
    }

    virtual double advanceWidth(int codePoint) override {
      auto it = _advanceWidths.find(codePoint);
      if (it != _advanceWidths.end()) {
        return it->second;
      }
      _findCodePoint(codePoint);
      return _advanceWidths[codePoint] = CTFontGetAdvancesForGlyphs(_fonts[_cachedFontIndex].get(), kCTFontOrientationDefault, _cachedGlyphs, nullptr, 1) / _pixelScale;
    }

    virtual Graphics::Glyph *renderGlyph(int codePoint) override {
      _findCodePoint(codePoint);
      const auto &font = _fonts[_cachedFontIndex];
      auto bounds = CTFontGetBoundingRectsForGlyphs(font.get(), kCTFontOrientationDefault, _cachedGlyphs, nullptr, 1);
      auto fontSize = _size * _pixelScale;

      // Make sure the context is big enough
      int minX = std::floor(bounds.origin.x) - 1;
      int minY = std::floor(bounds.origin.y - fontSize) - 1;
      int maxX = std::ceil(bounds.origin.x + bounds.size.width) + 2;
      int maxY = std::ceil(bounds.origin.y + bounds.size.height - fontSize) + 2;
      int width = maxX - minX;
      int height = maxY - minY;
      if (!_context || width > _width || height > _height) {
        _width = std::max(width * 2, _width);
        _height = std::max(height * 2, _height);
        _bytes.resize(_width * _height * 4);
        _context.reset(CGBitmapContextCreate(_bytes.data(), _width, _height, 8, _width * 4, _deviceRGB.get(), kCGImageAlphaPremultipliedLast));
      }

      auto mask = new Graphics::Mask(width, height);

      // Render the glyph three times at different offsets
      for (int i = 0; i < 3; i++) {
        auto position = CGPointMake(-minX + i / 3.0, -minY - fontSize);
        CGContextClearRect(_context.get(), CGRectMake(0, 0, width, height));
        CTFontDrawGlyphs(font.get(), _cachedGlyphs, &position, 1, _context.get());

        // Extract the mask (keep in mind CGContext is upside-down)
        auto from = _bytes.data() + (_height - height) * _width * 4 + 3;
        auto to = mask->pixels->bytesForCPP() + i;
        for (int y = 0; y < height; y++, from += (_width - width) * 4) {
          for (int x = 0; x < width; x++, to += 4, from += 4) {
            *to = *from;
          }
        }
      }

      return new Graphics::Glyph(codePoint, mask, -minX, maxY, 1 / _pixelScale, advanceWidth(codePoint));
    }

    #ifdef SKEW_GC_MARK_AND_SWEEP
      virtual void __gc_mark() override {
        UI::FontInstance::__gc_mark();

        Skew::GC::mark(_fontNames);
      }
    #endif

  private:
    bool _addFont(const char *name, double fontSize) {
      CFPtr<CFStringRef> holder(CFStringCreateWithCString(kCFAllocatorDefault, name, kCFStringEncodingUTF8));
      CFPtr<CTFontRef> font(CTFontCreateWithName(holder.get(), fontSize, nullptr));

      if (!font) {
        return false;
      }

      if (_flags & UI::FontFlags::BOLD) {
        CFPtr<CTFontRef> copy(CTFontCreateCopyWithSymbolicTraits(font.get(), 0, nullptr, kCTFontBoldTrait, kCTFontBoldTrait));
        if (copy) font = std::move(copy);
      }

      if (_flags & UI::FontFlags::ITALIC) {
        CFPtr<CTFontRef> copy(CTFontCreateCopyWithSymbolicTraits(font.get(), 0, nullptr, kCTFontItalicTrait, kCTFontItalicTrait));
        if (copy) font = std::move(copy);
      }

      _fonts.emplace_back(std::move(font));
      return true;
    }

    void _findCodePoint(int codePoint) {
      if (_cachedCodePoint == codePoint) {
        return;
      }

      _cachedCodePoint = codePoint;

      uint16_t codeUnits[2] = { 0, 0 };
      int codeUnitCount = 0;

      // The code point must be UTF-16 encoded
      if (codePoint < 0x10000) {
        codeUnits[0] = codePoint;
        codeUnitCount = 1;
      } else {
        codeUnits[0] = ((codePoint - 0x10000) >> 10) + 0xD800;
        codeUnits[1] = ((codePoint - 0x10000) & ((1 << 10) - 1)) + 0xDC00;
        codeUnitCount = 2;
      }

      // Search the entire font cascade
      for (int i = 0, length = (int)_fonts.size(); i < length; i++) {
        const auto &font = _fonts[i];
        if (CTFontGetGlyphsForCharacters(font.get(), codeUnits, _cachedGlyphs, codeUnitCount)) {
          _cachedFontIndex = i;
          return;
        }
      }

      // Give up after reaching the end
      _cachedGlyphs[0] = 0;
      _cachedFontIndex = 0;

      Log::warning("failed to find a glyph for code unit " + std::to_string(codePoint));
    }

    UI::Font _font = {};
    double _size = 0;
    double _lineHeight = 0;
    double _pixelScale = 0;
    int _flags = 0;
    std::unordered_map<int, double> _advanceWidths;

    // Stuff for rendering
    int _width = 0;
    int _height = 0;
    CGContextPtr _context = nullptr;
    CGColorSpacePtr _deviceRGB = CGColorSpacePtr(CGColorSpaceCreateDeviceRGB());
    std::vector<uint8_t> _bytes;

    // Stuff for font selection
    int _cachedCodePoint = -1;
    int _cachedFontIndex = -1;
    CGGlyph _cachedGlyphs[2] = {0, 0}; // CTFontGetGlyphsForCharacters requires two elements but uses one
    std::vector<CFPtr<CTFontRef>> _fonts;
    Skew::List<Skew::string> *_fontNames;
  };

  ////////////////////////////////////////////////////////////////////////////////

  struct AppWindow : UI::Window, private UI::PixelRenderer {
    AppWindow(NSWindow *window, AppView *appView, UI::Platform *platform) : _window(window), _appView(appView), _platform(platform) {
      _shortcuts = new Editor::ShortcutMap(platform);
      _translator = new UI::SemanticToPixelTranslator(this);
    }

    void triggerFrame();
    void handleResize();
    void handleKeyEvent(NSEvent *event);
    void handleInsertText(NSString *text);
    void handleMouseEvent(NSEvent *event);
    void handleAction(Editor::Action action);
    void handlePaste();

    void initializeOpenGL() {
      assert(_context == nullptr);
      _context = new OpenGL::Context();
      _solidBatch = new Graphics::SolidBatch(_context);
      _glyphBatch = new Graphics::GlyphBatch(_platform, _context);
      _dropShadow = new Graphics::DropShadow(_context);
      handleResize();
    }

    void setIsActive(bool isActive) {
    }

    virtual UI::SemanticRenderer *renderer() override {
      return _translator;
    }

    virtual void setTitle(Skew::string title) override {
      [_window setTitle:[NSString stringWithUTF8String:title.c_str()]];
    }

    virtual void setTheme(UI::Theme *theme) override {
      _translator->setTheme(theme);
    }

    virtual int width() override {
      return _width;
    }

    virtual int height() override {
      return _height;
    }

    virtual double pixelScale() override {
      return _pixelScale;
    }

    virtual UI::Platform *platform() override {
      return _platform;
    }

    virtual UI::FontInstance *fontInstance(UI::Font font) override {
      return _fontInstances[(int)font];
    }

    virtual void setCursor(UI::Cursor cursor) override {
      switch (cursor) {
        case UI::Cursor::ARROW: _cursor = [NSCursor arrowCursor]; break;
        case UI::Cursor::TEXT: _cursor = [NSCursor IBeamCursor]; break;
        default: Log::warning("attempted to set an unsupported cursor type"); break;
      }
    }

    virtual void setFont(UI::Font font, Skew::List<Skew::string> *names, double size, double height, int flags) override {
      _fontInstances[(int)font] = new FontInstance(font, names, size, height, flags, _pixelScale);
    }

    virtual void render() override;

    virtual void setViewport(double x, double y, double width, double height) override {
      if (_solidBatch != nullptr) {
        _solidBatch->flush();
        _solidBatch->resize(width, height, _pixelScale);
      }

      if (_glyphBatch != nullptr) {
        _glyphBatch->flush();
        _glyphBatch->resize(width, height, _pixelScale);
      }

      if (_dropShadow != nullptr) {
        _dropShadow->resize(width, height);
      }

      if (_context != nullptr) {
        _context->setViewport(
          std::round(x * _pixelScale),
          std::round(y * _pixelScale),
          std::round(width * _pixelScale),
          std::round(height * _pixelScale));
      }
    }

    virtual void setDefaultBackgroundColor(int color) override {
      _clearColor = color;
    }

    virtual void fillRect(double x, double y, double width, double height, int color) override {
      assert(_isRendering);

      _glyphBatch->flush();
      _solidBatch->fillRect(x, y, width, height, Graphics::RGBA::premultiplied(color));
    }

    virtual void fillRoundedRect(double x, double y, double width, double height, int color, double radius) override {
      assert(_isRendering);

      _glyphBatch->flush();
      _solidBatch->fillRoundedRect(x, y, width, height, Graphics::RGBA::premultiplied(color), radius);
    }

    virtual void strokePolyline(Skew::List<double> *coordinates, int color, double thickness) override {
      assert(_isRendering);
      assert(coordinates->count() % 2 == 0);

      _glyphBatch->flush();
      _solidBatch->strokeNonOverlappingPolyline(coordinates, Graphics::RGBA::premultiplied(color), thickness, Graphics::StrokeCap::OPEN);
    }

    virtual void renderText(double x, double y, Skew::string text, UI::Font font, int color) override {
      assert(_isRendering);

      auto fontInstance = _fontInstances[(int)font];
      if (fontInstance == nullptr || x >= _width || y >= _height || y + fontInstance->size() <= 0) {
        return;
      }

      _solidBatch->flush();
      color = Graphics::RGBA::premultiplied(color);

      for (const auto &codePoint : std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>().from_bytes(text.std_str())) {
        x += _glyphBatch->appendGlyph(fontInstance, codePoint, x, y, color);
      }
    }

    virtual void renderRectShadow(
      double boxX, double boxY, double boxWidth, double boxHeight,
      double clipX, double clipY, double clipWidth, double clipHeight,
      double shadowAlpha, double blurSigma) override {

      assert(_isRendering);

      _solidBatch->flush();
      _glyphBatch->flush();
      _dropShadow->render(boxX, boxY, boxWidth, boxHeight, clipX, clipY, clipWidth, clipHeight, shadowAlpha, blurSigma);
    }

    #ifdef SKEW_GC_MARK_AND_SWEEP
      virtual void __gc_mark() override {
        UI::Window::__gc_mark();
        UI::PixelRenderer::__gc_mark();

        Skew::GC::mark(_platform);
        Skew::GC::mark(_draggingView);
        Skew::GC::mark(_shortcuts);
        Skew::GC::mark(_translator);
        Skew::GC::mark(_context);
        Skew::GC::mark(_solidBatch);
        Skew::GC::mark(_glyphBatch);
        Skew::GC::mark(_dropShadow);

        for (const auto &it : _fontInstances) {
          Skew::GC::mark(it.second);
        }
      }
    #endif

  private:
    UI::MouseEvent *_mouseEventFromEvent(UI::EventType type, NSEvent *event, Vector *delta);

    int _width = 0;
    int _height = 0;
    int _clearColor = 0;
    bool _isRendering = false;
    double _pixelScale = 0;
    bool _needsToBeShown = true;
    NSWindow *_window = nullptr;
    NSCursor *_cursor = [NSCursor arrowCursor];
    AppView *_appView = nullptr;
    UI::Platform *_platform = nullptr;
    UI::View *_draggingView = nullptr;
    Editor::ShortcutMap *_shortcuts = nullptr;
    UI::SemanticToPixelTranslator *_translator = nullptr;
    Graphics::Context *_context = nullptr;
    Graphics::SolidBatch *_solidBatch = nullptr;
    Graphics::GlyphBatch *_glyphBatch = nullptr;
    Graphics::DropShadow *_dropShadow = nullptr;
    std::unordered_map<int, FontInstance *> _fontInstances;
  };

  ////////////////////////////////////////////////////////////////////////////////

  struct Platform : UI::Platform {
    virtual UI::OperatingSystem operatingSystem() override {
      return UI::OperatingSystem::OSX;
    }

    virtual UI::UserAgent userAgent() override {
      return UI::UserAgent::UNKNOWN;
    }

    virtual double nowInSeconds() override {
      timeval data;
      gettimeofday(&data, nullptr);
      return data.tv_sec + data.tv_usec / 1.0e6;
    }

    virtual UI::Window *createWindow() override;

    #ifdef SKEW_GC_MARK_AND_SWEEP
      virtual void __gc_mark() override {
        UI::Platform::__gc_mark();
      }
    #endif

  private:
    NSRect _boundsForNewWindow() {
      // Determine frame padding
      auto contentRect = NSMakeRect(0, 0, 256, 256);
      auto frameRect = [NSWindow frameRectForContentRect:contentRect styleMask:_styleMask];
      auto framePadding = NSMakeSize(
        NSWidth(frameRect) - NSWidth(contentRect),
        NSHeight(frameRect) - NSHeight(contentRect));

      // Determine content size
      auto screenRect = [[NSScreen mainScreen] visibleFrame];
      auto contentLimits = [NSWindow contentRectForFrameRect:screenRect styleMask:_styleMask];
      _newWindowBounds.size = NSMakeSize(
        std::fmin(800, NSWidth(contentLimits)),
        std::fmin(600, NSHeight(contentLimits)));

      // Center the first window in the screen
      if (_isFirstWindow) {
        _newWindowBounds.origin = NSMakePoint(
          NSMidX(contentLimits) - NSMidX(_newWindowBounds),
          NSMidY(contentLimits) - NSMidY(_newWindowBounds));
        _isFirstWindow = false;
      }

      // Offset subsequent windows
      else {
        auto offset = framePadding.height;
        _newWindowBounds.origin.x += offset;
        _newWindowBounds.origin.y -= offset;

        // Wrap in x
        if (NSMaxX(_newWindowBounds) > NSMaxX(contentLimits)) {
          _newWindowBounds.origin.x = std::fmin(NSMaxX(contentLimits) - NSWidth(_newWindowBounds),
            NSMinX(contentLimits) + std::fmod(NSMinX(_newWindowBounds) - NSMinX(contentLimits), offset));
        }

        // Wrap in y
        if (NSMinY(_newWindowBounds) < NSMinY(contentLimits)) {
          auto y = NSMinY(_newWindowBounds);
          y = NSHeight(contentLimits) - y - NSHeight(_newWindowBounds) + NSMinY(contentLimits);
          y = std::fmod(std::fmod(y, offset) + offset, offset);
          y = NSHeight(contentLimits) - y - NSHeight(_newWindowBounds) + NSMinY(contentLimits);
          _newWindowBounds.origin.y = std::fmax(y, NSMinY(contentLimits));
        }
      }

      return _newWindowBounds;
    }

    bool _isFirstWindow = true;
    NSRect _newWindowBounds = NSZeroRect;
    int _styleMask = NSClosableWindowMask | NSTitledWindowMask | NSResizableWindowMask | NSMiniaturizableWindowMask;
  };
}

////////////////////////////////////////////////////////////////////////////////

@interface AppView : NSOpenGLView <NSWindowDelegate> {
@public
  CVDisplayLinkRef displayLink;
  Skew::Root<OSX::AppWindow> appWindow;
}

@end

@implementation AppView

- (id)initWithFrame:(NSRect)frame window:(NSWindow *)window platform:(UI::Platform *)platform {
  NSOpenGLPixelFormatAttribute attributes[] = { NSOpenGLPFADoubleBuffer, 0 };
  auto format = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];

  if (self = [super initWithFrame:frame pixelFormat:format]) {
    [self setWantsBestResolutionOpenGLSurface:YES];
    appWindow = new OSX::AppWindow(window, self, platform);
    appWindow->handleResize();
  }

  return self;
}

- (void)dealloc {
  CVDisplayLinkRelease(displayLink);
}

static CVReturn displayLinkCallback(CVDisplayLinkRef displayLink, const CVTimeStamp *now,
    const CVTimeStamp *outputTime, CVOptionFlags flagsIn, CVOptionFlags *flagsOut, void *context) {
  [(__bridge AppView *)context performSelectorOnMainThread:@selector(invalidate) withObject:nil waitUntilDone:NO];
  return kCVReturnSuccess;
}

- (void)prepareOpenGL {
  int swap = 1;
  [[self openGLContext] makeCurrentContext];
  [[self openGLContext] setValues:&swap forParameter:NSOpenGLCPSwapInterval];

  CVDisplayLinkCreateWithActiveCGDisplays(&displayLink);
  CVDisplayLinkSetOutputCallback(displayLink, &displayLinkCallback, (__bridge void *)self);
  CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(displayLink,
    (CGLContextObj)[[self openGLContext] CGLContextObj],
    (CGLPixelFormatObj)[[self pixelFormat] CGLPixelFormatObj]);
  appWindow->initializeOpenGL();
}

- (void)invalidate {
  [self setNeedsDisplay:YES];

  #ifdef SKEW_GC_PARALLEL
    Skew::GC::parallelCollect();
  #endif
}

- (void)drawRect:(NSRect)rect {
  appWindow->triggerFrame();
}

- (void)windowDidResize:(NSNotification *)notification {
  appWindow->handleResize();
}

- (void)windowDidChangeBackingProperties:(NSNotification *)notification {
  appWindow->handleResize();
}

- (void)windowDidBecomeKey:(NSNotification *)notification {
  appWindow->setIsActive(true);
  CVDisplayLinkStart(displayLink);
}

- (void)windowDidResignKey:(NSNotification *)notification {
  appWindow->setIsActive(false);
  CVDisplayLinkStop(displayLink);
}

- (void)keyDown:(NSEvent *)event {
  appWindow->handleKeyEvent(event);
}

- (void)insertText:(NSString *)text {
  appWindow->handleInsertText(text);
}

- (void)mouseDown:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)mouseDragged:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)mouseUp:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)mouseMoved:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)rightMouseDown:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)rightMouseDragged:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)rightMouseUp:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)otherMouseDown:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)otherMouseDragged:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)otherMouseUp:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)scrollWheel:(NSEvent *)event {
  appWindow->handleMouseEvent(event);
}

- (void)undo:(id)sender {
  appWindow->handleAction(Editor::Action::UNDO);
}

- (void)redo:(id)sender {
  appWindow->handleAction(Editor::Action::REDO);
}

- (void)cut:(id)sender {
  appWindow->handleAction(Editor::Action::CUT);
}

- (void)copy:(id)sender {
  appWindow->handleAction(Editor::Action::COPY);
}

- (void)paste:(id)sender {
  appWindow->handleAction(Editor::Action::PASTE);
}

- (void)selectAll:(id)sender {
  appWindow->handleAction(Editor::Action::SELECT_ALL);
}

- (void)insertBacktab:(id)sender {
  appWindow->handleAction(Editor::Action::INSERT_TAB_BACKWARD);
}

- (void)insertTab:(id)sender {
  appWindow->handleAction(Editor::Action::INSERT_TAB_FORWARD);
}

@end

////////////////////////////////////////////////////////////////////////////////

void OSX::AppWindow::triggerFrame() {
  if (_delegate != nullptr) {
    _delegate->triggerFrame();
  }
  if (_isInvalid) {
    render();
    _isInvalid = false;
  }
}

void OSX::AppWindow::render() {
  _isRendering = true;
  [[_appView openGLContext] makeCurrentContext];

  _context->clear(_clearColor);
  _translator->renderView(_root);
  _solidBatch->flush();
  _glyphBatch->flush();

  [[_appView openGLContext] flushBuffer];
  _isRendering = false;

  if (_needsToBeShown) {
    [_window makeKeyAndOrderFront:nil];
    _needsToBeShown = false;
  }

  #ifndef SKEW_GC_PARALLEL
    Skew::GC::blockingCollect();
  #endif
}

void OSX::AppWindow::handleResize() {
  auto bounds = [_appView bounds];
  auto pixelSize = [_appView convertRectToBacking:bounds].size;

  _width = bounds.size.width;
  _height = bounds.size.height;
  _pixelScale = [_window backingScaleFactor];

  [[_appView openGLContext] makeCurrentContext];

  _handleResize(new Vector(_width, _height), _pixelScale);

  if (_context != nullptr) {
    _context->resize(pixelSize.width, pixelSize.height);
  }

  setViewport(0, 0, _width, _height);

  for (const auto &it : _fontInstances) {
    it.second->changePixelScale(_pixelScale);
  }
}

static UI::Key keyFromEvent(NSEvent *event) {
  static std::unordered_map<int, UI::Key> map = {
    { '.',                       UI::Key::PERIOD },
    { ';',                       UI::Key::SEMICOLON },

    { '0',                       UI::Key::NUMBER_0 },
    { '1',                       UI::Key::NUMBER_1 },
    { '2',                       UI::Key::NUMBER_2 },
    { '3',                       UI::Key::NUMBER_3 },
    { '4',                       UI::Key::NUMBER_4 },
    { '5',                       UI::Key::NUMBER_5 },
    { '6',                       UI::Key::NUMBER_6 },
    { '7',                       UI::Key::NUMBER_7 },
    { '8',                       UI::Key::NUMBER_8 },
    { '9',                       UI::Key::NUMBER_9 },

    { 'a',                       UI::Key::LETTER_A },
    { 'b',                       UI::Key::LETTER_B },
    { 'c',                       UI::Key::LETTER_C },
    { 'd',                       UI::Key::LETTER_D },
    { 'e',                       UI::Key::LETTER_E },
    { 'f',                       UI::Key::LETTER_F },
    { 'g',                       UI::Key::LETTER_G },
    { 'h',                       UI::Key::LETTER_H },
    { 'i',                       UI::Key::LETTER_I },
    { 'j',                       UI::Key::LETTER_J },
    { 'k',                       UI::Key::LETTER_K },
    { 'l',                       UI::Key::LETTER_L },
    { 'm',                       UI::Key::LETTER_M },
    { 'n',                       UI::Key::LETTER_N },
    { 'o',                       UI::Key::LETTER_O },
    { 'p',                       UI::Key::LETTER_P },
    { 'q',                       UI::Key::LETTER_Q },
    { 'r',                       UI::Key::LETTER_R },
    { 's',                       UI::Key::LETTER_S },
    { 't',                       UI::Key::LETTER_T },
    { 'u',                       UI::Key::LETTER_U },
    { 'v',                       UI::Key::LETTER_V },
    { 'w',                       UI::Key::LETTER_W },
    { 'x',                       UI::Key::LETTER_X },
    { 'y',                       UI::Key::LETTER_Y },
    { 'z',                       UI::Key::LETTER_Z },

    { 'A',                       UI::Key::LETTER_A },
    { 'B',                       UI::Key::LETTER_B },
    { 'C',                       UI::Key::LETTER_C },
    { 'D',                       UI::Key::LETTER_D },
    { 'E',                       UI::Key::LETTER_E },
    { 'F',                       UI::Key::LETTER_F },
    { 'G',                       UI::Key::LETTER_G },
    { 'H',                       UI::Key::LETTER_H },
    { 'I',                       UI::Key::LETTER_I },
    { 'J',                       UI::Key::LETTER_J },
    { 'K',                       UI::Key::LETTER_K },
    { 'L',                       UI::Key::LETTER_L },
    { 'M',                       UI::Key::LETTER_M },
    { 'N',                       UI::Key::LETTER_N },
    { 'O',                       UI::Key::LETTER_O },
    { 'P',                       UI::Key::LETTER_P },
    { 'Q',                       UI::Key::LETTER_Q },
    { 'R',                       UI::Key::LETTER_R },
    { 'S',                       UI::Key::LETTER_S },
    { 'T',                       UI::Key::LETTER_T },
    { 'U',                       UI::Key::LETTER_U },
    { 'V',                       UI::Key::LETTER_V },
    { 'W',                       UI::Key::LETTER_W },
    { 'X',                       UI::Key::LETTER_X },
    { 'Y',                       UI::Key::LETTER_Y },
    { 'Z',                       UI::Key::LETTER_Z },

    { 27,                        UI::Key::ESCAPE },
    { NSCarriageReturnCharacter, UI::Key::ENTER },
    { NSDeleteCharacter,         UI::Key::BACKSPACE },
    { NSDeleteFunctionKey,       UI::Key::DELETE },
    { NSDownArrowFunctionKey,    UI::Key::ARROW_DOWN },
    { NSEndFunctionKey,          UI::Key::END },
    { NSHomeFunctionKey,         UI::Key::HOME },
    { NSLeftArrowFunctionKey,    UI::Key::ARROW_LEFT },
    { NSPageDownFunctionKey,     UI::Key::PAGE_DOWN },
    { NSPageUpFunctionKey,       UI::Key::PAGE_UP },
    { NSRightArrowFunctionKey,   UI::Key::ARROW_RIGHT },
    { NSUpArrowFunctionKey,      UI::Key::ARROW_UP },
  };
  auto characters = [event charactersIgnoringModifiers];

  if ([characters length] == 1) {
    auto it = map.find([characters characterAtIndex:0]);

    if (it != map.end()) {
      return it->second;
    }
  }

  return UI::Key::NONE;
}

static int modifiersFromEvent(NSEvent *event) {
  auto flags = [event modifierFlags];
  return
    ((flags & NSShiftKeyMask) != 0 ? UI::Modifiers::SHIFT : 0) |
    ((flags & NSCommandKeyMask) != 0 ? UI::Modifiers::META : 0) |
    ((flags & NSAlternateKeyMask) != 0 ? UI::Modifiers::ALT : 0) |
    ((flags & NSControlKeyMask) != 0 ? UI::Modifiers::CONTROL : 0);
}

void OSX::AppWindow::handleKeyEvent(NSEvent *event) {
  auto key = keyFromEvent(event);

  if (key != UI::Key::NONE) {
    auto modifiers = modifiersFromEvent(event);
    auto action = _shortcuts->get(key, modifiers);

    // Keyboard shortcuts take precedence over text insertion
    if (action != Editor::Action::NONE) {
      handleAction(action);
      return;
    }

    // This isn't handled by interpretKeyEvents for some reason
    if (key == UI::Key::ENTER && modifiers == 0) {
      handleInsertText(@"\n");
      return;
    }
  }

  [_appView interpretKeyEvents:@[event]];
}

void OSX::AppWindow::handleInsertText(NSString *text) {
  dispatchEvent(new UI::TextEvent(UI::EventType::TEXT, viewWithFocus(), [text UTF8String], false));
}

UI::MouseEvent *OSX::AppWindow::_mouseEventFromEvent(UI::EventType type, NSEvent *event, Vector *delta) {
  auto point = [event locationInWindow];
  auto height = [[[event window] contentView] bounds].size.height;
  auto viewLocation = new Vector(point.x, height - point.y);
  auto target = _draggingView != nullptr ? _draggingView : viewFromLocation(viewLocation);
  int clickCount = type == UI::EventType::MOUSE_DOWN ? (int)[event clickCount] : 0;
  return new UI::MouseEvent(type, target, viewLocation, modifiersFromEvent(event), clickCount, delta);
}

void OSX::AppWindow::handleMouseEvent(NSEvent *event) {
  switch ([event type]) {
    case NSLeftMouseDown:
    case NSOtherMouseDown:
    case NSRightMouseDown: {
      _draggingView = nullptr;
      _draggingView = dispatchEvent(_mouseEventFromEvent(UI::EventType::MOUSE_DOWN, event, nullptr));
      break;
    }

    case NSMouseMoved:
    case NSLeftMouseDragged:
    case NSOtherMouseDragged:
    case NSRightMouseDragged: {
      dispatchEvent(_mouseEventFromEvent(UI::EventType::MOUSE_MOVE, event, nullptr));
      break;
    }

    case NSLeftMouseUp:
    case NSOtherMouseUp:
    case NSRightMouseUp: {
      dispatchEvent(_mouseEventFromEvent(UI::EventType::MOUSE_UP, event, nullptr));
      _draggingView = nullptr;
      break;
    }

    case NSScrollWheel: {
      dispatchEvent(_mouseEventFromEvent(UI::EventType::MOUSE_SCROLL, event, new Vector(-[event scrollingDeltaX], -[event scrollingDeltaY])));
      break;
    }
  }

  // Only show the cursor if the mouse is over the window
  if (NSPointInRect([event locationInWindow], [_appView frame])) {
    [_cursor set];
  } else {
    [[NSCursor arrowCursor] set];
  }
}

void OSX::AppWindow::handleAction(Editor::Action action) {
  switch (action) {
    case Editor::Action::CUT:
    case Editor::Action::COPY:
    case Editor::Action::PASTE: {
      auto clipboard = [NSPasteboard generalPasteboard];
      Skew::string text;

      // Load text from clipboard
      if (auto data = [clipboard stringForType:NSPasteboardTypeString]) {
        text = [data UTF8String];
      }

      // Send event
      auto type =
        action == Editor::Action::CUT ? UI::EventType::CLIPBOARD_CUT :
        action == Editor::Action::COPY ? UI::EventType::CLIPBOARD_COPY :
        UI::EventType::CLIPBOARD_PASTE;
      auto event = new UI::ClipboardEvent(type, viewWithFocus(), text);
      dispatchEvent(event);

      // Save text to clipboard
      if (event->text != text) {
        [clipboard clearContents];
        [clipboard setString:[NSString stringWithUTF8String:event->text.c_str()] forType:NSPasteboardTypeString];
      }
      break;
    }

    default: {
      if (_delegate != nullptr) {
        _delegate->triggerAction(action);
      }
      break;
    }
  }
}

UI::Window *OSX::Platform::createWindow() {
  auto bounds = _boundsForNewWindow();
  auto window = [[NSWindow alloc] initWithContentRect:bounds styleMask:_styleMask backing:NSBackingStoreBuffered defer:NO];
  auto appView = [[AppView alloc] initWithFrame:bounds window:window platform:this];

  [window setCollectionBehavior:[window collectionBehavior] | NSWindowCollectionBehaviorFullScreenPrimary];
  [window setContentMinSize:NSMakeSize(4, 4)];
  [window setAcceptsMouseMovedEvents:YES];
  [window setDelegate:appView];
  [window setContentView:appView];
  [window makeFirstResponder:appView];

  return appView->appWindow;
}

////////////////////////////////////////////////////////////////////////////////

@interface AppDelegate : NSObject <NSApplicationDelegate> {
  Skew::Root<Editor::App> app;
}

@end

@implementation AppDelegate

- (void)createNewWindow:(id)sender {
  app->createWindow();
}

- (void)openIssueTracker:(id)sender {
  [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:@"https://github.com/evanw/sky/issues"]];
}

- (void)applicationDidFinishLaunching:(id)sender {
  auto mainMenu = [[NSMenu alloc] init];
  auto name = [[NSProcessInfo processInfo] processName];

  auto appMenu = [[NSMenu alloc] init];
  [[mainMenu addItemWithTitle:@"" action:nil keyEquivalent:@""] setSubmenu:appMenu];
  [appMenu addItemWithTitle:[@"Hide " stringByAppendingString:name] action:@selector(hide:) keyEquivalent:@"h"];
  [[appMenu addItemWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"] setKeyEquivalentModifierMask:NSCommandKeyMask | NSAlternateKeyMask];
  [appMenu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""];
  [appMenu addItem:[NSMenuItem separatorItem]];
  [appMenu addItemWithTitle:[@"Quit " stringByAppendingString:name] action:@selector(terminate:) keyEquivalent:@"q"];

  auto fileMenu = [[NSMenu alloc] init];
  [fileMenu setTitle:@"File"];
  [[mainMenu addItemWithTitle:@"" action:nil keyEquivalent:@""] setSubmenu:fileMenu];
  [fileMenu addItemWithTitle:@"New" action:@selector(createNewWindow:) keyEquivalent:@"n"];
  [fileMenu addItemWithTitle:@"Close" action:@selector(performClose:) keyEquivalent:@"w"];

  auto editMenu = [[NSMenu alloc] init];
  [editMenu setTitle:@"Edit"];
  [[mainMenu addItemWithTitle:@"" action:nil keyEquivalent:@""] setSubmenu:editMenu];
  [editMenu addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
  [[editMenu addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"z"] setKeyEquivalentModifierMask:NSCommandKeyMask | NSShiftKeyMask];
  [editMenu addItem:[NSMenuItem separatorItem]];
  [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
  [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
  [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
  [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];

  auto helpMenu = [[NSMenu alloc] init];
  [helpMenu setTitle:@"Help"];
  [[mainMenu addItemWithTitle:@"" action:nil keyEquivalent:@""] setSubmenu:helpMenu];
  [helpMenu addItemWithTitle:@"Issue Tracker" action:@selector(openIssueTracker:) keyEquivalent:@""];

  [[NSApplication sharedApplication] setMainMenu:mainMenu];

  app = new Editor::App(new OSX::Platform());
}

@end

////////////////////////////////////////////////////////////////////////////////

int main() {
  @autoreleasepool {
    auto application = [NSApplication sharedApplication];
    auto delegate = [[AppDelegate alloc] init]; // This must be stored in a local variable because of ARC
    [application setDelegate:delegate];
    [application activateIgnoringOtherApps:YES];
    [application run];
  }
}
