#include <iostream>
#include <chrono>
#include <future>
#include "Main.h"
#include "Primitive/TerrainInterpolation.h"
#include "Primitive/TerrainComposition.h"
#include <nlohmann/json.hpp>

InputHandler::InputHandler(std::shared_ptr<Core> core) { _core = core; }

void InputHandler::cursorNotify(float xPos, float yPos) {}

void InputHandler::mouseNotify(int button, int action, int mods) {}

void InputHandler::keyNotify(int key, int scancode, int action, int mods) {
#ifndef __ANDROID__
  if ((action == GLFW_RELEASE && key == GLFW_KEY_C)) {
    if (_core->getEngineState()->getInput()->cursorEnabled()) {
      _core->getEngineState()->getInput()->showCursor(false);
    } else {
      _core->getEngineState()->getInput()->showCursor(true);
    }
  }
#endif
}

void InputHandler::charNotify(unsigned int code) {}

void InputHandler::scrollNotify(double xOffset, double yOffset) {}

void Main::_createTerrainPhong(std::string path) {
  auto heightmap = _core->loadImageCPU(path);
  switch (_interpolationMode) {
    case InrepolationMode::INTERPOLATION:
      _terrain = _core->createTerrainInterpolation(heightmap);
      if (_stripeLeft.has_value())
        std::dynamic_pointer_cast<TerrainInterpolation>(_terrain)->setStripes(_stripeLeft.value(), _stripeTop.value(),
                                                                              _stripeRight.value(), _stripeBot.value());
      break;
    case InrepolationMode::COMPOSITION:
      _terrain = _core->createTerrainComposition(heightmap);
      break;
  }
  _terrain->setPatchNumber(_patchX, _patchY);
  _terrain->setPatchRotations(_patchRotationsIndex);
  _terrain->setPatchTextures(_patchTextures);
  _terrain->initialize(_core->getCommandBufferApplication());
  _terrain->setMaterial(_materialPhong);
  _terrain->setScale(_terrainScale);
  _terrain->setTranslate(_terrainPosition);
  _terrain->setTessellationLevel(_minTessellationLevel, _maxTessellationLevel);
  _terrain->setTesselationDistance(_minDistance, _maxDistance);
  _terrain->setColorHeightLevels(_heightLevels);
  _terrain->setHeight(_heightScale, _heightShift);
}

void Main::_createTerrainPBR(std::string path) {
  auto heightmap = _core->loadImageCPU(path);
  switch (_interpolationMode) {
    case InrepolationMode::INTERPOLATION:
      _terrain = _core->createTerrainInterpolation(heightmap);
      if (_stripeLeft.has_value())
        std::dynamic_pointer_cast<TerrainInterpolation>(_terrain)->setStripes(_stripeLeft.value(), _stripeTop.value(),
                                                                              _stripeRight.value(), _stripeBot.value());
      break;
    case InrepolationMode::COMPOSITION:
      _terrain = _core->createTerrainComposition(heightmap);
      break;
  }
  _terrain->setPatchNumber(_patchX, _patchY);
  _terrain->setPatchRotations(_patchRotationsIndex);
  _terrain->setPatchTextures(_patchTextures);
  _terrain->initialize(_core->getCommandBufferApplication());
  _terrain->setMaterial(_materialPBR);
  _terrain->setScale(_terrainScale);
  _terrain->setTranslate(_terrainPosition);
  _terrain->setTessellationLevel(_minTessellationLevel, _maxTessellationLevel);
  _terrain->setTesselationDistance(_minDistance, _maxDistance);
  _terrain->setColorHeightLevels(_heightLevels);
  _terrain->setHeight(_heightScale, _heightShift);
}

void Main::_createTerrainColor(std::string path) {
  auto heightmap = _core->loadImageCPU(path);
  switch (_interpolationMode) {
    case InrepolationMode::INTERPOLATION:
      _terrain = _core->createTerrainInterpolation(heightmap);
      if (_stripeLeft.has_value())
        std::dynamic_pointer_cast<TerrainInterpolation>(_terrain)->setStripes(_stripeLeft.value(), _stripeTop.value(),
                                                                              _stripeRight.value(), _stripeBot.value());
      break;
    case InrepolationMode::COMPOSITION:
      _terrain = _core->createTerrainComposition(heightmap);
      break;
  }
  _terrain->setPatchNumber(_patchX, _patchY);
  _terrain->setPatchRotations(_patchRotationsIndex);
  _terrain->setPatchTextures(_patchTextures);
  _terrain->initialize(_core->getCommandBufferApplication());
  _terrain->setMaterial(_materialColor);
  _terrain->setScale(_terrainScale);
  _terrain->setTranslate(_terrainPosition);
  _terrain->setTessellationLevel(_minTessellationLevel, _maxTessellationLevel);
  _terrain->setTesselationDistance(_minDistance, _maxDistance);
  _terrain->setColorHeightLevels(_heightLevels);
  _terrain->setHeight(_heightScale, _heightShift);
}

void Main::_loadTerrain(std::string path) {
  std::ifstream file(path);
  nlohmann::json inputJSON;
  file >> inputJSON;
  if (inputJSON["stripe"].is_null() == false) {
    _stripeLeft = inputJSON["stripe"][0];
    _stripeRight = inputJSON["stripe"][1];
    _stripeTop = inputJSON["stripe"][2];
    _stripeBot = inputJSON["stripe"][3];
  }

  _patchX = inputJSON["patches"][0];
  _patchY = inputJSON["patches"][1];
  _patchRotationsIndex.resize(_patchX * _patchY);
  for (int i = 0; i < inputJSON["rotation"].size(); i++) {
    _patchRotationsIndex[i] = inputJSON["rotation"][i];
  }
  _patchTextures.resize(_patchX * _patchY);
  for (int i = 0; i < inputJSON["textures"].size(); i++) {
    _patchTextures[i] = inputJSON["textures"][i];
  }
}

void Main::_createTerrainDebug(std::string path) {
  switch (_interpolationMode) {
    case InrepolationMode::INTERPOLATION:
      _terrainDebug = std::make_shared<TerrainInterpolationDebug>(
          _core->loadImageCPU(path), std::pair{_patchX, _patchY}, _core->getCommandBufferApplication(), _core->getGUI(),
          _core->getGameState(), _core->getEngineState());
      break;
    case InrepolationMode::COMPOSITION:
      _terrainDebug = std::make_shared<TerrainCompositionDebug>(_core->loadImageCPU(path), std::pair{_patchX, _patchY},
                                                                _core->getCommandBufferApplication(), _core->getGUI(),
                                                                _core->getGameState(), _core->getEngineState());
      break;
  }

  _core->getEngineState()->getInput()->subscribe(std::dynamic_pointer_cast<InputSubscriber>(_terrainDebug));
  _terrainDebug->setTerrainPhysics(_terrainPhysics, _terrainCPU);
  _terrainDebug->setMaterial(_materialColor);

  _terrainDebug->setTessellationLevel(_minTessellationLevel, _maxTessellationLevel);
  _terrainDebug->setTesselationDistance(_minDistance, _maxDistance);
  _terrainDebug->setColorHeightLevels(_heightLevels);
  _terrainDebug->setHeight(_heightScale, _heightShift);
  _terrainDebug->patchEdge(_showPatches);
  _terrainDebug->showLoD(_showLoD);
  if (_showWireframe) {
    _terrainDebug->setDrawType(DrawType::WIREFRAME);
  }
  if (_showNormals) {
    _terrainDebug->setDrawType(DrawType::NORMAL);
  }
  if (_showWireframe == false && _showNormals == false) {
    _terrainDebug->setDrawType(DrawType::FILL);
  }
  _terrainDebug->setScale(_terrainScale);
  _terrainDebug->setTranslate(_terrainPositionDebug);
}

Main::Main() {
  int mipMapLevels = 4;
  auto settings = std::make_shared<Settings>();
  settings->setName("Terrain");
  settings->setClearColor({0.01f, 0.01f, 0.01f, 1.f});
  // TODO: fullscreen if resolution is {0, 0}
  // TODO: validation layers complain if resolution is {2560, 1600}
  settings->setResolution(std::tuple{1920, 1080});
  // for HDR, linear 16 bit per channel to represent values outside of 0-1 range (UNORM - float [0, 1], SFLOAT - float)
  // https://registry.khronos.org/vulkan/specs/1.1/html/vkspec.html#_identification_of_formats
  settings->setGraphicColorFormat(VK_FORMAT_R32G32B32A32_SFLOAT);
  settings->setSwapchainColorFormat(VK_FORMAT_B8G8R8A8_UNORM);
  // SRGB the same as UNORM but + gamma conversion out of box (!)
  settings->setLoadTextureColorFormat(VK_FORMAT_R8G8B8A8_SRGB);
  settings->setLoadTextureAuxilaryFormat(VK_FORMAT_R8G8B8A8_UNORM);
  settings->setAnisotropicSamples(4);
  settings->setDepthFormat(VK_FORMAT_D32_SFLOAT);
  settings->setMaxFramesInFlight(2);
  settings->setThreadsInPool(6);
  settings->setDesiredFPS(1000);

  _core = std::make_shared<Core>(settings);
  _core->initialize();
  _core->startRecording();
  _camera = std::make_shared<CameraFly>(_core->getEngineState());
  _camera->setProjectionParameters(60.f, 0.1f, 100.f);
  _camera->setSpeed(0.05f, 0.01f);
  _core->getEngineState()->getInput()->subscribe(std::dynamic_pointer_cast<InputSubscriber>(_camera));
  _inputHandler = std::make_shared<InputHandler>(_core);
  _core->getEngineState()->getInput()->subscribe(std::dynamic_pointer_cast<InputSubscriber>(_inputHandler));
  _core->setCamera(_camera);

  _pointLightVertical = _core->createPointLight();
  _pointLightVertical->setColor(glm::vec3(1.f, 1.f, 1.f));
  _pointLightHorizontal = _core->createPointLight();
  _pointLightHorizontal->setColor(glm::vec3(1.f, 1.f, 1.f));

  auto ambientLight = _core->createAmbientLight();
  ambientLight->setColor({0.1f, 0.1f, 0.1f});
  // cube colored light
  _cubeColoredLightVertical = _core->createShape3D(ShapeType::CUBE);
  _cubeColoredLightVertical->setScale(glm::vec3(0.3f, 0.3f, 0.3f));
  _cubeColoredLightVertical->getMesh()->setColor(
      std::vector{_cubeColoredLightVertical->getMesh()->getVertexData().size(), glm::vec3(1.f, 1.f, 1.f)},
      _core->getCommandBufferApplication());
  _core->addDrawable(_cubeColoredLightVertical);

  _cubeColoredLightHorizontal = _core->createShape3D(ShapeType::CUBE);
  _cubeColoredLightHorizontal->setScale(glm::vec3(0.3f, 0.3f, 0.3f));
  _cubeColoredLightHorizontal->getMesh()->setColor(
      std::vector{_cubeColoredLightHorizontal->getMesh()->getVertexData().size(), glm::vec3(1.f, 1.f, 1.f)},
      _core->getCommandBufferApplication());
  _core->addDrawable(_cubeColoredLightHorizontal);

  _sphereClickDebug = _core->createShape3D(ShapeType::SPHERE);
  _sphereClickDebug->setScale(glm::vec3(0.005f, 0.005f, 0.005f));

  auto fillMaterialPhong = [core = _core](std::shared_ptr<MaterialPhong> material) {
    if (material->getBaseColor().size() == 0)
      material->setBaseColor(std::vector{4, core->getResourceManager()->getTextureOne()});
    if (material->getNormal().size() == 0)
      material->setNormal(std::vector{4, core->getResourceManager()->getTextureZero()});
    if (material->getSpecular().size() == 0)
      material->setSpecular(std::vector{4, core->getResourceManager()->getTextureZero()});
  };

  auto fillMaterialPBR = [core = _core](std::shared_ptr<MaterialPBR> material) {
    if (material->getBaseColor().size() == 0)
      material->setBaseColor(std::vector{4, core->getResourceManager()->getTextureOne()});
    if (material->getNormal().size() == 0)
      material->setNormal(std::vector{4, core->getResourceManager()->getTextureZero()});
    if (material->getMetallic().size() == 0)
      material->setMetallic(std::vector{4, core->getResourceManager()->getTextureZero()});
    if (material->getRoughness().size() == 0)
      material->setRoughness(std::vector{4, core->getResourceManager()->getTextureZero()});
    if (material->getOccluded().size() == 0)
      material->setOccluded(std::vector{4, core->getResourceManager()->getTextureZero()});
    if (material->getEmissive().size() == 0)
      material->setEmissive(std::vector{4, core->getResourceManager()->getTextureZero()});
    material->setDiffuseIBL(core->getResourceManager()->getCubemapZero()->getTexture());
    material->setSpecularIBL(core->getResourceManager()->getCubemapZero()->getTexture(),
                             core->getResourceManager()->getTextureZero());
  };

  {
    auto tile0 = _core->createTexture("../assets/desert/albedo.png", settings->getLoadTextureColorFormat(),
                                      mipMapLevels);
    auto tile1 = _core->createTexture("../assets/grass/albedo.png", settings->getLoadTextureColorFormat(),
                                      mipMapLevels);
    auto tile2 = _core->createTexture("../assets/rock/albedo.png", settings->getLoadTextureColorFormat(), mipMapLevels);
    auto tile3 = _core->createTexture("../assets/ground/albedo.png", settings->getLoadTextureColorFormat(),
                                      mipMapLevels);
    _materialColor = _core->createMaterialColor(MaterialTarget::TERRAIN);
    _materialColor->setBaseColor({tile0, tile1, tile2, tile3});
  }

  {
    auto tile0Color = _core->createTexture("../assets/desert/albedo.png", settings->getLoadTextureColorFormat(),
                                           mipMapLevels);
    auto tile0Normal = _core->createTexture("../assets/desert/normal.png", settings->getLoadTextureAuxilaryFormat(),
                                            mipMapLevels);
    auto tile1Color = _core->createTexture("../assets/grass/albedo.png", settings->getLoadTextureColorFormat(),
                                           mipMapLevels);
    auto tile1Normal = _core->createTexture("../assets/grass/normal.png", settings->getLoadTextureAuxilaryFormat(),
                                            mipMapLevels);
    auto tile2Color = _core->createTexture("../assets/rock/albedo.png", settings->getLoadTextureColorFormat(),
                                           mipMapLevels);
    auto tile2Normal = _core->createTexture("../assets/rock/normal.png", settings->getLoadTextureAuxilaryFormat(),
                                            mipMapLevels);
    auto tile3Color = _core->createTexture("../assets/ground/albedo.png", settings->getLoadTextureColorFormat(),
                                           mipMapLevels);
    auto tile3Normal = _core->createTexture("../assets/ground/normal.png", settings->getLoadTextureAuxilaryFormat(),
                                            mipMapLevels);

    _materialPhong = _core->createMaterialPhong(MaterialTarget::TERRAIN);
    _materialPhong->setBaseColor({tile0Color, tile1Color, tile2Color, tile3Color});
    _materialPhong->setNormal({tile0Normal, tile1Normal, tile2Normal, tile3Normal});
    fillMaterialPhong(_materialPhong);
  }

  {
    auto tile0Color = _core->createTexture("../assets/desert/albedo.png", settings->getLoadTextureColorFormat(),
                                           mipMapLevels);
    auto tile0Normal = _core->createTexture("../assets/desert/normal.png", settings->getLoadTextureAuxilaryFormat(),
                                            mipMapLevels);
    auto tile0Metallic = _core->createTexture("../assets/desert/metallic.png", settings->getLoadTextureAuxilaryFormat(),
                                              mipMapLevels);
    auto tile0Roughness = _core->createTexture("../assets/desert/roughness.png",
                                               settings->getLoadTextureAuxilaryFormat(), mipMapLevels);
    auto tile0AO = _core->createTexture("../assets/desert/ao.png", settings->getLoadTextureAuxilaryFormat(),
                                        mipMapLevels);

    auto tile1Color = _core->createTexture("../assets/grass/albedo.png", settings->getLoadTextureColorFormat(),
                                           mipMapLevels);
    auto tile1Normal = _core->createTexture("../assets/grass/normal.png", settings->getLoadTextureAuxilaryFormat(),
                                            mipMapLevels);
    auto tile1Metallic = _core->createTexture("../assets/grass/metallic.png", settings->getLoadTextureAuxilaryFormat(),
                                              mipMapLevels);
    auto tile1Roughness = _core->createTexture("../assets/grass/roughness.png",
                                               settings->getLoadTextureAuxilaryFormat(), mipMapLevels);
    auto tile1AO = _core->createTexture("../assets/grass/ao.png", settings->getLoadTextureAuxilaryFormat(),
                                        mipMapLevels);

    auto tile2Color = _core->createTexture("../assets/rock/albedo.png", settings->getLoadTextureColorFormat(),
                                           mipMapLevels);
    auto tile2Normal = _core->createTexture("../assets/rock/normal.png", settings->getLoadTextureAuxilaryFormat(),
                                            mipMapLevels);
    auto tile2Metallic = _core->createTexture("../assets/rock/metallic.png", settings->getLoadTextureAuxilaryFormat(),
                                              mipMapLevels);
    auto tile2Roughness = _core->createTexture("../assets/rock/roughness.png", settings->getLoadTextureAuxilaryFormat(),
                                               mipMapLevels);
    auto tile2AO = _core->createTexture("../assets/rock/ao.png", settings->getLoadTextureAuxilaryFormat(),
                                        mipMapLevels);

    auto tile3Color = _core->createTexture("../assets/ground/albedo.png", settings->getLoadTextureColorFormat(),
                                           mipMapLevels);
    auto tile3Normal = _core->createTexture("../assets/ground/normal.png", settings->getLoadTextureAuxilaryFormat(),
                                            mipMapLevels);
    auto tile3Metallic = _core->createTexture("../assets/ground/metallic.png", settings->getLoadTextureAuxilaryFormat(),
                                              mipMapLevels);
    auto tile3Roughness = _core->createTexture("../assets/ground/roughness.png",
                                               settings->getLoadTextureAuxilaryFormat(), mipMapLevels);
    auto tile3AO = _core->createTexture("../assets/ground/ao.png", settings->getLoadTextureAuxilaryFormat(),
                                        mipMapLevels);

    _materialPBR = _core->createMaterialPBR(MaterialTarget::TERRAIN);
    _materialPBR->setBaseColor({tile0Color, tile1Color, tile2Color, tile3Color});
    _materialPBR->setNormal({tile0Normal, tile1Normal, tile2Normal, tile3Normal});
    _materialPBR->setMetallic({tile0Metallic, tile1Metallic, tile2Metallic, tile3Metallic});
    _materialPBR->setRoughness({tile0Roughness, tile1Roughness, tile2Roughness, tile3Roughness});
    _materialPBR->setOccluded({tile0AO, tile1AO, tile2AO, tile3AO});
    fillMaterialPBR(_materialPBR);
  }

  switch (_typeIndex) {
    case 0:
      _createTerrainColor("../assets/heightmap.png");
      break;
    case 1:
      _createTerrainPhong("../assets/heightmap.png");
      break;
    case 2:
      _createTerrainPBR("../assets/heightmap.png");
      break;
  }
  _core->addDrawable(_terrain);

  auto terrainCPU = _core->loadImageCPU("../assets/heightmap.png");
  auto [terrainWidth, terrainHeight] = terrainCPU->getResolution();
  _terrainPositionDebug = glm::vec3(_terrainPositionDebug.x, _terrainPositionDebug.y, _terrainPositionDebug.z);

  _physicsManager = std::make_shared<PhysicsManager>();
  _terrainPhysics = std::make_shared<TerrainPhysics>(_core->loadImageCPU("../assets/heightmap.png"),
                                                     _terrainPositionDebug, _terrainScale, std::tuple{64, 16},
                                                     _physicsManager, _core->getGameState(), _core->getEngineState());
  _terrainCPU = _core->createTerrainCPU(_terrainPhysics->getHeights(), terrainCPU->getResolution());
  _terrainCPU->setDrawType(DrawType::WIREFRAME);
  _terrainCPU->setTranslate(_terrainPositionDebug);
  _terrainCPU->setScale(_terrainScale);

  _createTerrainDebug("../assets/heightmap.png");

  _core->endRecording();

  _core->registerUpdate(std::bind(&Main::update, this));
  // can be lambda passed that calls reset
  _core->registerReset(std::bind(&Main::reset, this, std::placeholders::_1, std::placeholders::_2));
}

void Main::update() {
  static float i = 0;
  // update light position
  float radius = 15.f;
  static float angleHorizontal = 90.f;
  glm::vec3 lightPositionHorizontal = glm::vec3(radius * cos(glm::radians(angleHorizontal)), radius,
                                                radius * sin(glm::radians(angleHorizontal)));
  static float angleVertical = 0.f;
  glm::vec3 lightPositionVertical = glm::vec3(0.f, radius * sin(glm::radians(angleVertical)),
                                              radius * cos(glm::radians(angleVertical)));

  _pointLightVertical->getCamera()->setPosition(lightPositionVertical);
  _cubeColoredLightVertical->setTranslate(lightPositionVertical);
  _pointLightHorizontal->getCamera()->setPosition(lightPositionHorizontal);
  _cubeColoredLightHorizontal->setTranslate(lightPositionHorizontal);

  i += 0.1f;
  angleHorizontal += 0.05f;
  angleVertical += 0.1f;
  auto [FPSLimited, FPSReal] = _core->getFPS();
  auto [widthScreen, heightScreen] = _core->getEngineState()->getSettings()->getResolution();
  _core->getGUI()->startWindow("Terrain");
  _core->getGUI()->setWindowPosition({20, 20});
  if (_core->getGUI()->startTree("Info")) {
    _core->getGUI()->drawText({"Limited FPS: " + std::to_string(FPSLimited)});
    _core->getGUI()->drawText({"Maximum FPS: " + std::to_string(FPSReal)});
    _core->getGUI()->drawText({"Press 'c' to turn cursor on/off"});
    _core->getGUI()->endTree();
  }
  if (_core->getGUI()->startTree("Toggles")) {
    std::map<std::string, int*> terrainType;
    terrainType["##Type"] = &_typeIndex;
    if (_core->getGUI()->drawListBox({"Color", "Phong", "PBR"}, terrainType, 3)) {
      _core->startRecording();
      switch (_typeIndex) {
        case 0:
          _terrain->setMaterial(_materialColor);
          break;
        case 1:
          _terrain->setMaterial(_materialPhong);
          break;
        case 2:
          _terrain->setMaterial(_materialPBR);
          break;
      }
      _core->endRecording();
    }

    std::map<std::string, int*> patchesNumber{{"Patch x", &_patchX}, {"Patch y", &_patchY}};
    if (_core->getGUI()->drawInputInt(patchesNumber)) {
      _core->startRecording();
      if (_showTerrain) _core->removeDrawable(_terrain);
      switch (_typeIndex) {
        case 0:
          _createTerrainColor("../assets/heightmap.png");
          break;
        case 1:
          _createTerrainPhong("../assets/heightmap.png");
          break;
        case 2:
          _createTerrainPBR("../assets/heightmap.png");
          break;
      }
      if (_showTerrain) _core->addDrawable(_terrain);
      _core->endRecording();
    }

    std::map<std::string, int*> tesselationLevels{{"Tesselation min", &_minTessellationLevel},
                                                  {"Tesselation max", &_maxTessellationLevel}};
    if (_core->getGUI()->drawInputInt(tesselationLevels)) {
      _terrain->setTessellationLevel(_minTessellationLevel, _maxTessellationLevel);
    }

    std::map<std::string, int*> interpolationType;
    interpolationType["##Interpolation"] = &_interpolationIndex;
    if (_core->getGUI()->drawListBox({"Interpolation", "Composition"}, interpolationType, 2)) {
      switch (_interpolationIndex) {
        case 0:
          _interpolationMode = InrepolationMode::INTERPOLATION;
          break;
        case 1:
          _interpolationMode = InrepolationMode::COMPOSITION;
          break;
      }
      _core->startRecording();
      if (_showTerrain) _core->removeDrawable(_terrain);
      switch (_typeIndex) {
        case 0:
          _createTerrainColor("../assets/heightmap.png");
          break;
        case 1:
          _createTerrainPhong("../assets/heightmap.png");
          break;
        case 2:
          _createTerrainPBR("../assets/heightmap.png");
          break;
      }
      if (_showTerrain) _core->addDrawable(_terrain);

      if (_showDebug) _core->removeDrawable(_terrainDebug);
      _createTerrainDebug("../assets/heightmap.png");
      if (_showDebug) _core->addDrawable(_terrainDebug);

      _core->endRecording();
    }

    _core->getGUI()->drawInputText("##Path", _terrainPath, sizeof(_terrainPath));

    if (_core->getGUI()->drawButton("Load terrain")) {
      _core->startRecording();
      if (_showTerrain) _core->removeDrawable(_terrain);
      _loadTerrain(std::string(_terrainPath) + ".json");
      switch (_typeIndex) {
        case 0:
          _createTerrainColor(std::string(_terrainPath) + ".png");
          break;
        case 1:
          _createTerrainPhong(std::string(_terrainPath) + ".png");
          break;
        case 2:
          _createTerrainPBR(std::string(_terrainPath) + ".png");
          break;
      }
      if (_showTerrain) _core->addDrawable(_terrain);
      _core->endRecording();
    }

    _core->getGUI()->endTree();
  }

  if (_core->getGUI()->drawCheckbox({{"Show Terrain", &_showTerrain}})) {
    if (_showTerrain == false) {
      _core->removeDrawable(_terrain);
    } else {
      _core->addDrawable(_terrain);
    }
  }
  if (_core->getGUI()->drawCheckbox({{"Show Debug", &_showDebug}})) {
    if (_showDebug == false) {
      _core->removeDrawable(_terrainDebug);
      _core->removeDrawable(_terrainCPU);
    } else {
      _core->addDrawable(_terrainDebug);
      _core->addDrawable(_terrainCPU);
    }
  }
  _core->getGUI()->endWindow();

  auto hitPosition = _terrainDebug->getHitCoords();
  if (hitPosition) {
    _sphereClickDebug->setTranslate(hitPosition.value());
    _core->addDrawable(_sphereClickDebug);
  }

  if (_showDebug) {
    _core->getGUI()->startWindow("Editor");
    _core->getGUI()->setWindowPosition({widthScreen - std::get<0>(_core->getGUI()->getWindowSize()) - 20, 20});
    _core->startRecording();
    _terrainDebug->drawDebug(_core->getCommandBufferApplication());
    _core->endRecording();
    _core->getGUI()->endWindow();
  }
}

void Main::reset(int width, int height) { _camera->setAspect((float)width / (float)height); }

void Main::start() { _core->draw(); }

int main() {
  try {
    auto main = std::make_shared<Main>();
    main->start();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}