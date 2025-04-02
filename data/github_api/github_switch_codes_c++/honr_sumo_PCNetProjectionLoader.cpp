/****************************************************************************/
/// @file    PCNetProjectionLoader.cpp
/// @author  Daniel Krajzewicz
/// @date    Thu, 02.11.2006
/// @version $Id$
///
// A reader for a SUMO network's projection description
/****************************************************************************/
// SUMO, Simulation of Urban MObility; see http://sumo.sourceforge.net/
// Copyright 2001-2010 DLR (http://www.dlr.de/) and contributors
/****************************************************************************/
//
//   This program is free software; you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation; either version 2 of the License, or
//   (at your option) any later version.
//
/****************************************************************************/


// ===========================================================================
// included modules
// ===========================================================================
#ifdef _MSC_VER
#include <windows_config.h>
#else
#include <config.h>
#endif

#include <string>
#include <map>
#include <fstream>
#include <utils/options/OptionsCont.h>
#include <utils/options/Option.h>
#include <utils/common/StdDefs.h>
#include <polyconvert/PCPolyContainer.h>
#include "PCNetProjectionLoader.h"
#include <utils/common/RGBColor.h>
#include <utils/geom/GeomHelper.h>
#include <utils/geom/Boundary.h>
#include <utils/geom/Position2D.h>
#include <utils/geom/GeoConvHelper.h>
#include <utils/xml/XMLSubSys.h>
#include <utils/geom/GeomConvHelper.h>
#include <utils/common/MsgHandler.h>
#include <utils/common/FileHelpers.h>

#ifdef CHECK_MEMORY_LEAKS
#include <foreign/nvwa/debug_new.h>
#endif // CHECK_MEMORY_LEAKS


// ===========================================================================
// method definitions
// ===========================================================================
// ---------------------------------------------------------------------------
// static interface
// ---------------------------------------------------------------------------
void
PCNetProjectionLoader::loadIfSet(OptionsCont &oc,
                                 Position2D &netOffset, Boundary &origNetBoundary,
                                 Boundary &convNetBoundary,
                                 std::string &projParameter) throw(ProcessError) {
    if (!oc.isSet("net")) {
        return;
    }
    // check file
    std::string file = oc.getString("net");
    if (!FileHelpers::exists(file)) {
        throw ProcessError("Could not open net-file '" + file + "'.");
    }
    // build handler and parser
    PCNetProjectionLoader handler(netOffset, origNetBoundary, convNetBoundary, projParameter);
    handler.setFileName(file);
    XMLPScanToken token;
    XERCES_CPP_NAMESPACE_QUALIFIER SAX2XMLReader *parser = XMLSubSys::getSAXReader(handler);
    MsgHandler::getMessageInstance()->beginProcessMsg("Parsing network projection from '" + file + "'...");
    if (!parser->parseFirst(file.c_str(), token)) {
        delete parser;
        throw ProcessError("Can not read XML-file '" + handler.getFileName() + "'.");
    }
    // parse
    while (parser->parseNext(token) && !handler.hasReadAll());
    // clean up
    MsgHandler::getMessageInstance()->endProcessMsg("done.");
    if (!handler.hasReadAll()) {
        throw ProcessError("Could not find projection parameter in net.");
    }
    delete parser;
}



// ---------------------------------------------------------------------------
// handler methods
// ---------------------------------------------------------------------------
PCNetProjectionLoader::PCNetProjectionLoader(Position2D &netOffset,
        Boundary &origNetBoundary, Boundary &convNetBoundary,
        std::string &projParameter) throw()
        : SUMOSAXHandler("sumo-network"), myNetOffset(netOffset),
        myOrigNetBoundary(origNetBoundary), myConvNetBoundary(convNetBoundary),
        myProjParameter(projParameter),
        myFoundOffset(false), myFoundOrigNetBoundary(false),
        myFoundConvNetBoundary(false), myFoundProj(false) {}


PCNetProjectionLoader::~PCNetProjectionLoader() throw() {}


void
PCNetProjectionLoader::myStartElement(SumoXMLTag element,
                                      const SUMOSAXAttributes &attrs) throw(ProcessError) {
    if (element!=SUMO_TAG_LOCATION) {
        return;
    }
    bool ok = true;
    Position2DVector tmp = GeomConvHelper::parseShapeReporting(attrs.getOptStringReporting(SUMO_ATTR_NET_OFFSET, "net", 0, ok, ""), "net", 0, ok, false);
    if (ok) {
        myNetOffset = tmp[0];
    }
    myOrigNetBoundary = GeomConvHelper::parseBoundaryReporting(attrs.getOptStringReporting(SUMO_ATTR_ORIG_BOUNDARY, "net", 0, ok, ""), "net", 0, ok);
    myConvNetBoundary = GeomConvHelper::parseBoundaryReporting(attrs.getOptStringReporting(SUMO_ATTR_CONV_BOUNDARY, "net", 0, ok, ""), "net", 0, ok);
    myProjParameter = attrs.getOptStringReporting(SUMO_ATTR_ORIG_PROJ, "net", 0, ok, "");
    myFoundOffset = myFoundOrigNetBoundary = myFoundConvNetBoundary = myFoundProj = ok;
}


void
PCNetProjectionLoader::myCharacters(SumoXMLTag element,
                                    const std::string &chars) throw(ProcessError) {
    bool ok = true;
    switch (element) {
    case SUMO_TAG_ORIG_BOUNDARY:
        myOrigNetBoundary = GeomConvHelper::parseBoundaryReporting(chars, "net", 0, ok);
        myFoundOrigNetBoundary = ok;
        break;
    case SUMO_TAG_CONV_BOUNDARY:
        myConvNetBoundary = GeomConvHelper::parseBoundaryReporting(chars, "net", 0, ok);
        myFoundConvNetBoundary = ok;
        break;
    case SUMO_TAG_NET_OFFSET: {
        Position2DVector tmp = GeomConvHelper::parseShapeReporting(chars, "net", 0, ok, false);
        if (ok) {
            myNetOffset = tmp[0];
        }
        myFoundOffset = ok;
    }
    break;
    case SUMO_TAG_ORIG_PROJ:
        myProjParameter = chars;
        myFoundProj = true;
        break;
    default:
        break;
    }
}


bool
PCNetProjectionLoader::hasReadAll() const throw() {
    return myFoundOffset&&myFoundOrigNetBoundary&&myFoundConvNetBoundary&&myFoundProj;
}


/****************************************************************************/

