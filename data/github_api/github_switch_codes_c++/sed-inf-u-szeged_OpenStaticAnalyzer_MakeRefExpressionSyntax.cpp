/*
 *  This file is part of OpenStaticAnalyzer.
 *
 *  Copyright (c) 2004-2018 Department of Software Engineering - University of Szeged
 *
 *  Licensed under Version 1.2 of the EUPL (the "Licence");
 *
 *  You may not use this work except in compliance with the Licence.
 *
 *  You may obtain a copy of the Licence in the LICENSE file or at:
 *
 *  https://joinup.ec.europa.eu/software/page/eupl
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the Licence is distributed on an "AS IS" basis,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the Licence for the specific language governing permissions and
 *  limitations under the Licence.
 */

#include "csharp/inc/csharp.h"
#include "csharp/inc/Common.h"
#include "common/inc/WriteMessage.h"

#include "csharp/inc/messages.h"
#include <algorithm>
#include <string.h>
#include "common/inc/math/common.h"


namespace columbus { namespace csharp { namespace asg {

typedef boost::crc_32_type  Crc_type;

namespace expression { 
  MakeRefExpressionSyntax::MakeRefExpressionSyntax(NodeId _id, Factory *_factory) :
    ExpressionSyntax(_id, _factory),
    m_Expression(0)
  {
  }

  MakeRefExpressionSyntax::~MakeRefExpressionSyntax() {
  }

  void MakeRefExpressionSyntax::prepareDelete(bool tryOnVirtualParent){
    removeExpression();
    expression::ExpressionSyntax::prepareDelete(false);
  }

  NodeKind MakeRefExpressionSyntax::getNodeKind() const {
    return ndkMakeRefExpressionSyntax;
  }

  expression::ExpressionSyntax* MakeRefExpressionSyntax::getExpression() const {
    expression::ExpressionSyntax *_node = NULL;
    if (m_Expression != 0)
      _node = dynamic_cast<expression::ExpressionSyntax*>(factory->getPointer(m_Expression));
    if ( (_node == NULL) || factory->getIsFiltered(_node))
      return NULL;

    return _node;
  }

  bool MakeRefExpressionSyntax::setEdge(EdgeKind edgeKind, NodeId edgeEnd, bool tryOnVirtualParent) {
    switch (edgeKind) {
      case edkMakeRefExpressionSyntax_Expression:
        setExpression(edgeEnd);
        return true;
      default:
        break;
    }
    if (expression::ExpressionSyntax::setEdge(edgeKind, edgeEnd, false)) {
      return true;
    }
    return false;
  }

  bool MakeRefExpressionSyntax::removeEdge(EdgeKind edgeKind, NodeId edgeEnd, bool tryOnVirtualParent) {
    switch (edgeKind) {
      case edkMakeRefExpressionSyntax_Expression:
        removeExpression();
        return true;
      default:
        break;
    }
    if (expression::ExpressionSyntax::removeEdge(edgeKind, edgeEnd, false)) {
      return true;
    }
    return false;
  }

  void MakeRefExpressionSyntax::setExpression(NodeId _id) {
    expression::ExpressionSyntax *_node = NULL;
    if (_id) {
      if (!factory->getExist(_id))
        throw CsharpException(COLUMBUS_LOCATION, CMSG_EX_THE_END_POINT_OF_THE_EDGE_DOES_NOT_EXIST);

      _node = dynamic_cast<expression::ExpressionSyntax*> (factory->getPointer(_id));
      if ( _node == NULL) {
        throw CsharpException(COLUMBUS_LOCATION, CMSG_EX_INVALID_NODE_KIND);
      }
      if (&(_node->getFactory()) != this->factory)
        throw CsharpException(COLUMBUS_LOCATION, CMSG_EX_THE_FACTORY_OF_NODES_DOES_NOT_MATCH );

      if (m_Expression) {
        removeParentEdge(m_Expression);
        if (factory->getExistsReverseEdges())
          factory->reverseEdges->removeEdge(m_Expression, m_id, edkMakeRefExpressionSyntax_Expression);
      }
      m_Expression = _node->getId();
      if (m_Expression != 0)
        setParentEdge(factory->getPointer(m_Expression), edkMakeRefExpressionSyntax_Expression);
      if (factory->getExistsReverseEdges())
        factory->reverseEdges->insertEdge(m_Expression, this->getId(), edkMakeRefExpressionSyntax_Expression);
    } else {
      if (m_Expression) {
        throw CsharpException(COLUMBUS_LOCATION, CMSG_EX_CAN_T_SET_EDGE_TO_NULL);
      }
    }
  }

  void MakeRefExpressionSyntax::setExpression(expression::ExpressionSyntax *_node) {
    if (_node == NULL)
      throw CsharpException(COLUMBUS_LOCATION, CMSG_EX_CAN_T_SET_EDGE_TO_NULL);

    setExpression(_node->getId());
  }

  void MakeRefExpressionSyntax::removeExpression() {
      if (m_Expression) {
        removeParentEdge(m_Expression);
        if (factory->getExistsReverseEdges())
          factory->reverseEdges->removeEdge(m_Expression, m_id, edkMakeRefExpressionSyntax_Expression);
      }
      m_Expression = 0;
  }

  void MakeRefExpressionSyntax::accept(Visitor &visitor) const {
    visitor.visit(*this);
  }

  void MakeRefExpressionSyntax::acceptEnd(Visitor &visitor) const {
    visitor.visitEnd(*this);
  }

  double MakeRefExpressionSyntax::getSimilarity(const base::Base& base){
    if(base.getNodeKind() == getNodeKind()) {
      return 1.0;
    } else {
      return 0.0;
    }
  }

  void MakeRefExpressionSyntax::swapStringTable(RefDistributorStrTable& newStrTable, std::map<Key,Key>& oldAndNewStrKeyMap ){
    std::map<Key,Key>::iterator foundKeyId;

      m_position.swapStringTable(newStrTable,oldAndNewStrKeyMap);

  }

  NodeHashType MakeRefExpressionSyntax::getHash(std::set<NodeId>& travNodes) const {
    if (hashOk) return nodeHashCache;
    common::WriteMsg::write(CMSG_GET_THE_NODE_HASH_OF_NODE_BEGIN,this->getId()); 
    if (travNodes.count(getId())>0) {
      common::WriteMsg::write(CMSG_GET_THE_NODE_HASH_OF_NODE_SKIP);
      return 0;
    }
    travNodes.insert(getId());
    Crc_type resultHash;
    resultHash.process_bytes( "expression::MakeRefExpressionSyntax", strlen("expression::MakeRefExpressionSyntax"));
    common::WriteMsg::write(CMSG_GET_THE_NODE_HASH_OF_NODE_END,resultHash.checksum()); 
    nodeHashCache = resultHash.checksum();
    hashOk = true;
    return nodeHashCache;
  }

  void MakeRefExpressionSyntax::save(io::BinaryIO &binIo,bool withVirtualBase  /*= true*/) const {
    ExpressionSyntax::save(binIo,false);

    binIo.writeUInt4(m_Expression);

  }

  void MakeRefExpressionSyntax::load(io::BinaryIO &binIo, bool withVirtualBase /*= true*/) {
    ExpressionSyntax::load(binIo,false);

    m_Expression =  binIo.readUInt4();
    if (m_Expression != 0)
      setParentEdge(factory->getPointer(m_Expression),edkMakeRefExpressionSyntax_Expression);

  }


}


}}}
