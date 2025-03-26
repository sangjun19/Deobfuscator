// Repository: Ayden-123/plato
// File: common/sdk/net.go

package sdk

import (
	"fmt"
	"github.com/golang/protobuf/proto"
	"github.com/hardcore-os/plato/common/idl/message"
	"github.com/hardcore-os/plato/common/tcp"
	"net"
)

type connect struct {
	conn               *net.TCPConn
	sendChan, recvChan chan *Message
	connID             uint64
}

func newConnet(ip net.IP, port int, connID uint64) *connect {
	clientConn := &connect{
		sendChan: make(chan *Message),
		recvChan: make(chan *Message),
	}
	addr := &net.TCPAddr{IP: ip, Port: port}
	conn, err := net.DialTCP("tcp", nil, addr)
	if err != nil {
		fmt.Printf("DialTCP.err=%+v", err)
		return nil
	}
	clientConn.conn = conn
	if connID != 0 {
		clientConn.connID = connID
	}
	return clientConn
}

func handAckMsg(c *connect, data []byte) *Message {
	ackMsg := &message.ACKMsg{}
	proto.Unmarshal(data, ackMsg)
	switch ackMsg.Type {
	case message.CmdType_Login:
		c.connID = ackMsg.ConnID
	}
	return &Message{
		Type:       MsgTypeAck,
		Name:       "plato",
		FormUserID: "1212121",
		ToUserID:   "222212122",
		Content:    ackMsg.Msg,
	}
}

func (c *connect) send(ty message.CmdType, palyload []byte) {
	// 直接发送给接收方
	msgCmd := message.MsgCmd{
		Type:    ty,
		Payload: palyload,
	}
	msg, err := proto.Marshal(&msgCmd)
	if err != nil {
		panic(err)
	}
	dataPgk := tcp.DataPgk{
		Data: msg,
		Len:  uint32(len(msg)),
	}
	c.conn.Write(dataPgk.Marshal())
}

func (c *connect) recv() <-chan *Message {
	return c.recvChan
}

func (c *connect) close() {
	// 目前没啥值得回收的
}
