// Repository: xiaokangwang/VLite
// File: vendor2/github.com/golang-collections/go-datastructures/queue/ring.go

/*
Copyright 2014 Workiva, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package queue

import (
	"runtime"
	"sync/atomic"
)

// roundUp takes a uint32 greater than 0 and rounds it up to the next
// power of 2.
func roundUp(v uint32) uint32 {
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	v++
	return v
}

type node struct {
	position uint32
	data     interface{}
}

type nodes []*node

// RingBuffer is a MPMC buffer that achieves threadsafety with CAS operations
// only.  A put on full or get on empty call will block until an item
// is put or retrieved.  Calling Dispose on the RingBuffer will unblock
// any blocked threads with an error.  This buffer is similar to the buffer
// described here: http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
// with some minor additions.
type RingBuffer struct {
	queue, dequeue, mask, disposed uint32
	nodes                          nodes
}

func (rb *RingBuffer) init(size uint32) {
	size = roundUp(size)
	rb.nodes = make(nodes, size)
	for i := uint32(0); i < size; i++ {
		rb.nodes[i] = &node{position: i}
	}
	rb.mask = size - 1 // so we don't have to do this with every put/get operation
}

// Put adds the provided item to the queue.  If the queue is full, this
// call will block until an item is added to the queue or Dispose is called
// on the queue.  An error will be returned if the queue is disposed.
func (rb *RingBuffer) Put(item interface{}) error {
	var n *node
	pos := atomic.LoadUint32(&rb.queue)
L:
	for {
		if atomic.LoadUint32(&rb.disposed) == 1 {
			return disposedError
		}

		n = rb.nodes[pos&rb.mask]
		seq := atomic.LoadUint32(&n.position)
		switch dif := seq - pos; {
		case dif == 0:
			if atomic.CompareAndSwapUint32(&rb.queue, pos, pos+1) {
				break L
			}
		case dif < 0:
			panic(`Ring buffer in a compromised state during a put operation.`)
		default:
			pos = atomic.LoadUint32(&rb.queue)
		}
		runtime.Gosched() // free up the cpu before the next iteration
	}

	n.data = item
	atomic.StoreUint32(&n.position, pos+1)
	return nil
}

// Get will return the next item in the queue.  This call will block
// if the queue is empty.  This call will unblock when an item is added
// to the queue or Dispose is called on the queue.  An error will be returned
// if the queue is disposed.
func (rb *RingBuffer) Get() (interface{}, error) {
	var n *node
	pos := atomic.LoadUint32(&rb.dequeue)
L:
	for {
		if atomic.LoadUint32(&rb.disposed) == 1 {
			return nil, disposedError
		}

		n = rb.nodes[pos&rb.mask]
		seq := atomic.LoadUint32(&n.position)
		switch dif := seq - (pos + 1); {
		case dif == 0:
			if atomic.CompareAndSwapUint32(&rb.dequeue, pos, pos+1) {
				break L
			}
		case dif < 0:
			panic(`Ring buffer in compromised state during a get operation.`)
		default:
			pos = atomic.LoadUint32(&rb.dequeue)
		}
		runtime.Gosched() // free up cpu before next iteration
	}
	data := n.data
	n.data = nil
	atomic.StoreUint32(&n.position, pos+rb.mask+1)
	return data, nil
}

// Len returns the number of items in the queue.
func (rb *RingBuffer) Len() uint32 {
	return atomic.LoadUint32(&rb.queue) - atomic.LoadUint32(&rb.dequeue)
}

// Cap returns the capacity of this ring buffer.
func (rb *RingBuffer) Cap() uint32 {
	return uint32(len(rb.nodes))
}

// Dispose will dispose of this queue and free any blocked threads
// in the Put and/or Get methods.  Calling those methods on a disposed
// queue will return an error.
func (rb *RingBuffer) Dispose() {
	atomic.CompareAndSwapUint32(&rb.disposed, 0, 1)
}

// IsDisposed will return a bool indicating if this queue has been
// disposed.
func (rb *RingBuffer) IsDisposed() bool {
	return atomic.LoadUint32(&rb.disposed) == 1
}

// NewRingBuffer will allocate, initialize, and return a ring buffer
// with the specified size.
func NewRingBuffer(size uint32) *RingBuffer {
	rb := &RingBuffer{}
	rb.init(size)
	return rb
}
