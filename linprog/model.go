package linprog

import (
	"github.com/dane-unltd/linalg/mat"
	"time"
)

type Model struct {
	c, b mat.Vec
	a    *mat.Dense

	x, y, s mat.Vec

	iter int
	time time.Duration

	callbacks []func(m *Model)
}

func NewStandard(c mat.Vec, A *mat.Dense, b mat.Vec) *Model {
	m, n := A.Dims()
	if len(c) != n || len(b) != m {
		panic("linprog: dimension mismatch")
	}
	return &Model{
		c: c,
		b: b,
		a: A,
	}
}

func (m *Model) X() mat.Vec {
	return m.x
}
func (m *Model) Y() mat.Vec {
	return m.y
}
func (m *Model) S() mat.Vec {
	return m.s
}

func (m *Model) Iter() int {
	return m.iter
}
func (m *Model) Time() time.Duration {
	return m.time
}

func (m *Model) AddCallback(cb func(m *Model)) {
	m.callbacks = append(m.callbacks, cb)
}
func (m *Model) DoCallbacks() {
	for _, cb := range m.callbacks {
		cb(m)
	}
}
func (m *Model) ClearCallbacks() {
	m.callbacks = m.callbacks[0:0]
}
