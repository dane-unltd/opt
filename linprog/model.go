package linprog

import (
	"github.com/dane-unltd/linalg/mat"
	"time"
)

type Model struct {
	C, B mat.Vec
	A    *mat.Dense

	X, Y, S mat.Vec

	Iter int
	Time time.Duration

	callbacks []func(m *Model)
}

func NewStandard(c mat.Vec, A *mat.Dense, b mat.Vec) *Model {
	m, n := A.Dims()
	if len(c) != n || len(b) != m {
		panic("linprog: dimension mismatch")
	}
	return &Model{
		C: c,
		B: b,
		A: A,
	}
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
