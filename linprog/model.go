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

	Rp, Rd, Rs mat.Vec

	Params Params

	initialTime time.Time
	callbacks   []func(m *Model) Status
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
		Params: Params{
			Infeasibility: 1e-6,
			DualityGap:    1e-6,
			IterMax:       1000,
			TimeMax:       time.Minute,
		},
	}
}

func (m *Model) AddCallback(cb func(m *Model) Status) {
	m.callbacks = append(m.callbacks, cb)
}

func (m *Model) ClearCallbacks() {
	m.callbacks = m.callbacks[0:0]
}

func (m *Model) doCallbacks() Status {
	var status Status
	for _, cb := range m.callbacks {
		st := cb(m)
		if st != 0 {
			status = st
		}
	}
	return status
}

func (m *Model) checkConvergence() Status {
	if m.Rd.Asum() < m.Params.Infeasibility &&
		m.Rp.Asum() < m.Params.Infeasibility &&
		m.Rs.Asum() < m.Params.DualityGap {
		return Success
	}

	if m.Iter > m.Params.IterMax {
		return IterLimit
	}
	if m.Time > m.Params.TimeMax {
		return TimeLimit
	}
	return NotTerminated
}

func (m *Model) init() {
	rows, cols := m.A.Dims()
	m.Rd = mat.NewVec(cols)
	m.Rp = mat.NewVec(rows)
	m.Rs = mat.NewVec(cols)

	m.Iter = 0
	m.initialTime = time.Now()
}

func (m *Model) update() Status {
	m.Time = time.Since(m.initialTime)

	At := m.A.TrView()

	m.Rd.Sub(m.C, m.S)
	m.Rd.AddMul(At, m.Y, -1)
	m.Rp.Apply(m.A, m.X)
	m.Rp.Sub(m.B, m.Rp)
	m.Rs.Mul(m.X, m.S)
	m.Rs.Neg(m.Rs)

	if status := m.doCallbacks(); status != 0 {
		return status
	}
	if status := m.checkConvergence(); status != 0 {
		return status
	}

	m.Iter++

	return NotTerminated
}
