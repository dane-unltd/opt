package uni

import (
	"math"
	"time"
)

type Callback interface {
	Update(m *Model) Status
}

//Model for a general univariate optimization problem.
//Only change the fields directly if you know what you are doing
//otherwise use the provided methods.
type Model struct {
	Obj    Function
	deriv  Deriv
	deriv2 Deriv2

	X       float64
	ObjX    float64
	DerivX  float64
	Deriv2X float64

	oldX      float64
	oldObjX   float64
	oldDerivX float64

	x0, f0, d0 float64

	initialInterval float64
	initialTime     time.Time

	LB      float64
	ObjLB   float64
	DerivLB float64

	UB      float64
	ObjUB   float64
	DerivUB float64

	Iter        int
	Time        time.Duration
	FunEvals    int
	DerivEvals  int
	Deriv2Evals int

	Params Params
	Solver Solver

	Status Status

	callbacks []Callback
}

func NewModel(obj Function) *Model {
	m := &Model{
		Obj: obj,

		X:       math.NaN(),
		ObjX:    math.NaN(),
		DerivX:  math.NaN(),
		Deriv2X: math.NaN(),

		oldX:    math.NaN(),
		oldObjX: math.NaN(),

		x0: math.NaN(),
		f0: math.NaN(),
		d0: math.NaN(),

		initialInterval: math.Inf(1),

		LB:      0,
		ObjLB:   math.NaN(),
		DerivLB: math.NaN(),

		UB:      math.Inf(1),
		ObjUB:   math.NaN(),
		DerivUB: math.NaN(),

		callbacks: make([]Callback, 0),

		Params: Params{
			FunTolAbs: 1e-15,
			FunTolRel: 1e-15,
			XTolAbs:   1e-6,
			XTolRel:   1e-2,

			Inexact:   true,
			Armijo:    0.2,
			Curvature: 0.9,

			IterMax:    1000,
			TimeMax:    time.Second,
			FunEvalMax: 1000,
		},
	}
	return m
}

func (m *Model) ChangeFun(obj Function) {
	m.Obj = obj
	m.deriv = nil
	m.deriv2 = nil

	m.ObjX = math.NaN()
	m.DerivX = math.NaN()
	m.Deriv2X = math.NaN()

	m.ObjLB = math.NaN()
	m.DerivLB = math.NaN()

	m.ObjUB = math.NaN()
	m.DerivUB = math.NaN()
}

//sets x, objX, derivX, deriv2X, in that order
func (m *Model) SetX(xs ...float64) {
	m.X = math.NaN()
	m.ObjX = math.NaN()
	m.DerivX = math.NaN()
	m.Deriv2X = math.NaN()

	if len(xs) > 0 {
		m.X = xs[0]
		if len(xs) > 1 {
			m.ObjX = xs[1]
			if len(xs) > 2 {
				m.DerivX = xs[2]
				if len(xs) > 3 {
					m.Deriv2X = xs[3]
				}
			}
		}
	}
}

//sets lb, objLB, derivLB, in that order
func (m *Model) SetLB(lbs ...float64) {
	m.LB = 0
	m.ObjLB = math.NaN()
	m.DerivLB = math.NaN()

	if len(lbs) > 0 {
		m.LB = lbs[0]
		if m.UB < m.LB {
			panic("uni: upper bound has to at least as high as the lower bound")
		}
		if len(lbs) > 1 {
			m.ObjLB = lbs[1]
			if len(lbs) > 2 {
				m.DerivLB = lbs[2]
			}
		}
	}
}

//sets ub, objUB, derivUB, in that order
func (m *Model) SetUB(ubs ...float64) {
	m.UB = math.Inf(1)
	m.ObjUB = math.NaN()
	m.DerivUB = math.NaN()

	if len(ubs) > 0 {
		m.UB = ubs[0]
		if m.UB < m.LB {
			panic("uni: upper bound has to at least as high as the lower bound")
		}
		if len(ubs) > 1 {
			m.ObjUB = ubs[1]
			if len(ubs) > 2 {
				m.DerivUB = ubs[2]
			}
		}
	}
}

func (m *Model) AddCallback(cb Callback) {
	m.callbacks = append(m.callbacks, cb)
}

func (m *Model) ClearCallbacks() {
	m.callbacks = m.callbacks[0:0]
}

func (m *Model) doCallbacks() Status {
	var status Status
	for _, cb := range m.callbacks {
		st := cb.Update(m)
		if st != 0 {
			status = st
		}
	}
	return status
}

func (m *Model) checkConvergence() Status {
	if m.Params.Inexact {
		if math.Abs(m.DerivX/m.d0) < m.Params.Curvature &&
			m.ObjX-m.f0 < m.Params.Armijo*(m.X-m.x0)*m.d0 {
			return WolfeConv
		}
	}
	if math.Abs(m.UB-m.LB) < m.Params.XTolAbs {
		return XAbsConv
	}
	if math.Abs((m.UB-m.LB)/m.initialInterval) < m.Params.XTolRel {
		return XRelConv
	}
	if math.Abs(m.DerivX) < m.Params.FunTolAbs {
		return DerivAbsConv
	}
	if math.Abs(m.DerivX/m.d0) < m.Params.FunTolRel {
		return DerivRelConv
	}
	if math.Abs(m.ObjX-m.oldObjX) < m.Params.FunTolAbs {
		return ObjAbsConv
	}
	if math.Abs((m.ObjX-m.oldObjX)/m.ObjX) < m.Params.FunTolRel {
		return ObjRelConv
	}

	if m.Iter > m.Params.IterMax {
		return IterLimit
	}
	if m.Time > m.Params.TimeMax {
		return TimeLimit
	}
	if m.FunEvals > m.Params.FunEvalMax {
		return FunEvalLimit
	}
	return NotTerminated
}

func (m *Model) init(useDeriv, useDeriv2 bool) {
	m.initialTime = time.Now()
	m.initialInterval = m.UB - m.LB
	if math.IsInf(m.initialInterval, 1) {
		m.initialInterval = 0
	}
	m.Iter = 0
	m.FunEvals = 0
	m.DerivEvals = 0
	m.Deriv2Evals = 0

	m.Status = NotTerminated

	if math.IsNaN(m.X) || m.X <= m.LB || m.X >= m.UB {
		if math.IsInf(m.UB, 1) {
			m.X = m.LB + 1
		} else {
			m.X = (m.LB + m.UB) / 2
		}

	}
	if useDeriv {
		m.deriv = m.Obj.(Deriv)
		m.ObjX, m.DerivX = m.deriv.ValDeriv(m.X)
		m.FunEvals++
		m.DerivEvals++
		if math.IsNaN(m.DerivLB) {
			m.ObjLB, m.DerivLB = m.deriv.ValDeriv(m.LB)
			m.FunEvals++
			m.DerivEvals++
		}
	}
	if useDeriv2 {
		m.deriv2 = m.Obj.(Deriv2)
	}

	if math.IsNaN(m.ObjLB) {
		m.ObjLB = m.Obj.Val(m.LB)
		m.FunEvals++
	}
	m.x0 = m.LB
	m.f0 = m.ObjLB
	m.d0 = m.DerivLB
}

func (m *Model) update() Status {
	m.Time = time.Since(m.initialTime)
	if status := m.doCallbacks(); status != 0 {
		return status
	}
	if status := m.checkConvergence(); status != 0 {
		return status
	}

	m.oldX = m.X
	m.oldDerivX = m.DerivX
	m.oldObjX = m.ObjX
	m.Iter++

	return NotTerminated
}
