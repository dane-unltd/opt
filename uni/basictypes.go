package uni

import (
	"math"
	"time"
)

type Solver interface {
	Solve(m *Model) error
}

//Model for a general univariate optimization problem.
//Only change the fields directly if you know what you are doing
//otherwise use the provided methods.
type Model struct {
	Obj    func(x float64) float64
	Deriv  func(x float64) float64
	Deriv2 func(x float64) float64

	X       float64
	ObjX    float64
	DerivX  float64
	Deriv2X float64

	LB      float64
	ObjLB   float64
	DerivLB float64

	UB      float64
	ObjUB   float64
	DerivUB float64

	Iter int
	Time time.Duration

	callbacks []func(m *Model)
}

func NewModel(obj, deriv func(float64) float64) *Model {
	m := &Model{
		Obj:     obj,
		Deriv:   deriv,
		X:       math.NaN(),
		ObjX:    math.NaN(),
		DerivX:  math.NaN(),
		Deriv2X: math.NaN(),
		LB:      0,
		ObjLB:   math.NaN(),
		DerivLB: math.NaN(),
		UB:      math.Inf(1),
		ObjUB:   math.NaN(),
		DerivUB: math.NaN(),
	}
	return m
}

func (m *Model) ChangeFun(obj func(float64) float64, deriv func(float64) float64) {
	m.Obj = obj
	m.Deriv = deriv

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
			panic("uni: upperbound has to at least as high as the lower bound")
		}
		if len(ubs) > 1 {
			m.ObjUB = ubs[1]
			if len(ubs) > 2 {
				m.DerivUB = ubs[2]
			}
		}
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
