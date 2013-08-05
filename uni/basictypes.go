package uni

import (
	"math"
	"time"
)

type Solver interface {
	Solve(m *Model) error
}

type Model struct {
	obj    func(x float64) float64
	deriv  func(x float64) float64
	deriv2 func(x float64) float64

	x       float64
	objX    float64
	derivX  float64
	deriv2X float64

	lb      float64
	objLB   float64
	derivLB float64

	ub      float64
	objUB   float64
	derivUB float64

	iter int
	time time.Duration

	callbacks []func(m *Model)
}

func NewModel(obj, deriv func(float64) float64) *Model {
	m := &Model{
		obj:     obj,
		deriv:   deriv,
		x:       math.NaN(),
		objX:    math.NaN(),
		derivX:  math.NaN(),
		deriv2X: math.NaN(),
		lb:      0,
		objLB:   math.NaN(),
		derivLB: math.NaN(),
		ub:      math.Inf(1),
		objUB:   math.NaN(),
		derivUB: math.NaN(),
	}
	return m
}

func (m *Model) ChangeFun(obj func(float64) float64, deriv func(float64) float64) {
	m.obj = obj
	m.deriv = deriv

	m.objX = math.NaN()
	m.derivX = math.NaN()
	m.deriv2X = math.NaN()

	m.objLB = math.NaN()
	m.derivLB = math.NaN()

	m.objUB = math.NaN()
	m.derivUB = math.NaN()
}

//sets x, objX, derivX, deriv2X, in that order
func (m *Model) SetX(xs ...float64) {
	m.x = math.NaN()
	m.objX = math.NaN()
	m.derivX = math.NaN()
	m.deriv2X = math.NaN()

	if len(xs) > 0 {
		m.x = xs[0]
		if len(xs) > 1 {
			m.objX = xs[1]
			if len(xs) > 2 {
				m.derivX = xs[2]
				if len(xs) > 3 {
					m.deriv2X = xs[3]
				}
			}
		}
	}
}

//sets lb, objLB, derivLB, in that order
func (m *Model) SetLB(lbs ...float64) {
	m.lb = 0
	m.objLB = math.NaN()
	m.derivLB = math.NaN()

	if len(lbs) > 0 {
		m.lb = lbs[0]
		if len(lbs) > 1 {
			m.objLB = lbs[1]
			if len(lbs) > 2 {
				m.derivLB = lbs[2]
			}
		}
	}
}

//sets ub, objUB, derivUB, in that order
func (m *Model) SetUB(ubs ...float64) {
	m.ub = math.Inf(1)
	m.objUB = math.NaN()
	m.derivUB = math.NaN()

	if len(ubs) > 0 {
		m.ub = ubs[0]
		if m.ub < m.lb {
			panic("uni: upperbound has to at least as high as the lower bound")
		}
		if len(ubs) > 1 {
			m.objUB = ubs[1]
			if len(ubs) > 2 {
				m.derivUB = ubs[2]
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

func (m *Model) DerivX() float64 {
	return m.derivX
}

func (m *Model) X() float64 {
	return m.x
}

func (m *Model) ObjX() float64 {
	return m.objX
}

func (m *Model) Iter() int {
	return m.iter
}

func (m *Model) Time() time.Duration {
	return m.time
}
