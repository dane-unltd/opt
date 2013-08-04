package uni

import (
	"math"
	"time"
)

type Solver interface {
	Solve(m *Model) error
}

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

	callback func(m *Model)
}

func NewModel(f, g func(float64) float64) *Model {
	m := &Model{
		Obj:     f,
		Deriv:   g,
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
