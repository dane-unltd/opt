package linesearch

import "math"

type Siso func(float64) float64

type Solver interface {
	Solve(m *Model) (*Result, error)
}

type Model struct {
	F, G     Siso
	LB, UB   float64
	LBF, LBG float64
	X        float64
}

type Result struct {
	X, F, G float64
	Iter    int
}

func NewModel(f, g Siso) *Model {
	m := &Model{F: f, G: g, UB: math.Inf(1),
		LBF: math.NaN(), LBG: math.NaN(), X: math.NaN()}
	return m
}
