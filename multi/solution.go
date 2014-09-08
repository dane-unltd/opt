package multi

import (
	"math"
	"time"
)

var nan = math.NaN()

type Stats struct {
	Iter      int
	Time      time.Duration
	FunEvals  int
	GradEvals int
}

type Solution struct {
	X       []float64
	Obj     float64
	Grad    []float64
	LastDir []float64
}

func NewSolution(x []float64) *Solution {
	return &Solution{
		X:   x,
		Obj: nan,
	}
}

func (s *Solution) check(obj FdF) {
	if s.Grad == nil {
		s.Grad = make([]float64, len(s.X))
		s.Obj = obj.FdF(s.X, s.Grad)
	}
	if math.IsNaN(s.Obj) {
		s.Obj = obj.F(s.X)
	}

	if s.LastDir == nil {
		s.LastDir = make([]float64, len(s.X))
	}
}

func (s *Solution) SetX(x []float64, cpy bool) {
	if cpy {
		if s.X == nil {
			s.X = make([]float64, len(x))
		}
		copy(s.X, x)
	} else {
		s.X = x
	}
	s.Obj = nan
	s.Grad = nil
}

func (s *Solution) AddVar(x float64) {
	s.X = append(s.X, x)

	s.Obj = nan
	s.Grad = nil
}
