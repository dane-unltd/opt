package uni

import (
	"math"
)

var nan = math.NaN()

type Solution struct {
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
}

func NewSolution() *Solution {
	return &Solution{
		X:       nan,
		ObjX:    nan,
		DerivX:  nan,
		Deriv2X: nan,

		LB:      0,
		ObjLB:   nan,
		DerivLB: nan,

		UB:      math.Inf(1),
		ObjUB:   nan,
		DerivUB: nan,
	}
}

func (s *Solution) init(obj ObjWrapper) {
	if math.IsNaN(s.X) || s.X <= s.LB || s.X >= s.UB {
		if math.IsInf(s.UB, 1) {
			s.X = s.LB + 1
		} else {
			s.X = (s.LB + s.UB) / 2
		}

	}
	if math.IsNaN(s.ObjLB) {
		s.ObjLB = obj.Val(s.LB)
	}
}

func (s *Solution) initDeriv(obj ObjDerivWrapper) {
	if math.IsNaN(s.X) || s.X <= s.LB || s.X >= s.UB {
		if math.IsInf(s.UB, 1) {
			s.X = s.LB + 1
		} else {
			s.X = (s.LB + s.UB) / 2
		}

	}
	s.ObjX, s.DerivX = obj.ValDeriv(s.X)
	if math.IsNaN(s.DerivLB) {
		s.ObjLB, s.DerivLB = obj.ValDeriv(s.LB)
	}

	if math.IsNaN(s.ObjLB) {
		s.ObjLB = obj.Val(s.LB)
	}
}

//sets X, ObjX, DerivX, Deriv2X, in that order
func (s *Solution) SetX(xs ...float64) {
	s.X = nan
	s.ObjX = nan
	s.DerivX = nan
	s.Deriv2X = nan

	if len(xs) > 0 {
		s.X = xs[0]
		if len(xs) > 1 {
			s.ObjX = xs[1]
			if len(xs) > 2 {
				s.DerivX = xs[2]
				if len(xs) > 3 {
					s.Deriv2X = xs[3]
				}
			}
		}
	}
}

//sets LB, ObjLB, DerivLB, in that order
func (s *Solution) SetLB(lbs ...float64) {
	s.LB = 0
	s.ObjLB = nan
	s.DerivLB = nan

	if len(lbs) > 0 {
		s.LB = lbs[0]
		if s.UB < s.LB {
			panic("uni: upper bound has to at least as high as the lower bound")
		}
		if len(lbs) > 1 {
			s.ObjLB = lbs[1]
			if len(lbs) > 2 {
				s.DerivLB = lbs[2]
			}
		}
	}
}

//sets UB, ObjUB, DerivUB, in that order
func (s *Solution) SetUB(ubs ...float64) {
	s.UB = math.Inf(1)
	s.ObjUB = nan
	s.DerivUB = nan

	if len(ubs) > 0 {
		s.UB = ubs[0]
		if s.UB < s.LB {
			panic("uni: upper bound has to at least as high as the lower bound")
		}
		if len(ubs) > 1 {
			s.ObjUB = ubs[1]
			if len(ubs) > 2 {
				s.DerivUB = ubs[2]
			}
		}
	}
}
