package uni

import (
	"math"
)

var nan = math.NaN()

type Solution struct {
	X      float64
	Obj    float64
	Deriv  float64
	Deriv2 float64

	XLower     float64
	ObjLower   float64
	DerivLower float64

	XUpper     float64
	ObjUpper   float64
	DerivUpper float64
}

func NewSolution() *Solution {
	return &Solution{
		X:      nan,
		Obj:    nan,
		Deriv:  nan,
		Deriv2: nan,

		XLower:     0,
		ObjLower:   nan,
		DerivLower: nan,

		XUpper:     math.Inf(1),
		ObjUpper:   nan,
		DerivUpper: nan,
	}
}

func (s *Solution) initF(obj fWrapper) {
	if math.IsNaN(s.X) || s.X <= s.XLower || s.X >= s.XUpper {
		if math.IsInf(s.XUpper, 1) {
			s.X = s.XLower + 1
		} else {
			s.X = (s.XLower + s.XUpper) / 2
		}

	}
	if math.IsNaN(s.ObjLower) {
		s.ObjLower = obj.F(s.XLower)
	}
}

func (s *Solution) initFdF(obj fdfWrapper) {
	if math.IsNaN(s.X) || s.X <= s.XLower || s.X >= s.XUpper {
		if math.IsInf(s.XUpper, 1) {
			s.X = s.XLower + 1
		} else {
			s.X = (s.XLower + s.XUpper) / 2
		}

	}
	s.Obj, s.Deriv = obj.FdF(s.X)
	if math.IsNaN(s.DerivLower) {
		s.ObjLower, s.DerivLower = obj.FdF(s.XLower)
	}

	if math.IsNaN(s.ObjLower) {
		s.ObjLower = obj.F(s.XLower)
	}
}

//sets X, Obj, Deriv, Deriv2, in that order
func (s *Solution) Set(xs ...float64) {
	s.X = nan
	s.Obj = nan
	s.Deriv = nan
	s.Deriv2 = nan

	if len(xs) > 0 {
		s.X = xs[0]
		if len(xs) > 1 {
			s.Obj = xs[1]
			if len(xs) > 2 {
				s.Deriv = xs[2]
				if len(xs) > 3 {
					s.Deriv2 = xs[3]
				}
			}
		}
	}
}

//sets XLower, ObjLower, DerivLower, in that order
func (s *Solution) SetLower(lbs ...float64) {
	s.XLower = 0
	s.ObjLower = nan
	s.DerivLower = nan

	if len(lbs) > 0 {
		s.XLower = lbs[0]
		if s.XUpper < s.XLower {
			panic("uni: upper bound has to at least as high as the lower bound")
		}
		if len(lbs) > 1 {
			s.ObjLower = lbs[1]
			if len(lbs) > 2 {
				s.DerivLower = lbs[2]
			}
		}
	}
}

//sets XUpper, ObjUpper, DerivUpper, in that order
func (s *Solution) SetUpper(ubs ...float64) {
	s.XUpper = math.Inf(1)
	s.ObjUpper = nan
	s.DerivUpper = nan

	if len(ubs) > 0 {
		s.XUpper = ubs[0]
		if s.XUpper < s.XLower {
			panic("uni: upper bound has to at least as high as the lower bound")
		}
		if len(ubs) > 1 {
			s.ObjUpper = ubs[1]
			if len(ubs) > 2 {
				s.DerivUpper = ubs[2]
			}
		}
	}
}
