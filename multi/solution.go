package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
)

var nan = math.NaN()

type Solution struct {
	X    mat.Vec
	Obj  float64
	Grad mat.Vec
}

func NewSolution(x mat.Vec) *Solution {
	return &Solution{
		X:    x,
		Obj:  nan,
		Grad: nil,
	}
}

func (s *Solution) initF(obj fWrapper) {
	if math.IsNaN(s.Obj) {
		s.Obj = obj.F(s.X)
	}
}

func (s *Solution) initFdF(obj fdfWrapper) {
	if s.Grad == nil {
		s.Grad = make(mat.Vec, len(s.X))
		s.Obj = obj.FdF(s.X, s.Grad)
	}
	if math.IsNaN(s.Obj) {
		s.Obj = obj.F(s.X)
	}
}

func (s *Solution) SetX(x mat.Vec, cpy bool) {
	if cpy {
		if s.X == nil {
			s.X = make(mat.Vec, len(x))
		}
		s.X.Copy(x)
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
