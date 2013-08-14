package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
)

var nan = math.NaN()

type Solution struct {
	X     mat.Vec
	ObjX  float64
	GradX mat.Vec
}

func NewSolution(x mat.Vec) *Solution {
	return &Solution{
		X:     x,
		ObjX:  nan,
		GradX: nil,
	}
}

func (s *Solution) init(obj ObjWrapper) {
	if math.IsNaN(s.ObjX) {
		s.ObjX = obj.Val(s.X)
	}
}

func (s *Solution) initGrad(obj ObjGradWrapper) {
	if s.GradX == nil {
		s.GradX = make(mat.Vec, len(s.X))
		s.ObjX = obj.ValGrad(s.X, s.GradX)
	}
	if math.IsNaN(s.ObjX) {
		s.ObjX = obj.Val(s.X)
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
	s.ObjX = nan
	s.GradX = nil
}

func (s *Solution) AddVar(x float64) {
	s.X = append(s.X, x)

	s.ObjX = nan
	s.GradX = nil
}
