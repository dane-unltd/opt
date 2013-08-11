package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
)

type Solution struct {
	X     mat.Vec
	ObjX  float64
	GradX mat.Vec
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
	s.ObjX = math.NaN()
	s.GradX = nil
}

func (s *Solution) AddVar(x float64) {
	s.X = append(s.X, x)

	s.ObjX = math.NaN()
	s.GradX = nil
}
