package multi

import (
	"github.com/gonum/blas/dbw"
)

type SteepestDescent struct{}

func (SteepestDescent) SearchDirection(s Solution, d []float64) {
	dbw.Copy(dbw.NewVector(s.Grad), dbw.NewVector(d))
	dbw.Scal(-1, dbw.NewVector(d))
}
