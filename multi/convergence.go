package multi

import (
	"github.com/gonum/blas/blasw"
)

type DeltaXConv struct {
	Tol  float64
	oldX []float64
	tmp  []float64
}

func NewDeltaXConv(tol float64) *DeltaXConv {
	return &DeltaXConv{Tol: tol}
}

func (dx *DeltaXConv) Update(r *Result) Status {
	if dx.tmp == nil {
		dx.tmp = make([]float64, len(r.X))
		dx.oldX = make([]float64, len(r.X))
		return NotTerminated
	}
	copy(dx.tmp, r.X)
	blasw.Axpy(-1, blasw.NewVector(dx.oldX), blasw.NewVector(dx.tmp))
	if blasw.Nrm2(blasw.NewVector(dx.tmp)) < dx.Tol {
		return XAbsConv
	}
	copy(dx.oldX, r.X)
	return NotTerminated
}

type GradConv struct {
	Tol float64
}

func (gc GradConv) Update(r *Result) Status {
	if blasw.Nrm2(blasw.NewVector(r.Grad)) < gc.Tol {
		return GradAbsConv
	}
	return NotTerminated
}
