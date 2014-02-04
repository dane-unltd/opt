package multi

import (
	"github.com/gonum/blas/dbw"
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
	dbw.Axpy(-1, dbw.NewVector(dx.oldX), dbw.NewVector(dx.tmp))
	if dbw.Nrm2(dbw.NewVector(dx.tmp)) < dx.Tol {
		return XAbsConv
	}
	copy(dx.oldX, r.X)
	return NotTerminated
}

type GradConv struct {
	Tol float64
}

func (gc GradConv) Update(r *Result) Status {
	if dbw.Nrm2(dbw.NewVector(r.Grad)) < gc.Tol {
		return GradAbsConv
	}
	return NotTerminated
}
