package multi

import (
	"github.com/gonum/blas/dblas"
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
	dblas.Daxpy(-1, dblas.NewVector(dx.oldX), dblas.NewVector(dx.tmp))
	if dblas.Dnrm2(dblas.NewVector(dx.tmp)) < dx.Tol {
		return XAbsConv
	}
	copy(dx.oldX, r.X)
	return NotTerminated
}

type GradConv struct {
	Tol float64
}

func (gc GradConv) Update(r *Result) Status {
	if dblas.Dnrm2(dblas.NewVector(r.Grad)) < gc.Tol {
		return GradAbsConv
	}
	return NotTerminated
}
