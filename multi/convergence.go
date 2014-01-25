package multi

import (
	"github.com/dane-unltd/goblas"
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
	goblas.Daxpy(-1, goblas.NewVector(dx.oldX), goblas.NewVector(dx.tmp))
	if goblas.Dnrm2(goblas.NewVector(dx.tmp)) < dx.Tol {
		return XAbsConv
	}
	copy(dx.oldX, r.X)
	return NotTerminated
}

type GradConv struct {
	Tol float64
}

func (gc GradConv) Update(r *Result) Status {
	if goblas.Dnrm2(goblas.NewVector(r.Grad)) < gc.Tol {
		return GradAbsConv
	}
	return NotTerminated
}
