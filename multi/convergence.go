package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type DeltaXConv struct {
	Tol  float64
	oldX mat.Vec
	tmp  mat.Vec
}

func NewDeltaXConv(tol float64) *DeltaXConv {
	return &DeltaXConv{Tol: tol}
}

func (dx *DeltaXConv) Update(r *Result) Status {
	if dx.tmp == nil {
		dx.tmp = mat.NewVec(len(r.X))
		dx.oldX = mat.NewVec(len(r.X))
		return NotTerminated
	}
	dx.tmp.Sub(r.X, dx.oldX)
	if dx.tmp.Nrm2() < dx.Tol {
		return XAbsConv
	}
	dx.oldX.Copy(r.X)
	return NotTerminated
}

type GradConv struct {
	Tol float64
}

func (gc GradConv) Update(r *Result) Status {
	if r.Grad.Nrm2() < gc.Tol {
		return GradAbsConv
	}
	return NotTerminated
}
