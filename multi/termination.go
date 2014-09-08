package multi

import (
	"github.com/gonum/blas/dbw"

	"time"
)

type Termination struct {
	IterMax int
	TimeMax time.Duration
}

func (t Termination) Update(sol *Solution, stats *Stats) Status {
	if stats.Iter >= t.IterMax {
		return IterLimit
	}
	if stats.Time >= t.TimeMax {
		return TimeLimit
	}
	return NotTerminated
}

type DeltaXConv struct {
	Tol  float64
	oldX []float64
	tmp  []float64
}

func NewDeltaXConv(tol float64) *DeltaXConv {
	return &DeltaXConv{Tol: tol}
}

func (dx *DeltaXConv) Update(sol *Solution, stats *Stats) Status {
	if dx.tmp == nil {
		dx.tmp = make([]float64, len(sol.X))
		dx.oldX = make([]float64, len(sol.X))
		return NotTerminated
	}
	copy(dx.tmp, sol.X)
	dbw.Axpy(-1, dbw.NewVector(dx.oldX), dbw.NewVector(dx.tmp))
	if dbw.Nrm2(dbw.NewVector(dx.tmp)) < dx.Tol {
		return XAbsConv
	}
	copy(dx.oldX, sol.X)
	return NotTerminated
}

type GradConv struct {
	Tol float64
}

func (gc GradConv) Update(sol *Solution, stats *Stats) Status {
	if dbw.Nrm2(dbw.NewVector(sol.Grad)) < gc.Tol {
		return GradAbsConv
	}
	return NotTerminated
}
