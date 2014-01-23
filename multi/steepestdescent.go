package multi

import (
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/blasw"
	"time"
)

type SteepestDescent struct {
	Termination
	LineSearch uni.FdFOptimizer
}

func NewSteepestDescent() *SteepestDescent {
	s := &SteepestDescent{
		Termination: Termination{
			IterMax: 1000,
			TimeMax: time.Minute,
		},
		LineSearch: uni.NewBacktracking(),
	}
	return s
}

func (sol SteepestDescent) OptimizeFdF(o FdF, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := fdfWrapper{r: r, fdf: o}
	r.initFdF(obj)

	upd = append(upd, sol.Termination)

	initialTime := time.Now()

	n := len(r.X)
	s := 1.0 //initial step size

	x := blasw.NewVector(r.X)
	g := blasw.NewVector(r.Grad)
	d := blasw.NewVector(make([]float64, n))

	gLin := -blasw.Dot(g, g)

	blasw.Copy(g, d)
	blasw.Scal(-1, d)

	lineFunc := NewLineFdF(obj, r.X, d.Data)
	lsInit := uni.NewSolution()

	for doUpdates(r, initialTime, upd) == 0 {
		s = s * 2

		wolfe := uni.Wolfe{
			Armijo:    0.2,
			Curvature: 0.9,
			X0:        0,
			F0:        r.Obj,
			Deriv0:    gLin,
		}

		lsInit.Set(s)
		lsInit.SetLower(0, r.Obj, gLin)
		lsRes := sol.LineSearch.OptimizeFdF(lineFunc, lsInit, wolfe)
		if lsRes.Status < 0 {
			r.Status = Status(lsRes.Status)
			break
		}
		s, r.Obj = lsRes.X, lsRes.Obj

		blasw.Axpy(s, d, x)

		obj.DF(r.X, r.Grad)
		blasw.Copy(g, d)
		blasw.Scal(-1, d)

		gLin = -blasw.Dot(d, d)
	}
	return r
}
