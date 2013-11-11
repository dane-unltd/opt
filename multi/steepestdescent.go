package multi

import (
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/matrix/mat64"
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
	gLin := -r.Grad.Nrm2Sq()

	d := make(mat64.Vec, n)
	copy(d, r.Grad)
	d.Scal(-1)

	lineFunc := NewLineFdF(obj, r.X, d)
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

		r.X.Axpy(s, d)

		obj.DF(r.X, r.Grad)
		d.Copy(r.Grad)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()
	}
	return r
}
