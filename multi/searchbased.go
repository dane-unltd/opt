package multi

import (
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/dbw"
	"time"
)

type SearchDirectioner interface {
	SearchDirection(s Solution, d []float64)
}

type SearchBased struct {
	sd SearchDirectioner
	ls uni.FdFOptimizer
}

func NewSearchBased(sd SearchDirectioner, ls uni.FdFOptimizer) *SearchBased {
	return &SearchBased{
		sd: sd,
		ls: ls,
	}
}

func (sol SearchBased) OptimizeFdF(o FdF, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := fdfWrapper{r: r, fdf: o}
	r.initFdF(obj)

	if len(upd) == 0 {
		upd = append(upd, GradConv{1e-6})
	}

	initialTime := time.Now()

	n := len(r.X)
	s := 1.0 //initial step size

	x := dbw.NewVector(r.X)
	g := dbw.NewVector(r.Grad)
	d := dbw.NewVector(make([]float64, n))

	sol.sd.SearchDirection(r.Solution, d.Data)

	gLin := dbw.Dot(g, d)

	for doUpdates(r, initialTime, upd) == 0 {
		wolfe := uni.Wolfe{
			Armijo:    0.2,
			Curvature: 0.9,
			X0:        0,
			F0:        r.Obj,
			Deriv0:    gLin,
		}

		lineFunc := NewLineFdF(obj, r.X, d.Data)
		lsInit := uni.NewSolution()
		lsInit.Set(s)
		lsInit.SetLower(0, r.Obj, gLin)
		lsRes := sol.ls.OptimizeFdF(lineFunc, lsInit, wolfe)
		if lsRes.Status < 0 {
			r.Status = Status(lsRes.Status)
			break
		}
		s, r.Obj = lsRes.X, lsRes.Obj

		dbw.Axpy(s, d, x)
		obj.DF(r.X, r.Grad)

		sol.sd.SearchDirection(r.Solution, d.Data)

		gLin = dbw.Dot(g, d)
	}
	return r
}
