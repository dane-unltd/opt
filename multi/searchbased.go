package multi

import (
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/dbw"

	"time"
)

type SearchDirectioner interface {
	SearchDirection(s *Solution, d []float64)
}

type SearchBased struct {
	sd SearchDirectioner
	ls uni.FdFOptimizer

	stats Stats
}

func NewSearchBased(sd SearchDirectioner, ls uni.FdFOptimizer) *SearchBased {
	return &SearchBased{
		sd: sd,
		ls: ls,
	}
}

func (sb *SearchBased) Stats() *Stats {
	return &sb.stats
}

func (sb *SearchBased) Optimize(o FdF, sol *Solution, upd ...Updater) Status {

	obj := Wrapper{Stats: &sb.stats, Func: o}
	sol.check(obj)

	if len(upd) == 0 {
		upd = append(upd, GradConv{1e-6})
	}

	initialTime := time.Now()

	s := 1.0 //initial step size

	x := dbw.NewVector(sol.X)
	g := dbw.NewVector(sol.Grad)
	d := dbw.NewVector(sol.LastDir)

	var status Status
	for ; status == NotTerminated; status = doUpdates(sol, &sb.stats, initialTime, upd) {

		s = 1.0

		sb.sd.SearchDirection(sol, d.Data)
		gLin := dbw.Dot(g, d)

		wolfe := uni.Wolfe{
			Armijo:    0.2,
			Curvature: 0.9,
			X0:        0,
			F0:        sol.Obj,
			Deriv0:    gLin,
		}

		lineFunc := NewLineFdF(obj, sol.X, d.Data)
		lsInit := uni.NewSolution()
		lsInit.Set(s)
		lsInit.SetLower(0, sol.Obj, gLin)
		lsRes := sb.ls.OptimizeFdF(lineFunc, lsInit, wolfe)
		if lsRes.Status < 0 {
			status = Status(lsRes.Status)
			break
		}
		s, sol.Obj = lsRes.X, lsRes.Obj

		dbw.Axpy(s, d, x)
		obj.DF(sol.X, sol.Grad)

	}
	return status
}
